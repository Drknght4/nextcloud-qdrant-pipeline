#!/usr/bin/env python3
"""
RAG Pipeline — PDF Chunker

Parses a policy handbook PDF into section-aware JSON chunks with metadata.
Designed for Qdrant ingestion.

Architecture:
  1. PyMuPDF (fitz) extracts text per page with font-size data
  2. Section headers detected via font size heuristic + regex patterns
  3. LangChain RecursiveCharacterTextSplitter splits oversized sections
  4. Output: JSON array of chunks with source_doc, page_number, section_title

Usage:
  python chunk_pdf.py handbook.pdf
  python chunk_pdf.py handbook.pdf -o output.json --chunk-size 512 --chunk-overlap 64
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Header Detection ──────────────────────────────────────────────

# Patterns that indicate a section header (ordered by specificity)
HEADER_PATTERNS = [
    # "SECTION 2.1: Title" / "Section 2.1 - Title" / "Sec. 2.1: Title"
    re.compile(
        r"^(?:SECTION|Section|Sec\.)\s+(\d+(?:\.\d+)*)\s*[:.\-–—]\s*(.+)$"
    ),
    # "2.1 Title Here" / "1.2.3 Some Policy Name"
    re.compile(r"^(\d+(?:\.\d+)+)\s+([A-Z][^\n]{2,80})$"),
    # "CHAPTER 3: Title" / "Chapter 3 - Title"
    re.compile(
        r"^(?:CHAPTER|Chapter)\s+(\d+)\s*[:.\-–—]?\s*(.+)$", re.IGNORECASE
    ),
    # ALL CAPS headers (policy docs love these)
    re.compile(r"^([A-Z][A-Z\s&/:,.!\-]{2,80})$"),
    # Title Case headers with at least 2 words
    re.compile(r"^([A-Z][a-z]+(?:\s+[A-Za-z][a-z]+){1,10})$"),
]

# Patterns that look like headers but are noise — reject these
NOISE_PATTERNS = [
    re.compile(r"^\d+$"),                      # bare numbers (page nums)
    re.compile(r"^[A-Z]\.$"),                   # single letter + period
    re.compile(r"^page\s+\d+", re.IGNORECASE),  # "Page 3"
    re.compile(r"^\d+\s*[-–—]\s*\d+$"),         # "3 - 7" (page ranges)
    re.compile(r"^P\s*a\s*g\s*e\s*\|\s*\d+$", re.IGNORECASE),  # "P a g e  |  5" (PDF footer)
    re.compile(r"^\d{7,}"),                     # document IDs like "1103195324"
]

# Minimum characters for a header to be considered valid
MIN_HEADER_LEN = 3

# Section titles to skip entirely (ToC, legal boilerplate, date artifacts)
SKIP_SECTIONS = {
    "contents", "table of contents", "index",
}

# Regex patterns for section titles to skip (applied after lowercase strip)
SKIP_SECTION_PATTERNS = [
    re.compile(r"^(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}$"),  # month-year running headers
]

# Prefixes indicating front-matter / ToC pages (skip these sections)
SKIP_SECTION_PREFIXES = (
    "contents",
)

# Page ranges to skip (1-indexed, inclusive). Default: skip ToC/front-matter.
# Most PDFs have ToC in the first few pages. Set to empty list to disable.
DEFAULT_SKIP_PAGES = (1, 7)  # Cover + ToC (pages 1-7 in most handbooks)


# ── PDF Extraction ─────────────────────────────────────────────────

def extract_blocks(pdf_path: Path) -> list[dict]:
    """
    Extract text blocks from PDF with font-size metadata.
    Each block: {text, font_size, page, is_bold}
    Strips running headers/footers (document IDs, page footers).
    """
    # Pattern for the running document ID header/footer (e.g., "1103195324\1AMERICAS")
    DOC_ID_PATTERN = re.compile(r"^\d{7,}")
    # Pattern for "P a g e  |  N" footer
    PAGE_FOOTER_PATTERN = re.compile(r"^P\s*a\s*g\s*e\s*\|\s*\d+$", re.IGNORECASE)

    doc = fitz.open(str(pdf_path))
    blocks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in text_dict["blocks"]:
            if block["type"] != 0:  # skip image blocks
                continue

            for line in block["lines"]:
                line_parts = []
                max_font_size = 0.0
                is_bold = False

                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    line_parts.append(text)
                    max_font_size = max(max_font_size, span["size"])
                    # Detect bold via font name
                    font_name = span.get("font", "").lower()
                    if "bold" in font_name or "heavy" in font_name or "black" in font_name:
                        is_bold = True

                line_text = " ".join(line_parts).strip()
                if not line_text:
                    continue

                # Skip running headers/footers
                if DOC_ID_PATTERN.match(line_text):
                    continue
                if PAGE_FOOTER_PATTERN.match(line_text):
                    continue

                blocks.append({
                    "text": line_text,
                    "font_size": round(max_font_size, 1),
                    "page": page_num + 1,  # 1-indexed
                    "is_bold": is_bold,
                })

    doc.close()
    return blocks


def compute_body_font_size(blocks: list[dict]) -> float:
    """Determine the body text font size (most common)."""
    if not blocks:
        return 12.0

    sizes = Counter(b["font_size"] for b in blocks)
    return sizes.most_common(1)[0][0]


# ── Section Detection ──────────────────────────────────────────────

def is_header(block: dict, body_font_size: float) -> bool:
    """Determine if a text block is a section header."""
    text = block["text"].strip()

    if len(text) < MIN_HEADER_LEN:
        return False

    # Reject known noise
    if any(p.match(text) for p in NOISE_PATTERNS):
        return False

    # Font-size heuristic: significantly larger than body = header
    font_ratio = block["font_size"] / body_font_size if body_font_size > 0 else 1.0
    is_large = font_ratio >= 1.15  # 15%+ larger than body

    # Pattern match: looks like a structured header
    matches_pattern = any(p.match(text) for p in HEADER_PATTERNS)

    # Bold + short text lines are often headers
    is_bold_short = block["is_bold"] and len(text) < 100

    # Need at least two signals for a header (reduce false positives)
    signals = sum([is_large, matches_pattern, is_bold_short])
    return signals >= 2 or (is_large and font_ratio >= 1.3)


def group_into_sections(blocks: list[dict], body_font_size: float) -> list[dict]:
    """
    Walk through blocks and group into sections.
    Each section: {section_title, start_page, content}
    """
    sections = []
    current_title = "Document Start"
    current_page = blocks[0]["page"] if blocks else 1
    section_lines = []

    for block in blocks:
        if is_header(block, body_font_size):
            # Flush current section
            if section_lines:
                content = "\n".join(section_lines).strip()
                if content:
                    sections.append({
                        "section_title": current_title,
                        "start_page": current_page,
                        "content": content,
                    })
            # Start new section
            current_title = block["text"].strip()
            current_page = block["page"]
            section_lines = []
        else:
            if not section_lines:
                current_page = block["page"]
            section_lines.append(block["text"].strip())

    # Flush final section
    if section_lines:
        content = "\n".join(section_lines).strip()
        if content:
            sections.append({
                "section_title": current_title,
                "start_page": current_page,
                "content": content,
            })

    return sections


# ── Chunk Splitting ────────────────────────────────────────────────

def split_sections(
    sections: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Split sections that exceed chunk_size into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    chunk_index = 0

    for section in sections:
        content = section["content"].strip()
        if not content:
            continue

        section_title = section["section_title"]

        if len(content) <= chunk_size:
            chunks.append({
                "chunk_index": chunk_index,
                "source_doc": "",  # filled by caller
                "section_title": section_title,
                "page_number": section["start_page"],
                "content": content,
            })
            chunk_index += 1
        else:
            splits = splitter.split_text(content)
            num_parts = len(splits)
            for i, split in enumerate(splits):
                part_label = f" (part {i + 1}/{num_parts})" if num_parts > 1 else ""
                chunks.append({
                    "chunk_index": chunk_index,
                    "source_doc": "",  # filled by caller
                    "section_title": f"{section_title}{part_label}",
                    "page_number": section["start_page"],
                    "content": split.strip(),
                })
                chunk_index += 1

    return chunks


# ── Main Pipeline ──────────────────────────────────────────────────

def chunk_pdf(
    pdf_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    skip_pages: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Full pipeline: PDF → blocks → sections → chunks."""
    blocks = extract_blocks(pdf_path)
    body_font_size = compute_body_font_size(blocks)

    # Filter out front-matter / ToC pages
    if skip_pages is None:
        skip_pages = [DEFAULT_SKIP_PAGES]

    skip_ranges = skip_pages or []
    if skip_ranges:
        before = len(blocks)
        blocks = [
            b for b in blocks
            if not any(lo <= b["page"] <= hi for lo, hi in skip_ranges)
        ]
        skipped = before - len(blocks)
        if skipped:
            print(f"  Skipped {skipped} blocks from front-matter pages {skip_ranges}")

    sections = group_into_sections(blocks, body_font_size)

    # Filter out ToC / index / date-header sections by title
    before_sections = len(sections)
    filtered_sections = []
    for s in sections:
        title_lower = s["section_title"].strip().lower()
        # Exact match skip list
        if title_lower in SKIP_SECTIONS:
            continue
        # Prefix match skip list
        if any(title_lower.startswith(pfx) for pfx in SKIP_SECTION_PREFIXES):
            continue
        # Regex pattern skip (e.g., month-year running headers)
        if any(p.match(title_lower) for p in SKIP_SECTION_PATTERNS):
            continue
        filtered_sections.append(s)
    sections = filtered_sections
    skipped_sections = before_sections - len(sections)
    if skipped_sections:
        print(f"  Skipped {skipped_sections} ToC/index sections")

    chunks = split_sections(sections, chunk_size, chunk_overlap)

    # Fill source_doc
    source_name = pdf_path.stem
    for chunk in chunks:
        chunk["source_doc"] = source_name

    return chunks


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chunk a PDF policy handbook into section-aware JSON for RAG ingestion."
    )
    parser.add_argument("pdf", type=Path, help="Input PDF file path")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output JSON path (default: <pdf_stem>_chunks.json in same dir)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512,
        help="Max characters per chunk (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=64,
        help="Overlap characters between chunks (default: 64)",
    )
    parser.add_argument(
        "--min-chunk-size", type=int, default=50,
        help="Discard chunks below this size (noise filter, default: 50)",
    )
    parser.add_argument(
        "--skip-pages", type=str, default=None,
        help="Page ranges to skip as 'lo-hi' pairs, comma-separated (default: '2-4' for front-matter). Example: '2-5,8-9'",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Print stats without writing output file",
    )
    args = parser.parse_args()

    # Validate input
    if not args.pdf.exists():
        print(f"Error: {args.pdf} not found", file=sys.stderr)
        sys.exit(1)

    if args.pdf.suffix.lower() != ".pdf":
        print(f"Error: {args.pdf} is not a PDF file", file=sys.stderr)
        sys.exit(1)

    # Parse skip-pages
    if args.skip_pages is None:
        skip_pages = [DEFAULT_SKIP_PAGES]
    elif args.skip_pages.strip().lower() in ("none", "0", "off", ""):
        skip_pages = []
    else:
        skip_pages = []
        for pair in args.skip_pages.split(","):
            lo, hi = pair.strip().split("-")
            skip_pages.append((int(lo.strip()), int(hi.strip())))

    output_path = args.output or args.pdf.with_name(f"{args.pdf.stem}_chunks.json")

    # Run pipeline
    print(f"Input:    {args.pdf}")
    print(f"Settings: chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}, skip_pages={skip_pages}")

    chunks = chunk_pdf(args.pdf, args.chunk_size, args.chunk_overlap, skip_pages=skip_pages)

    # Filter noise (tiny chunks)
    before = len(chunks)
    chunks = [c for c in chunks if len(c["content"]) >= args.min_chunk_size]
    filtered = before - len(chunks)

    if not chunks:
        print("\nError: No valid chunks produced. PDF may be image-based (needs OCR).", file=sys.stderr)
        sys.exit(1)

    # Stats
    total_chars = sum(len(c["content"]) for c in chunks)
    avg_chars = total_chars / len(chunks)
    sections = sorted(set(c["section_title"] for c in chunks))
    pages = sorted(set(c["page_number"] for c in chunks))

    print(f"\nResults:")
    print(f"  {len(chunks)} chunks from {len(sections)} sections, pages {pages[0]}–{pages[-1]}")
    print(f"  {total_chars:,} total chars, avg {avg_chars:.0f} chars/chunk")
    if filtered:
        print(f"  Filtered {filtered} chunks below {args.min_chunk_size} chars")

    print(f"\nSections detected:")
    for title in sections:
        count = sum(1 for c in chunks if c["section_title"] == title or c["section_title"].startswith(title + " (part"))
        print(f"  • {title} ({count} chunk{'s' if count > 1 else ''})")

    # Preview
    print(f"\nPreview (first 3 chunks):")
    for chunk in chunks[:3]:
        preview = chunk["content"][:150].replace("\n", " ")
        print(f"  [{chunk['chunk_index']}] p.{chunk['page_number']} — {chunk['section_title']}")
        print(f"      \"{preview}...\"")

    if len(chunks) > 3:
        print(f"  ... and {len(chunks) - 3} more")

    # Write output
    if not args.stats_only:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"\nOutput: {output_path} ({len(chunks)} chunks, {output_path.stat().st_size:,} bytes)")
    else:
        print(f"\n(stats-only mode, no file written)")


if __name__ == "__main__":
    main()