#!/usr/bin/env python3
"""
RAG Pipeline — Inbox Sync & Auto-Ingest Daemon  (v2)

Polls Nextcloud WebDAV for new PDFs in 'Qdrant Inbox', downloads them,
runs the full pipeline (chunk → embed via Infinity → upsert to Qdrant),
and routes files to processed/ or failed/ based on outcome.

v2: SHA256 hashing + duplicate detection. Before ingesting, compute the
PDF hash and check Qdrant metadata. Same hash → skip. Same filename but
different hash → delete old vectors and re-ingest.

Collection naming: filename stem, lowercased, hyphens for spaces.
  e.g. "Employee Handbook 2025.pdf" → collection "employee-handbook-2025"

Runs as a systemd user service: rag-inbox.service
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Configuration ──────────────────────────────────────────────────

NEXTCLOUD_URL = os.getenv("NEXTCLOUD_URL", "http://localhost:8667")
NEXTCLOUD_USER = os.getenv("NEXTCLOUD_USER", "")
NEXTCLOUD_PASSWORD = os.getenv("NEXTCLOUD_APP_PASSWORD", "")
NEXTCLOUD_INBOX = os.getenv("NEXTCLOUD_INBOX", "Qdrant Inbox")

INFINITY_URL = os.getenv("INFINITY_URL", "http://localhost:7997")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

BASE_DIR = Path(os.getenv("RAG_PIPELINE_DIR", Path.home() / "rag-pipeline"))
INBOX_DIR = BASE_DIR / "inbox"
PROCESSED_DIR = BASE_DIR / "processed"
FAILED_DIR = BASE_DIR / "failed"
CHUNK_SCRIPT = BASE_DIR / "chunk_pdf.py"
INGEST_SCRIPT = BASE_DIR / "ingest.py"

# Telegram notification (optional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))  # seconds between polls

# WebDAV base path
WEBDAV_BASE = f"{NEXTCLOUD_URL}/remote.php/dav/files/{NEXTCLOUD_USER}"

# ── Helpers ───────────────────────────────────────────────────────

DAV_NS = {"d": "DAV:"}


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def collection_name_from_filename(filename: str) -> str:
    """Derive Qdrant collection name from PDF filename.

    Rules:
      - Strip .pdf extension
      - Lowercase
      - Replace spaces/underscores with hyphens
      - Strip non-alphanumeric chars except hyphens
      - Collapse multiple hyphens
      - Remove leading/trailing hyphens
    """
    stem = Path(filename).stem
    name = stem.lower()
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"[^a-z0-9\-]", "", name)
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip("-")
    return name or "untitled"


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hash_in_qdrant(collection: str, file_hash: str) -> str | None:
    """Check if a hash already exists in Qdrant collection metadata.

    Returns:
      - "match" if the hash exists (exact duplicate — skip)
      - "conflict" if the collection has vectors for this source_doc
        but with a different hash (re-ingest needed)
      - None if no existing vectors found for this source_doc
    """
    try:
        resp = requests.get(
            f"{QDRANT_URL}/collections/{collection}", timeout=10
        )
        if resp.status_code != 200:
            return None
        points_count = resp.json()["result"]["points_count"]
        if points_count == 0:
            return None
    except Exception:
        return None

    # Scroll through points with source_doc filter to find existing hash
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=30)
        existing_hash = None
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source_doc", match=MatchValue(value=collection))]
                ),
                limit=10,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                if point.payload and "hash" in point.payload:
                    existing_hash = point.payload["hash"]
                    break
            if existing_hash or offset is None:
                break

        if existing_hash is None:
            return None
        elif existing_hash == file_hash:
            return "match"
        else:
            return "conflict"
    except Exception as e:
        log(f"  Hash check query failed: {e}")
        return None


def delete_vectors_by_source(client, collection: str, source_doc: str) -> int:
    """Delete all vectors matching source_doc from a collection.
    Returns the number of points deleted.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList

    # Scroll to find all matching point IDs
    point_ids = []
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="source_doc", match=MatchValue(value=source_doc))]
            ),
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        point_ids.extend(p.id for p in results)
        if offset is None:
            break

    if point_ids:
        client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=point_ids),
        )
    return len(point_ids)


def send_telegram(message: str) -> bool:
    """Send notification via Telegram bot. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram credentials not configured — skipping notification")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            log("Telegram notification sent")
            return True
        else:
            log(f"Telegram failed: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        log(f"Telegram error: {e}")
        return False


# ── WebDAV Sync ──────────────────────────────────────────────────


def webdav_list_pdfs() -> list[dict]:
    """List PDF files in the Nextcloud inbox via PROPFIND.
    Returns list of {filename, href} dicts.
    """
    url = f"{WEBDAV_BASE}/{urllib.parse.quote(NEXTCLOUD_INBOX)}/"
    try:
        resp = requests.request(
            "PROPFIND",
            url,
            auth=(NEXTCLOUD_USER, NEXTCLOUD_PASSWORD),
            headers={"Depth": "1"},
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        log(f"WebDAV PROPFIND failed: {e}")
        return []

    files = []
    try:
        root = ET.fromstring(resp.text)
        for response in root.findall("d:response", DAV_NS):
            href_el = response.find("d:href", DAV_NS)
            if href_el is None or href_el.text is None:
                continue
            href = href_el.text
            # Decode the URL-encoded path
            decoded = urllib.parse.unquote(href)
            filename = decoded.rstrip("/").split("/")[-1]

            # Skip the directory itself and non-PDF files
            if not filename.lower().endswith(".pdf"):
                continue

            # Check content type to confirm it's a file, not a collection
            restype = response.find(".//d:resourcetype", DAV_NS)
            if restype is not None and restype.find("{DAV:}collection") is not None:
                continue

            files.append({"filename": filename, "href": href})
    except ET.ParseError:
        log("Failed to parse WebDAV response XML")

    return files


def webdav_download(filename: str, href: str) -> bool:
    """Download a file from WebDAV to local inbox. Returns True on success."""
    download_url = f"{NEXTCLOUD_URL}{href}"
    local_path = INBOX_DIR / filename

    try:
        with requests.get(
            download_url,
            auth=(NEXTCLOUD_USER, NEXTCLOUD_PASSWORD),
            stream=True,
            timeout=60,
        ) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        log(f"Download failed for {filename}: {e}")
        if local_path.exists():
            local_path.unlink()
        return False


def sync_inbox() -> list[str]:
    """Sync new PDFs from Nextcloud to local inbox.
    Returns list of newly downloaded filenames.
    """
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    remote_files = webdav_list_pdfs()
    if not remote_files:
        return []

    # Build sets of local files (inbox + processed + failed) to avoid re-download
    local_exists = set()
    for d in (INBOX_DIR, PROCESSED_DIR, FAILED_DIR):
        if d.exists():
            local_exists.update(f.name for f in d.glob("*.pdf"))

    new_files = []
    for entry in remote_files:
        filename = entry["filename"]
        if filename in local_exists:
            continue
        log(f"Downloading: {filename}")
        if webdav_download(filename, entry["href"]):
            new_files.append(filename)
            log(f"Downloaded: {filename}")

    return new_files


# ── Pipeline ───────────────────────────────────────────────────────


def run_pipeline(pdf_filename: str) -> tuple[bool, str, str]:
    """Run chunk → embed → upsert for a single PDF.

    Returns (success: bool, collection_name: str, status: str).
    Status is one of: "ingested", "skipped_duplicate", "re_ingested", "failed".
    """
    collection = collection_name_from_filename(pdf_filename)
    pdf_path = INBOX_DIR / pdf_filename
    chunks_path = INBOX_DIR / f"{Path(pdf_filename).stem}_chunks.json"

    log(f"Pipeline start: {pdf_filename} → collection '{collection}'")

    # Step 0: Compute SHA256 and check for duplicates
    file_hash = sha256_file(pdf_path)
    log(f"  [0/4] SHA256: {file_hash[:16]}...")

    hash_status = check_hash_in_qdrant(collection, file_hash)
    if hash_status == "match":
        log(f"  ✓ Duplicate detected — same hash already in '{collection}'. Skipping.")
        return True, collection, "skipped_duplicate"
    elif hash_status == "conflict":
        log(f"  ⚠ Hash mismatch — collection '{collection}' exists with different content. Deleting old vectors...")
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, timeout=30)
        deleted = delete_vectors_by_source(client, collection, collection)
        log(f"  ✓ Deleted {deleted} old vectors from '{collection}'. Re-ingesting.")

    # Step 1: Chunk the PDF
    log(f"  [1/4] Chunking {pdf_filename}...")
    chunk_result = subprocess.run(
        [
            sys.executable,
            str(CHUNK_SCRIPT),
            str(pdf_path),
            "-o",
            str(chunks_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if chunk_result.returncode != 0:
        log(f"  ✗ Chunking failed: {chunk_result.stderr[-500:]}")
        return False, collection, "failed"
    log(f"  ✓ Chunked → {chunks_path.name}")

    # Step 2+3: Embed + Upsert via ingest.py (with --hash flag)
    log(f"  [2/4] Embedding + [3/4] Upserting to Qdrant...")
    ingest_result = subprocess.run(
        [
            sys.executable,
            str(INGEST_SCRIPT),
            "--chunks", str(chunks_path),
            "--collection", collection,
            "--hash", file_hash,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if ingest_result.returncode != 0:
        log(f"  ✗ Ingest failed: {ingest_result.stderr[-500:]}")
        return False, collection, "failed"
    log(f"  ✓ Ingested into '{collection}'")

    status = "re_ingested" if hash_status == "conflict" else "ingested"
    return True, collection, status


def process_file(pdf_filename: str) -> None:
    """Process a single PDF: run pipeline, route to processed/ or failed/, notify."""
    start = time.time()
    success, collection, status = run_pipeline(pdf_filename)
    elapsed = time.time() - start

    pdf_path = INBOX_DIR / pdf_filename
    chunks_path = INBOX_DIR / f"{Path(pdf_filename).stem}_chunks.json"

    # Always move out of inbox (even duplicates — they've been processed)
    dest_dir = PROCESSED_DIR if success else FAILED_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_pdf = dest_dir / pdf_filename
    if dest_pdf.exists():
        dest_pdf.unlink()
    shutil.move(str(pdf_path), str(dest_pdf))
    if chunks_path.exists():
        dest_chunks = dest_dir / chunks_path.name
        shutil.move(str(chunks_path), str(dest_chunks))

    if success:
        if status == "skipped_duplicate":
            log(f"→ Skipped (duplicate): {pdf_filename}")
            send_telegram(
                f"⏭ *RAG Ingest Skipped*\\n"
                f"📄 `{pdf_filename}`\\n"
                f"📦 Collection: `{collection}`\\n"
                f"ℹ Already ingested (same hash)"
            )
            return

        log(f"→ Moved to processed/{pdf_filename} ({elapsed:.1f}s)")

        # Get point count
        points_count = "N/A"
        try:
            resp = requests.get(
                f"{QDRANT_URL}/collections/{collection}", timeout=10
            )
            if resp.status_code == 200:
                points_count = resp.json()["result"]["points_count"]
        except Exception:
            pass

        label = "Re-ingested" if status == "re_ingested" else "Ingest Complete"
        emoji = "🔄" if status == "re_ingested" else "✅"
        send_telegram(
            f"{emoji} *RAG {label}*\\n"
            f"📄 `{pdf_filename}`\\n"
            f"📦 Collection: `{collection}`\\n"
            f"📊 Points: {points_count}\\n"
            f"⏱ {elapsed:.1f}s"
        )
    else:
        log(f"→ Moved to failed/{pdf_filename}")
        send_telegram(
            f"❌ *RAG Ingest Failed*\\n"
            f"📄 `{pdf_filename}`\\n"
            f"📦 Collection: `{collection}`\\n"
            f"Check logs: `journalctl --user -u rag-inbox`"
        )


# ── Main Loop ─────────────────────────────────────────────────────


def process_existing_files() -> None:
    """Process any PDFs already in the local inbox (from previous crashes/restarts)."""
    pdfs = sorted(INBOX_DIR.glob("*.pdf"))
    if pdfs:
        log(f"Found {len(pdfs)} existing PDF(s) in inbox — processing...")
        for pdf in pdfs:
            process_file(pdf.name)


def main() -> None:
    log("RAG Inbox Sync starting (v2 — hash-based dedup)")
    log(f"  Nextcloud: {NEXTCLOUD_URL} (inbox: '{NEXTCLOUD_INBOX}')")
    log(f"  Infinity:  {INFINITY_URL}")
    log(f"  Qdrant:    {QDRANT_URL}")
    log(f"  Python:    {sys.executable}")
    log(f"  Local:     {INBOX_DIR}")
    log(f"  Poll:      {POLL_INTERVAL}s")

    # Create directories
    for d in (INBOX_DIR, PROCESSED_DIR, FAILED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Process any leftover files from a previous run
    process_existing_files()

    # Main poll loop
    log(f"Polling Nextcloud inbox every {POLL_INTERVAL}s...")
    try:
        while True:
            try:
                new_files = sync_inbox()
                for filename in new_files:
                    process_file(filename)
            except Exception as e:
                log(f"Poll error: {e}")
                traceback.print_exc()
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        log("Shutting down (KeyboardInterrupt)")


if __name__ == "__main__":
    main()