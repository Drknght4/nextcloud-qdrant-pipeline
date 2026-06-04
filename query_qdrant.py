#!/usr/bin/env python3
"""
RAG Pipeline — Qdrant Query

Embeds a natural-language query via bge-m3 and searches Qdrant for
relevant document chunks. Returns top-K results with metadata.

Usage:
  python query_qdrant.py "What is the vacation policy?"
  python query_qdrant.py "dress code" --limit 5 --threshold 0.6
  python query_qdrant.py "PTO accrual rate" --collection my-docs --format json
"""

import argparse
import json
import os
import sys

import requests
from qdrant_client import QdrantClient

# ── Configuration ──────────────────────────────────────────────────

INFINITY_URL = os.getenv("INFINITY_URL", "http://localhost:7997")
INFINITY_MODEL = os.getenv("INFINITY_MODEL", "BAAI/bge-m3")
INFINITY_EMBED_ENDPOINT = f"{INFINITY_URL}/embeddings"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "default")

DEFAULT_LIMIT = 5
DEFAULT_THRESHOLD = 0.45  # Cosine similarity — calibrated for bge-m3 dense + 512-char chunks
# Score distribution observed on EG America handbook:
#   0.70+  highly specific procedural matches
#   0.55-0.70  good policy matches
#   0.45-0.55  relevant but broad
#   below 0.45  increasingly tangential

# ── Embedding ──────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """Embed a query string via Infinity bge-m3."""
    response = requests.post(
        INFINITY_EMBED_ENDPOINT,
        json={"model": INFINITY_MODEL, "input": query},
        timeout=120,
    )
    response.raise_for_status()
    vec = response.json()["data"][0]["embedding"]
    if len(vec) != 1024:
        raise ValueError(f"Expected 1024 dims, got {len(vec)}. Model may need reload.")
    return vec


# ── Search ─────────────────────────────────────────────────────────

def search_qdrant(
    query_vector: list[float],
    collection: str = DEFAULT_COLLECTION,
    limit: int = DEFAULT_LIMIT,
    threshold: float = DEFAULT_THRESHOLD,
    source_filter: str | None = None,
) -> list[dict]:
    """Search Qdrant and return results above threshold."""
    client = QdrantClient(url=QDRANT_URL, timeout=30)

    # Build optional payload filter
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_filter = None
    if source_filter:
        query_filter = Filter(
            must=[FieldCondition(key="source_doc", match=MatchValue(value=source_filter))]
        )

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        score_threshold=threshold,
        with_payload=True,
    )

    hits = []
    for point in results.points:
        payload = point.payload
        hits.append({
            "score": round(point.score, 4),
            "section_title": payload.get("section_title", ""),
            "page_number": payload.get("page_number", 0),
            "source_doc": payload.get("source_doc", ""),
            "content": payload.get("content", ""),
        })

    return hits


# ── Formatting ────────────────────────────────────────────────────

def format_markdown(hits: list[dict], query: str) -> str:
    """Format results as readable Markdown."""
    if not hits:
        return f'No results found above threshold for: "{query}"'

    lines = [f'## Results for: "{query}"\n']
    for i, hit in enumerate(hits, 1):
        lines.append(f"### {i}. {hit['section_title']} (p.{hit['page_number']}) — score {hit['score']}")
        lines.append(f"**Source:** {hit['source_doc']}")
        lines.append(f"\n{hit['content']}\n")
        lines.append("---\n")
    return "\n".join(lines)


def format_compact(hits: list[dict], query: str) -> str:
    """Format results as compact context for LLM injection."""
    if not hits:
        return f'No relevant results found for: "{query}"'

    lines = [f"Query: {query}\nRelevant context from {hits[0]['source_doc']}:\n"]
    for hit in hits:
        lines.append(
            f"[{hit['section_title']}, p.{hit['page_number']}, score={hit['score']}] "
            f"{hit['content']}"
        )
    return "\n\n".join(lines)


def format_json(hits: list[dict]) -> str:
    """Format results as JSON."""
    return json.dumps(hits, indent=2, ensure_ascii=False)


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Search Qdrant for relevant document chunks using bge-m3 embeddings."
    )
    parser.add_argument("query", help="Natural language query to search for")
    parser.add_argument(
        "-c", "--collection", default=DEFAULT_COLLECTION,
        help=f"Qdrant collection (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "-k", "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Max results to return (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Minimum cosine similarity (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "-s", "--source", type=str, default=None,
        help="Filter by source_doc name (e.g., handbook filename without extension)",
    )
    parser.add_argument(
        "-f", "--format", choices=["markdown", "compact", "json"],
        default="compact",
        help="Output format (default: compact)",
    )
    args = parser.parse_args()

    # Embed
    try:
        query_vector = embed_query(args.query)
    except Exception as e:
        print(f"Error embedding query: {e}", file=sys.stderr)
        sys.exit(1)

    # Search
    try:
        hits = search_qdrant(
            query_vector=query_vector,
            collection=args.collection,
            limit=args.limit,
            threshold=args.threshold,
            source_filter=args.source,
        )
    except Exception as e:
        print(f"Error searching Qdrant: {e}", file=sys.stderr)
        sys.exit(1)

    # Format
    if args.format == "json":
        print(format_json(hits))
    elif args.format == "markdown":
        print(format_markdown(hits, args.query))
    else:
        print(format_compact(hits, args.query))

    # Exit code: 0 if results found, 1 if empty
    sys.exit(0 if hits else 1)


if __name__ == "__main__":
    main()