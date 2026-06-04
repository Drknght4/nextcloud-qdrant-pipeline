#!/usr/bin/env python3
"""
RAG Pipeline — Ingestion Script  (v2)

Reads chunked JSON, embeds via bge-m3 on Infinity, upserts to Qdrant.

Usage:
  python ingest.py                          # Full ingestion
  python ingest.py --dry-run                 # Validate connectivity + first embedding
  python ingest.py --batch-size 50          # Custom batch size for Qdrant upserts
  python ingest.py --chunks other.json       # Use a different chunks file

Architecture:
  • Embed: bge-m3 via Infinity (dense, 1024 dims)
  • Store: Qdrant (cosine similarity)
  • Payload: source_doc, section_title, page_number, content

Note: bge-m3 also supports sparse vectors for hybrid search. Infinity's
/embeddings endpoint returns dense vectors only. To enable hybrid (dense+sparse)
search later, you'll need to either:
  1. Run bge-m3 via FlagEmbedding library directly for sparse output, or
  2. Use Qdrant's built-in BM25 sparse indexing on the chunk text payload.
  For now, this script stores dense vectors + full payload for future hybrid upgrade.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import qdrant_client
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
import requests

# ── Configuration ──────────────────────────────────────────────────

INFINITY_URL = os.getenv("INFINITY_URL", "http://localhost:7997")
INFINITY_MODEL = os.getenv("INFINITY_MODEL", "BAAI/bge-m3")
INFINITY_EMBED_ENDPOINT = f"{INFINITY_URL}/embeddings"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "default")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))  # bge-m3 dense output dimension
BATCH_SIZE = 50     # Qdrant upsert batch size
MAX_RETRIES = 3     # Retry count for connection errors
RETRY_DELAY = 5     # Seconds between retries

DEFAULT_CHUNKS_FILE = Path.home() / "rag-pipeline" / "handbook_chunks.json"


# ── Infinity Embedding ────────────────────────────────────────────

def embed_text(text: str, model: str = INFINITY_MODEL) -> list[float]:
    """Embed a single text string via Infinity's OpenAI-compatible /embeddings endpoint.
    Retries on connection errors only — Infinity is stable under load.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                INFINITY_EMBED_ENDPOINT,
                json={"model": model, "input": text},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * attempt
                print(f"  ⚠ Embedding error (attempt {attempt}/{MAX_RETRIES}): {e}")
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def embed_batch(texts: list[str], model: str = INFINITY_MODEL) -> list[list[float] | None]:
    """Embed multiple texts via Infinity. Returns list of embeddings, None for failures."""
    embeddings: list[list[float] | None] = []
    for i, text in enumerate(texts):
        try:
            embeddings.append(embed_text(text, model))
        except Exception as e:
            print(f"  ✗ Failed to embed chunk {i}/{len(texts)}: {e}")
            embeddings.append(None)
        if (i + 1) % 25 == 0:
            ok = sum(1 for e in embeddings if e is not None)
            print(f"  Embedded {i + 1}/{len(texts)} chunks ({ok} ok, {i + 1 - ok} failed)")
    return embeddings


# ── Qdrant ─────────────────────────────────────────────────────────

def get_qdrant_client() -> qdrant_client.QdrantClient:
    """Create a Qdrant client pointing at the configured server."""
    return qdrant_client.QdrantClient(url=QDRANT_URL, timeout=120)


def ensure_collection(client: qdrant_client.QdrantClient, collection_name: str = QDRANT_COLLECTION) -> None:
    """Create the collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        print(f"Collection '{collection_name}' exists — will upsert into it.")
    else:
        print(f"Creating collection '{collection_name}' (vector_size={VECTOR_SIZE}, distance=Cosine)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Collection created.")


def upsert_batch(
    client: qdrant_client.QdrantClient,
    chunks: list[dict],
    embeddings: list[list[float] | None],
    collection_name: str = QDRANT_COLLECTION,
) -> tuple[int, int]:
    """Upsert a batch of points into Qdrant. Skips failed (None) embeddings.
    Returns (upserted, skipped) counts.
    """
    points = []
    skipped = 0
    for chunk, embedding in zip(chunks, embeddings):
        if embedding is None:
            skipped += 1
            continue
        points.append(
            PointStruct(
                id=chunk["chunk_index"],
                vector=embedding,
                payload={
                    "source_doc": chunk["source_doc"],
                    "section_title": chunk["section_title"],
                    "page_number": chunk["page_number"],
                    "content": chunk["content"],
                },
            )
        )
    if points:
        client.upsert(collection_name=collection_name, points=points)
    return len(points), skipped


# ── Connectivity Checks ───────────────────────────────────────────

def check_infinity() -> bool:
    """Verify Infinity is reachable and bge-m3 model is loaded."""
    print(f"Checking Infinity at {INFINITY_URL}...")
    try:
        # Health check
        resp = requests.get(f"{INFINITY_URL}/health", timeout=10)
        resp.raise_for_status()
        print(f"  ✓ Infinity health check passed")

        # Model check — Infinity serves /models, not /v1/models
        models_resp = requests.get(f"{INFINITY_URL}/models", timeout=10)
        models_resp.raise_for_status()
        model_ids = [m["id"] for m in models_resp.json().get("data", [])]
        found = INFINITY_MODEL in model_ids or any(INFINITY_MODEL in m for m in model_ids)
        if found:
            print(f"  ✓ Infinity reachable, {INFINITY_MODEL} model available")
            return True
        else:
            print(f"  ✗ Infinity reachable, but {INFINITY_MODEL} not found in models:")
            for m in model_ids:
                print(f"    - {m}")
            return False
    except requests.RequestException as e:
        print(f"  ✗ Cannot reach Infinity at {INFINITY_URL}: {e}")
        return False


def check_qdrant() -> bool:
    """Verify Qdrant is reachable."""
    print(f"Checking Qdrant at {QDRANT_URL}...")
    try:
        resp = requests.get(f"{QDRANT_URL}/collections", timeout=10)
        resp.raise_for_status()
        collections = resp.json().get("result", {}).get("collections", [])
        print(f"  ✓ Qdrant reachable, {len(collections)} existing collections")
        return True
    except requests.RequestException as e:
        print(f"  ✗ Cannot reach Qdrant at {QDRANT_URL}: {e}")
        return False


def test_embedding() -> bool:
    """Test embedding a sample string and print the shape."""
    print(f"Testing embedding via {INFINITY_MODEL}...")
    try:
        start = time.time()
        vec = embed_text("Test embedding for RAG pipeline validation.")
        elapsed = time.time() - start
        print(f"  ✓ Embedding shape: ({len(vec)},)")
        print(f"  ✓ Dimensions: {len(vec)} (expected {VECTOR_SIZE})")
        print(f"  ✓ Sample values: [{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}, ...]")
        print(f"  ✓ Latency: {elapsed:.2f}s")

        if len(vec) != VECTOR_SIZE:
            print(f"  ✗ Dimension mismatch! Got {len(vec)}, expected {VECTOR_SIZE}")
            print(f"  Update VECTOR_SIZE in the script or .env and re-run.")
            return False
        return True
    except Exception as e:
        print(f"  ✗ Embedding failed: {e}")
        return False


# ── Main Pipeline ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest chunked JSON into Qdrant via bge-m3 embeddings (Infinity)."
    )
    parser.add_argument(
        "--chunks", type=Path, default=DEFAULT_CHUNKS_FILE,
        help=f"Path to chunks JSON file (default: {DEFAULT_CHUNKS_FILE})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Qdrant upsert batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--fill-gaps", action="store_true",
        help="Check Qdrant for missing chunk IDs and re-ingest only those",
    )
    parser.add_argument(
        "--resume-from", type=int, default=0,
        help="Skip chunks with chunk_index < N (useful for resuming after failures)",
    )
    parser.add_argument(
        "--collection", type=str, default=None,
        help=f"Qdrant collection name (default: {QDRANT_COLLECTION})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate connectivity + test embedding, don't write to Qdrant",
    )
    args = parser.parse_args()

    # Allow CLI override of collection name
    collection_name = args.collection or QDRANT_COLLECTION

    # ── Dry Run ────────────────────────────────────────────
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — Validating connectivity and embedding")
        print("=" * 60)
        print()

        all_ok = True
        all_ok &= check_infinity()
        all_ok &= check_qdrant()
        all_ok &= test_embedding()

        if all_ok:
            print()
            print("✓ All checks passed. Ready for ingestion.")
            print(f"  Run without --dry-run to ingest {args.chunks}")
        else:
            print()
            print("✗ Some checks failed. Fix the issues above before ingesting.")
            sys.exit(1)
        return

    # ── Full Ingestion ──────────────────────────────────────
    # Load chunks
    if not args.chunks.exists():
        print(f"Error: {args.chunks} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.chunks, encoding="utf-8") as f:
        chunks = json.load(f)

    # Fill-gaps mode: only re-ingest missing chunk IDs
    if args.fill_gaps:
        client = get_qdrant_client()
        ensure_collection(client, collection_name)
        # Scroll through all existing points to find which IDs are present
        existing_ids = set()
        offset = None
        while True:
            result, offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=False,
            )
            existing_ids.update(p.id for p in result)
            if offset is None:
                break
        all_ids = set(c["chunk_index"] for c in chunks)
        missing_ids = sorted(all_ids - existing_ids)
        print(f"Existing: {len(existing_ids)} | Total needed: {len(all_ids)} | Missing: {len(missing_ids)}")
        if not missing_ids:
            print("✓ No gaps — all chunks already in Qdrant.")
            return
        chunks = [c for c in chunks if c["chunk_index"] in set(missing_ids)]
        print(f"Filling {len(chunks)} gaps: IDs {missing_ids[0]}–{missing_ids[-1]}")

    # Resume: skip already-ingested chunks
    if args.resume_from > 0:
        before = len(chunks)
        chunks = [c for c in chunks if c["chunk_index"] >= args.resume_from]
        print(f"Resuming from chunk_index >= {args.resume_from}: skipping {before - len(chunks)} already-ingested chunks")
        if not chunks:
            print("No chunks remaining to ingest. Done.")
            return

    print(f"Loaded {len(chunks)} chunks from {args.chunks}")
    print(f"  Source doc: {chunks[0]['source_doc']}")
    print(f"  Pages: {chunks[0]['page_number']}–{chunks[-1]['page_number']}")
    print()

    # Connect to Qdrant
    client = get_qdrant_client()
    ensure_collection(client, collection_name)

    # Embed and upsert in batches
    total = len(chunks)
    batch_size = args.batch_size
    total_start = time.time()
    total_upserted = 0
    total_skipped = 0

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"Batch {batch_num}/{total_batches}: embedding {len(batch_chunks)} chunks...")

        batch_start = time.time()
        embeddings = embed_batch([c["content"] for c in batch_chunks])
        embed_time = time.time() - batch_start

        upsert_start = time.time()
        upserted, skipped = upsert_batch(client, batch_chunks, embeddings, collection_name)
        upsert_time = time.time() - upsert_start

        elapsed = time.time() - batch_start
        print(f"  Embed: {embed_time:.1f}s | Upsert: {upsert_time:.1f}s | Upserted: {upserted}, Skipped: {skipped} | Total: {elapsed:.1f}s")
        total_upserted += upserted
        total_skipped += skipped

    total_time = time.time() - total_start
    print()
    print(f"✓ Ingestion complete: {total_upserted} upserted, {total_skipped} skipped in {total_time:.1f}s")
    print(f"  Collection: {collection_name}")
    if total_upserted > 0:
        print(f"  Avg: {total_time/total_upserted:.2f}s/chunk")

    # Verify
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"  Points in collection: {collection_info.points_count}")

if __name__ == "__main__":
    main()