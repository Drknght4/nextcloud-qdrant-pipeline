#!/usr/bin/env python3
"""
RAG Pipeline — Purge Script

Delete vectors from Qdrant by source document or entire collection.
Confirmation prompt before any destructive operation.

Usage:
  python purge.py --source handbook.pdf          # Delete all vectors for a source_doc
  python purge.py --source handbook.pdf --collection my-docs  # Specify collection
  python purge.py --collection my-docs            # Wipe an entire collection
  python purge.py --collection my-docs --yes      # Skip confirmation prompt
"""

import argparse
import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList

# ── Configuration ──────────────────────────────────────────────────

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def get_client() -> QdrantClient:
    """Create a Qdrant client."""
    return QdrantClient(url=QDRANT_URL, timeout=60)


def purge_by_source(client: QdrantClient, collection: str, source_doc: str) -> int:
    """Delete all vectors matching source_doc from a collection.
    Returns the number of points deleted.
    """
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

    if not point_ids:
        print(f"No vectors found for source_doc='{source_doc}' in collection '{collection}'.")
        return 0

    client.delete(
        collection_name=collection,
        points_selector=PointIdsList(points=point_ids),
    )
    return len(point_ids)


def purge_collection(client: QdrantClient, collection: str) -> bool:
    """Delete an entire collection. Returns True on success."""
    collections = [c.name for c in client.get_collections().collections]
    if collection not in collections:
        print(f"Collection '{collection}' does not exist.")
        return False

    client.delete_collection(collection_name=collection)
    return True


def confirm(prompt: str) -> bool:
    """Ask for user confirmation. Returns True if user types 'y' or 'yes'."""
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in ("y", "yes")


def main():
    parser = argparse.ArgumentParser(
        description="Purge vectors from Qdrant by source document or entire collection."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-s", "--source", type=str,
        help="Delete all vectors matching this source_doc (filename stem)",
    )
    group.add_argument(
        "-c", "--collection", type=str,
        help="Delete an entire Qdrant collection",
    )
    parser.add_argument(
        "--collection-override", type=str, default=None,
        help="Collection to search when using --source (required if collection name differs from source doc stem)",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    client = get_client()

    if args.collection:
        # Wipe entire collection
        collection = args.collection
        try:
            info = client.get_collection(collection_name=collection)
            count = info.points_count
        except Exception:
            print(f"Collection '{collection}' not found.")
            sys.exit(1)

        print(f"Collection: {collection}")
        print(f"  Points: {count}")
        print()

        if not args.yes:
            if not confirm(f"Delete entire collection '{collection}' ({count} points)?"):
                print("Cancelled.")
                return

        success = purge_collection(client, collection)
        if success:
            print(f"✓ Collection '{collection}' deleted.")
        else:
            print(f"✗ Failed to delete collection '{collection}'.")
            sys.exit(1)

    elif args.source:
        # Delete by source_doc
        collection = args.collection_override or args.source.lower().replace(" ", "-").replace("_", "-")
        # Remove extension if provided
        source_doc = args.source.replace(".pdf", "") if args.source.endswith(".pdf") else args.source

        # Count matching points
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

        if not point_ids:
            print(f"No vectors found for source_doc='{source_doc}' in collection '{collection}'.")
            sys.exit(0)

        print(f"Collection: {collection}")
        print(f"  source_doc: {source_doc}")
        print(f"  Points to delete: {len(point_ids)}")
        print()

        if not args.yes:
            if not confirm(f"Delete {len(point_ids)} vectors for '{source_doc}' from '{collection}'?"):
                print("Cancelled.")
                return

        deleted = purge_by_source(client, collection, source_doc)
        print(f"✓ Deleted {deleted} vectors for '{source_doc}' from '{collection}'.")


if __name__ == "__main__":
    main()