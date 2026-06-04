# nextcloud-qdrant-pipeline

Automated RAG pipeline: Nextcloud inbox → PDF chunking → bge-m3 embedding → Qdrant vector store.

Drop a PDF into your Nextcloud "Qdrant Inbox" folder and it's automatically chunked, embedded, and stored — ready for semantic search. SHA256 dedup prevents re-ingesting unchanged files, and enriched metadata (hash + timestamp) enables auditability.

## Architecture

```
Nextcloud WebDAV (Qdrant Inbox)
  ↓ sync_inbox.py polls every 30s
  ↓ SHA256 hash computed → checked against Qdrant metadata
  ↓ same hash → skip | same filename, different hash → delete old + re-ingest
Local inbox/
  ↓ chunk_pdf.py
Section-aware JSON chunks
  ↓ ingest.py → Infinity bge-m3
1024-dim dense embeddings + enriched metadata (hash, ingested_at)
  ↓ upsert to Qdrant
Vector collection (cosine similarity)
  ↓ query_qdrant.py
Semantic search results
```

## Components

| File | Purpose |
|------|---------|
| `sync_inbox.py` | WebDAV poll daemon — SHA256 dedup, downloads PDFs, runs pipeline, routes to processed/ or failed/, Telegram notifications |
| `chunk_pdf.py` | Section-aware PDF chunker — PyMuPDF + font-size heuristics + LangChain RecursiveCharacterTextSplitter |
| `ingest.py` | Embed via Infinity bge-m3, upsert to Qdrant with enriched metadata — supports dry-run, resume, gap-fill, --hash |
| `query_qdrant.py` | Semantic search — embeds query, searches Qdrant, outputs markdown/compact/JSON |
| `purge.py` | Delete vectors by source document or wipe entire collections — confirmation prompt required |
| `rag-inbox.service` | systemd user unit for the sync daemon |
| `requirements.txt` | Python dependencies |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and configure environment
cp .env.example .env
# Edit .env with your Nextcloud, Infinity, and Qdrant URLs + credentials

# 3. Test connectivity
python ingest.py --dry-run

# 4. Chunk a PDF manually
python chunk_pdf.py handbook.pdf

# 5. Ingest chunks into Qdrant (with hash for dedup)
python ingest.py --chunks handbook_chunks.json --collection my-docs --hash abc123...

# 6. Query the collection
python query_qdrant.py "What is the vacation policy?" --collection my-docs
```

## SHA256 Dedup

Before ingesting, `sync_inbox.py` computes the SHA256 hash of each PDF and checks it against existing Qdrant metadata:

- **Same hash** → skip. No re-ingest, no duplicate vectors. Logged and notified via Telegram.
- **Same filename, different hash** → the PDF was updated. Old vectors are deleted, new ones ingested. Logged as "re-ingested" with a 🔁 notification.
- **No existing hash** → normal first-time ingest.

The hash is passed to `ingest.py` via `--hash` and stored in every chunk's payload alongside an `ingested_at` ISO timestamp. This enables:

- Audit trail: when was each chunk ingested, from which version of the PDF
- Manual dedup checks: query Qdrant by hash to see if a document version already exists
- Re-ingest detection: compare hashes across source_docs to find stale data

## Enriched Metadata

Every chunk in Qdrant now carries:

```json
{
  "source_doc": "employee_handbook_2024",
  "section_title": "3.2 Paid Time Off",
  "page_number": 15,
  "content": "Full-time employees accrue PTO at a rate of...",
  "hash": "a3f2b8c1d4e5...",
  "ingested_at": "2026-06-04T21:30:00.123456+00:00"
}
```

- `hash` — SHA256 of the source PDF (set via `--hash` flag on ingest.py)
- `ingested_at` — ISO 8601 UTC timestamp of when the chunk was upserted

## Purge Script

```bash
# Delete all vectors for a source document (auto-derives collection name from filename)
python purge.py --source employee_handbook_2024

# Delete by source with explicit collection
python purge.py --source employee_handbook_2024 --collection-override my-custom-collection

# Wipe an entire collection
python purge.py --collection my-docs

# Skip confirmation prompt (for scripts/automation)
python purge.py --collection old-data --yes
```

Both modes show a count of affected points and require confirmation (`[y/N]`) before proceeding. Use `--yes` or `-y` to skip the prompt.

## Automated Ingest (Daemon)

Run `sync_inbox.py` as a systemd service to auto-ingest any PDF dropped into your Nextcloud inbox:

```bash
# Install the service
cp rag-inbox.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now rag-inbox

# Check logs
journalctl --user -u rag-inbox -f
```

The daemon polls your Nextcloud "Qdrant Inbox" folder every 30 seconds (configurable via `POLL_INTERVAL`). SHA256 dedup runs automatically. Successfully processed PDFs move to `processed/`, failures to `failed/`, duplicates are skipped. Telegram notifications for all outcomes.

Telegram notification states:
- ✅ **Ingest Complete** — first-time ingest
- 🔁 **Re-ingested** — same filename, different hash (updated PDF)
- ⏭ **Skipped** — exact duplicate (same hash)
- ❌ **Failed** — chunking or embedding error

## Chunking Strategy

`chunk_pdf.py` uses a multi-signal approach for policy/HR handbooks:

1. **PyMuPDF** extracts text blocks with font-size metadata per page
2. **Header detection** — font size ratio ≥ 1.15× body + pattern match + bold/short (need 2+ signals to reduce false positives)
3. **Noise rejection** — page numbers, document IDs, "P a g e | N" footers, ToC sections
4. **Section grouping** — blocks between headers form a section, preserving document structure
5. **LangChain** splits oversized sections with `\n\n → \n → . → space` separator priority
6. **Noise filter** — discards chunks below 50 chars

Output per chunk:
```json
{
  "chunk_index": 0,
  "source_doc": "employee_handbook_2024",
  "section_title": "3.2 Paid Time Off",
  "page_number": 15,
  "content": "Full-time employees accrue PTO at a rate of..."
}
```

## Embedding

Uses [Infinity](https://github.com/michaelfeil/infinity) serving **BAAI/bge-m3** (1024-dim dense vectors). bge-m3 also supports sparse vectors for hybrid search — Infinity currently returns dense only. For future hybrid search, either run bge-m3 via FlagEmbedding directly or use Qdrant's built-in BM25 sparse indexing.

## Query

```bash
# Basic
python query_qdrant.py "dress code policy"

# Top 10 results, higher threshold
python query_qdrant.py "PTO accrual" --limit 10 --threshold 0.6

# JSON output for programmatic use
python query_qdrant.py "benefits" --format json

# Filter by source document
python query_qdrant.py "holidays" --source employee_handbook_2024
```

Score distribution (bge-m3 dense + 512-char chunks):
- **0.70+** — highly specific procedural matches
- **0.55–0.70** — good policy matches
- **0.45–0.55** — relevant but broad
- **below 0.45** — increasingly tangential

## Requirements

- **Nextcloud** with WebDAV enabled (for inbox sync)
- **Infinity** serving bge-m3 (for embeddings)
- **Qdrant** (for vector storage)
- Python 3.11+

## License

MIT

---

Built by [Drknght4](https://github.com/Drknght4)