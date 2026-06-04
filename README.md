# nextcloud-qdrant-pipeline

Automated RAG pipeline: Nextcloud inbox → PDF chunking → bge-m3 embedding → Qdrant vector store.

Drop a PDF into your Nextcloud "Qdrant Inbox" folder and it's automatically chunked, embedded, and stored — ready for semantic search.

## Architecture

```
Nextcloud WebDAV (Qdrant Inbox)
  ↓ sync_inbox.py polls every 30s
Local inbox/
  ↓ chunk_pdf.py
Section-aware JSON chunks
  ↓ ingest.py → Infinity bge-m3
1024-dim dense embeddings
  ↓ upsert to Qdrant
Vector collection (cosine similarity)
  ↓ query_qdrant.py
Semantic search results
```

## Components

| File | Purpose |
|------|---------|
| `sync_inbox.py` | WebDAV poll daemon — downloads PDFs, runs pipeline, routes to processed/ or failed/, sends Telegram notifications |
| `chunk_pdf.py` | Section-aware PDF chunker — PyMuPDF + font-size heuristics + LangChain RecursiveCharacterTextSplitter |
| `ingest.py` | Embed via Infinity bge-m3, upsert to Qdrant — supports dry-run, resume, gap-fill |
| `query_qdrant.py` | Semantic search — embeds query, searches Qdrant, outputs markdown/compact/JSON |
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

# 5. Ingest chunks into Qdrant
python ingest.py --chunks handbook_chunks.json --collection my-docs

# 6. Query the collection
python query_qdrant.py "What is the vacation policy?" --collection my-docs
```

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

The daemon polls your Nextcloud "Qdrant Inbox" folder every 30 seconds (configurable via `POLL_INTERVAL`). Successfully processed PDFs move to `processed/`, failures to `failed/`. Optional Telegram notifications on completion/failure.

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