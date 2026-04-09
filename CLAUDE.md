# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Key Commands

```bash
# Ingest/re-embed compliance data from Excel (run once, or after Excel changes)
python data_embedding.py

# Ingest PPT files into the same collection (run after data_embedding.py)
python data_embedding_ppt.py

# Ingest ISO 42001 reference content (controls list, terms, objectives, risk sources)
python data_embedding_ref.py

# Run the Streamlit app
streamlit run app.py

# Download models from HuggingFace (edit db.py for the target repo)
python db.py

# Activate virtual environment
source .venv/bin/activate
```

No test suite or linter is configured.

## Architecture

The system has two independent phases:

### Phase 1 — Ingestion (`data_embedding.py` + `data_embedding_ppt.py`)
- `data_embedding.py` reads `file/Combined_EU_NIST_ISO_AI_Compliance_2311.xlsx`, processes each sheet, embeds rows using BGE-Large, and writes to a local Qdrant collection (`./data`, collection `grc_docs`).
- `data_embedding_ppt.py` reads 5 ISO 42001 PPT files from `GRC_Requirement/`, extracts text slide-by-slide, and upserts into the same `grc_docs` collection.

Each stored point carries:
- `text` — full embedding text (prefixed with framework + sheet context)
- `bm25_text` — keyword search text with cross-reference columns stripped
- `framework` — detected from sheet name (`EU AI Act`, `NIST AI RMF`, `ISO`, `General`)
- `sheet` — source sheet name or PPT filename (without extension)

Critical ingestion details:
- Dashboard/utility sheets are skipped via `SKIP_SHEETS`
- ISO hierarchical sheets (`ISO-42K Management Clauses`, `ISO-42K-Annex A`) get clause numbers forward-filled so sub-rows inherit parent context
- Columns with >50% placeholder values (TBD/TBA/N/A) are dropped
- BM25 text deliberately excludes cross-reference columns (e.g. "ISO 42K References") to prevent false-positive clause matches
- Vectors are 1024-dim cosine (BGE-Large output size)
- PPT slides with < 30 characters of text are skipped (blank/image-only slides)

### Phase 2 — Query (`app.py`)
At startup, loads all Qdrant points into memory to build a `BM25Okapi` index alongside the vector store. Per query:
1. **Hybrid retrieve** — BM25 top-40 + Qdrant vector top-40, merged and deduplicated
2. **Rerank** — BGE-Reranker-Large cross-encoder scores all candidates, keeps top-10
3. **Generate** — Mistral-Small 3.2 via Ollama generates a grounded answer from the top-10 context

Framework filtering (sidebar checkboxes) applies to both BM25 and vector search legs.

## Model Paths

Models are stored locally under `./models/` and referenced by absolute path in both scripts. If the project is moved, update `EMBED_MODEL_PATH` and `RERANK_MODEL` in `app.py` and `MODEL_PATH` in `data_embedding.py`.

| Model | Local path | HuggingFace ID |
|-------|-----------|----------------|
| Embedder | `./models/bge-large` | `BAAI/bge-large-en-v1.5` |
| Reranker | `./models/bge-reranker-large` | `BAAI/bge-reranker-large` |

## Data Sources

- **Excel** (`file/Combined_EU_NIST_ISO_AI_Compliance_2311.xlsx`) — EU AI Act, NIST AI RMF, ISO 42001 compliance controls
- **PPT files** (`GRC_Requirement/`) — 5 ISO 42001 implementation/training decks (AIMS-IMP-03/04/05, AIMS-PPT-01/02)

The `grc_docs_v2` entry in `data/meta.json` is a stale leftover from an earlier rename. Only `grc_docs` is used.

## LLM Dependency

Ollama must be running locally with `mistral-small3.2` pulled before starting `app.py`:
```bash
ollama pull mistral-small3.2
ollama serve   # if not already running as a service
```
