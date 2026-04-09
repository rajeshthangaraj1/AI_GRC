# AI GRC Compliance Advisor

A fully local, privacy-first AI chatbot for querying **EU AI Act**, **NIST AI RMF**, and **ISO 42001** compliance requirements. Built with hybrid RAG (BM25 + vector search), neural reranking, and a local LLM — no data ever leaves your environment.

---

## Features

- **Hybrid Retrieval** — combines BM25 keyword search and dense vector search for high recall
- **Neural Reranking** — BGE-Reranker-Large cross-encoder re-scores results for precision
- **Three Frameworks** — EU AI Act, NIST AI RMF, ISO 42001 (all in one Excel source)
- **Framework Filtering** — query one or all frameworks simultaneously
- **Local LLM** — Mistral-Small 3.2 via Ollama; no API keys, no cloud calls
- **Streamlit UI** — chat interface with source attribution and sample questions

---

## Architecture

```
Excel + PPT Source Data
       │
       ▼
data_embedding.py             # Ingestion pipeline (Excel)
data_embedding_ppt.py         # Ingestion pipeline (PPT slides)
  ├── Sheet/slide parsing & cleaning
  ├── Forward-fill ISO clause hierarchy
  ├── BGE-Large embeddings (CUDA)
  └── Qdrant vector store (./data, collection: grc_docs)
       │
       ▼
app.py                        # Streamlit app
  ├── BM25 index (in-memory)
  ├── Hybrid retrieval (BM25 + Qdrant)
  ├── BGE-Reranker-Large cross-encoder
  └── Mistral-Small 3.2 via Ollama → Answer
```

---

## Tech Stack

| Component        | Library / Tool                     | Version  |
|------------------|------------------------------------|----------|
| UI               | Streamlit                          | 1.55.0   |
| Embeddings       | sentence-transformers (BGE-Large)  | 5.3.0    |
| Reranking        | CrossEncoder (BGE-Reranker-Large)  | 5.3.0    |
| Vector DB        | Qdrant (local/persistent)          | 1.17.1   |
| Keyword Search   | rank-bm25                          | 0.2.2    |
| LLM              | Ollama — mistral-small3.2          | —        |
| LLM wrapper      | langchain-ollama                   | 1.0.1    |
| Data processing  | pandas / numpy                     | 2.3.3 / 2.4.3 |
| Model download   | huggingface_hub                    | 1.7.1    |

---

## Project Structure

```
AI_GRC/
├── app.py                     # Main Streamlit app
├── data_embedding.py          # Data ingestion pipeline (Excel)
├── data_embedding_ppt.py      # Data ingestion pipeline (PPT slides)
├── db.py                      # Utility: model download helper
├── file/
│   └── Combined_EU_NIST_ISO_AI_Compliance_2311.xlsx   # Source compliance data
├── GRC_Requirement/           # ISO 42001 PPT files (5 files)
├── models/
│   ├── bge-large/             # BAAI/bge-large-en-v1.5 (downloaded locally)
│   └── bge-reranker-large/    # BAAI/bge-reranker-large (downloaded locally)
├── data/                      # Qdrant persistent vector store
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (recommended for embedding speed)
- [Ollama](https://ollama.com) installed and running

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd AI_GRC
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull the LLM via Ollama

```bash
ollama pull mistral-small3.2
```

### 3. Download embedding & reranker models

Edit `db.py` to point to the correct model and run:

```bash
python db.py
```

This downloads `BAAI/bge-large-en-v1.5` and `BAAI/bge-reranker-large` into `./models/`.

### 4. Ingest compliance data

```bash
python data_embedding.py        # Excel → grc_docs collection
python data_embedding_ppt.py    # PPT slides → upserted into same collection
```

Run `data_embedding.py` first (creates the collection), then `data_embedding_ppt.py` (upserts PPT slides).

### 5. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Data Source

**Excel** (`Combined_EU_NIST_ISO_AI_Compliance_2311.xlsx`) — structured compliance controls:

- **EU AI Act** — prohibited practices, high-risk requirements, transparency obligations
- **NIST AI RMF** — GOVERN, MAP, MEASURE, MANAGE functions
- **ISO 42001** — management clauses and Annex A controls

**PPT files** (`GRC_Requirement/`) — ISO 42001 implementation content (5 decks covering implementation phases, certification process, training, and integrated management systems)

---

## Configuration

Key constants in `app.py` and `data_embedding.py`:

| Constant          | Default                  | Description                          |
|-------------------|--------------------------|--------------------------------------|
| `QDRANT_PATH`     | `./data`                 | Qdrant persistent storage path       |
| `COLLECTION_NAME` | `grc_docs`            | Qdrant collection name               |
| `EMBED_MODEL_PATH`| `./models/bge-large`     | Local path to BGE-Large model        |
| `RERANK_MODEL`    | `./models/bge-reranker-large` | Local path to reranker model    |
| `OLLAMA_MODEL`    | `mistral-small3.2`       | Ollama model name                    |
| `TOP_K_RETRIEVE`  | `40`                     | Candidates retrieved per search leg  |
| `TOP_K_RERANK`    | `10`                     | Final docs passed to LLM             |

---

## Privacy

All processing is local:
- Embeddings computed on your GPU
- Vectors stored on disk (Qdrant local mode)
- LLM runs via Ollama on your machine
- No API keys required, no data sent externally
