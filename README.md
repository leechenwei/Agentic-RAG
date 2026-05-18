# Agentic RAG

A minimal but production-shaped Agentic Retrieval-Augmented Generation system.
The LLM is given a `retrieve` tool and decides *when*, *what*, and *how many
times* to call it — enabling query rewriting, multi-step retrieval, and honest
refusal when the corpus contains no answer.

Built as a tech-test submission for the MaiStorage AI Engineer role.

---

## What it does

- Ingests `.pdf`, `.txt`, `.md` files into a local vector database
- Answers questions over the indexed corpus with inline citations
- Shows the agent's reasoning trace — query rewrites, retrieval rounds, scores
- Supports live document upload and deletion through the UI
- Includes an evaluation suite with retrieval recall@k and refusal tests

---

## Quickstart

```bash
# 1. Clone and enter
git clone <your-repo-url> && cd agentic_rag

# 2. Install dependencies (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Add your Gemini API key
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...

# 4. Ingest the sample corpus
python -m rag.ingest

# 5. Launch the UI
streamlit run app.py
# → http://localhost:8501
```

Get a free Gemini API key at <https://aistudio.google.com/apikey>.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit UI (app.py)                  │
│  · Chat interface                                       │
│  · Sidebar: upload, list, delete documents              │
│  · Agent trace panel with citations                     │
└──────────────────────┬──────────────────────────────────┘
                       │
       ┌───────────────▼────────────────┐
       │      Agent Loop (rag/agent.py) │  ◄── Gemini 2.5 Flash
       │                                │      (function calling)
       │   while not done:              │
       │     LLM → wants to retrieve?   │
       │       └→ call retriever        │
       │     else → return answer       │
       └───────────────┬────────────────┘
                       │
       ┌───────────────▼────────────────┐
       │   Retriever (rag/retriever.py) │
       │   · retrieve(query, k)         │
       │   · list_sources()             │
       │   · delete_source(name)        │
       └───────────────┬────────────────┘
                       │
       ┌───────────────▼────────────────┐
       │      ChromaDB (local disk)     │
       │   · 384-dim vectors (MiniLM)   │
       │   · HNSW index, cosine space   │
       │   · source + chunk_index meta  │
       └───────────────▲────────────────┘
                       │  one-time / on upload
       ┌───────────────┴────────────────┐
       │     Ingest (rag/ingest.py)     │
       │   files → chunk (500/80)       │
       │         → embed (MiniLM)       │
       │         → upsert with hash IDs │
       └────────────────────────────────┘
```

Each layer owns one responsibility and depends only inward. Swapping ChromaDB
for Qdrant touches one file (`retriever.py` + `ingest.py`); swapping Gemini for
Claude touches only `agent.py`; swapping Streamlit for FastAPI touches only
`app.py`.

---

## Why Agentic RAG (vs the alternatives)

| | Naive RAG | Advanced RAG | **Agentic RAG (this)** | Graph RAG |
|---|---|---|---|---|
| Retrieval | Always, once | Always, once + rerank | **LLM decides via tool call** | Graph traversal |
| Query rewriting | No | Static rewrite | **LLM rewrites per call** | Entity extraction |
| Multi-hop | No | Weak | **Yes — decomposes** | Native |
| Skip retrieval | No | No | **Yes (small talk, math)** | No |
| Self-correction | No | No | **Yes — re-query if weak** | Limited |
| Cost / query | $ | $$ | $$ | $$$ |
| Setup effort | 1 day | 3 days | 1 week | 2–4 weeks |

**Why this tier**: the use case is conversational document Q&A. Naive RAG
breaks on multi-hop questions; Graph RAG is overkill for unstructured slides
and prose. Agentic RAG is the sweet spot — handles multi-hop and ambiguous
questions, and the architecture extends naturally to additional tools
(`web_search`, `sql_query`, `calculator`) without rewriting the loop.

---

## Design decisions (and the reasoning)

### Chunking: 500 characters with 80-character overlap

- **500 chars** ≈ one paragraph — long enough to contain a complete thought,
  short enough to keep the embedding focused on a single topic
- **80 chars overlap** (~16%) keeps boundary phrases intact across adjacent
  chunks, so a phrase like "Software Engineer Intern at Dell" doesn't get
  sliced
- Character-based chunking was chosen for **demo clarity**. For production
  I would switch to semantic / slide-aware chunking
  (e.g. `RecursiveCharacterTextSplitter`, one chunk per slide for PPTX)

### Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

- 384-dimensional output — strong accuracy-to-cost ratio
- Runs locally on CPU — zero API cost for indexing
- Deterministic — useful for reproducible eval runs
- Production upgrade: `bge-large-en-v1.5` (1024 dims) for higher recall,
  or `text-embedding-3-small` for managed simplicity

### Vector database: ChromaDB (local, persistent)

- Embedded, single-file deployment — no infrastructure for the demo
- Cosine distance configured via `metadata={"hnsw:space": "cosine"}`
- Stores metadata alongside vectors — citations come for free
- Production upgrade: Qdrant or Pinecone for horizontal scaling and
  multi-tenant isolation

### Ingestion idempotency

Stable IDs (`md5(filename:chunk_index:content_prefix)`) plus ChromaDB's
`upsert` semantics mean re-running `ingest.py` is a no-op for unchanged
content and only re-embeds new or modified chunks. Production would extend
this to **delete chunks for files removed from the source directory** (the
current implementation orphans them).

### Agent loop: `MAX_AGENT_STEPS = 4`

- Each step = one LLM call + optional `retrieve` call
- Cap prevents runaway loops if the model gets confused
- Tuned for Gemini free-tier (5 requests / minute) — production with paid
  tier could raise to 8–10 for harder multi-hop questions

### Rate-limit handling

429 responses are caught and retried with backoff that honors the
provider-suggested `retryDelay`. For production scale this would be replaced
with a request queue and quota-aware throttling.

### Citations

Every retrieved chunk carries `source` and `chunk_index` metadata. The system
prompt instructs the LLM to cite inline as `[source#chunkN]`. The UI also
shows each retrieved chunk with its cosine-similarity score in the trace
panel — so users can verify *which* chunk the answer came from, not just
*that* it had a source.

### Honest refusal

The system prompt explicitly tells the model to say it does not know rather
than fabricate when retrieved chunks don't contain the answer. This is
validated in the eval suite by an "unanswerable question" test case.

---

## Testing & quality assurance

```bash
pytest -v
```

Three layers of test:

1. **Retrieval recall@k** (`tests/test_retrieval.py::test_retrieval_recall_at_k`)
   For each positive eval case, the expected source document must appear in
   the top-k chunks. Aggregate threshold: ≥80% across all positives.

2. **End-to-end agent answer** (`test_agent_end_to_end`)
   Final answer must contain expected keywords (case-insensitive). Slow tier
   (`@pytest.mark.slow`) because it hits the LLM.

3. **Negative refusal** (case in `eval_dataset.py` with `expect_refusal: True`)
   For a question whose answer is not in the corpus, the agent must produce
   a refusal phrase (`"don't know"`, `"no information"`, etc.) and must not
   fabricate. This is the single most important test — without it, the
   system's failure mode is silent hallucination.

The eval dataset (`tests/eval_dataset.py`) is the **source of truth for what
"working" means**. Adding a new failure mode means adding a case here first.

---

## Project structure

```
agentic_rag/
├── app.py                       Streamlit UI
├── requirements.txt
├── .env.example
├── README.md                    This file
│
├── data/                        Source documents
│   ├── maistorage_intro.txt     Candidate intro deck (extracted from PPTX)
│   ├── rag_overview.txt         Reference: RAG concepts
│   ├── vector_db_overview.txt   Reference: vector databases
│   └── embeddings_overview.txt  Reference: embeddings
│
├── rag/
│   ├── ingest.py                File → chunk → embed → ChromaDB
│   ├── retriever.py             Cosine retrieval + source list/delete
│   └── agent.py                 Gemini function-calling loop
│
├── tests/
│   ├── eval_dataset.py          Hand-curated Q&A pairs
│   └── test_retrieval.py        pytest suite
│
└── chroma_db/                   (auto-generated; gitignored)
```

---

## Limitations & next steps

| Limitation | Production fix |
|---|---|
| Character chunking can split sentences | Semantic / sentence-aware splitting; per-slide chunking for PPTX |
| Pure dense retrieval — misses exact identifiers and acronyms | Hybrid search: BM25 + dense, fused via Reciprocal Rank Fusion |
| Top-k ordering relies solely on bi-encoder | Cross-encoder reranker on top 20 → top 4 |
| Orphaned chunks when source files are deleted | Track ingested-files manifest; remove chunks whose source no longer exists |
| Manual re-ingest required | File watcher (`watchdog`) or event-driven (S3 → SQS → worker) |
| Single-process Streamlit | FastAPI + SSE streaming, horizontal scaling |
| ChromaDB local cap (~1M vectors, single writer) | Qdrant Cloud / Pinecone / Weaviate |
| No observability | LangFuse / Arize for traces, latency, cost, drift |
| Eval thresholds are hand-coded | LLM-as-judge for answer faithfulness; RAGAS framework |

---

## Tech stack

- **Python 3.11+**
- **Gemini 2.5 Flash** — LLM, function calling, low-cost
- **sentence-transformers / all-MiniLM-L6-v2** — 384-dim embeddings, CPU
- **ChromaDB** — embedded vector database with HNSW index
- **Streamlit** — chat UI and document management
- **pytest** — evaluation harness

---

## Author

**Chen Wei, Lee** — AI Engineer candidate
[LinkedIn](https://linkedin.com/in/lcw02) · [GitHub](https://github.com/leechenwei)
