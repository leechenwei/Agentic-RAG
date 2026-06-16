# Self-Evaluating Agentic RAG

A production-shaped Retrieval-Augmented Generation chatbot **with a built-in
evaluation workflow**. Users curate a golden Q&A dataset alongside their
documents, run regression-style evaluations on demand, and track metric
deltas across configuration changes — so "is the chatbot still accurate?"
stops being a vibe check and becomes a number you can graph.

> **The interview question this answers:** *"How do you make sure the RAG
> stays relevant as you tune chunking, embeddings, prompts, or models?"*
> You curate a golden dataset, you measure Hit@k / MRR / faithfulness on
> every change, and you persist runs to a history that highlights
> regressions. **This app makes that workflow visible to users.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)                       │
│  💬 Chat   📚 KB   🎯 Eval Dataset   📊 Run Eval   📈 History    │
└──────────────────┬──────────────────────────────────┬───────────┘
                   │ chat                              │ eval
       ┌───────────▼────────────┐         ┌────────────▼──────────┐
       │   Streaming Agent      │         │   Eval Runner         │
       │   (rag/agent.py)       │         │   (eval/runner.py)    │
       │   Gemini 2.5 Flash     │         │   Hit@k / MRR /       │
       │   tool: retrieve()     │         │   keyword / refusal   │
       └───────────┬────────────┘         └────────────┬──────────┘
                   │                                    │
                   └────────────────┬───────────────────┘
                                    ▼
                  ┌─────────────────────────────────┐
                  │      Retrieval Pipeline         │
                  │   (rag/retriever.py)            │
                  │                                 │
                  │  Dense  ┐                       │
                  │  (Chroma + MiniLM 384-d)        │
                  │         ├─► RRF fuse ─► Rerank  │
                  │  BM25   ┘   (k=60)     (BGE)    │
                  │  (rag/bm25_store.py)            │
                  │                                 │
                  │   → top-K parent blocks → LLM   │
                  └─────────────────────────────────┘
                                    ▲
                                    │
                  ┌─────────────────┴───────────────┐
                  │   Per-session Chroma (ephemeral) │
                  │   (rag/session.py)               │
                  │   · Parent-child chunking         │
                  │   · Multi-tenant safe             │
                  └─────────────────────────────────┘
```

---

## Features

### Retrieval
- **Hybrid search** — dense vector (ChromaDB + `all-MiniLM-L6-v2`, 384 dims) + BM25 lexical, fused via Reciprocal Rank Fusion (k=60)
- **Cross-encoder reranker** — `BAAI/bge-reranker-base` re-scores the top-20 candidates jointly with the query, narrowing to top-K most relevant
- **Parent-child chunking** — small children for retrieval precision (256 tokens), larger parents for LLM context (1024 tokens), via LangChain's `RecursiveCharacterTextSplitter`
- **HNSW index** with cosine similarity for sub-millisecond ANN

### Agentic loop
- Gemini 2.5 Flash with function-calling — the LLM decides when to retrieve, what query to use, whether to retrieve again, and when to answer
- **Streaming output** — tokens appear as they arrive (peek-then-stream architecture)
- Multi-step retrieval with query rewriting and honest refusal

### Self-evaluation (the differentiator)
- **User-curated golden dataset** — add Q&A pairs in the UI, JSON export/import for portability
- **Metrics** — Hit@k, Mean Reciprocal Rank, keyword pass rate, refusal accuracy
- **Run history** — timestamped, configuration-snapshotted, with deltas vs the previous run highlighted
- **Two-stage eval** — fast retrieval-only (free, no LLM calls) + slower agent-end-to-end (uses LLM quota)

### Privacy + deploy
- **Per-session multi-tenancy** — ephemeral in-memory ChromaDB per browser session, isolated from other users
- **BYOK** — users paste their own Gemini API key, validated on submission, stored in session only (never disk, never logs)
- **Zero server-side persistence** — refresh = clean slate; users export their dataset/history as JSON if they want to save
- **Sample preload** — one-click "Load sample docs" + "Load sample eval set" for instant demo

---

## Live demo

> **[Insert Streamlit Cloud URL here once deployed]**
>
> Bring your own free Gemini API key from
> [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

The sample corpus is a small portfolio of Chen Wei's projects — load them
in the Knowledge Base tab and try questions like:
- *"Tell me about Chen Wei's production AI work"*
- *"How does the reranker work in this project?"* (meta-recursive — the demo answers questions about itself)
- *"What's the connection between the FYP's hybrid routing and Inside Advisory?"*

Then load the sample eval set and click **▶ Run evaluation** to see Hit@k and MRR computed live.

---

## Run locally

```bash
git clone https://github.com/leechenwei/Agentic-RAG.git
cd Agentic-RAG

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
# → http://localhost:8501
```

Then paste your Gemini API key in the sidebar.

For convenience during local dev you can also set it in `.env`:
```bash
cp .env.example .env
echo "GEMINI_API_KEY=your-key-here" >> .env
```
(The deployed version ignores env vars — users always BYOK.)

---

## Tests

```bash
pytest -v
```

| Test | What it checks |
|---|---|
| `test_production_retrieval_aggregate` | Hybrid + reranked pipeline must hit ≥90% recall@5 across the labeled cases |
| `test_dense_only_recall_baseline` | Dense-only baseline ≥70% — used as a baseline to demonstrate the lift from hybrid + rerank |
| `test_agent_end_to_end` (`@slow`) | Final agent answers must contain expected keywords or refuse for negative cases |

`tests/conftest.py` ingests `data/` into the test session's ephemeral Chroma
once per pytest run, so tests don't depend on persistent state.

---

## Project structure

```
agentic_rag/
├── app.py                          Streamlit UI (5 tabs)
├── requirements.txt
├── README.md
├── .env.example                    (local-dev only; deployed version uses BYOK)
│
├── data/                           Sample corpus (5 portfolio docs)
│   ├── about_chen_wei.txt
│   ├── inside_assistant_project.txt
│   ├── agentic_rag_project.txt     ← meta: doc describing THIS app
│   ├── fyp_dialogue_system.txt
│   └── dell_internship.txt
│
├── rag/
│   ├── session.py                  Per-session state (Chroma, BM25, API key, dataset)
│   ├── ingest.py                   Parent-child chunking + recursive splitter
│   ├── bm25_store.py               BM25 lexical index
│   ├── retriever.py                Dense + BM25 + RRF + reranker pipeline
│   ├── reranker.py                 BGE cross-encoder
│   └── agent.py                    Streaming Gemini agent with retrieve tool
│
├── eval/
│   ├── dataset_store.py            Session-scoped golden Q&A CRUD
│   ├── runner.py                   Hit@k / MRR / keyword / refusal metrics
│   ├── history.py                  Per-session run history with deltas
│   └── golden_dataset.json         19 sample cases tied to the bundled docs
│
└── tests/
    ├── conftest.py                 Ingests data/ into test session
    ├── eval_dataset.py             Loads from golden_dataset.json (single source of truth)
    └── test_retrieval.py
```

---

## Design decisions (and the reasoning)

### Why hybrid retrieval (dense + BM25) instead of pure dense?
Dense embeddings blur exact-token matches — IDs, code identifiers, acronyms, version numbers all get treated as "generic invoice-like words." BM25 catches those literally. RRF combines the two rankings using only ranks (not raw scores), so the incompatible cosine and BM25 score scales never need normalizing. The constant `k=60` is from the original RRF paper.

### Why a cross-encoder reranker?
Bi-encoders (the dense retriever) embed query and doc independently — fast, but the embedding has to compress all possible meanings of a doc into one vector. Cross-encoders read query and doc *jointly* through the same model, so attention captures direct relevance. Far more accurate, but expensive — which is why we only run it on the top-20 candidates from broad retrieval.

### Why parent-child chunking?
Small chunks (~256 tokens) embed precisely — retrieval matches the right *passage*. Large chunks (~1024 tokens) give the LLM enough surrounding context to write a coherent answer. Parent-child lets us have both: embed children for retrieval precision, return parents to the LLM for context.

### Why ephemeral per-session storage?
Public demo on free-tier hosting + uploaded user documents = a privacy and cost minefield if persisted. Ephemeral storage means each browser session is isolated, nothing is logged server-side, and storage costs stay at zero. Users export their dataset/history as JSON if they want to save it.

### Why BYOK (Bring Your Own Key)?
A shared key on a public demo gets exhausted by the first curious visitor — Gemini's free tier is 25 requests/day. BYOK eliminates that abuse vector, eliminates the maintainer's API bill, and is the production-correct pattern. The trade-off (users need to get a key) is well-understood by the AI-tooling audience this demo targets.

### Why streaming on the final turn only?
Tool-calling turns can't be safely streamed — partial function-call structured output would garble the UI. We use a "peek-then-stream" approach: one non-streaming call to detect whether the turn is text or tool, then re-call with streaming for the final answer. Costs one extra LLM call on the final turn but gives a clean per-token typing UX.

---

## Tech stack

- **Python 3.11+**
- **Streamlit** — UI
- **ChromaDB** (ephemeral mode) — vector store
- **sentence-transformers / all-MiniLM-L6-v2** — embeddings (384 dims, CPU)
- **rank-bm25** — lexical retrieval
- **BAAI/bge-reranker-base** — cross-encoder reranker
- **LangChain text-splitters** — `RecursiveCharacterTextSplitter`
- **Google Gemini SDK** — `gemini-2.5-flash` LLM
- **pytest** — eval harness

---

## What's next

- LLM-as-judge faithfulness scoring (RAGAS-style)
- Adaptive routing — skip retrieval entirely for trivial queries
- MCP server wrapping the `retrieve` tool
- Multi-provider LLM abstraction (OpenAI / Claude alongside Gemini)
- Cohere Rerank as a managed alternative to BGE

---

## Author

**Chen Wei, Lee** — AI Engineer · Malaysia
[LinkedIn](https://linkedin.com/in/lcw02) · [GitHub](https://github.com/leechenwei) · LuisLCW02@gmail.com
