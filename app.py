"""Streamlit UI for the Self-Evaluating Agentic RAG.

Tabs:
  💬 Chat              streaming agent with hybrid + reranked retrieval
  📚 Knowledge Base    upload, list, delete documents
  🎯 Eval Dataset      manage the golden Q&A pairs that define "correct"
  📊 Run Evaluation    run the dataset through the pipeline, show metrics
  📈 History           track metric deltas across runs (regression / win)

The eval tabs are the differentiator vs NotebookLM-style demos: you can
PROVE the system stays relevant as you change chunking / embeddings / prompts.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from eval.dataset_store import (
    add_case,
    delete_case,
    import_from_json,
    load_dataset,
    save_dataset,
    update_case,
)
from eval.history import delta_vs_previous, load_history, record_run
from eval.runner import run_evaluation
from rag.agent import run_agent_stream
from rag.ingest import ingest_directory, ingest_file
from rag.reranker import confidence_badge
from rag.retriever import delete_source, list_sources
from rag.session import (
    MissingAPIKey,
    get_api_key,
    get_session,
    reset_session,
    set_api_key,
)

load_dotenv()

st.set_page_config(page_title="Self-Evaluating RAG", layout="wide")
st.title("🔍 Self-Evaluating Agentic RAG")
st.caption(
    "Hybrid retrieval (dense + BM25 + RRF) → BGE cross-encoder reranker → "
    "agentic LLM with streaming. Built-in eval workflow proves it stays "
    "relevant as you tune it."
)

# ----- Ephemeral-mode banner + sidebar session controls --------------------
st.warning(
    "⚡ **Demo mode** — your documents, dataset, and history live only in "
    "this browser session. Refresh or close the tab and they're gone. "
    "Use **💾 Export session** in the sidebar to save your dataset + history."
)

with st.sidebar:
    # --- BYOK: Gemini API key ----------------------------------------------
    st.header("🔑 Gemini API key")
    _current_key = get_api_key()
    if _current_key:
        st.success(f"Key set (`…{_current_key[-4:]}`)")
        if st.button("Clear key", use_container_width=True):
            set_api_key(None)
            st.rerun()
    else:
        st.caption(
            "**Required for chat + agent eval.** "
            "Free key: [aistudio.google.com/apikey]"
            "(https://aistudio.google.com/apikey)"
        )
        with st.form("api_key_form", clear_on_submit=False):
            new_key = st.text_input(
                "Paste key", type="password", placeholder="AIza...",
            )
            if st.form_submit_button("Validate & save", use_container_width=True):
                if not new_key.strip():
                    st.error("Key cannot be empty.")
                else:
                    try:
                        from google import genai
                        from google.genai import types as _types
                        _client = genai.Client(api_key=new_key.strip())
                        _client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents="hi",
                            config=_types.GenerateContentConfig(
                                max_output_tokens=1
                            ),
                        )
                    except Exception as e:
                        st.error(f"Key rejected by Gemini: {e}")
                    else:
                        set_api_key(new_key)
                        st.success("Validated — key saved for this session.")
                        st.rerun()

    st.markdown("---")
    st.header("⚙️ Session")
    _session = get_session()
    st.caption(f"Session ID: `{_session['session_id']}`")

    # --- Export current session's dataset + history -----------------------
    export_payload = {
        "golden_dataset": _session.get("golden_dataset", []),
        "eval_history": _session.get("eval_history", []),
    }
    st.download_button(
        "💾 Export session (JSON)",
        data=json.dumps(export_payload, indent=2, ensure_ascii=False),
        file_name=f"rag_session_{_session['session_id']}.json",
        mime="application/json",
        use_container_width=True,
    )

    # --- Import a previously exported session -----------------------------
    imported = st.file_uploader(
        "📂 Restore session", type=["json"], key="session_import"
    )
    if imported is not None:
        try:
            payload = json.loads(imported.read().decode("utf-8"))
            _session["golden_dataset"] = list(payload.get("golden_dataset", []))
            _session["eval_history"] = list(payload.get("eval_history", []))
            st.success("Session restored. (You'll need to re-upload documents.)")
        except Exception as e:
            st.error(f"Import failed: {e}")

    # --- Hard reset -------------------------------------------------------
    if st.button("🔄 Reset session", use_container_width=True):
        reset_session()
        st.rerun()

    st.markdown("---")
    st.caption(
        "🔒 Your data and API key live in-session only. Nothing is logged "
        "or stored server-side."
    )

TAB_CHAT, TAB_KB, TAB_DATASET, TAB_EVAL, TAB_HISTORY = st.tabs(
    ["💬 Chat", "📚 Knowledge Base", "🎯 Eval Dataset",
     "📊 Run Evaluation", "📈 History"]
)


# =============================================================================
# 💬 CHAT TAB
# =============================================================================
with TAB_CHAT:
    if not get_api_key():
        st.info(
            "👋 **Welcome!** To chat with the agent, paste your Gemini API key "
            "in the sidebar (left). Don't have one? Get a free key in 30 "
            "seconds at [aistudio.google.com/apikey]"
            "(https://aistudio.google.com/apikey).\n\n"
            "**Already have a key?** While you set that up, head to "
            "**📚 Knowledge Base** to load the sample docs, or **🎯 Eval Dataset** "
            "to load the sample evaluation cases — both work without a key."
        )
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the indexed documents...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            trace_holder: dict = {}

            def text_stream():
                """Adapter — yields text tokens, captures the final AgentTrace
                in trace_holder for the trace panel."""
                for item in run_agent_stream(
                    question,
                    history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.history
                    ],
                ):
                    if isinstance(item, str):
                        yield item
                    else:
                        trace_holder["trace"] = item

            try:
                with st.spinner("Retrieving + reasoning..."):
                    st.write_stream(text_stream())
            except MissingAPIKey as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Agent error: {e}")
                with st.expander("Debug details"):
                    st.exception(e)
                st.stop()

            trace = trace_holder.get("trace")
            if trace:
                with st.expander(
                    f"🔍 Agent trace — {trace.steps} step(s) · "
                    f"{len(trace.tool_calls)} retrieval(s) · "
                    f"{len(trace.all_chunks)} chunk(s) considered"
                ):
                    st.markdown("**Retrieval calls**")
                    for i, call in enumerate(trace.tool_calls, 1):
                        st.markdown(
                            f"  {i}. query=`{call['query']}` · k={call['k']} "
                            f"· {call['n_results']} chunks"
                        )

                    st.markdown("**Cited sources (with reranker confidence)**")
                    seen_keys = set()
                    for c in sorted(
                        trace.all_chunks, key=lambda x: x.score, reverse=True
                    ):
                        key = (c.source, c.parent_index)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        st.markdown(
                            f"{confidence_badge(c.score)} **{c.cite()}** · "
                            f"score `{c.score:.3f}`"
                        )
                        st.markdown(
                            f"> {c.parent_text[:300].strip()}"
                            + ("..." if len(c.parent_text) > 300 else "")
                        )

        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append(
            {"role": "assistant", "content": trace.answer if trace else ""}
        )


# =============================================================================
# 📚 KNOWLEDGE BASE TAB
# =============================================================================
with TAB_KB:
    # --- Quick-start: load the bundled sample corpus ---
    col_sample, col_caption = st.columns([1, 4])
    with col_sample:
        if st.button("📥 Load sample docs", use_container_width=True):
            with st.spinner("Indexing 4 sample documents..."):
                try:
                    n = ingest_directory("data")
                    st.success(f"Indexed {n} child chunks from the sample corpus.")
                except Exception as e:
                    st.error(f"Failed to load samples: {e}")
    with col_caption:
        st.caption(
            "Loads 4 short technical docs (RAG, embeddings, vector DBs, a sample "
            "presentation). Lets you try the system immediately without uploading."
        )

    st.markdown("---")
    st.subheader("Upload your own document")
    uploaded = st.file_uploader(
        "PDF, TXT, MD, or DOCX (max 10MB)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=False,
        help="Files are chunked (parent-child), embedded, indexed in Chroma + BM25. "
             "Kept in your session only — never written to disk on the server.",
    )
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)
        final_path = tmp_path.with_name(uploaded.name)
        tmp_path.rename(final_path)
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                n = ingest_file(final_path)
                st.success(f"Added {n} child chunks from {uploaded.name}")
            except Exception as e:
                st.error(f"Ingest failed: {e}")
            finally:
                final_path.unlink(missing_ok=True)

    st.markdown("---")
    st.subheader("Indexed sources")
    sources = list_sources()
    if not sources:
        st.info("No documents indexed yet. Upload one above.")
    else:
        total = sum(sources.values())
        st.caption(f"{len(sources)} file(s) · {total} child chunks total")
        for name, count in sources.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{name}** · `{count} chunks`")
            with col2:
                if st.button("Delete", key=f"del-{name}"):
                    n = delete_source(name)
                    st.success(f"Removed {n} chunks")
                    st.rerun()


# =============================================================================
# 🎯 EVAL DATASET TAB
# =============================================================================
with TAB_DATASET:
    st.markdown(
        "Define what *correct* looks like for your knowledge base. "
        "Each Q&A case below is checked when you run an evaluation — "
        "this is how you make sure the RAG stays relevant after every "
        "change to chunking, embeddings, prompts, or models."
    )

    # --- Quick-start: load the bundled sample dataset ---
    col_smp, col_smp_cap = st.columns([1, 4])
    with col_smp:
        if st.button("📥 Load sample eval set", use_container_width=True):
            sample_path = Path("eval/golden_dataset.json")
            try:
                payload = json.loads(sample_path.read_text(encoding="utf-8"))
                n = import_from_json(payload)
                st.success(f"Loaded {n} sample eval cases.")
                st.rerun()
            except FileNotFoundError:
                st.error(
                    "Sample dataset file missing — re-clone the repo or "
                    "import your own JSON below."
                )
            except Exception as e:
                st.error(f"Failed to load samples: {e}")
    with col_smp_cap:
        st.caption(
            "Loads 19 hand-labeled cases covering the bundled sample docs "
            "(positive + multi-hop + refusal). Pairs with **Load sample docs** "
            "in the Knowledge Base tab for an instant working demo."
        )
    st.markdown("---")

    # ----- Add new case -----
    with st.expander("➕ Add a new case", expanded=False):
        with st.form("add_case_form", clear_on_submit=True):
            q = st.text_area("Question", placeholder="e.g., What is HNSW?")
            es = st.text_input(
                "Expected source filename (optional)",
                placeholder="e.g., vector_db_overview.txt — leave blank for refusal cases",
            )
            ek = st.text_input(
                "Expected keywords (comma-separated)",
                placeholder="e.g., hnsw, navigable",
            )
            er = st.checkbox(
                "Refusal case (agent must say 'I don't know')",
                value=False,
                help="Use for questions whose answer is NOT in the corpus.",
            )
            if st.form_submit_button("Add case"):
                if not q.strip():
                    st.error("Question is required.")
                else:
                    keywords = [k.strip() for k in ek.split(",") if k.strip()]
                    case = add_case(q, es.strip() or None, keywords, er)
                    st.success(f"Added case {case['id']}: {case['question'][:60]}")
                    st.rerun()

    # ----- Bulk import / export -----
    with st.expander("📂 Import / export full dataset", expanded=False):
        col_imp, col_exp = st.columns(2)
        with col_imp:
            up = st.file_uploader("Import JSON", type=["json"], key="dataset_import")
            if up is not None:
                try:
                    payload = json.loads(up.read().decode("utf-8"))
                    n = import_from_json(payload)
                    st.success(f"Imported {n} cases (replaced existing dataset).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")
        with col_exp:
            current = load_dataset()
            if current:
                st.download_button(
                    "Download current dataset (JSON)",
                    data=json.dumps(current, indent=2, ensure_ascii=False),
                    file_name="golden_dataset.json",
                    mime="application/json",
                )

    st.markdown("---")
    # ----- Cases table -----
    cases = load_dataset()
    st.subheader(f"Current dataset · {len(cases)} case(s)")

    if not cases:
        st.info("No cases yet. Add one above, or import a JSON dataset.")
    else:
        for case in cases:
            with st.container(border=True):
                col_main, col_actions = st.columns([5, 1])
                with col_main:
                    label = "🔴 refusal" if case["expect_refusal"] else "🟢 positive"
                    st.markdown(f"**{label}** · `{case['id']}` · {case['question']}")
                    parts = []
                    if case["expected_source"]:
                        parts.append(f"source: `{case['expected_source']}`")
                    if case["expected_keywords"]:
                        parts.append(
                            "keywords: " + ", ".join(
                                f"`{k}`" for k in case["expected_keywords"]
                            )
                        )
                    if parts:
                        st.caption(" · ".join(parts))
                with col_actions:
                    if st.button("Delete", key=f"del-case-{case['id']}"):
                        delete_case(case["id"])
                        st.rerun()


# =============================================================================
# 📊 RUN EVALUATION TAB
# =============================================================================
with TAB_EVAL:
    cases = load_dataset()
    st.markdown(
        "Run the full pipeline against every case in your golden dataset. "
        "Retrieval metrics (Hit@k, MRR) are computed deterministically; "
        "answer-quality metrics also require LLM calls per case."
    )

    if not cases:
        st.warning(
            "No cases in the golden dataset. Add some in the **Eval Dataset** tab first."
        )
    else:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            top_k = st.number_input("top_k", min_value=1, max_value=10, value=5)
        with col2:
            candidate_k = st.number_input(
                "candidate_k", min_value=5, max_value=50, value=20
            )
        with col3:
            run_agent_too = st.checkbox(
                "Also run agent end-to-end (slower; needs API quota)",
                value=False,
                help=(
                    "Off: retrieval-only eval (Hit@k, MRR). Fast. "
                    "On: also calls the LLM per question and checks keyword/refusal."
                ),
            )

        note = st.text_input(
            "Note for this run (optional)",
            placeholder="e.g., 'switched chunk_size 512→1024'",
            help="Annotate what you changed since the last run. Shows in History.",
        )

        # Block agent-mode eval if no API key is set
        if run_agent_too and not get_api_key():
            st.warning(
                "⚠️  Agent eval requires a Gemini API key — paste one in the "
                "sidebar, OR uncheck the box above to run retrieval-only eval "
                "(no API needed)."
            )

        run_button = st.button(
            "▶ Run evaluation",
            type="primary",
            disabled=(run_agent_too and not get_api_key()),
        )

        if run_button:
            progress = st.progress(0.0)
            status = st.empty()

            def _on_progress(idx: int, total: int, result: dict):
                progress.progress(idx / total)
                status.write(
                    f"[{idx}/{total}] {result['question'][:80]} → "
                    f"hit={result['hit_at_k']} rr={result['reciprocal_rank']}"
                )

            try:
                result = run_evaluation(
                    cases,
                    top_k=int(top_k),
                    candidate_k=int(candidate_k),
                    run_agent_too=run_agent_too,
                    on_progress=_on_progress,
                )
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
            else:
                progress.empty()
                status.empty()
                record_run(result, note=note)
                st.session_state["last_eval"] = result
                st.success(
                    f"Done in {result['aggregate']['elapsed_seconds']}s. "
                    "Result recorded to History."
                )

        # Show the most recent run's results, if any
        result = st.session_state.get("last_eval")
        if result:
            agg = result["aggregate"]
            st.markdown("---")
            st.markdown("### Aggregate metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Hit@k",
                f"{agg['hit_at_k']:.2%}" if agg.get("hit_at_k") is not None else "—",
            )
            m2.metric(
                "MRR",
                f"{agg['mrr']:.3f}" if agg.get("mrr") is not None else "—",
            )
            m3.metric(
                "Keyword acc.",
                f"{agg['answer_keyword_acc']:.2%}"
                if agg.get("answer_keyword_acc") is not None else "—",
            )
            m4.metric(
                "Refusal acc.",
                f"{agg['refusal_acc']:.2%}"
                if agg.get("refusal_acc") is not None else "—",
            )

            st.markdown("### Per-case results")
            rows = []
            for r in result["per_case"]:
                rows.append({
                    "id": r.get("id"),
                    "question": r["question"],
                    "expected_source": r.get("expected_source") or "—",
                    "Hit@k": r["hit_at_k"] if r["hit_at_k"] is not None else "—",
                    "RR": r["reciprocal_rank"]
                          if r["reciprocal_rank"] is not None else "—",
                    "keyword_pass": r.get("keyword_pass"),
                    "refusal_pass": r.get("refusal_pass"),
                    "top_sources": ", ".join(
                        f"{s['source']}#{s['chunk_index']}"
                        for s in r["retrieved"][:3]
                    ),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# =============================================================================
# 📈 HISTORY TAB
# =============================================================================
with TAB_HISTORY:
    runs = load_history(limit=20)
    annotated = delta_vs_previous(runs)

    if not annotated:
        st.info(
            "No evaluation runs yet. Run an evaluation in the **Run Evaluation** "
            "tab to start tracking history."
        )
    else:
        st.markdown(
            "Each row is one run. Deltas compare against the previous (older) run "
            "— green ↑ for improvement, red ↓ for regression."
        )
        rows = []
        for run in annotated:
            agg = run["aggregate"]
            d = run.get("delta", {})
            def _fmt(key: str, kind: str = "pct") -> str:
                v = agg.get(key)
                if v is None:
                    return "—"
                dv = d.get(key)
                base = f"{v:.2%}" if kind == "pct" else f"{v:.3f}"
                if dv is None:
                    return base
                arrow = "↑" if dv > 0 else ("↓" if dv < 0 else "·")
                dstr = f"{dv:+.2%}" if kind == "pct" else f"{dv:+.3f}"
                return f"{base} ({arrow} {dstr})"

            rows.append({
                "When (UTC)": run["timestamp"],
                "Note": run.get("note", ""),
                "Hit@k": _fmt("hit_at_k"),
                "MRR": _fmt("mrr", kind="num"),
                "Keyword acc.": _fmt("answer_keyword_acc"),
                "Refusal acc.": _fmt("refusal_acc"),
                "Elapsed (s)": agg.get("elapsed_seconds"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
