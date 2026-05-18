"""Streamlit demo UI for the Agentic RAG system."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.agent import run_agent
from rag.ingest import ingest_file
from rag.retriever import delete_source, list_sources

load_dotenv()

st.set_page_config(page_title="Agentic RAG Demo", layout="wide")
st.title("Agentic RAG")
st.caption(
    "The LLM decides when and what to retrieve. The trace panel shows "
    "the agent's actions - query rewrites, multi-step retrieval, citations."
)

# ----------------------------- Sidebar: knowledge base management -----
with st.sidebar:
    st.header("Knowledge Base")

    st.subheader("Upload document")
    uploaded = st.file_uploader(
        "PDF, TXT, or MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
        help="Files are chunked, embedded, and added to the vector index immediately.",
    )
    if uploaded is not None:
        # Save the upload to a temp path so ingest_file can read it via Path.
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)
        # Rename so the source metadata uses the original filename, not the
        # temp file's randomized name.
        final_path = tmp_path.with_name(uploaded.name)
        tmp_path.rename(final_path)
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                n = ingest_file(final_path)
                st.success(f"Added {n} chunks from {uploaded.name}")
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
        st.caption(f"{len(sources)} file(s) - {total} chunks total")
        for name, count in sources.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{name}**  \n`{count} chunks`")
            with col2:
                if st.button("Delete", key=f"del-{name}"):
                    n = delete_source(name)
                    st.success(f"Removed {n} chunks")
                    st.rerun()

# ----------------------------- Main panel: chat ---------------------------
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
        with st.spinner("Agent thinking..."):
            trace = run_agent(
                question,
                history=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.history
                ],
            )
        st.markdown(trace.answer)

        with st.expander(f"Agent trace - {trace.steps} step(s) - "
                         f"{len(trace.tool_calls)} retrieval(s)"):
            for i, call in enumerate(trace.tool_calls, 1):
                st.markdown(
                    f"**Step {i}** - `retrieve(query={call['query']!r}, "
                    f"k={call['k']})` returned {call['n_results']} chunks"
                )
            st.markdown("---")
            st.markdown("**Retrieved chunks (citations)**")
            for c in trace.all_chunks:
                st.markdown(
                    f"- {c.cite()} *(score {c.score:.3f})* - "
                    f"{c.text[:150]}..."
                )

    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": trace.answer})
