"""Per-session state container.

Each Streamlit user (each browser session) gets isolated:
  - ChromaDB client (in-memory)  — their documents
  - BM25 cache                    — their lexical index
  - API key                       — their Gemini credentials (never persisted)
  - Golden dataset                — their eval Q&A pairs
  - Eval history                  — their past evaluation runs

This module abstracts away Streamlit so `rag/` and `eval/` don't import
streamlit directly. Tests / CLI get a module-global fallback session.
"""
from __future__ import annotations

import os
import uuid

import chromadb
from chromadb.utils import embedding_functions

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "docs"

# Single shared session for tests, CLI, and other non-Streamlit contexts.
_fallback_session: dict | None = None


def _new_session() -> dict:
    """Initialize all per-session state for a new user."""
    return {
        "session_id": uuid.uuid4().hex[:8],
        "chroma_client": chromadb.EphemeralClient(),
        "bm25_cache": None,
        "api_key": None,
        "golden_dataset": [],
        "eval_history": [],
    }


def get_session() -> dict:
    """Return the active session state.

    Inside Streamlit: state scoped to the user's browser tab via session_state.
    Outside Streamlit: a single module-global session for tests / CLI.
    """
    try:
        import streamlit as st
        # Accessing session_state outside a Streamlit script run raises.
        if "rag_session" not in st.session_state:
            st.session_state["rag_session"] = _new_session()
        return st.session_state["rag_session"]
    except Exception:
        global _fallback_session
        if _fallback_session is None:
            _fallback_session = _new_session()
        return _fallback_session


def reset_session() -> None:
    """Re-initialize the current session (used by 'reset' buttons in the UI)."""
    try:
        import streamlit as st
        st.session_state["rag_session"] = _new_session()
        return
    except Exception:
        global _fallback_session
        _fallback_session = _new_session()


def get_collection():
    """Get the user's Chroma collection (creates it if needed)."""
    client = get_session()["chroma_client"]
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Gemini API key — BYOK (Bring Your Own Key)
# ---------------------------------------------------------------------------

class MissingAPIKey(RuntimeError):
    """Raised when an LLM call is attempted with no Gemini API key set.
    The Streamlit UI catches this and shows a friendly sidebar prompt."""


def get_api_key() -> str | None:
    """Return the active Gemini key. Session takes precedence over env var
    so a logged-in user always uses their own key; env var is a local-dev
    convenience that won't override a user's choice."""
    return get_session().get("api_key") or os.environ.get("GEMINI_API_KEY")


def set_api_key(key: str | None) -> None:
    """Store (or clear) the session's Gemini API key. Never persisted to disk."""
    get_session()["api_key"] = (key or "").strip() or None


def require_api_key() -> str:
    """Return the API key or raise MissingAPIKey with a clear message."""
    key = get_api_key()
    if not key:
        raise MissingAPIKey(
            "No Gemini API key set. Paste your key in the sidebar to enable "
            "chat and agent-based evaluation. Get a free key at "
            "https://aistudio.google.com/apikey"
        )
    return key
