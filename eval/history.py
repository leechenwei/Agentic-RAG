"""Per-session evaluation history.

Each run records timestamp + config snapshot + aggregate metrics, kept in
session state so users see only their own runs. Persists for the lifetime
of the browser tab — users can export the current dataset+history via the
UI's Export button.
"""
from __future__ import annotations

from datetime import datetime

from rag.session import get_session


def record_run(result: dict, note: str = "") -> dict:
    """Append a run summary to this session's history."""
    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "note": note.strip(),
        "config": result.get("config", {}),
        "aggregate": result.get("aggregate", {}),
    }
    history = get_session().setdefault("eval_history", [])
    history.append(summary)
    return summary


def load_history(limit: int = 20) -> list[dict]:
    """Return the latest `limit` runs (newest first)."""
    history = get_session().get("eval_history", [])
    return list(reversed(history))[:limit]


def delta_vs_previous(history: list[dict]) -> list[dict]:
    """Annotate each run with deltas vs the previous (older) run."""
    annotated = []
    for i, run in enumerate(history):
        deltas: dict[str, float] = {}
        if i + 1 < len(history):
            prev = history[i + 1]
            for key in ("hit_at_k", "mrr", "answer_keyword_acc", "refusal_acc"):
                cur_val = run["aggregate"].get(key)
                prev_val = prev["aggregate"].get(key)
                if cur_val is not None and prev_val is not None:
                    deltas[key] = round(cur_val - prev_val, 4)
        annotated.append({**run, "delta": deltas})
    return annotated
