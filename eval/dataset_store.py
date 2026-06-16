"""Per-session golden Q&A dataset.

Each case the user labels:
    {"id", "question", "expected_source" (or None),
     "expected_keywords", "expect_refusal"}

Dataset lives in the per-session state (rag/session.py), so different users
never see each other's eval cases. Persistence is the user's responsibility
via the Export/Import buttons in the UI (download JSON, upload it next time).

This is the differentiator vs NotebookLM: the user OWNS the eval dataset
and can run regressions whenever they tweak chunking/embedding/prompts.
"""
from __future__ import annotations

import uuid
from typing import Iterable

from rag.session import get_session


def load_dataset() -> list[dict]:
    """Return the current session's eval dataset (list of case dicts)."""
    return get_session().get("golden_dataset", [])


def save_dataset(cases: list[dict]) -> None:
    """Replace the session's dataset with the given list."""
    get_session()["golden_dataset"] = list(cases)


def add_case(
    question: str,
    expected_source: str | None = None,
    expected_keywords: Iterable[str] | None = None,
    expect_refusal: bool = False,
) -> dict:
    """Append a new case and return it (with assigned id)."""
    cases = load_dataset()
    case = {
        "id": uuid.uuid4().hex[:8],
        "question": question.strip(),
        "expected_source": expected_source or None,
        "expected_keywords": [k.strip() for k in (expected_keywords or []) if k.strip()],
        "expect_refusal": bool(expect_refusal),
    }
    cases.append(case)
    save_dataset(cases)
    return case


def update_case(case_id: str, **fields) -> dict | None:
    cases = load_dataset()
    for c in cases:
        if c["id"] == case_id:
            c.update({k: v for k, v in fields.items() if k != "id"})
            save_dataset(cases)
            return c
    return None


def delete_case(case_id: str) -> bool:
    cases = load_dataset()
    new_cases = [c for c in cases if c["id"] != case_id]
    if len(new_cases) == len(cases):
        return False
    save_dataset(new_cases)
    return True


def import_from_json(payload: list[dict]) -> int:
    """Bulk import — replaces the entire dataset. Returns the new count."""
    normalized = []
    for c in payload:
        if "question" not in c:
            continue
        normalized.append({
            "id": c.get("id") or uuid.uuid4().hex[:8],
            "question": str(c["question"]).strip(),
            "expected_source": c.get("expected_source") or None,
            "expected_keywords": list(c.get("expected_keywords", [])),
            "expect_refusal": bool(c.get("expect_refusal", False)),
        })
    save_dataset(normalized)
    return len(normalized)
