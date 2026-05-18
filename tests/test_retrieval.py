"""Retrieval quality tests.

Two layers of testing:
  1. UNIT: deterministic retrieval — does the right source appear in top-k?
  2. END-TO-END: does the agent produce an answer mentioning required keywords
     and properly cite (or refuse if answer absent)?

Run with: pytest -v
"""
from __future__ import annotations

import pytest

from rag.retriever import retrieve
from rag.agent import run_agent
from tests.eval_dataset import EVAL_SET


@pytest.mark.parametrize("case", [c for c in EVAL_SET if c["expected_source"]])
def test_retrieval_recall_at_k(case):
    """Top-k retrieval must include the expected source document."""
    chunks = retrieve(case["question"], k=4)
    sources = {c.source for c in chunks}
    assert case["expected_source"] in sources, (
        f"Expected {case['expected_source']} in top-4 retrieval for "
        f"{case['question']!r}, got {sources}"
    )


def test_retrieval_recall_at_k_aggregate():
    """Aggregate recall@4 across all positive cases must be >= 80%."""
    positive = [c for c in EVAL_SET if c["expected_source"]]
    hits = 0
    for case in positive:
        sources = {c.source for c in retrieve(case["question"], k=4)}
        if case["expected_source"] in sources:
            hits += 1
    recall = hits / len(positive)
    assert recall >= 0.8, f"Recall@4 = {recall:.2f}, below 0.8 threshold"


@pytest.mark.slow
@pytest.mark.parametrize("case", EVAL_SET)
def test_agent_end_to_end(case):
    """Agent answers should contain expected keywords (or refuse for negatives)."""
    result = run_agent(case["question"])
    answer = result.answer.lower()
    for kw in case["expected_keywords"]:
        assert kw.lower() in answer, (
            f"Expected keyword {kw!r} in answer to {case['question']!r}, "
            f"got: {result.answer}"
        )
    if case.get("expect_refusal"):
        # Refusal cases should NOT have retrieved chunks that were used
        # to fabricate an answer with confident facts.
        assert any(
            phrase in answer
            for phrase in ["don't know", "not", "no information", "cannot"]
        )
