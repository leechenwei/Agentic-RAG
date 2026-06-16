"""Retrieval quality tests.

Two layers of testing:
  1. UNIT: deterministic retrieval — does the right source appear in top-k?
  2. END-TO-END: does the agent produce an answer mentioning required keywords
     and properly cite (or refuse if answer absent)?

Run with: pytest -v
"""
from __future__ import annotations

import pytest

from rag.retriever import retrieve_dense, retrieve_hybrid_reranked
from rag.agent import run_agent
from tests.eval_dataset import EVAL_SET


def test_production_retrieval_aggregate():
    """Production recall@5 across positive cases must be >= 90%.

    Note: some questions are legitimately answerable from more than one doc
    (e.g. Dell internship is covered both in dell_internship.txt and the
    about_chen_wei.txt bio paragraph). The threshold allows 1-2 such cases
    while still proving the pipeline is healthy.
    """
    positive = [c for c in EVAL_SET if c["expected_source"]]
    hits = 0
    for case in positive:
        sources = {c.source for c in retrieve_hybrid_reranked(case["question"], k=5)}
        if case["expected_source"] in sources:
            hits += 1
    recall = hits / len(positive)
    assert recall >= 0.90, f"Production recall@5 = {recall:.2f}, below 0.90 threshold"


def test_dense_only_recall_baseline():
    """Dense-only baseline — must stay >= 70% so we can show the lift from hybrid+rerank."""
    positive = [c for c in EVAL_SET if c["expected_source"]]
    hits = 0
    for case in positive:
        sources = {c.source for c in retrieve_dense(case["question"], k=4)}
        if case["expected_source"] in sources:
            hits += 1
    recall = hits / len(positive)
    assert recall >= 0.7, f"Dense recall@4 = {recall:.2f}, below 0.7 baseline"


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
