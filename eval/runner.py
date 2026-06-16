"""Evaluation runner — measures whether the RAG pipeline answers correctly.

Two layers of metrics:

  RETRIEVAL (does the right source come back?)
    - Hit@k        : binary — was any relevant doc in top-k?
    - Recall@k     : fraction — what fraction of relevant docs are in top-k?
                     (Equal to Hit@k when each case has exactly 1 expected source.)
    - MRR          : Mean Reciprocal Rank — 1/rank_of_first_relevant, averaged.

  GENERATION (does the answer actually address the question?)
    - keyword_hit  : all required keywords appear in the answer (case-insensitive)
    - refusal_ok   : for negative cases, did the agent honestly refuse?

The runner returns a structured dict with per-case results AND aggregates,
which the UI uses for the metrics dashboard and history tracking.
"""
from __future__ import annotations

import time
from typing import Callable

from rag.agent import run_agent
from rag.retriever import retrieve_hybrid_reranked, RetrievedChunk

REFUSAL_PHRASES = ("don't know", "do not know", "no information", "cannot",
                   "not in", "isn't in", "not provided")


# ---------------------------------------------------------------------------
# Per-case scoring helpers
# ---------------------------------------------------------------------------

def _hit_at_k(chunks: list[RetrievedChunk], expected_source: str | None) -> int:
    if not expected_source:
        return 0
    return 1 if any(c.source == expected_source for c in chunks) else 0


def _reciprocal_rank(chunks: list[RetrievedChunk], expected_source: str | None) -> float:
    """1 / rank_of_first_match (1-indexed). 0 if no match."""
    if not expected_source:
        return 0.0
    for i, c in enumerate(chunks, start=1):
        if c.source == expected_source:
            return 1.0 / i
    return 0.0


def _keyword_check(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    lower = answer.lower()
    return all(k.lower() in lower for k in keywords)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    cases: list[dict],
    *,
    top_k: int = 5,
    candidate_k: int = 20,
    run_agent_too: bool = True,
    on_progress: Callable[[int, int, dict], None] | None = None,
) -> dict:
    """Evaluate the RAG pipeline against a golden dataset.

    Args:
      cases: list of dicts with question / expected_source / expected_keywords /
             expect_refusal.
      top_k: number of chunks to consider when computing Hit@k / MRR.
      candidate_k: candidates per retriever before fusion (for the production pipeline).
      run_agent_too: if True, also runs the agent end-to-end for keyword/refusal
                     scoring. Set False for fast retrieval-only eval.
      on_progress: optional callback(idx, total, per_case_result) for UI updates.

    Returns:
      A dict with `per_case` results and `aggregate` metrics, ready to display
      and persist to history.
    """
    started = time.time()
    per_case: list[dict] = []
    n = len(cases)

    hits = 0
    mrr_sum = 0.0
    answer_ok = 0
    refusal_ok = 0
    refusal_cases = 0
    answer_cases = 0

    for idx, case in enumerate(cases):
        question = case["question"]
        expected_source = case.get("expected_source")
        expected_keywords = case.get("expected_keywords") or []
        expect_refusal = bool(case.get("expect_refusal"))

        # Stage A — retrieval-only metrics (fast, deterministic)
        try:
            chunks = retrieve_hybrid_reranked(
                question, k=top_k, candidate_k=candidate_k
            )
        except Exception as e:
            chunks = []
            retrieval_error = str(e)
        else:
            retrieval_error = None

        hit = _hit_at_k(chunks, expected_source)
        rr = _reciprocal_rank(chunks, expected_source)
        if expected_source:
            hits += hit
            mrr_sum += rr

        retrieved_sources = [
            {"source": c.source, "chunk_index": c.chunk_index, "score": round(c.score, 4)}
            for c in chunks
        ]

        # Stage B — answer quality (only if requested; LLM calls cost money/time)
        agent_answer: str | None = None
        keyword_pass: bool | None = None
        refusal_pass: bool | None = None
        agent_error: str | None = None
        if run_agent_too:
            try:
                trace = run_agent(question)
                agent_answer = trace.answer
            except Exception as e:
                agent_error = str(e)

            if agent_answer is not None:
                if expect_refusal:
                    refusal_cases += 1
                    lower = agent_answer.lower()
                    rok = any(p in lower for p in REFUSAL_PHRASES)
                    refusal_pass = rok
                    if rok:
                        refusal_ok += 1
                else:
                    answer_cases += 1
                    kok = _keyword_check(agent_answer, expected_keywords)
                    keyword_pass = kok
                    if kok:
                        answer_ok += 1

        result = {
            "id": case.get("id"),
            "question": question,
            "expected_source": expected_source,
            "expected_keywords": expected_keywords,
            "expect_refusal": expect_refusal,
            "hit_at_k": hit if expected_source else None,
            "reciprocal_rank": round(rr, 4) if expected_source else None,
            "retrieved": retrieved_sources,
            "retrieval_error": retrieval_error,
            "agent_answer": agent_answer,
            "keyword_pass": keyword_pass,
            "refusal_pass": refusal_pass,
            "agent_error": agent_error,
        }
        per_case.append(result)
        if on_progress:
            on_progress(idx + 1, n, result)

    positive_cases = sum(1 for c in cases if c.get("expected_source"))
    aggregate = {
        "n_cases": n,
        "n_positive": positive_cases,
        "n_refusal": refusal_cases if run_agent_too else None,
        "hit_at_k": round(hits / positive_cases, 4) if positive_cases else None,
        "mrr": round(mrr_sum / positive_cases, 4) if positive_cases else None,
        "answer_keyword_acc": round(answer_ok / answer_cases, 4) if answer_cases else None,
        "refusal_acc": round(refusal_ok / refusal_cases, 4) if refusal_cases else None,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    return {
        "per_case": per_case,
        "aggregate": aggregate,
        "config": {
            "top_k": top_k,
            "candidate_k": candidate_k,
            "run_agent_too": run_agent_too,
        },
    }
