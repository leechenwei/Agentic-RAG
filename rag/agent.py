"""Agentic RAG loop using Gemini function calling.

The LLM is given a `retrieve` tool and decides:
  - WHEN to retrieve (some questions need no retrieval)
  - WHAT to query (it can rewrite/decompose the user's question)
  - WHETHER to retrieve again (if first results were weak)
  - WHEN it has enough context to answer

Retrieval pipeline used by the tool:
  hybrid (dense + BM25 + RRF)  →  rerank (BGE cross-encoder)  →  return PARENTS

Two entry points:
  - run_agent(question)        : non-streaming, returns full AgentTrace
  - run_agent_stream(question) : streaming, yields tokens then final AgentTrace
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from .retriever import RetrievedChunk, retrieve_hybrid_reranked, to_llm_context
from .session import require_api_key

MODEL = "gemini-2.5-flash"
MAX_AGENT_STEPS = 4
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BASE_WAIT = 5  # seconds

# Retrieval shape — how many candidates per retriever, how many final chunks.
RETRIEVE_CANDIDATE_K = 20   # broad net for dense + BM25 each before fusion
RETRIEVE_TOP_K = 5          # after reranker, how many chunks reach the LLM

SYSTEM_PROMPT = """You are a careful research assistant answering questions using a knowledge base.

You have ONE tool: `retrieve(query, k)` — hybrid search (dense + BM25 + reranker)
over the indexed documents. It returns the top-k most relevant PARENT blocks
with their citation tags.

Strategy:
1. For factual questions, ALWAYS call `retrieve` first. Do not answer from prior knowledge.
2. Rewrite the user's question into a focused search query (extract key entities/terms).
3. If the first retrieval is weak or incomplete, retrieve again with a different query.
4. For complex questions, decompose into sub-queries and retrieve for each.
5. Cite sources inline using bracketed numbers in order, like [1], [2], [3] — they refer
   to the chunks you retrieved, in the order you used them. The UI maps these to the
   source files separately. Keep prose clean: a sentence ends with "...[1]." not "...[file.txt#chunk7]."
6. If retrieved chunks don't contain the answer, say so honestly — do not fabricate.
"""

RETRIEVE_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="retrieve",
            description=(
                "Hybrid semantic + lexical search over the indexed document "
                "collection, with cross-encoder reranking. Returns top-k chunks."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="Search query — should contain key entities/terms.",
                    ),
                    "k": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of chunks to return (default 5, max 8).",
                    ),
                },
                required=["query"],
            ),
        )
    ]
)


@dataclass
class AgentTrace:
    """Record of one agent run — used by the UI and for debugging."""
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    all_chunks: list[RetrievedChunk] = field(default_factory=list)
    steps: int = 0


# ---------------------------------------------------------------------------
# Low-level wrappers (rate-limit retry, history conversion)
# ---------------------------------------------------------------------------

def _call_with_retry(api_method, *, model, contents, config):
    """Wrap any Gemini generation call with 429 exponential backoff.

    Pass `client.models.generate_content` (non-streaming) or
    `client.models.generate_content_stream` (streaming) as api_method.
    """
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return api_method(model=model, contents=contents, config=config)
        except ClientError as e:
            if e.code != 429 or attempt == RATE_LIMIT_RETRIES - 1:
                raise
            wait = RATE_LIMIT_BASE_WAIT * (2 ** attempt)
            # Honor server's RetryInfo hint if present
            try:
                for detail in e.details.get("error", {}).get("details", []):
                    if detail.get("@type", "").endswith("RetryInfo"):
                        d = detail.get("retryDelay", "")
                        if d.endswith("s"):
                            wait = max(wait, int(float(d[:-1])) + 1)
            except (AttributeError, KeyError, ValueError):
                pass
            print(f"[agent] rate-limited; waiting {wait}s (attempt {attempt + 1})")
            time.sleep(wait)


def _history_to_contents(history: list[dict]) -> list[types.Content]:
    """Convert simple {role, content} dicts into Gemini Content objects."""
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )
    return contents


def _execute_tool(
    fc: types.FunctionCall, trace: AgentTrace
) -> types.Part:
    """Execute one tool call. Records the call to the trace."""
    if fc.name != "retrieve":
        return types.Part.from_function_response(
            name=fc.name, response={"result": f"Unknown tool: {fc.name}"}
        )
    query = fc.args["query"]
    k = min(int(fc.args.get("k", RETRIEVE_TOP_K)), 8)
    chunks = retrieve_hybrid_reranked(
        query, k=k, candidate_k=RETRIEVE_CANDIDATE_K
    )
    trace.tool_calls.append(
        {"query": query, "k": k, "n_results": len(chunks)}
    )
    trace.all_chunks.extend(chunks)
    return types.Part.from_function_response(
        name="retrieve",
        response={"result": to_llm_context(chunks) or "(no results)"},
    )


# ---------------------------------------------------------------------------
# Non-streaming agent (original behavior, used by tests)
# ---------------------------------------------------------------------------

def run_agent(question: str, history: list[dict] | None = None) -> AgentTrace:
    """Run the agent loop. Returns full AgentTrace with the final answer."""
    client = genai.Client(api_key=require_api_key())
    trace = AgentTrace()

    contents = _history_to_contents(history or [])
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=question)])
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[RETRIEVE_TOOL],
        temperature=0.2,
    )

    for step in range(MAX_AGENT_STEPS):
        trace.steps = step + 1
        resp = _call_with_retry(
            client.models.generate_content,
            model=MODEL, contents=contents, config=config,
        )
        candidate = resp.candidates[0]
        parts = candidate.content.parts or []
        function_calls = [p.function_call for p in parts if p.function_call]

        if not function_calls:
            trace.answer = "".join(p.text for p in parts if p.text)
            return trace

        contents.append(candidate.content)
        response_parts = [_execute_tool(fc, trace) for fc in function_calls]
        contents.append(types.Content(role="user", parts=response_parts))

    trace.answer = "(agent hit max steps without producing a final answer)"
    return trace


# ---------------------------------------------------------------------------
# Streaming agent (Approach B: peek-then-stream)
# ---------------------------------------------------------------------------
# Why two calls on the final iteration:
#   We can't safely stream tokens until we know the turn is a TEXT turn
#   (and not a tool-calling turn whose structured output would garble the UI).
#   So we first do a non-streaming "peek" to detect the turn type. If it's a
#   tool call, we execute and loop. If it's text, we re-call with streaming
#   and yield tokens. Cost: one extra LLM call on the final turn. Benefit:
#   simple, predictable, never shows partial garbage to the user.
#
# A more efficient "Approach C" streams from the start and discriminates
# token-by-token. Cleaner but trickier under thinking-token interleaving.

def run_agent_stream(
    question: str, history: list[dict] | None = None
) -> Iterator[str | AgentTrace]:
    """Streaming agent loop.

    Yields:
      - Each token of the final answer as it arrives (strings)
      - Finally, the AgentTrace object (for the UI to render the trace panel)
    """
    client = genai.Client(api_key=require_api_key())
    trace = AgentTrace()

    contents = _history_to_contents(history or [])
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=question)])
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[RETRIEVE_TOOL],
        temperature=0.2,
    )

    for step in range(MAX_AGENT_STEPS):
        trace.steps = step + 1

        # PEEK (non-streaming): determine whether this turn is tools or text.
        resp = _call_with_retry(
            client.models.generate_content,
            model=MODEL, contents=contents, config=config,
        )
        candidate = resp.candidates[0]
        parts = candidate.content.parts or []
        function_calls = [p.function_call for p in parts if p.function_call]

        if function_calls:
            # Tool turn — execute, loop. Never streamed anything yet.
            contents.append(candidate.content)
            response_parts = [_execute_tool(fc, trace) for fc in function_calls]
            contents.append(types.Content(role="user", parts=response_parts))
            continue

        # TEXT turn — re-call with streaming and yield tokens live.
        full_text = ""
        stream = _call_with_retry(
            client.models.generate_content_stream,
            model=MODEL, contents=contents, config=config,
        )
        for chunk in stream:
            if chunk.text:
                full_text += chunk.text
                yield chunk.text
        trace.answer = full_text
        yield trace
        return

    trace.answer = "(agent hit max steps without producing a final answer)"
    yield trace
