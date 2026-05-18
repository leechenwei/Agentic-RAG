"""Agentic RAG loop using Gemini function calling.

The LLM is given a `retrieve` tool and decides:
  - WHEN to retrieve (some questions need no retrieval)
  - WHAT to query (it can rewrite/decompose the user's question)
  - WHETHER to retrieve again (if first results were weak)
  - WHEN it has enough context to answer
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from .retriever import RetrievedChunk, retrieve

MODEL = "gemini-2.5-flash"
MAX_AGENT_STEPS = 4   # tighter cap to stay within free-tier 5 req/min
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BASE_WAIT = 5  # seconds — Gemini free tier resets every minute

SYSTEM_PROMPT = """You are a careful research assistant answering questions using a knowledge base.

You have ONE tool: `retrieve(query, k)` — semantic search over indexed documents.

Strategy:
1. For factual questions, ALWAYS call `retrieve` first. Do not answer from prior knowledge.
2. Rewrite the user's question into a focused search query (extract key entities/terms).
3. If the first retrieval is weak or incomplete, retrieve again with a different query.
4. For complex questions, decompose into sub-queries and retrieve for each.
5. Cite sources inline using the format [source#chunkN] from the retrieved chunks.
6. If retrieved chunks don't contain the answer, say so honestly — do not fabricate.
"""

# Gemini function declaration — same concept as Anthropic's tool schema,
# different naming convention.
RETRIEVE_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="retrieve",
            description="Semantic search over the indexed document collection. Returns top-k chunks.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="Search query — should contain the key entities/terms.",
                    ),
                    "k": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of chunks to return (default 4, max 8).",
                    ),
                },
                required=["query"],
            ),
        )
    ]
)


@dataclass
class AgentTrace:
    """Record of one agent run — useful for the UI and for debugging."""
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    all_chunks: list[RetrievedChunk] = field(default_factory=list)
    steps: int = 0


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"{c.cite()} (score={c.score:.3f})\n{c.text}" for c in chunks
    )


def _generate_with_retry(client, *, model, contents, config):
    """Call generate_content with retry-with-backoff on 429 rate-limit errors.

    Gemini returns a `retryDelay` hint in the error payload. We honor it when
    present, otherwise fall back to exponential backoff (5s, 10s, 20s).
    """
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except ClientError as e:
            if e.code != 429 or attempt == RATE_LIMIT_RETRIES - 1:
                raise
            # Try to parse the provider-suggested retry delay
            wait = RATE_LIMIT_BASE_WAIT * (2 ** attempt)
            try:
                for detail in e.details.get("error", {}).get("details", []):
                    if detail.get("@type", "").endswith("RetryInfo"):
                        # Format like "4s"
                        d = detail.get("retryDelay", "")
                        if d.endswith("s"):
                            wait = max(wait, int(float(d[:-1])) + 1)
            except (AttributeError, KeyError, ValueError):
                pass
            print(f"[agent] rate-limited; waiting {wait}s (attempt {attempt + 1})")
            time.sleep(wait)


def _history_to_contents(history: list[dict]) -> list[types.Content]:
    """Convert simple {role, content} dicts into Gemini Content objects.

    Gemini uses 'user' and 'model' roles (not 'user' and 'assistant').
    """
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )
    return contents


def run_agent(question: str, history: list[dict] | None = None) -> AgentTrace:
    """Run the agentic RAG loop. Returns answer + full trace."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
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
        resp = _generate_with_retry(
            client, model=MODEL, contents=contents, config=config
        )

        candidate = resp.candidates[0]
        parts = candidate.content.parts or []

        # Collect any function calls the model wants to make this turn
        function_calls = [p.function_call for p in parts if p.function_call]

        if not function_calls:
            # No tool call → model is producing the final answer
            trace.answer = "".join(p.text for p in parts if p.text)
            return trace

        # Echo the model turn back into history so the next call sees it
        contents.append(candidate.content)

        # Execute each function call and append results
        response_parts = []
        for fc in function_calls:
            if fc.name == "retrieve":
                query = fc.args["query"]
                k = min(int(fc.args.get("k", 4)), 8)
                chunks = retrieve(query, k=k)
                trace.tool_calls.append(
                    {"query": query, "k": k, "n_results": len(chunks)}
                )
                trace.all_chunks.extend(chunks)
                response_parts.append(
                    types.Part.from_function_response(
                        name="retrieve",
                        response={"result": _format_chunks(chunks) or "(no results)"},
                    )
                )

        contents.append(types.Content(role="user", parts=response_parts))

    trace.answer = "(agent hit max steps without producing a final answer)"
    return trace
