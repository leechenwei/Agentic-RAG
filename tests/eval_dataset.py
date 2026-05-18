"""Hand-curated eval set tied to the actual ingested documents.

Each case has:
  - question: the user query
  - expected_source: filename the answer MUST come from (or None for refusals)
  - expected_keywords: substrings that should appear in the final answer
  - expect_refusal (optional): True if the agent must say "I don't know"
"""

EVAL_SET = [
    # ---------- Single-hop, MaiStorage intro deck ----------
    {
        "question": "Where did Chen Wei study and what was his CGPA?",
        "expected_source": "maistorage_intro.txt",
        "expected_keywords": ["university of malaya", "3.80"],
    },
    {
        "question": "What internship did Chen Wei do at Dell?",
        "expected_source": "maistorage_intro.txt",
        "expected_keywords": ["dell", "software engineer", "intern"],
    },
    {
        "question": "What was the final year project about?",
        "expected_source": "maistorage_intro.txt",
        "expected_keywords": ["dialogue", "rasa"],
    },
    {
        "question": "What confidence threshold did the dialogue system use?",
        "expected_source": "maistorage_intro.txt",
        "expected_keywords": ["70"],
    },

    # ---------- Single-hop, reference docs ----------
    {
        "question": "What is HNSW and why is it used in vector databases?",
        "expected_source": "vector_db_overview.txt",
        "expected_keywords": ["hnsw", "navigable"],
    },
    {
        "question": "How many dimensions does all-MiniLM-L6-v2 produce?",
        "expected_source": "embeddings_overview.txt",
        "expected_keywords": ["384"],
    },
    {
        "question": "What is the difference between a bi-encoder and a cross-encoder?",
        "expected_source": "embeddings_overview.txt",
        "expected_keywords": ["bi-encoder", "cross-encoder"],
    },

    # ---------- Multi-hop / multi-source (showcase agentic behavior) ----------
    # These force the agent to retrieve more than once because the answer
    # spans multiple files in the corpus.
    {
        "question": "Which vector database is the project using, and why is it a good choice for prototypes?",
        "expected_source": "vector_db_overview.txt",
        "expected_keywords": ["chromadb"],
    },
    {
        "question": "Chen Wei built agentic AI workflows — what tools did he use and what is agentic RAG in general?",
        "expected_source": "maistorage_intro.txt",  # primary source
        "expected_keywords": ["n8n", "dify", "agentic"],
    },

    # ---------- Negative test (refusal) ----------
    # Answer is NOT in the corpus. The agent must refuse, not hallucinate.
    # This visibly proves the system handles "no answer found" correctly.
    {
        "question": "What is the airspeed velocity of an unladen swallow?",
        "expected_source": None,
        "expected_keywords": [],
        "expect_refusal": True,
    },
]
