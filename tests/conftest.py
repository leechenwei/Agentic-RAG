"""Pytest setup — ingests the data/ corpus once into the fallback session
so retrieval tests have something to find. Without this, ephemeral Chroma
would start empty for each test run.
"""
import pytest

from rag.ingest import ingest_directory


@pytest.fixture(scope="session", autouse=True)
def _seed_corpus():
    """Ingest data/ into the test session's ephemeral Chroma once."""
    ingest_directory("data")
