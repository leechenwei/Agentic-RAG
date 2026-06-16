"""Test eval set — loaded from the canonical golden_dataset.json so tests
and the live app share the same source of truth. If you change the dataset
in the Streamlit UI, export it as JSON back to eval/golden_dataset.json
to keep tests in sync.
"""
import json
from pathlib import Path

EVAL_SET = json.loads(
    Path("eval/golden_dataset.json").read_text(encoding="utf-8")
)
