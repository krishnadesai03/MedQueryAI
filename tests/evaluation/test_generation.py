"""
Generation evaluation — answer quality tests.

Checks:
  1. Answers contain at least one expected keyword (groundedness proxy)
  2. Answers contain inline citations [1], [2], etc.
  3. Out-of-scope questions are refused (no hallucinated answer)
  4. Disclaimer is present in every answer

Run with:
    pytest tests/evaluation/test_generation.py -v

Note: This calls the Claude API for each question — keep the golden set
      small to control costs during development.
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.generation.generator import ask

GOLDEN_QA_PATH = Path(__file__).parent / "golden_qa.json"
ALL_QA = json.loads(GOLDEN_QA_PATH.read_text(encoding="utf-8"))
IN_SCOPE  = [q for q in ALL_QA if not q.get("expect_refusal")]
OUT_SCOPE = [q for q in ALL_QA if q.get("expect_refusal")]

CITATION_PATTERN = re.compile(r'\[\d+\]')
DISCLAIMER_KEYWORDS = ["informational", "consult", "clinician", "medical decisions"]


# ---------------------------------------------------------------------------
# In-scope questions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("item", [
    pytest.param(item, id=item["id"]) for item in IN_SCOPE
])
def test_answer_contains_keywords(item):
    """Answer must contain at least one expected keyword (case-insensitive)."""
    response = ask(item["question"])
    answer_lower = response.answer.lower()
    matched = [kw for kw in item["expected_keywords"] if kw.lower() in answer_lower]
    assert len(matched) > 0, (
        f"[{item['id']}] No expected keywords found in answer.\n"
        f"Expected any of: {item['expected_keywords']}\n"
        f"Answer: {response.answer[:300]}"
    )


@pytest.mark.parametrize("item", [
    pytest.param(item, id=item["id"]) for item in IN_SCOPE
])
def test_answer_contains_citation(item):
    """Every in-scope answer must have at least one inline citation [N]."""
    response = ask(item["question"])
    assert CITATION_PATTERN.search(response.answer), (
        f"[{item['id']}] No citation found in answer.\n"
        f"Answer: {response.answer[:300]}"
    )


@pytest.mark.parametrize("item", [
    pytest.param(item, id=item["id"]) for item in IN_SCOPE
])
def test_answer_contains_disclaimer(item):
    """Every answer must contain the safety disclaimer."""
    response = ask(item["question"])
    answer_lower = response.answer.lower()
    assert any(kw in answer_lower for kw in DISCLAIMER_KEYWORDS), (
        f"[{item['id']}] Disclaimer missing from answer.\n"
        f"Answer: {response.answer[:300]}"
    )


# ---------------------------------------------------------------------------
# Out-of-scope questions — must be refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("item", [
    pytest.param(item, id=item["id"]) for item in OUT_SCOPE
])
def test_out_of_scope_is_refused(item):
    """Questions outside the dataset must trigger a refusal."""
    response = ask(item["question"])
    assert response.refused, (
        f"[{item['id']}] Expected refusal but got an answer.\n"
        f"Answer: {response.answer[:300]}"
    )
