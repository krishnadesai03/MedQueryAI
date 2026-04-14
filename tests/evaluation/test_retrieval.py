"""
Retrieval evaluation — recall@K tests.

For each in-scope golden question, checks that at least one retrieved
document belongs to the expected disease category.

Run with:
    pytest tests/evaluation/test_retrieval.py -v
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.retrieval.hybrid_retriever import HybridRetriever

GOLDEN_QA_PATH = Path(__file__).parent / "golden_qa.json"
TOP_K = 3  # must find the right disease in top-3 results


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever()


@pytest.fixture(scope="module")
def golden_qa():
    with open(GOLDEN_QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    # Only in-scope questions have an expected_disease
    return [q for q in data if q.get("expected_disease")]


def test_golden_qa_file_exists():
    assert GOLDEN_QA_PATH.exists(), "golden_qa.json not found"


def test_golden_qa_not_empty(golden_qa):
    assert len(golden_qa) > 0, "No in-scope questions in golden QA set"


@pytest.mark.parametrize("item", [
    pytest.param(item, id=item["id"])
    for item in json.loads(GOLDEN_QA_PATH.read_text(encoding="utf-8"))
    if item.get("expected_disease")
])
def test_retrieval_recall(retriever, item):
    """Top-K retrieved docs must include at least one from the expected disease."""
    docs = retriever.retrieve(item["question"])[:TOP_K]
    retrieved_diseases = [
        doc.metadata.get("disease", "").lower() for doc in docs
    ]
    expected = item["expected_disease"].lower()
    assert any(expected in d for d in retrieved_diseases), (
        f"[{item['id']}] Expected disease '{item['expected_disease']}' not found in top-{TOP_K}.\n"
        f"Retrieved diseases: {retrieved_diseases}"
    )


@pytest.mark.parametrize("item", [
    pytest.param(item, id=item["id"])
    for item in json.loads(GOLDEN_QA_PATH.read_text(encoding="utf-8"))
    if item.get("expect_refusal")
])
def test_out_of_scope_retrieval(retriever, item):
    """Out-of-scope questions should still return docs (retrieval always returns something),
    but the generation layer should refuse. Just verify retrieval doesn't crash."""
    docs = retriever.retrieve(item["question"])
    assert isinstance(docs, list), "Retriever must return a list"
