"""Hand-calculated metric tests plus edge cases."""

from __future__ import annotations

import math

import pytest

from rag_benchmark.evaluation import metrics as M


def test_recall_at_k_hand_calculated():
    relevant = {"a", "b", "c"}
    retrieved = ["a", "d", "b", "e"]
    assert M.recall_at_k(relevant, retrieved, 1) == pytest.approx(1 / 3)
    assert M.recall_at_k(relevant, retrieved, 2) == pytest.approx(1 / 3)
    assert M.recall_at_k(relevant, retrieved, 3) == pytest.approx(2 / 3)
    assert M.recall_at_k(relevant, retrieved, 4) == pytest.approx(2 / 3)


def test_recall_at_k_no_relevant_is_none():
    assert M.recall_at_k(set(), ["a", "b"], 2) is None


def test_mrr_examples():
    assert M.reciprocal_rank({"a"}, ["d", "b", "a"]) == pytest.approx(1 / 3)
    assert M.reciprocal_rank({"b"}, ["b", "a"]) == 1.0
    assert M.reciprocal_rank({"z"}, ["a", "b"]) == 0.0
    assert M.reciprocal_rank(set(), ["a"]) is None


def test_ndcg_perfect_ranking_is_one():
    graded = {"a": 2, "b": 1}
    assert M.ndcg_at_k(graded, ["a", "b"], 2) == pytest.approx(1.0)


def test_ndcg_hand_calculated():
    graded = {"a": 2, "b": 1}
    retrieved = ["x", "a"]
    dcg = 3 / math.log2(3)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert M.ndcg_at_k(graded, retrieved, 2) == pytest.approx(dcg / idcg, abs=1e-4)
    assert M.ndcg_at_k(graded, retrieved, 2) == pytest.approx(0.5213, abs=1e-3)


def test_ndcg_no_relevance_is_none():
    assert M.ndcg_at_k({}, ["a", "b"], 2) is None


def test_context_completeness():
    assert M.context_completeness(["A", "B"], ["contains A here"]) == pytest.approx(0.5)
    assert M.context_completeness(["A", "B"], ["A and B"]) == pytest.approx(1.0)
    assert M.context_completeness(["A"], ["nothing"]) == pytest.approx(0.0)
    assert M.context_completeness([], ["anything"]) == pytest.approx(1.0)


def test_missing_facts():
    assert M.missing_facts(["A", "B"], ["A only"]) == ["B"]


def test_citation_accuracy_correct_and_incorrect():
    predicted = [{"source_name": "a.md", "section": "X"}]
    expected = [{"source_name": "a.md", "section": "X"}]
    result = M.citation_accuracy(predicted, expected)
    assert result["precision"] == 1.0
    assert result["coverage"] == 1.0
    assert result["uncovered_evidence"] == 0

    wrong = M.citation_accuracy([{"source_name": "a.md", "section": "Y"}], expected)
    assert wrong["precision"] == 0.0
    assert wrong["uncovered_evidence"] == 1


def test_citation_accuracy_no_citations():
    result = M.citation_accuracy([], [{"source_name": "a.md"}])
    assert result["precision"] is None
    assert result["total_citations"] == 0
    assert result["uncovered_evidence"] == 1


def test_duplicate_chunk_rate():
    assert M.duplicate_chunk_rate(["a", "a", "b"]) == pytest.approx(1 / 3)
    assert M.duplicate_chunk_rate([]) == 0.0


def test_chunk_size_stats():
    stats = M.chunk_size_stats([10, 20, 30])
    assert stats["count"] == 3
    assert stats["avg"] == 20
    assert stats["min"] == 10
    assert stats["max"] == 30
    assert stats["median"] == 20
    empty = M.chunk_size_stats([])
    assert empty["avg"] is None


def test_aggregation_ignores_none():
    assert M.mean_ignoring_none([1.0, None, 0.0]) == pytest.approx(0.5)
    assert M.mean_ignoring_none([None, None]) is None
    assert M.count_defined([1.0, None]) == 1
