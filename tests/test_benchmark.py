"""End-to-end benchmark engine tests with a fake embedder (no network)."""

from __future__ import annotations

import json

import pytest

from rag_benchmark.chunking.base import ChunkingConfig
from rag_benchmark.evaluation import (
    BenchmarkConfig,
    BenchmarkRunner,
    best_strategy,
    comparison_dataframe,
    export_all,
    key_metrics_dataframe,
    leaderboard_dataframe,
    load_dataset,
    passages_dataframe,
    per_query_dataframe,
)
from rag_benchmark.evaluation.dataset import relevance_grade


DOC = """# Security

Northwind uses AES-256 encryption at rest and TLS 1.3 in transit.

# Pricing

| Plan | Price |
| --- | --- |
| Basic | $5 |
| Enterprise | $49 |
"""


@pytest.fixture
def bench_env(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "doc1.md").write_text(DOC, encoding="utf-8")

    dataset_data = {
        "corpus_id": "test-corpus",
        "documents": ["doc1.md"],
        "questions": [
            {
                "question_id": "t1",
                "question": "What encryption is used at rest?",
                "category": "factual_lookup",
                "expected_sources": ["doc1.md"],
                "essential_facts": ["AES-256"],
                "primary_fact": "AES-256",
            },
            {
                "question_id": "t2",
                "question": "What is the price of the Enterprise plan?",
                "category": "table_lookup",
                "expected_sources": ["doc1.md"],
                "essential_facts": ["$49"],
                "primary_fact": "$49",
            },
        ],
    }
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(dataset_data), encoding="utf-8")
    dataset = load_dataset(path)
    return dataset, docs_dir, tmp_path


def _config(**kwargs):
    base = dict(
        embedding_model="fake",
        strategies=["fixed", "structural"],
        chunking_config=ChunkingConfig(target_tokens=40, max_tokens=80, overlap_tokens=0, min_tokens=5),
        top_k=3,
        retrieval_mode="dense",
        recall_ks=[1, 3],
        ndcg_k=3,
        persist_indexes=False,
    )
    base.update(kwargs)
    return BenchmarkConfig(**base)


def test_relevance_grade():
    dataset_question = None
    from rag_benchmark.evaluation.dataset import BenchmarkQuestion

    q = BenchmarkQuestion(
        question_id="x", question="?", category="c",
        expected_sources=["doc1.md"], essential_facts=["AES-256", "TLS 1.3"],
        primary_fact="AES-256",
    )
    from rag_benchmark.chunking.base import Chunk

    primary = Chunk(chunk_id="c1", source_id="s", source_name="doc1.md",
                    text="uses AES-256", content_type="paragraph", chunking_strategy="x",
                    chunk_index=0, token_count=3)
    secondary = Chunk(chunk_id="c2", source_id="s", source_name="doc1.md",
                      text="uses TLS 1.3", content_type="paragraph", chunking_strategy="x",
                      chunk_index=1, token_count=3)
    other = Chunk(chunk_id="c3", source_id="s", source_name="other.md",
                  text="uses AES-256", content_type="paragraph", chunking_strategy="x",
                  chunk_index=2, token_count=3)
    assert relevance_grade(primary, q) == 2
    assert relevance_grade(secondary, q) == 1
    assert relevance_grade(other, q) == 0


def test_benchmark_runs_all_strategies_fairly(bench_env, fake_embedder):
    dataset, docs_dir, _ = bench_env
    runner = BenchmarkRunner(dataset, docs_dir, _config(), embedder=fake_embedder)
    report = runner.run_all()

    assert len(report.strategies) == 2
    for run in report.strategies:
        assert run.retrieval_mode == "dense"
        assert run.top_k == 3
        assert run.embedding_model == "fake"
        assert len(run.per_query) == 2
        assert run.chunk_stats["total_chunks"] > 0
        assert "recall@1" in run.metrics
        assert run.metrics["recall@1"] is not None
        assert run.metrics["citation_accuracy"] is None
        for q in run.per_query:
            assert q.metrics["context_completeness"] is not None
            assert q.retrieved


def test_benchmark_relevance_is_high_when_retrieval_works(bench_env, fake_embedder):
    dataset, docs_dir, _ = bench_env
    runner = BenchmarkRunner(dataset, docs_dir, _config(), embedder=fake_embedder)
    run = runner.run("fixed")
    # The relevant chunk should be found within top-2 for these simple questions.
    high = sum(1 for q in run.per_query if q.metrics["mrr"] >= 0.5)
    assert high >= 1


def test_benchmark_identical_questions_across_strategies(bench_env, fake_embedder):
    dataset, docs_dir, _ = bench_env
    runner = BenchmarkRunner(dataset, docs_dir, _config(), embedder=fake_embedder)
    report = runner.run_all()
    question_sets = [
        {q.question_id for q in run.per_query} for run in report.strategies
    ]
    assert question_sets[0] == question_sets[1]


def test_export_all_writes_artifacts(bench_env, fake_embedder, tmp_path):
    dataset, docs_dir, _ = bench_env
    runner = BenchmarkRunner(dataset, docs_dir, _config(), embedder=fake_embedder)
    report = runner.run_all()
    out = tmp_path / "reports"
    paths = export_all(report, out)

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    comparison = comparison_dataframe(report)
    assert len(comparison) == 2
    assert "recall@1" in comparison.columns

    key = key_metrics_dataframe(report)
    assert list(key.columns) == ["strategy", "recall@3", "mrr", "context_completeness"]
    assert len(key) == 2

    board = leaderboard_dataframe(report)
    assert list(board.columns) == ["rank", "strategy", "overall", "recall@3", "mrr", "context_completeness"]
    assert list(board["rank"]) == list(range(1, len(board) + 1))
    assert board["overall"].is_monotonic_decreasing

    per_query = per_query_dataframe(report)
    assert len(per_query) == 4  # 2 strategies x 2 questions

    passages = passages_dataframe(report)
    assert not passages.empty

    loaded = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    assert len(loaded["strategies"]) == 2


def test_best_strategy_selection(bench_env, fake_embedder):
    dataset, docs_dir, _ = bench_env
    runner = BenchmarkRunner(dataset, docs_dir, _config(), embedder=fake_embedder)
    report = runner.run_all()
    winner = best_strategy(report, "recall@1")
    assert winner is not None
    assert winner[0].startswith(("A.", "B.", "C.", "D.", "E.", "F."))


def test_empty_corpus_is_safe(tmp_path, fake_embedder):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "empty.md").write_text("", encoding="utf-8")
    data = {
        "corpus_id": "empty",
        "documents": ["empty.md"],
        "questions": [
            {"question_id": "e1", "question": "anything?", "expected_sources": ["empty.md"],
             "essential_facts": ["missing"], "primary_fact": "missing"},
        ],
    }
    path = tmp_path / "q.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    dataset = load_dataset(path)
    runner = BenchmarkRunner(dataset, docs_dir, _config(strategies=["fixed"]), embedder=fake_embedder)
    report = runner.run_all()
    run = report.strategies[0]
    assert run.chunk_stats["total_chunks"] == 0
    assert run.per_query[0].retrieved == []
    assert run.metrics["recall@1"] is None  # no relevant chunks -> undefined
