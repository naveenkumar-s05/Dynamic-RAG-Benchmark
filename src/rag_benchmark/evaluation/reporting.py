"""Reporting and export helpers for benchmark results.

Produces comparison tables, per-query tables and passage tables, and writes CSV
and JSON artifacts from an executed run. Nothing here fabricates values; exports
only contain what the benchmark actually measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .benchmark import BenchmarkReport, save_report

METRIC_COLUMNS = [
    "recall@1",
    "recall@3",
    "recall@5",
    "mrr",
    "ndcg@5",
    "context_completeness",
    "citation_accuracy",
]

STAT_COLUMNS = [
    "total_chunks",
    "avg_chunk_tokens",
    "min_chunk_tokens",
    "max_chunk_tokens",
    "oversized_chunks",
    "oversized_rate",
    "duplicate_rate",
]

TIMING_COLUMNS = [
    "chunking_seconds",
    "embedding_seconds",
    "indexing_seconds",
    "retrieval_seconds",
    "total_seconds",
]

# The three scores that answer "did it work?" for a first look. Everything else
# is supporting/diagnostic detail and stays available in the full comparison.
KEY_METRIC_COLUMNS = ["mrr", "context_completeness"]


def comparison_dataframe(report: BenchmarkReport) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in report.strategies:
        row: Dict[str, Any] = {"strategy": run.label, "strategy_key": run.strategy}
        for col in METRIC_COLUMNS:
            row[col] = run.metrics.get(col)
        row["avg_chunk_tokens"] = run.chunk_stats.get("avg")
        row["min_chunk_tokens"] = run.chunk_stats.get("min")
        row["max_chunk_tokens"] = run.chunk_stats.get("max")
        for col in ("total_chunks", "oversized_chunks", "oversized_rate", "duplicate_rate"):
            row[col] = run.chunk_stats.get(col)
        for col in TIMING_COLUMNS:
            row[col] = run.timings.get(col)
        row["index_size_bytes"] = run.index_metadata.get("index_size_bytes")
        rows.append(row)
    return pd.DataFrame(rows)


def key_metrics_dataframe(report: BenchmarkReport) -> pd.DataFrame:
    """A slimmed-down view: strategy + the headline retrieval scores only.

    Shows Recall@k (at the run's ``top_k``), MRR and context completeness — the
    three numbers that answer "did this strategy retrieve the right content?".
    Full detail remains in :func:`comparison_dataframe` and the CSV/JSON exports.
    """
    df = comparison_dataframe(report)
    recall_col = f"recall@{report.top_k}"
    if recall_col not in df.columns:
        recall_cols = [c for c in df.columns if c.startswith("recall@")]
        recall_col = recall_cols[-1] if recall_cols else None

    columns = ["strategy"]
    if recall_col:
        columns.append(recall_col)
    for col in KEY_METRIC_COLUMNS:
        if col in df.columns:
            columns.append(col)
    return df[columns]


def leaderboard_dataframe(report: BenchmarkReport) -> pd.DataFrame:
    """Rank strategies by an ``overall`` score.

    ``overall`` is the mean of the headline retrieval scores (Recall@k at the
    run's ``top_k``, MRR and context completeness). All three are 0–1, so the
    mean is a simple, transparent combined score. Strategies are returned
    best-first with a 1-based ``rank`` column; ties keep report order.
    """
    df = comparison_dataframe(report)
    recall_col = f"recall@{report.top_k}"
    if recall_col not in df.columns:
        recall_cols = [c for c in df.columns if c.startswith("recall@")]
        recall_col = recall_cols[-1] if recall_cols else None

    metric_cols = ([recall_col] if recall_col else []) + list(KEY_METRIC_COLUMNS)
    metric_cols = [c for c in metric_cols if c in df.columns]

    numeric = df[metric_cols].apply(pd.to_numeric, errors="coerce")
    df["overall"] = numeric.mean(axis=1, skipna=True)
    df = df.sort_values("overall", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df[["rank", "strategy", "overall", *metric_cols]]


def per_query_dataframe(report: BenchmarkReport) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in report.strategies:
        for q in run.per_query:
            rows.append(
                {
                    "strategy": run.strategy,
                    "question_id": q.question_id,
                    "category": q.category,
                    "question": q.question,
                    "num_relevant_chunks": q.num_relevant_chunks,
                    "returned": len(q.retrieved),
                    "top1_source": q.retrieved[0]["source_name"] if q.retrieved else None,
                    "top1_chunk_id": q.retrieved[0]["chunk_id"] if q.retrieved else None,
                    "top1_relevant": q.retrieved[0]["relevant"] if q.retrieved else None,
                    "mrr": q.metrics.get("mrr"),
                    "recall@k": q.metrics.get(f"recall@{report.top_k}"),
                    "context_completeness": q.metrics.get("context_completeness"),
                    "missing_facts": "; ".join(q.missing_facts),
                    "latency_ms": round(q.latency_seconds * 1000, 3),
                }
            )
    return pd.DataFrame(rows)


def passages_dataframe(report: BenchmarkReport) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in report.strategies:
        for q in run.per_query:
            for hit in q.retrieved:
                rows.append(
                    {
                        "strategy": run.strategy,
                        "question_id": q.question_id,
                        "rank": hit["rank"],
                        "chunk_id": hit["chunk_id"],
                        "score": hit["score"],
                        "relevant": hit["relevant"],
                        "source_name": hit["source_name"],
                        "source_type": hit["source_type"],
                        "page_range": hit["page_range"],
                        "section_path": " > ".join(hit["section_path"]),
                        "content_type": hit["content_type"],
                        "retrieval_mode": hit["retrieval_mode"],
                        "text_preview": hit["text_preview"],
                    }
                )
    return pd.DataFrame(rows)


def best_strategy(report: BenchmarkReport, metric: str) -> Optional[Tuple[str, float]]:
    """Return (strategy_label, value) with the highest value for ``metric``."""
    best: Optional[Tuple[str, float]] = None
    for run in report.strategies:
        value = run.metrics.get(metric)
        if value is None:
            continue
        if best is None or value > best[1]:
            best = (run.label, float(value))
    return best


def export_all(report: BenchmarkReport, out_dir: str | Path) -> Dict[str, Path]:
    """Write comparison CSV, per-query CSV, passages CSV and the full JSON report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = report.run_id

    comparison_path = out_dir / f"{base}_comparison.csv"
    per_query_path = out_dir / f"{base}_per_query.csv"
    passages_path = out_dir / f"{base}_retrieved_passages.csv"
    json_path = out_dir / f"{base}_report.json"

    comparison_dataframe(report).to_csv(comparison_path, index=False, encoding="utf-8")
    per_query_dataframe(report).to_csv(per_query_path, index=False, encoding="utf-8")
    passages_dataframe(report).to_csv(passages_path, index=False, encoding="utf-8")
    save_report(report, json_path)

    return {
        "comparison_csv": comparison_path,
        "per_query_csv": per_query_path,
        "passages_csv": passages_path,
        "report_json": json_path,
    }


def comparison_markdown(report: BenchmarkReport) -> str:
    df = comparison_dataframe(report)
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)
