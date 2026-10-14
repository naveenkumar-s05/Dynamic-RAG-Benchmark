"""Evaluation layer: dataset, metrics, benchmark engine and reporting."""

from .benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkRunner,
    QueryResult,
    StrategyRun,
    load_report,
    save_report,
)
from .dataset import (
    BenchmarkDataset,
    BenchmarkQuestion,
    EvidenceLocation,
    load_dataset,
    relevance_grade,
    relevant_chunk_ids,
)
from . import metrics
from .reporting import (
    best_strategy,
    comparison_dataframe,
    comparison_markdown,
    export_all,
    key_metrics_dataframe,
    leaderboard_dataframe,
    passages_dataframe,
    per_query_dataframe,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkRunner",
    "QueryResult",
    "StrategyRun",
    "load_report",
    "save_report",
    "BenchmarkDataset",
    "BenchmarkQuestion",
    "EvidenceLocation",
    "load_dataset",
    "relevance_grade",
    "relevant_chunk_ids",
    "metrics",
    "best_strategy",
    "comparison_dataframe",
    "comparison_markdown",
    "key_metrics_dataframe",
    "leaderboard_dataframe",
    "export_all",
    "passages_dataframe",
    "per_query_dataframe",
]
