"""Command-line benchmark runner.

Usage:

    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --strategies fixed structural hybrid --top-k 5
    python scripts/run_benchmark.py --retrieval-mode hybrid --no-index-persist

Writes CSV and JSON reports to ``reports/`` and prints a comparison table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
for _p in (str(PROJECT_ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (  # noqa: E402
    BENCHMARK_DIR,
    INDEX_DIR,
    REPORTS_DIR,
    SAMPLE_DOCS_DIR,
    SETTINGS,
    ensure_directories,
)
from rag_benchmark.chunking import STRATEGIES  # noqa: E402
from rag_benchmark.chunking.base import ChunkingConfig  # noqa: E402
from rag_benchmark.evaluation import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkRunner,
    export_all,
    leaderboard_dataframe,
    load_dataset,
)
from rag_benchmark.indexing import get_embedding_service  # noqa: E402
from rag_benchmark.utils.logging_config import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the chunking-strategy benchmark.")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-mode", default="dense", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--target-tokens", type=int, default=400)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--overlap-tokens", type=int, default=60)
    parser.add_argument("--embedding-model", default=SETTINGS.embedding_model)
    parser.add_argument("--no-index-persist", action="store_true",
                        help="Do not write FAISS indexes to disk (size is estimated).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    setup_logging(SETTINGS.log_level, force=True)

    dataset = load_dataset(BENCHMARK_DIR / "questions.json")
    chunking_config = ChunkingConfig(
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    config = BenchmarkConfig(
        embedding_model=args.embedding_model,
        strategies=args.strategies,
        chunking_config=chunking_config,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        recall_ks=sorted({1, 3, args.top_k}),
        ndcg_k=args.top_k,
        persist_indexes=not args.no_index_persist,
    )

    print(f"Corpus: {dataset.corpus_id} ({len(dataset.questions)} questions, "
          f"{len(dataset.documents)} documents)")
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"Retrieval: {args.retrieval_mode}, top-k={args.top_k}\n")

    embedder = get_embedding_service(args.embedding_model)
    runner = BenchmarkRunner(
        dataset, SAMPLE_DOCS_DIR, config, embedder=embedder,
        index_base_dir=INDEX_DIR if config.persist_indexes else None,
    )
    report = runner.run_all(args.strategies)

    paths = export_all(report, REPORTS_DIR)
    print("Ranking (overall = mean of recall@k, mrr, context_completeness):\n")
    board = leaderboard_dataframe(report).round(3)
    print(board.to_string(index=False))
    print("\nFull detail in the reports below. Reports written:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
