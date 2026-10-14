"""Fair benchmarking engine.

For every strategy the engine starts from the same parsed documents, chunks them
with the same configuration factory, embeds with the same model, builds an
independent FAISS index, runs the identical question set with the same retrieval
mode and top-k, and computes the same metrics. Only the chunking strategy varies.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..chunking import STRATEGY_LABELS, STRATEGIES, chunk_document
from ..chunking.base import Chunk, ChunkingConfig
from ..indexing.embeddings import EmbeddingService, get_embedding_service
from ..indexing.vector_store import VectorStore, index_directory, make_index_key
from ..parsers import parse_document
from ..parsers.models import ParsedDocument
from ..retrieval import get_retriever
from ..utils.logging_config import get_logger
from ..utils.tokens import tokenizer_name
from . import metrics as M
from .dataset import BenchmarkDataset, BenchmarkQuestion, relevant_chunk_ids

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    strategies: List[str] = field(default_factory=lambda: list(STRATEGIES))
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    top_k: int = 5
    retrieval_mode: str = "dense"
    recall_ks: List[int] = field(default_factory=lambda: [1, 3, 5])
    ndcg_k: int = 5
    expand_parents: bool = False
    expand_neighbors: int = 0
    persist_indexes: bool = True
    embed_batch_size: int = 64

    def validate(self) -> None:
        self.chunking_config.validate()
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        unknown = set(self.strategies) - set(STRATEGIES)
        if unknown:
            raise ValueError(f"Unknown strategies: {sorted(unknown)}")
        if self.retrieval_mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"Unknown retrieval mode '{self.retrieval_mode}'.")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("chunking_config")
        data["chunking_config"] = self.chunking_config.to_dict()
        return data


@dataclass
class QueryResult:
    question_id: str
    question: str
    category: str
    retrieved: List[Dict[str, Any]]
    metrics: Dict[str, Optional[float]]
    missing_facts: List[str]
    num_relevant_chunks: int
    latency_seconds: float
    retrieval_mode: str = "dense"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyRun:
    strategy: str
    label: str
    chunking_config: Dict[str, Any]
    retrieval_mode: str
    top_k: int
    embedding_model: str
    metrics: Dict[str, Optional[float]]
    chunk_stats: Dict[str, Any]
    timings: Dict[str, float]
    per_query: List[QueryResult]
    index_metadata: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["per_query"] = [q.to_dict() for q in self.per_query]
        return data


@dataclass
class BenchmarkReport:
    run_id: str
    timestamp: str
    corpus_id: str
    config: Dict[str, Any]
    embedding_model: str
    retrieval_mode: str
    top_k: int
    chunking_config: Dict[str, Any]
    tokenizer: str
    dataset: Dict[str, Any]
    parse_time_seconds: float
    document_warnings: Dict[str, List[str]]
    strategies: List[StrategyRun]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["strategies"] = [s.to_dict() for s in self.strategies]
        return data


class BenchmarkRunner:
    """Runs a fair comparison of chunking strategies over a fixed dataset."""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        documents_dir: str | Path,
        config: Optional[BenchmarkConfig] = None,
        embedder: Optional[EmbeddingService] = None,
        index_base_dir: Optional[str | Path] = None,
        enable_ocr: bool = False,
    ) -> None:
        self.dataset = dataset
        self.documents_dir = Path(documents_dir)
        self.config = config or BenchmarkConfig()
        self.config.validate()
        self.embedder = embedder or get_embedding_service(self.config.embedding_model)
        self.index_base_dir = Path(index_base_dir) if index_base_dir else None
        self.enable_ocr = enable_ocr
        self._parsed: Optional[List[ParsedDocument]] = None
        self.parse_warnings: Dict[str, List[str]] = {}

    # ---------------------------------------------------------------- corpus
    def prepare_corpus(self) -> List[ParsedDocument]:
        if self._parsed is not None:
            return self._parsed
        parsed: List[ParsedDocument] = []
        self.parse_warnings = {}
        for name in self.dataset.documents:
            path = self.documents_dir / name
            doc = parse_document(path, enable_ocr=self.enable_ocr)
            parsed.append(doc)
            if doc.warnings:
                self.parse_warnings[name] = list(doc.warnings)
        self._parsed = parsed
        return parsed

    # ------------------------------------------------------------------- run
    def run(self, strategy: str) -> StrategyRun:
        parsed = self.prepare_corpus()
        warnings: List[str] = []

        # 1. Chunk (same config factory for every strategy).
        t0 = time.perf_counter()
        all_children: List[Chunk] = []
        parents: Dict[str, Chunk] = {}
        for doc in parsed:
            children, doc_parents = chunk_document(
                doc, strategy, self.config.chunking_config, embedder=self.embedder.encode
                if strategy == "semantic" else None
            )
            all_children.extend(children)
            for parent in doc_parents:
                parents[parent.chunk_id] = parent
        chunking_time = time.perf_counter() - t0

        # 2. Embed + index (independent per strategy).
        store = VectorStore(
            dimension=self.embedder.dimension,
            embedding_model=self.config.embedding_model,
            strategy=strategy,
            chunking_config=self.config.chunking_config,
            corpus_id=self.dataset.corpus_id,
        )
        embedding_time = 0.0
        index_add_time = 0.0
        if all_children:
            texts = [c.retrieval_text() for c in all_children]
            vectors_parts = []
            for i in range(0, len(texts), self.config.embed_batch_size):
                batch = texts[i : i + self.config.embed_batch_size]
                t_emb = time.perf_counter()
                vectors_parts.append(self.embedder.encode(batch))
                embedding_time += time.perf_counter() - t_emb
            vectors = np.vstack(vectors_parts) if vectors_parts else np.zeros((0, store.dimension))
            t_add = time.perf_counter()
            store.add(vectors, all_children)
            index_add_time = time.perf_counter() - t_add
        indexing_time = embedding_time + index_add_time

        index_meta = store.metadata().to_dict()
        index_size_bytes = self._measure_index_size(store, strategy, index_meta)

        # 3. Retrieve + evaluate.
        retriever = get_retriever(self.config.retrieval_mode, store, self.embedder)
        per_query: List[QueryResult] = []
        for question in self.dataset.questions:
            per_query.append(self._evaluate_question(question, retriever, all_children, parents))

        metrics = self._aggregate(per_query)
        chunk_stats = self._chunk_stats(all_children)
        timings = {
            "chunking_seconds": chunking_time,
            "embedding_seconds": embedding_time,
            "index_add_seconds": index_add_time,
            "indexing_seconds": indexing_time,
            "retrieval_seconds": sum(q.latency_seconds for q in per_query),
            "total_seconds": chunking_time + indexing_time + sum(q.latency_seconds for q in per_query),
        }

        return StrategyRun(
            strategy=strategy,
            label=STRATEGY_LABELS.get(strategy, strategy),
            chunking_config=self.config.chunking_config.to_dict(),
            retrieval_mode=self.config.retrieval_mode,
            top_k=self.config.top_k,
            embedding_model=self.config.embedding_model,
            metrics=metrics,
            chunk_stats=chunk_stats,
            timings=timings,
            per_query=per_query,
            index_metadata={**index_meta, "index_size_bytes": index_size_bytes},
            warnings=warnings,
        )

    def run_all(self, strategies: Optional[Sequence[str]] = None) -> BenchmarkReport:
        chosen = list(strategies or self.config.strategies)
        t_parse = time.perf_counter()
        self.prepare_corpus()
        parse_time = time.perf_counter() - t_parse

        runs: List[StrategyRun] = []
        for strategy in chosen:
            logger.info("Benchmarking strategy '%s'...", strategy)
            runs.append(self.run(strategy))

        notes = [
            "All strategies use the same parsed documents, embedding model, "
            f"retrieval mode ({self.config.retrieval_mode}), top-k "
            f"({self.config.top_k}) and question set.",
            f"Token counting uses {tokenizer_name()}.",
            "In retrieval-only mode citation accuracy is not applicable and is "
            "reported as null; it is evaluated when an LLM generates cited answers.",
        ]

        return BenchmarkReport(
            run_id=f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            corpus_id=self.dataset.corpus_id,
            config=self.config.to_dict(),
            embedding_model=self.config.embedding_model,
            retrieval_mode=self.config.retrieval_mode,
            top_k=self.config.top_k,
            chunking_config=self.config.chunking_config.to_dict(),
            tokenizer=tokenizer_name(),
            dataset={
                "corpus_id": self.dataset.corpus_id,
                "num_questions": len(self.dataset.questions),
                "documents": list(self.dataset.documents),
                "relevance_rule": self.dataset.relevance_rule,
            },
            parse_time_seconds=parse_time,
            document_warnings=self.parse_warnings,
            strategies=runs,
            notes=notes,
        )

    # -------------------------------------------------------------- helpers
    def _evaluate_question(
        self,
        question: BenchmarkQuestion,
        retriever,
        all_children: List[Chunk],
        parents: Dict[str, Chunk],
    ) -> QueryResult:
        t0 = time.perf_counter()
        results = retriever.retrieve(
            question.question,
            k=self.config.top_k,
            expand_parents=self.config.expand_parents,
            parents=parents,
            expand_neighbors=self.config.expand_neighbors,
        )
        latency = time.perf_counter() - t0

        retrieved_ids = [r.chunk_id for r in results]
        graded = relevant_chunk_ids(all_children, question)
        relevant_set = set(graded)
        retrieved_texts = [r.text for r in results]

        metrics: Dict[str, Optional[float]] = {}
        for k in self.config.recall_ks:
            metrics[f"recall@{k}"] = M.recall_at_k(relevant_set, retrieved_ids, k)
        metrics["mrr"] = M.reciprocal_rank(relevant_set, retrieved_ids)
        metrics[f"ndcg@{self.config.ndcg_k}"] = M.ndcg_at_k(
            graded, retrieved_ids, self.config.ndcg_k
        )
        metrics["context_completeness"] = M.context_completeness(
            question.essential_facts, retrieved_texts
        )
        metrics["citation_accuracy"] = None  # requires generated answers

        retrieved_summary = [
            {
                "rank": r.rank,
                "chunk_id": r.chunk_id,
                "score": r.score,
                "source_name": r.source_name,
                "source_type": r.source_type,
                "page_range": r.page_range,
                "section_path": list(r.section_path),
                "content_type": r.content_type,
                "relevant": graded.get(r.chunk_id, 0),
                "text_preview": (r.text[:240] + "...") if len(r.text) > 240 else r.text,
                "retrieval_mode": r.retrieval_mode,
            }
            for r in results
        ]

        return QueryResult(
            question_id=question.question_id,
            question=question.question,
            category=question.category,
            retrieved=retrieved_summary,
            metrics=metrics,
            missing_facts=M.missing_facts(question.essential_facts, retrieved_texts),
            num_relevant_chunks=len(relevant_set),
            latency_seconds=latency,
            retrieval_mode=self.config.retrieval_mode,
        )

    def _aggregate(self, per_query: List[QueryResult]) -> Dict[str, Optional[float]]:
        if not per_query:
            return {}
        keys = set()
        for q in per_query:
            keys.update(q.metrics.keys())
        aggregate: Dict[str, Optional[float]] = {}
        for key in sorted(keys):
            values = [q.metrics.get(key) for q in per_query]
            aggregate[key] = M.mean_ignoring_none(values)
            aggregate[f"{key}__defined"] = M.count_defined(values)
        aggregate["retrieval_latency_ms_mean"] = (
            sum(q.latency_seconds for q in per_query) / len(per_query) * 1000.0
        )
        return aggregate

    def _chunk_stats(self, children: List[Chunk]) -> Dict[str, Any]:
        tokens = [c.token_count for c in children]
        stats = M.chunk_size_stats(tokens)
        oversized = sum(1 for c in children if c.token_count > self.config.chunking_config.max_tokens)
        stats.update(
            {
                "total_chunks": len(children),
                "oversized_chunks": oversized,
                "oversized_rate": (oversized / len(children)) if children else 0.0,
                "duplicate_rate": M.duplicate_chunk_rate([c.text for c in children]),
                "total_tokens": sum(tokens) if tokens else 0,
            }
        )
        return stats

    def _measure_index_size(
        self, store: VectorStore, strategy: str, index_meta: Dict[str, Any]
    ) -> Optional[int]:
        if not self.config.persist_indexes:
            # Estimate: float32 vectors + UTF-8 text.
            text_bytes = sum(len(c.text.encode("utf-8")) for c in store.chunks)
            return store.size * store.dimension * 4 + text_bytes
        if self.index_base_dir is None:
            return None
        key = make_index_key(
            strategy,
            self.config.chunking_config,
            self.config.embedding_model,
            self.dataset.corpus_id,
        )
        directory = index_directory(self.index_base_dir, key)
        store.save(directory)
        total = 0
        for path in directory.iterdir():
            if path.is_file():
                total += path.stat().st_size
        return total


# ------------------------------------------------------------------- export
def save_report(report: BenchmarkReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=_json_default)
    path.write_text(payload, encoding="utf-8")
    logger.info("Saved benchmark report to %s", path)
    return path


def load_report(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
