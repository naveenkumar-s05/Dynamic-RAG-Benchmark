"""Optional BM25 keyword retrieval and hybrid (dense + lexical) retrieval.

Hybrid retrieval fuses dense and BM25 rankings with Reciprocal Rank Fusion
(RRF). During a benchmark, the *same* retrieval mode is used for every chunking
strategy so differences can be attributed to chunking rather than to retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..chunking.base import Chunk
from ..indexing.embeddings import EmbeddingService
from ..indexing.vector_store import VectorStore
from ..utils.logging_config import get_logger
from .retriever import DenseRetriever, RetrievalResult, build_predicate

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Retriever:
    """Lexical retrieval over the same chunk set using BM25."""

    mode = "bm25"

    def __init__(self, store: VectorStore) -> None:
        # BM25Plus is used instead of BM25Okapi because it has a positive IDF
        # floor (delta term), which avoids zero scores on very small corpora.
        from rank_bm25 import BM25Plus

        self.store = store
        self.chunks: List[Chunk] = list(store.chunks)
        corpus = [tokenize(c.retrieval_text()) or ["_empty_"] for c in self.chunks]
        self._bm25 = BM25Plus(corpus) if corpus else None

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        filters: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> List[RetrievalResult]:
        if self._bm25 is None or not self.chunks:
            return []
        predicate = build_predicate(filters)
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: List[RetrievalResult] = []
        for idx in ranked:
            chunk = self.chunks[idx]
            if predicate is not None and not predicate(chunk):
                continue
            if scores[idx] <= 0:
                # Skip clearly non-matching documents.
                if results:
                    break
            results.append(
                RetrievalResult.from_chunk(
                    chunk, query=query, rank=len(results) + 1,
                    score=float(scores[idx]), retrieval_mode=self.mode,
                )
            )
            if len(results) >= k:
                break
        return results


class HybridRetriever:
    """Reciprocal Rank Fusion of dense and BM25 results."""

    mode = "hybrid"

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingService,
        *,
        rrf_k: int = 60,
    ) -> None:
        self.dense = DenseRetriever(store, embedder)
        self.bm25 = BM25Retriever(store)
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        filters: Optional[Dict[str, Any]] = None,
        expand_parents: bool = False,
        parents: Optional[Dict[str, Chunk]] = None,
        expand_neighbors: int = 0,
    ) -> List[RetrievalResult]:
        fetch = max(k * 4, k)
        dense_hits = self.dense.retrieve(query, fetch, filters=filters)
        bm25_hits = self.bm25.retrieve(query, fetch, filters=filters)

        fused: Dict[str, Dict[str, Any]] = {}
        for hits in (dense_hits, bm25_hits):
            for hit in hits:
                entry = fused.setdefault(
                    hit.chunk_id,
                    {"result": hit, "score": 0.0, "best": hit.score},
                )
                entry["score"] += 1.0 / (self.rrf_k + hit.rank)
                entry["best"] = max(entry["best"], hit.score)

        ordered = sorted(fused.values(), key=lambda e: e["score"], reverse=True)
        results: List[RetrievalResult] = []
        for rank, entry in enumerate(ordered[:k], start=1):
            result = entry["result"]
            result.rank = rank
            result.score = float(entry["score"])
            result.retrieval_mode = self.mode
            result.meta = {**result.meta, "rrf_score": entry["score"], "best_component_score": entry["best"]}
            results.append(result)

        if expand_parents and parents:
            results = self.dense._expand_parents(results, parents)
        if expand_neighbors > 0:
            results = self.dense._expand_neighbors(results, expand_neighbors)
        return results


def get_retriever(
    mode: str,
    store: VectorStore,
    embedder: Optional[EmbeddingService] = None,
):
    """Return a retriever for ``mode``: ``dense`` | ``bm25`` | ``hybrid``."""
    mode = (mode or "dense").lower()
    if mode == "dense":
        if embedder is None:
            raise ValueError("Dense retrieval requires an embedding service.")
        return DenseRetriever(store, embedder)
    if mode == "bm25":
        return BM25Retriever(store)
    if mode == "hybrid":
        if embedder is None:
            raise ValueError("Hybrid retrieval requires an embedding service.")
        return HybridRetriever(store, embedder)
    raise ValueError(f"Unknown retrieval mode '{mode}'.")
