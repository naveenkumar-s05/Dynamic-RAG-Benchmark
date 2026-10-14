"""Dense vector retrieval with filtering and optional context expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..chunking.base import Chunk
from ..indexing.embeddings import EmbeddingService
from ..indexing.vector_store import VectorStore
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    query: str
    chunk_id: str
    rank: int
    score: float
    text: str
    source_id: str
    source_name: str
    source_type: str = ""
    page_number: Optional[int] = None
    page_range: Optional[str] = None
    section_path: List[str] = field(default_factory=list)
    content_type: str = ""
    chunking_strategy: str = ""
    chunk_index: int = 0
    block_ids: List[str] = field(default_factory=list)
    parent_chunk_id: Optional[str] = None
    retrieval_mode: str = "dense"
    expanded_from: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "score": self.score,
            "text": self.text,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "page_range": self.page_range,
            "section_path": list(self.section_path),
            "content_type": self.content_type,
            "chunking_strategy": self.chunking_strategy,
            "chunk_index": self.chunk_index,
            "block_ids": list(self.block_ids),
            "parent_chunk_id": self.parent_chunk_id,
            "retrieval_mode": self.retrieval_mode,
            "expanded_from": self.expanded_from,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        *,
        query: str,
        rank: int,
        score: float,
        retrieval_mode: str = "dense",
    ) -> "RetrievalResult":
        return cls(
            query=query,
            chunk_id=chunk.chunk_id,
            rank=rank,
            score=score,
            text=chunk.text,
            source_id=chunk.source_id,
            source_name=chunk.source_name,
            source_type=chunk.source_type,
            page_number=chunk.page_number,
            page_range=chunk.page_range,
            section_path=list(chunk.section_path),
            content_type=chunk.content_type,
            chunking_strategy=chunk.chunking_strategy,
            chunk_index=chunk.chunk_index,
            block_ids=list(chunk.block_ids),
            parent_chunk_id=chunk.parent_chunk_id,
            retrieval_mode=retrieval_mode,
            meta=dict(chunk.meta),
        )


def build_predicate(filters: Optional[Dict[str, Any]]):
    """Translate a filter dict into a chunk predicate.

    Supported keys: ``source_id``, ``source_name``, ``content_type``,
    ``source_type`` (each may be a value or a list of values).
    """
    if not filters:
        return None

    def _match(value, expected):
        if isinstance(expected, (list, tuple, set)):
            return value in expected
        return value == expected

    def predicate(chunk: Chunk) -> bool:
        if "source_id" in filters and not _match(chunk.source_id, filters["source_id"]):
            return False
        if "source_name" in filters and not _match(chunk.source_name, filters["source_name"]):
            return False
        if "content_type" in filters and not _match(chunk.content_type, filters["content_type"]):
            return False
        if "source_type" in filters and not _match(chunk.source_type, filters["source_type"]):
            return False
        return True

    return predicate


class DenseRetriever:
    """Embeds the query and searches the FAISS index."""

    mode = "dense"

    def __init__(self, store: VectorStore, embedder: EmbeddingService) -> None:
        self.store = store
        self.embedder = embedder

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
        predicate = build_predicate(filters)
        qv = self.embedder.encode([query])[0]
        hits = self.store.search(qv, k=k, predicate=predicate)
        results = [
            RetrievalResult.from_chunk(
                hit.chunk, query=query, rank=hit.rank, score=hit.score, retrieval_mode=self.mode
            )
            for hit in hits
        ]

        if expand_parents and parents:
            results = self._expand_parents(results, parents)
        if expand_neighbors > 0:
            results = self._expand_neighbors(results, expand_neighbors)
        return results

    def _expand_parents(
        self, results: List[RetrievalResult], parents: Dict[str, Chunk]
    ) -> List[RetrievalResult]:
        for result in results:
            if result.parent_chunk_id and result.parent_chunk_id in parents:
                parent = parents[result.parent_chunk_id]
                result.expanded_from = "child"
                result.meta = {**result.meta, "child_text": result.text}
                result.text = parent.text
        return results

    def _expand_neighbors(
        self, results: List[RetrievalResult], window: int
    ) -> List[RetrievalResult]:
        by_key = {
            (c.source_id, c.chunk_index): c
            for c in self.store.chunks
            if not c.meta.get("is_parent")
        }
        for result in results:
            if result.expanded_from is not None:
                continue
            base = result.chunk_index
            pieces: List[str] = []
            for offset in range(-window, window + 1):
                idx = base + offset
                if idx < 0:
                    continue
                neighbor = by_key.get((result.source_id, idx))
                if neighbor is not None:
                    pieces.append(neighbor.text)
            if pieces:
                result.text = "\n\n".join(pieces)
                result.expanded_from = "neighbors"
        return results
