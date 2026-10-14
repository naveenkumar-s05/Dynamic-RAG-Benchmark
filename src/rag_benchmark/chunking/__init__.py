"""Chunking strategies and a common factory."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ..parsers.models import ParsedDocument
from .base import BaseChunker, Chunk, ChunkingConfig
from .fixed_size import FixedSizeChunker
from .hybrid_dynamic import HybridDynamicChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .sentence_based import SentenceChunker
from .structural import StructuralChunker

_CHUNKERS = {
    "fixed": FixedSizeChunker,
    "sentence": SentenceChunker,
    "structural": StructuralChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "hybrid": HybridDynamicChunker,
}

STRATEGIES: List[str] = ["fixed", "sentence", "structural", "recursive", "semantic", "hybrid"]

STRATEGY_LABELS: Dict[str, str] = {
    "fixed": "A. Fixed-size",
    "sentence": "B. Sentence-based",
    "structural": "C. Structural",
    "recursive": "D. Recursive",
    "semantic": "E. Semantic",
    "hybrid": "F. Hybrid / dynamic",
}


def get_chunker(
    strategy: str,
    config: Optional[ChunkingConfig] = None,
    embedder: Optional[Callable] = None,
) -> BaseChunker:
    """Instantiate a chunker by strategy key."""
    if strategy not in _CHUNKERS:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Choose from {STRATEGIES}."
        )
    cls = _CHUNKERS[strategy]
    if strategy == "semantic":
        return cls(config, embedder=embedder)
    return cls(config)


def chunk_document(
    doc: ParsedDocument,
    strategy: str,
    config: Optional[ChunkingConfig] = None,
    embedder: Optional[Callable] = None,
) -> Tuple[List[Chunk], List[Chunk]]:
    """Chunk a parsed document, returning ``(children, parents)``.

    Parents are only produced by the structural and hybrid strategies; other
    strategies return an empty parent list.
    """
    chunker = get_chunker(strategy, config, embedder=embedder)
    if hasattr(chunker, "chunk_with_parents"):
        return chunker.chunk_with_parents(
            doc.blocks,
            source_id=doc.source_id,
            source_name=doc.source_name,
            source_type=doc.source_type,
            document_version=doc.document_version,
        )
    return (
        chunker.chunk(
            doc.blocks,
            source_id=doc.source_id,
            source_name=doc.source_name,
            source_type=doc.source_type,
            document_version=doc.document_version,
        ),
        [],
    )


__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkingConfig",
    "STRATEGIES",
    "STRATEGY_LABELS",
    "get_chunker",
    "chunk_document",
    "FixedSizeChunker",
    "SentenceChunker",
    "StructuralChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "HybridDynamicChunker",
]
