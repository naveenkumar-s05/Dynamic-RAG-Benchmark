"""High-level helpers wiring parsing, chunking, indexing and retrieval.

Kept intentionally small so the Streamlit app and command-line scripts share the
same behavior and so benchmarking and interactive use cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .chunking import chunk_document
from .chunking.base import Chunk, ChunkingConfig
from .indexing.embeddings import EmbeddingService, get_embedding_service
from .indexing.vector_store import VectorStore
from .parsers import parse_document
from .parsers.models import ParsedDocument
from .retrieval import get_retriever


def parse_paths(paths: Sequence[str | Path], *, enable_ocr: bool = False) -> List[ParsedDocument]:
    """Parse a list of files into :class:`ParsedDocument` objects."""
    return [parse_document(Path(p), enable_ocr=enable_ocr) for p in paths]


def collect_chunks(
    documents: Sequence[ParsedDocument],
    strategy: str,
    config: Optional[ChunkingConfig] = None,
    embedder: Optional[EmbeddingService] = None,
) -> Tuple[List[Chunk], Dict[str, Chunk]]:
    """Chunk every document with one strategy.

    Returns ``(children, parents_by_id)``. Parents are only produced by the
    structural and hybrid strategies.
    """
    config = config or ChunkingConfig()
    children: List[Chunk] = []
    parents: Dict[str, Chunk] = {}
    embed_fn = embedder.encode if (embedder and strategy == "semantic") else None
    for doc in documents:
        doc_children, doc_parents = chunk_document(doc, strategy, config, embedder=embed_fn)
        children.extend(doc_children)
        for parent in doc_parents:
            parents[parent.chunk_id] = parent
    return children, parents


def build_index(
    documents: Sequence[ParsedDocument],
    strategy: str,
    *,
    config: Optional[ChunkingConfig] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedder: Optional[EmbeddingService] = None,
    corpus_id: str = "interactive",
    persist_dir: Optional[str | Path] = None,
) -> Tuple[VectorStore, List[Chunk], Dict[str, Chunk]]:
    """Build a FAISS index for one strategy over the given documents."""
    config = config or ChunkingConfig()
    embedder = embedder or get_embedding_service(embedding_model)
    children, parents = collect_chunks(documents, strategy, config, embedder)

    store = VectorStore(
        dimension=embedder.dimension,
        embedding_model=embedding_model,
        strategy=strategy,
        chunking_config=config,
        corpus_id=corpus_id,
    )
    store.build_from_chunks(children, embedder.encode)
    if persist_dir is not None and children:
        store.save(persist_dir)
    return store, children, parents


def retrieve(
    store: VectorStore,
    query: str,
    *,
    mode: str = "dense",
    top_k: int = 5,
    embedder: Optional[EmbeddingService] = None,
    filters: Optional[dict] = None,
    expand_parents: bool = False,
    parents: Optional[Dict[str, Chunk]] = None,
    expand_neighbors: int = 0,
):
    """Convenience wrapper around :func:`get_retriever`."""
    retriever = get_retriever(mode, store, embedder)
    return retriever.retrieve(
        query,
        k=top_k,
        filters=filters,
        expand_parents=expand_parents,
        parents=parents,
        expand_neighbors=expand_neighbors,
    )
