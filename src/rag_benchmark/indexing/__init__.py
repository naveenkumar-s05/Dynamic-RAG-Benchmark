"""Embedding and vector-index layer."""

from .embeddings import (
    EmbeddingInfo,
    EmbeddingService,
    EmbeddingUnavailableError,
    embedding_available,
    get_embedding_service,
)
from .vector_store import (
    IndexMetadata,
    SearchHit,
    VectorStore,
    VectorStoreError,
    index_directory,
    make_index_key,
)

__all__ = [
    "EmbeddingInfo",
    "EmbeddingService",
    "EmbeddingUnavailableError",
    "embedding_available",
    "get_embedding_service",
    "IndexMetadata",
    "SearchHit",
    "VectorStore",
    "VectorStoreError",
    "index_directory",
    "make_index_key",
]
