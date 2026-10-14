"""FAISS-backed vector index with chunk records and metadata filtering.

Embeddings are normalized by the embedding service, so an inner-product index
(``IndexFlatIP``) computes cosine similarity. Each strategy/config combination
gets its own index directory so indexes never overwrite one another and can be
rebuilt independently.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..chunking.base import Chunk, ChunkingConfig
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

INDEX_VERSION = "1.0.0"


class VectorStoreError(RuntimeError):
    pass


@dataclass
class IndexMetadata:
    embedding_model: str
    dimension: int
    strategy: str
    chunking_config: Dict[str, Any]
    corpus_id: str = ""
    source_ids: List[str] = field(default_factory=list)
    index_version: str = INDEX_VERSION
    created_at: str = ""
    chunk_count: int = 0
    build_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    rank: int


class VectorStore:
    """An in-memory FAISS index plus its chunk records and metadata."""

    def __init__(
        self,
        dimension: int,
        embedding_model: str,
        strategy: str,
        chunking_config: Optional[ChunkingConfig] = None,
        corpus_id: str = "",
    ) -> None:
        import faiss  # local import keeps the module import-light

        self._faiss = faiss
        self.dimension = int(dimension)
        self.embedding_model = embedding_model
        self.strategy = strategy
        self.chunking_config = (chunking_config or ChunkingConfig()).to_dict()
        self.corpus_id = corpus_id
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[Chunk] = []
        self.build_time_seconds = 0.0

    # ------------------------------------------------------------------ build
    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def add(self, vectors: np.ndarray, chunks: Sequence[Chunk]) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding matrix shape {vectors.shape} does not match index "
                f"dimension {self.dimension}."
            )
        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError(
                f"Received {len(chunks)} chunks but {vectors.shape[0]} vectors."
            )
        if vectors.shape[0] == 0:
            return
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def build_from_chunks(
        self,
        chunks: Sequence[Chunk],
        embedder: Callable[[Sequence[str]], np.ndarray],
        *,
        batch_size: int = 64,
    ) -> None:
        """Embed and add chunks in batches. Empty input yields an empty index."""
        start = time.perf_counter()
        chunks = list(chunks)
        if not chunks:
            self.build_time_seconds = 0.0
            return
        texts = [c.retrieval_text() for c in chunks]
        batch_vectors = []
        for i in range(0, len(texts), batch_size):
            batch_vectors.append(embedder(texts[i : i + batch_size]))
        vectors = np.vstack(batch_vectors) if batch_vectors else np.zeros((0, self.dimension))
        self.add(vectors, chunks)
        self.build_time_seconds = time.perf_counter() - start

    # ----------------------------------------------------------------- search
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        *,
        predicate: Optional[Callable[[Chunk], bool]] = None,
    ) -> List[SearchHit]:
        """Return the top-``k`` chunks, optionally filtered by a predicate.

        When a predicate is supplied, candidates are over-fetched so filtering
        still yields up to ``k`` results where possible.
        """
        if self.size == 0 or k <= 0:
            return []
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if query_vector.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Query dimension {query_vector.shape[1]} != index dimension {self.dimension}."
            )
        fetch = self.size if predicate is not None else min(k, self.size)
        scores, indices = self.index.search(query_vector, fetch)
        hits: List[SearchHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if predicate is not None and not predicate(chunk):
                continue
            hits.append(SearchHit(chunk=chunk, score=float(score), rank=len(hits) + 1))
            if len(hits) >= k:
                break
        return hits

    # -------------------------------------------------------------- metadata
    def metadata(self) -> IndexMetadata:
        return IndexMetadata(
            embedding_model=self.embedding_model,
            dimension=self.dimension,
            strategy=self.strategy,
            chunking_config=dict(self.chunking_config),
            corpus_id=self.corpus_id,
            source_ids=sorted({c.source_id for c in self.chunks}),
            chunk_count=self.size,
            build_time_seconds=self.build_time_seconds,
        )

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        meta = self.metadata()
        meta.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        (directory / "meta.json").write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved index (%d chunks) to %s", self.size, directory)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        import faiss

        directory = Path(directory)
        meta_path = directory / "meta.json"
        chunks_path = directory / "chunks.json"
        index_path = directory / "index.faiss"
        if not (meta_path.exists() and chunks_path.exists() and index_path.exists()):
            raise VectorStoreError(f"Incomplete index at {directory}.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(
            dimension=meta["dimension"],
            embedding_model=meta["embedding_model"],
            strategy=meta["strategy"],
            corpus_id=meta.get("corpus_id", ""),
        )
        store.chunking_config = meta.get("chunking_config", {})
        store.index = faiss.read_index(str(index_path))
        store.chunks = [
            Chunk.from_dict(c) for c in json.loads(chunks_path.read_text(encoding="utf-8"))
        ]
        store.build_time_seconds = float(meta.get("build_time_seconds", 0.0))
        return store


def index_directory(base_dir: str | Path, key: str) -> Path:
    """Return the on-disk directory for an index identified by ``key``."""
    from ..utils.validation import safe_filename

    return Path(base_dir) / safe_filename(key)


def make_index_key(strategy: str, config: ChunkingConfig, embedding_model: str,
                   corpus_id: str = "") -> str:
    """Deterministic, filesystem-safe key identifying an index configuration."""
    import hashlib

    payload = json.dumps(
        {
            "strategy": strategy,
            "config": config.to_dict(),
            "model": embedding_model,
            "corpus": corpus_id,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
    return f"{strategy}-{digest}"
