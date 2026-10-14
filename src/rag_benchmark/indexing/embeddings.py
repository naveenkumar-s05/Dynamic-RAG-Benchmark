"""Embedding service backed by sentence-transformers.

The same model and normalization settings are used across every chunking
strategy during a benchmark. Model loading is lazy and cached. If the model or
its dependency cannot be loaded, a clear, actionable error is raised rather than
silently degrading to keyword search.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the embedding model or its dependency cannot be loaded."""


@dataclass
class EmbeddingInfo:
    model_name: str
    dimension: int
    normalized: bool = True


class EmbeddingService:
    """Lazy wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 normalize: bool = True, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = None
        self._dimension: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - dependency missing
                raise EmbeddingUnavailableError(
                    "sentence-transformers is not installed. Install it with "
                    "`python -m pip install sentence-transformers`."
                ) from exc
            try:
                logger.info("Loading embedding model '%s'...", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = int(self._model.get_embedding_dimension())
                else:  # older sentence-transformers
                    self._dimension = int(self._model.get_sentence_embedding_dimension())
            except Exception as exc:  # pragma: no cover - network/model issues
                raise EmbeddingUnavailableError(
                    f"Could not load embedding model '{self.model_name}': {exc}. "
                    "Check your network connection or pre-download the model with "
                    "`python -c \"from sentence_transformers import "
                    "SentenceTransformer; SentenceTransformer("
                    f"'{self.model_name}')\"`."
                ) from exc
            return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_model()
        return int(self._dimension)

    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(self.model_name, self.dimension, self.normalize)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into a normalized ``float32`` matrix.

        Returns a ``(len(texts), dim)`` array. Batches are processed in chunks so
        large corpora are not held in a single request.
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype="float32").ravel()
        b = np.asarray(b, dtype="float32").ravel()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)


_EMBEDDERS: dict = {}
_EMBEDDERS_LOCK = threading.Lock()


def get_embedding_service(model_name: str) -> EmbeddingService:
    """Return a process-wide singleton embedding service per model name."""
    with _EMBEDDERS_LOCK:
        if model_name not in _EMBEDDERS:
            _EMBEDDERS[model_name] = EmbeddingService(model_name)
        return _EMBEDDERS[model_name]


def embedding_available(model_name: str) -> tuple:
    """Return ``(available, message)`` without raising."""
    try:
        service = get_embedding_service(model_name)
        service._ensure_model()
        return True, f"Model '{model_name}' loaded ({service.dimension} dims)."
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
