"""Embedding and vector-store tests using a deterministic fake embedder."""

from __future__ import annotations

import os

import numpy as np
import pytest

from rag_benchmark.chunking.base import Chunk, ChunkingConfig
from rag_benchmark.indexing.vector_store import VectorStore, VectorStoreError


def _make_chunk(i, text, source_id="s") -> Chunk:
    return Chunk(
        chunk_id=f"{source_id}:c{i:05d}", source_id=source_id, source_name="d",
        text=text, content_type="paragraph", chunking_strategy="fixed",
        chunk_index=i, token_count=len(text.split()),
    )


def _build(embedder, chunks):
    store = VectorStore(
        dimension=embedder.dimension,
        embedding_model="fake",
        strategy="fixed",
        chunking_config=ChunkingConfig(),
    )
    store.build_from_chunks(chunks, embedder.encode)
    return store


def test_empty_index_returns_empty_search(fake_embedder):
    store = _build(fake_embedder, [])
    assert store.size == 0
    assert store.search(fake_embedder.encode(["anything"])[0], k=5) == []


def test_vector_to_chunk_mapping(fake_embedder):
    chunks = [
        _make_chunk(0, "cats are furry animals"),
        _make_chunk(1, "the eiffel tower is in paris"),
        _make_chunk(2, "python is a programming language"),
    ]
    store = _build(fake_embedder, chunks)
    hits = store.search(fake_embedder.encode(["furry cats"])[0], k=1)
    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "s:c00000"


def test_dimension_mismatch_raises(fake_embedder):
    store = VectorStore(dimension=fake_embedder.dimension, embedding_model="fake", strategy="fixed")
    wrong = np.zeros((1, fake_embedder.dimension + 5), dtype="float32")
    with pytest.raises(VectorStoreError):
        store.add(wrong, [_make_chunk(0, "x")])


def test_build_from_empty_chunks(fake_embedder):
    store = _build(fake_embedder, [])
    assert store.size == 0
    assert store.build_time_seconds == 0.0


def test_index_persistence_and_reload(fake_embedder, tmp_path):
    chunks = [_make_chunk(0, "alpha beta"), _make_chunk(1, "gamma delta")]
    store = _build(fake_embedder, chunks)
    directory = tmp_path / "idx"
    store.save(directory)

    reloaded = VectorStore.load(directory)
    assert reloaded.size == store.size
    assert [c.chunk_id for c in reloaded.chunks] == [c.chunk_id for c in store.chunks]
    assert reloaded.embedding_model == "fake"
    hits = reloaded.search(fake_embedder.encode(["alpha"])[0], k=1)
    assert hits[0].chunk.chunk_id == "s:c00000"


def test_load_incomplete_index_raises(tmp_path):
    with pytest.raises(VectorStoreError):
        VectorStore.load(tmp_path / "missing")


@pytest.mark.skipif(
    os.getenv("RUN_EMBEDDING_TESTS") != "1",
    reason="Set RUN_EMBEDDING_TESTS=1 to run the real sentence-transformers integration test.",
)
def test_real_embedding_model_integration():
    from rag_benchmark.indexing.embeddings import get_embedding_service

    service = get_embedding_service("sentence-transformers/all-MiniLM-L6-v2")
    vectors = service.encode(["hello world", "goodbye world"])
    assert vectors.shape == (2, service.dimension)
    # Normalized vectors have unit length.
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
