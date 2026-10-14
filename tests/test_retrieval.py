"""Retrieval correctness, filtering and top-k tests."""

from __future__ import annotations

from rag_benchmark.chunking.base import Chunk, ChunkingConfig
from rag_benchmark.indexing.vector_store import VectorStore
from rag_benchmark.retrieval import get_retriever


def _chunk(i, text, source_id="s1", source_name="doc1") -> Chunk:
    return Chunk(
        chunk_id=f"{source_id}:c{i:05d}", source_id=source_id, source_name=source_name,
        text=text, content_type="paragraph", chunking_strategy="fixed",
        chunk_index=i, token_count=len(text.split()),
    )


def _store(fake_embedder, chunks):
    store = VectorStore(dimension=fake_embedder.dimension, embedding_model="fake",
                        strategy="fixed", chunking_config=ChunkingConfig())
    store.build_from_chunks(chunks, fake_embedder.encode)
    return store


def test_dense_ranks_relevant_first(fake_embedder):
    store = _store(
        fake_embedder,
        [
            _chunk(0, "cats are furry animals that purr"),
            _chunk(1, "the eiffel tower is located in paris france"),
            _chunk(2, "python is a programming language for data science"),
        ],
    )
    results = get_retriever("dense", store, fake_embedder).retrieve("furry cat animals", k=3)
    assert results[0].chunk_id == "s1:c00000"


def test_top_k_respected(fake_embedder):
    store = _store(fake_embedder, [_chunk(i, f"token{i} alpha beta") for i in range(5)])
    results = get_retriever("dense", store, fake_embedder).retrieve("alpha", k=2)
    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]


def test_metadata_filter_by_source_id(fake_embedder):
    chunks = [
        _chunk(0, "alpha beta", source_id="s1", source_name="a"),
        _chunk(1, "alpha beta gamma", source_id="s2", source_name="b"),
    ]
    store = _store(fake_embedder, chunks)
    results = get_retriever("dense", store, fake_embedder).retrieve(
        "alpha beta", k=5, filters={"source_id": "s2"}
    )
    assert results
    assert all(r.source_id == "s2" for r in results)


def test_bm25_word_match(fake_embedder):
    store = _store(
        fake_embedder,
        [
            _chunk(0, "cats are furry animals"),
            _chunk(1, "the eiffel tower is in paris"),
        ],
    )
    results = get_retriever("bm25", store).retrieve("eiffel tower", k=1)
    assert results[0].chunk_id == "s1:c00001"


def test_hybrid_returns_fused_results(fake_embedder):
    store = _store(
        fake_embedder,
        [
            _chunk(0, "cats are furry animals"),
            _chunk(1, "the eiffel tower is in paris"),
        ],
    )
    results = get_retriever("hybrid", store, fake_embedder).retrieve("eiffel tower", k=2)
    assert results
    assert results[0].retrieval_mode == "hybrid"
    assert "rrf_score" in results[0].meta


def test_empty_store_returns_empty(fake_embedder):
    store = _store(fake_embedder, [])
    assert get_retriever("dense", store, fake_embedder).retrieve("anything", k=5) == []


def test_retrieval_result_fields(fake_embedder):
    store = _store(fake_embedder, [_chunk(0, "alpha beta")])
    result = get_retriever("dense", store, fake_embedder).retrieve("alpha", k=1)[0]
    payload = result.to_dict()
    for field in ("query", "chunk_id", "rank", "score", "text", "source_name"):
        assert field in payload
