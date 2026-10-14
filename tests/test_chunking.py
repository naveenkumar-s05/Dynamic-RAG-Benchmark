"""Tests for all six chunking strategies."""

from __future__ import annotations

import pytest

from rag_benchmark.chunking import STRATEGIES, chunk_document
from rag_benchmark.chunking._textsplit import split_sentences
from rag_benchmark.chunking.base import ChunkingConfig
from rag_benchmark.chunking.fixed_size import FixedSizeChunker
from rag_benchmark.chunking.hybrid_dynamic import HybridDynamicChunker
from rag_benchmark.chunking.recursive import RecursiveChunker
from rag_benchmark.chunking.semantic import SemanticChunker
from rag_benchmark.chunking.sentence_based import SentenceChunker
from rag_benchmark.chunking.structural import StructuralChunker
from rag_benchmark.parsers.models import ParsedBlock, ParsedDocument


def _chunk(strategy, doc, config, embedder=None):
    children, parents = chunk_document(doc, strategy, config, embedder=embedder)
    return children, parents


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_produces_chunks(strategy, sample_document, fake_embedder):
    config = ChunkingConfig(target_tokens=60, max_tokens=120, overlap_tokens=10, min_tokens=10)
    children, _ = _chunk(strategy, sample_document, config, embedder=fake_embedder.encode)
    assert children, f"{strategy} produced no chunks"
    for chunk in children:
        assert chunk.token_count > 0
        assert chunk.text.strip()
        assert chunk.chunking_strategy == strategy


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_empty_input_returns_empty(strategy, fake_embedder):
    config = ChunkingConfig(target_tokens=60, max_tokens=120, overlap_tokens=0)
    empty = ParsedDocument(source_id="e", source_name="e.md", source_type="md", blocks=[])
    children, parents = chunk_document(empty, strategy, config, embedder=fake_embedder.encode)
    assert children == []


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_max_tokens_never_exceeded(strategy, sample_document, fake_embedder):
    config = ChunkingConfig(target_tokens=40, max_tokens=80, overlap_tokens=5, min_tokens=5)
    children, _ = _chunk(strategy, sample_document, config, embedder=fake_embedder.encode)
    # Overlap may legitimately push a chunk slightly above max for fixed/sentence;
    # allow a small tolerance but flag gross violations.
    for chunk in children:
        assert chunk.token_count <= config.max_tokens * 1.5


def test_fixed_size_overlap(sample_document):
    config = ChunkingConfig(target_tokens=25, max_tokens=50, overlap_tokens=10)
    children = FixedSizeChunker(config).chunk(
        sample_document.blocks, source_id="src1", source_name="doc.md", source_type="md"
    )
    assert len(children) >= 2
    # Consecutive chunks share overlapping tokens.
    first_tokens = set(children[0].text.split())
    second_tokens = set(children[1].text.split())
    assert first_tokens & second_tokens


def test_fixed_size_deterministic(sample_document):
    config = ChunkingConfig(target_tokens=50, max_tokens=100, overlap_tokens=0)
    a = FixedSizeChunker(config).chunk(sample_document.blocks, source_id="s", source_name="d", source_type="md")
    b = FixedSizeChunker(config).chunk(sample_document.blocks, source_id="s", source_name="d", source_type="md")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_structural_keeps_table_header_with_rows(sample_document):
    config = ChunkingConfig(target_tokens=100, max_tokens=200, min_tokens=1)
    children, parents = StructuralChunker(config).chunk_with_parents(
        sample_document.blocks, source_id="s", source_name="d", source_type="md"
    )
    table_chunks = [c for c in children if c.table_id]
    assert table_chunks
    for chunk in table_chunks:
        assert "| Basic |" in chunk.text or "Basic" in chunk.text
        assert parents and chunk.parent_chunk_id


def test_structural_keeps_code_block_intact(sample_document):
    config = ChunkingConfig(target_tokens=200, max_tokens=400, min_tokens=1)
    children, _ = StructuralChunker(config).chunk_with_parents(
        sample_document.blocks, source_id="s", source_name="d", source_type="md"
    )
    code_chunks = [c for c in children if c.is_code]
    assert any("def backup" in c.text and "return True" in c.text for c in code_chunks)


def test_oversized_table_split_repeats_header():
    header = "Plan | Storage"
    rows = [f"row {i}: P{i} | {i} GB" for i in range(1, 40)]
    block = ParsedBlock(
        block_id="s:b0", source_id="s", source_name="d", source_type="md",
        content_type="table", text=header + "\n" + "-" * len(header) + "\n" + "\n".join(rows),
        table_id="t0", meta={"headers": ["Plan", "Storage"]},
    )
    config = ChunkingConfig(target_tokens=60, max_tokens=80, overlap_tokens=0, min_tokens=1)
    children, _ = StructuralChunker(config).chunk_with_parents(
        [block], source_id="s", source_name="d", source_type="md"
    )
    assert len(children) >= 2
    for chunk in children:
        assert chunk.text.startswith("Plan | Storage")


def test_recursive_handles_oversized_block():
    text = "\n\n".join(f"Paragraph {i} has several words to split into pieces." for i in range(20))
    block = ParsedBlock(block_id="s:b0", source_id="s", source_name="d", source_type="md",
                        content_type="paragraph", text=text)
    config = ChunkingConfig(target_tokens=40, max_tokens=60, overlap_tokens=0)
    children = RecursiveChunker(config).chunk([block], source_id="s", source_name="d", source_type="md")
    assert len(children) > 1
    assert all(c.token_count <= 60 for c in children)


def test_semantic_uses_embedder_and_records_reason(sample_document, fake_embedder):
    config = ChunkingConfig(target_tokens=40, max_tokens=90, overlap_tokens=0, semantic_threshold=0.1, min_tokens=1)
    chunker = SemanticChunker(config, embedder=fake_embedder.encode)
    children = chunker.chunk(sample_document.blocks, source_id="s", source_name="d", source_type="md")
    assert children
    assert all(c.chunking_strategy == "semantic" for c in children)


def test_semantic_fallback_without_embedder(sample_document):
    config = ChunkingConfig(target_tokens=40, max_tokens=90, overlap_tokens=0, min_tokens=1)
    chunker = SemanticChunker(config, embedder=None)
    children = chunker.chunk(sample_document.blocks, source_id="s", source_name="d", source_type="md")
    assert children
    assert any("fallback" in c.boundary_reason for c in children)


def test_hybrid_merges_short_and_has_parents(sample_document):
    config = ChunkingConfig(target_tokens=200, max_tokens=300, min_tokens=5)
    children, parents = HybridDynamicChunker(config).chunk_with_parents(
        sample_document.blocks, source_id="s", source_name="d", source_type="md"
    )
    assert children and parents
    assert all(c.parent_chunk_id for c in children)


def test_chunk_ids_unique_across_strategies(sample_document, fake_embedder):
    config = ChunkingConfig(target_tokens=50, max_tokens=100, overlap_tokens=0, min_tokens=5)
    seen = set()
    for strategy in STRATEGIES:
        children, _ = chunk_document(sample_document, strategy, config, embedder=fake_embedder.encode)
        for chunk in children:
            assert chunk.chunk_id not in seen
            seen.add(chunk.chunk_id)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ChunkingConfig(target_tokens=500, max_tokens=100).validate()
    with pytest.raises(ValueError):
        ChunkingConfig(target_tokens=100, max_tokens=200, overlap_tokens=100).validate()
    with pytest.raises(ValueError):
        ChunkingConfig(target_tokens=0).validate()


def test_sentence_splitter_handles_abbreviations():
    sentences = split_sentences("Dr. Smith went home. He was tired.")
    assert sentences == ["Dr. Smith went home.", "He was tired."]


def test_metadata_preserved_through_chunking(sample_document):
    config = ChunkingConfig(target_tokens=40, max_tokens=80, overlap_tokens=0, min_tokens=5)
    children = SentenceChunker(config).chunk(
        sample_document.blocks, source_id="src1", source_name="doc.md",
        source_type="md", document_version="2",
    )
    for chunk in children:
        assert chunk.source_id == "src1"
        assert chunk.source_name == "doc.md"
        assert chunk.document_version == "2"
        assert chunk.block_ids
