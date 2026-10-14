"""Metadata model round-trip and provenance tests."""

from __future__ import annotations

from rag_benchmark.chunking import chunk_document
from rag_benchmark.chunking.base import Chunk, ChunkingConfig
from rag_benchmark.parsers.models import ParsedBlock, ParsedDocument


def test_parsed_block_round_trip():
    block = ParsedBlock(
        block_id="s:b0", source_id="s", source_name="d.md", source_type="md",
        content_type="paragraph", text="hello", page_number=3,
        section_path=["A", "B"], meta={"x": 1},
    )
    restored = ParsedBlock.from_dict(block.to_dict())
    assert restored == block


def test_parsed_document_round_trip(sample_document):
    restored = ParsedDocument.from_dict(sample_document.to_dict())
    assert restored.source_id == sample_document.source_id
    assert restored.block_count == sample_document.block_count
    assert restored.full_text == sample_document.full_text


def test_chunk_round_trip(sample_document):
    config = ChunkingConfig(target_tokens=50, max_tokens=100, overlap_tokens=0, min_tokens=5)
    children, _ = chunk_document(sample_document, "structural", config)
    chunk = children[0]
    restored = Chunk.from_dict(chunk.to_dict())
    assert restored.chunk_id == chunk.chunk_id
    assert restored.token_count == chunk.token_count
    assert restored.block_ids == chunk.block_ids


def test_chunk_page_range_and_section():
    chunk = Chunk(
        chunk_id="c1", source_id="s", source_name="d", text="t", content_type="paragraph",
        chunking_strategy="fixed", chunk_index=0, token_count=1, page_number=1, page_end=3,
        section_path=["A", "B"],
    )
    assert chunk.page_range == "1-3"
    assert chunk.section_path_str == "A > B"


def test_retrieval_text_includes_provenance(sample_document):
    config = ChunkingConfig(target_tokens=200, max_tokens=400, min_tokens=1)
    children, _ = chunk_document(sample_document, "structural", config)
    text = children[0].retrieval_text()
    assert "doc.md" in text
    assert text.endswith(children[0].text)


def test_block_ids_unique_within_document(sample_document):
    ids = [b.block_id for b in sample_document.blocks]
    assert len(ids) == len(set(ids))
