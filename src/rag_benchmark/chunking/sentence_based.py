"""Strategy B: sentence-based chunking.

Sentences are extracted in document order and greedily grouped until the target
token size is reached. Sentence order is preserved and a configurable overlap
carries trailing sentences into the next chunk. Section changes force a boundary
when ``respect_sections`` is enabled (no overlap is carried across sections).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from ..parsers.models import ParsedBlock
from ..utils.tokens import count_tokens
from ._textsplit import split_sentences
from .base import BaseChunker, Chunk, make_chunk


class SentenceChunker(BaseChunker):
    strategy_name = "sentence"

    def chunk(
        self,
        blocks: Sequence[ParsedBlock],
        *,
        source_id: str,
        source_name: str,
        source_type: str,
        document_version: str = "1",
    ) -> List[Chunk]:
        sentences: List[Tuple[str, ParsedBlock]] = []
        for block in blocks:
            if not (block.text or "").strip():
                continue
            for sentence in split_sentences(block.text):
                sentences.append((sentence, block))
        if not sentences:
            return []

        chunks: List[Chunk] = []
        buffer: List[Tuple[str, ParsedBlock]] = []
        buffer_tokens = 0
        state = {"index": 0}

        def flush(with_overlap: bool) -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            text = " ".join(s for s, _ in buffer).strip()
            chunks.append(
                make_chunk(
                    strategy=self.strategy_name,
                    source_id=source_id,
                    source_name=source_name,
                    source_type=source_type,
                    document_version=document_version,
                    text=text,
                    content_type="mixed",
                    chunk_index=state["index"],
                    blocks=_unique_blocks(buffer),
                    boundary_reason="sentence group reaching target size or section change",
                    extra_meta={
                        "sentence_count": len(buffer),
                        "configured_target_tokens": self.config.target_tokens,
                    },
                )
            )
            state["index"] += 1

            if with_overlap and self.config.overlap_tokens > 0:
                overlap: List[Tuple[str, ParsedBlock]] = []
                tokens = 0
                for sentence, block in reversed(buffer):
                    st = count_tokens(sentence)
                    if tokens + st > self.config.overlap_tokens:
                        break
                    overlap.insert(0, (sentence, block))
                    tokens += st
                buffer = overlap
                buffer_tokens = tokens
            else:
                buffer = []
                buffer_tokens = 0

        for sentence, block in sentences:
            st = count_tokens(sentence)
            section_change = (
                self.config.respect_sections
                and buffer
                and block.section_path != buffer[-1][1].section_path
            )
            if buffer and section_change:
                flush(with_overlap=False)
            elif buffer and buffer_tokens + st > self.config.target_tokens:
                flush(with_overlap=True)
            buffer.append((sentence, block))
            buffer_tokens += st

        flush(with_overlap=False)
        return chunks


def _unique_blocks(pairs) -> List[ParsedBlock]:
    seen = set()
    result: List[ParsedBlock] = []
    for _, block in pairs:
        if block.block_id not in seen:
            seen.add(block.block_id)
            result.append(block)
    return result
