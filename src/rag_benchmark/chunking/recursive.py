"""Strategy D: recursive chunking.

Content is packed block-by-block up to the hard maximum. Any block that is still
too large is split recursively at progressively smaller separators (section break
-> paragraph -> line -> sentence -> word). Source metadata survives every split
and empty chunks are never emitted. Optional overlap is carried as the tail of
the previous chunk.
"""

from __future__ import annotations

from typing import List, Sequence

from ..parsers.models import ParsedBlock
from ..utils.tokens import count_tokens, decode_text, encode_text
from ._textsplit import recursive_split
from .base import BaseChunker, Chunk, make_chunk


class RecursiveChunker(BaseChunker):
    strategy_name = "recursive"

    def chunk(
        self,
        blocks: Sequence[ParsedBlock],
        *,
        source_id: str,
        source_name: str,
        source_type: str,
        document_version: str = "1",
    ) -> List[Chunk]:
        live = [b for b in blocks if (b.text or "").strip()]
        if not live:
            return []

        chunks: List[Chunk] = []
        chunk_index = 0
        buffer: List[ParsedBlock] = []
        buffer_tokens = 0
        last_tail = ""

        def emit(text: str, member_blocks: List[ParsedBlock], reason: str) -> None:
            nonlocal chunk_index, last_tail
            body = text.strip()
            if not body:
                return
            if last_tail and self.config.overlap_tokens > 0:
                body = f"{last_tail}\n{body}"
            if self.config.overlap_tokens > 0:
                last_tail = decode_text(
                    encode_text(body)[-self.config.overlap_tokens:]
                )
            is_code = all(b.content_type == "code" for b in member_blocks)
            chunks.append(
                make_chunk(
                    strategy=self.strategy_name,
                    source_id=source_id,
                    source_name=source_name,
                    source_type=source_type,
                    document_version=document_version,
                    text=body,
                    content_type="mixed",
                    chunk_index=chunk_index,
                    blocks=member_blocks,
                    is_code=is_code,
                    boundary_reason=reason,
                    extra_meta={"configured_max_tokens": self.config.max_tokens},
                )
            )
            chunk_index += 1

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            text = "\n\n".join(b.text for b in buffer)
            emit(text, list(buffer), "recursive block packing up to max tokens")
            buffer = []
            buffer_tokens = 0

        for block in live:
            block_tokens = count_tokens(block.text)
            if block_tokens > self.config.max_tokens:
                flush()
                for piece in recursive_split(block.text, self.config.max_tokens):
                    if piece.strip():
                        emit(piece, [block], "recursive split at decreasing separators")
                continue

            if buffer and buffer_tokens + block_tokens > self.config.max_tokens:
                flush()

            buffer.append(block)
            buffer_tokens += block_tokens

        flush()
        return chunks
