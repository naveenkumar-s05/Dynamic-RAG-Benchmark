"""Strategy A: fixed-size token-window chunking.

Content is flattened into a token stream and sliced into fixed windows with a
configurable overlap. Token-level provenance is preserved, so every window knows
exactly which source blocks (and therefore pages/sections) it came from. Windows
are decoded back to text at block boundaries to avoid gluing unrelated blocks
together mid-word.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from ..parsers.models import ContentType, ParsedBlock
from ..utils.tokens import encode_text
from .base import BaseChunker, Chunk, ChunkingConfig, build_text_from_token_units, make_chunk


class FixedSizeChunker(BaseChunker):
    strategy_name = "fixed"

    def chunk(
        self,
        blocks: Sequence[ParsedBlock],
        *,
        source_id: str,
        source_name: str,
        source_type: str,
        document_version: str = "1",
    ) -> List[Chunk]:
        # Build a flat token stream that remembers the owning block.
        units: List[Tuple[ParsedBlock, object]] = []
        for block in blocks:
            if not (block.text or "").strip():
                continue
            for token in encode_text(block.text):
                units.append((block, token))

        total = len(units)
        if total == 0:
            return []

        window = self.config.target_tokens
        step = max(1, self.config.target_tokens - self.config.overlap_tokens)

        chunks: List[Chunk] = []
        chunk_index = 0
        start = 0
        while start < total:
            end = min(start + window, total)
            slice_units = units[start:end]
            text = build_text_from_token_units(slice_units)
            if text.strip():
                contributing = _unique_blocks(slice_units)
                is_code = all(
                    b.content_type == ContentType.CODE for b in contributing
                )
                chunks.append(
                    make_chunk(
                        strategy=self.strategy_name,
                        source_id=source_id,
                        source_name=source_name,
                        source_type=source_type,
                        document_version=document_version,
                        text=text,
                        content_type="mixed",
                        chunk_index=chunk_index,
                        blocks=contributing,
                        is_code=is_code,
                        boundary_reason=(
                            "fixed token window"
                            if start == 0
                            else f"token window with {self.config.overlap_tokens} overlap"
                        ),
                        extra_meta={
                            "window_start_token": start,
                            "window_end_token": end - 1,
                            "configured_target_tokens": self.config.target_tokens,
                            "configured_overlap_tokens": self.config.overlap_tokens,
                        },
                    )
                )
                chunk_index += 1
            if end >= total:
                break
            start += step

        return chunks


def _unique_blocks(slice_units) -> List[ParsedBlock]:
    seen = set()
    result: List[ParsedBlock] = []
    for block, _ in slice_units:
        if block.block_id not in seen:
            seen.add(block.block_id)
            result.append(block)
    return result
