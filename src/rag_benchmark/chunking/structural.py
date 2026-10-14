"""Strategy C: structural chunking.

Splits primarily along document structure: sections defined by headings, then
paragraphs, lists, tables and code blocks. Related content stays under its
heading; table headers stay attached to their rows; code blocks are kept intact
unless they exceed the hard maximum, in which case they are split at line
boundaries. Oversized sections are split safely. A parent chunk is emitted for
each section to support optional parent expansion during retrieval.
"""

from __future__ import annotations

from itertools import groupby
from typing import List, Sequence, Tuple

from ..parsers.models import ContentType, ParsedBlock
from ..utils.tokens import count_tokens
from ._textsplit import split_paragraphs, split_sentences
from .base import BaseChunker, Chunk, make_chunk

_SUB_CONTENT = "chunk_text"
_SUB_REASON = "reason"


class StructuralChunker(BaseChunker):
    strategy_name = "structural"

    def chunk(
        self,
        blocks: Sequence[ParsedBlock],
        *,
        source_id: str,
        source_name: str,
        source_type: str,
        document_version: str = "1",
    ) -> List[Chunk]:
        children, _ = self.chunk_with_parents(
            blocks,
            source_id=source_id,
            source_name=source_name,
            source_type=source_type,
            document_version=document_version,
        )
        return children

    def chunk_with_parents(
        self,
        blocks: Sequence[ParsedBlock],
        *,
        source_id: str,
        source_name: str,
        source_type: str,
        document_version: str = "1",
    ) -> Tuple[List[Chunk], List[Chunk]]:
        children: List[Chunk] = []
        parents: List[Chunk] = []
        child_index = 0
        parent_index = 0

        live = [b for b in blocks if (b.text or "").strip()]
        grouped = groupby(live, key=lambda b: tuple(b.section_path))

        for section_key, section_iter in grouped:
            section_blocks = list(section_iter)

            # Parent chunk for the whole section.
            parent_text = "\n\n".join(b.text for b in section_blocks).strip()
            parent = make_chunk(
                strategy=self.strategy_name,
                source_id=source_id,
                source_name=source_name,
                source_type=source_type,
                document_version=document_version,
                text=parent_text,
                content_type="mixed",
                chunk_index=parent_index,
                blocks=section_blocks,
                boundary_reason="section parent",
                extra_meta={"is_parent": True},
            )
            parent.chunk_id = f"{self.strategy_name}:{source_id}:p{parent_index:05d}"
            parents.append(parent)
            parent_index += 1

            for text, reason, member_blocks, is_code in self._split_section(section_blocks):
                if not text.strip():
                    continue
                child = make_chunk(
                    strategy=self.strategy_name,
                    source_id=source_id,
                    source_name=source_name,
                    source_type=source_type,
                    document_version=document_version,
                    text=text,
                    content_type="mixed",
                    chunk_index=child_index,
                    blocks=member_blocks,
                    parent_chunk_id=parent.chunk_id,
                    is_code=is_code,
                    boundary_reason=reason,
                    extra_meta={"section": list(section_key)},
                )
                children.append(child)
                child_index += 1

        return children, parents

    def _split_section(
        self, section_blocks: List[ParsedBlock]
    ) -> List[Tuple[str, str, List[ParsedBlock], bool]]:
        results: List[Tuple[str, str, List[ParsedBlock], bool]] = []
        buffer: List[ParsedBlock] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            text = "\n\n".join(b.text for b in buffer).strip()
            is_code = all(b.content_type == ContentType.CODE for b in buffer)
            results.append((text, "structural section boundary", list(buffer), is_code))
            buffer = []
            buffer_tokens = 0

        for block in section_blocks:
            block_tokens = count_tokens(block.text)

            if block_tokens > self.config.max_tokens:
                flush()
                for text, reason, is_code in self._split_oversized(block):
                    results.append((text, reason, [block], is_code))
                continue

            # Keep distinct structural elements from being merged when they are
            # "hard" units (tables/code) that the user expects intact.
            if (
                buffer
                and block.content_type in (ContentType.TABLE, ContentType.CODE)
                and buffer_tokens >= self.config.min_tokens
            ):
                flush()

            if buffer and buffer_tokens + block_tokens > self.config.target_tokens:
                flush()

            buffer.append(block)
            buffer_tokens += block_tokens

        flush()
        return results

    def _split_oversized(
        self, block: ParsedBlock
    ) -> List[Tuple[str, str, bool]]:
        if block.content_type == ContentType.TABLE:
            return self._split_table(block)
        if block.content_type == ContentType.CODE:
            return self._split_code(block)
        return self._split_text(block)

    def _split_table(self, block: ParsedBlock) -> List[Tuple[str, str, bool]]:
        lines = [ln for ln in block.text.split("\n") if ln.strip()]
        if len(lines) < 3:
            return [(block.text, "oversized single-line table", False)]
        header = lines[0]
        rows = lines[2:]
        results: List[Tuple[str, str, bool]] = []
        current: List[str] = []
        current_tokens = count_tokens(header)
        for row in rows:
            rt = count_tokens(row)
            if current and current_tokens + rt > self.config.target_tokens:
                results.append(
                    (header + "\n" + "\n".join(current), "oversized table split by rows", False)
                )
                current = []
                current_tokens = count_tokens(header)
            current.append(row)
            current_tokens += rt
        if current:
            results.append(
                (header + "\n" + "\n".join(current),
                 "oversized table split by rows (header repeated)",
                 False)
            )
        return results

    def _split_code(self, block: ParsedBlock) -> List[Tuple[str, str, bool]]:
        lines = block.text.split("\n")
        results: List[Tuple[str, str, bool]] = []
        current: List[str] = []
        tokens = 0
        for line in lines:
            lt = count_tokens(line)
            if current and tokens + lt > self.config.target_tokens:
                results.append(("\n".join(current), "oversized code split by lines", True))
                current = []
                tokens = 0
            current.append(line)
            tokens += lt
        if current:
            results.append(("\n".join(current), "oversized code split by lines", True))
        return results

    def _split_text(self, block: ParsedBlock) -> List[Tuple[str, str, bool]]:
        results: List[Tuple[str, str, bool]] = []
        for paragraph in split_paragraphs(block.text):
            if count_tokens(paragraph) <= self.config.max_tokens:
                results.append((paragraph, "oversized paragraph promoted", False))
                continue
            current: List[str] = []
            tokens = 0
            for sentence in split_sentences(paragraph):
                st = count_tokens(sentence)
                if current and tokens + st > self.config.target_tokens:
                    results.append(
                        (" ".join(current), "oversized text split at sentence", False)
                    )
                    current = []
                    tokens = 0
                current.append(sentence)
                tokens += st
            if current:
                results.append(
                    (" ".join(current), "oversized text split at sentence", False)
                )
        return results
