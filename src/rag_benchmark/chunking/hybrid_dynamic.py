"""Strategy F: hybrid / dynamic chunking (the primary adaptive strategy).

The strategy is deterministic and explainable. It works in three stages:

1. **Structural pass** -- content is partitioned into sections by heading path.
2. **Content-aware unitization** -- each section is turned into typed units:
   headings, prose paragraphs (with FAQ Q/A pairs fused), ordered/unordered
   lists, tables, code blocks and CSV records. Tables and code stay intact and
   never merge with prose. Adjacent list items are kept together so ordered
   steps and definitions survive.
3. **Adaptive packing** -- compatible prose-like units are merged while they fit
   the target size; oversized units are split at safe boundaries (table rows with
   the header repeated, code lines, sentence boundaries). Compatible short units
   in the same section are merged afterwards. A hard maximum is always enforced.

Every emitted boundary records a human-readable reason. A parent chunk is
produced for each section to enable optional parent expansion at retrieval time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
from typing import List, Optional, Sequence, Tuple

from ..parsers.models import ContentType, ParsedBlock
from ..utils.tokens import count_tokens
from ._textsplit import recursive_split
from .base import BaseChunker, Chunk, make_chunk

_PROSE_FAMILY = {ContentType.HEADING, ContentType.PARAGRAPH, ContentType.OTHER}
_ATOMIC_KINDS = {ContentType.TABLE, ContentType.CODE}


@dataclass
class _Unit:
    kind: str  # heading|prose|list|point|table|code|record|faq
    blocks: List[ParsedBlock]
    text: str
    tokens: int


@dataclass
class _PackedChunk:
    text: str
    blocks: List[ParsedBlock]
    kind: str
    is_code: bool
    reason: str
    meta: dict = field(default_factory=dict)


class HybridDynamicChunker(BaseChunker):
    strategy_name = "hybrid"

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
        live = [b for b in blocks if (b.text or "").strip()]
        if not live:
            return [], []

        children: List[Chunk] = []
        parents: List[Chunk] = []
        child_index = 0
        parent_index = 0

        for section_key, section_iter in groupby(live, key=lambda b: tuple(b.section_path)):
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

            units = self._build_units(section_blocks)
            packed = self._pack(units)
            packed = self._merge_short(packed)
            packed = self._enforce_max(packed)

            for piece in packed:
                if not piece.text.strip():
                    continue
                child = make_chunk(
                    strategy=self.strategy_name,
                    source_id=source_id,
                    source_name=source_name,
                    source_type=source_type,
                    document_version=document_version,
                    text=piece.text,
                    content_type="mixed",
                    chunk_index=child_index,
                    blocks=piece.blocks,
                    parent_chunk_id=parent.chunk_id,
                    is_code=piece.is_code,
                    boundary_reason=piece.reason,
                    extra_meta={**piece.meta, "section": list(section_key)},
                )
                children.append(child)
                child_index += 1

        return children, parents

    # ------------------------------------------------------------------ units
    def _build_units(self, blocks: List[ParsedBlock]) -> List[_Unit]:
        units: List[_Unit] = []
        i = 0
        n = len(blocks)
        while i < n:
            block = blocks[i]
            ct = block.content_type
            if ct == ContentType.HEADING:
                units.append(_Unit("heading", [block], block.text, count_tokens(block.text)))
                i += 1
            elif ct == ContentType.LIST_ITEM:
                group = [block]
                j = i + 1
                while j < n and blocks[j].content_type == ContentType.LIST_ITEM:
                    group.append(blocks[j])
                    j += 1
                text = "\n".join(b.text for b in group)
                units.append(_Unit("list", group, text, count_tokens(text)))
                i = j
            elif ct == ContentType.TABLE:
                units.append(_Unit("table", [block], block.text, count_tokens(block.text)))
                i += 1
            elif ct == ContentType.CODE:
                units.append(_Unit("code", [block], block.text, count_tokens(block.text)))
                i += 1
            elif ct == ContentType.RECORD:
                units.append(_Unit("record", [block], block.text, count_tokens(block.text)))
                i += 1
            else:  # paragraph / other
                # Fuse FAQ-style question with its answer so context is preserved.
                if self._is_question(block.text) and i + 1 < n and blocks[i + 1].content_type in (
                    ContentType.PARAGRAPH,
                    ContentType.LIST_ITEM,
                ):
                    group = [block, blocks[i + 1]]
                    text = block.text + "\n" + blocks[i + 1].text
                    units.append(_Unit("faq", group, text, count_tokens(text)))
                    i += 2
                else:
                    units.append(_Unit("prose", [block], block.text, count_tokens(block.text)))
                    i += 1
        return units

    @staticmethod
    def _is_question(text: str) -> bool:
        t = (text or "").strip().lower()
        return t.endswith("?") or t.startswith("q:") or t.startswith("question:")

    # ------------------------------------------------------------------- pack
    def _pack(self, units: List[_Unit]) -> List[_PackedChunk]:
        packed: List[_PackedChunk] = []
        pending: List[_Unit] = []
        pending_tokens = 0

        def flush(reason: str) -> None:
            nonlocal pending, pending_tokens
            if not pending:
                return
            text = _join_units(pending)
            kind = pending[0].kind
            is_code = all(u.kind == "code" for u in pending)
            member_blocks: List[ParsedBlock] = []
            for u in pending:
                member_blocks.extend(u.blocks)
            packed.append(_PackedChunk(text, member_blocks, kind, is_code, reason))
            pending = []
            pending_tokens = 0

        for unit in units:
            if unit.tokens > self.config.max_tokens:
                flush("flushed before oversized unit")
                for piece, is_code in self._split_unit(unit):
                    if piece.strip():
                        packed.append(
                            _PackedChunk(
                                piece,
                                list(unit.blocks),
                                unit.kind,
                                is_code,
                                f"oversized {unit.kind} split at safe boundary",
                            )
                        )
                continue

            if not pending:
                pending = [unit]
                pending_tokens = unit.tokens
                continue

            mergeable = _mergeable(pending[-1].kind, unit.kind)
            if mergeable and pending_tokens + unit.tokens <= self.config.target_tokens:
                pending.append(unit)
                pending_tokens += unit.tokens
            elif (
                mergeable
                and pending_tokens + unit.tokens <= self.config.max_tokens
                and _same_atomic_kind(pending[-1].kind, unit.kind)
            ):
                pending.append(unit)
                pending_tokens += unit.tokens
            else:
                flush("content-type change or target size reached")
                pending = [unit]
                pending_tokens = unit.tokens

        flush("end of section")
        return packed

    def _split_unit(self, unit: _Unit) -> List[Tuple[str, bool]]:
        if unit.kind == "table":
            return [(t, False) for t in self._split_table(unit.blocks[0].text)]
        if unit.kind == "code":
            return [(t, True) for t in self._split_code(unit.blocks[0].text)]
        if unit.kind == "list":
            return [(t, False) for t in self._split_list(unit.blocks)]
        return [(t, False) for t in recursive_split(unit.text, self.config.max_tokens)]

    def _split_table(self, text: str) -> List[str]:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if len(lines) < 3:
            return [text]
        header, rows = lines[0], lines[2:]
        out: List[str] = []
        current: List[str] = []
        tokens = count_tokens(header)
        for row in rows:
            rt = count_tokens(row)
            if current and tokens + rt > self.config.target_tokens:
                out.append(header + "\n" + "\n".join(current))
                current = []
                tokens = count_tokens(header)
            current.append(row)
            tokens += rt
        if current:
            out.append(header + "\n" + "\n".join(current))
        return out

    def _split_code(self, text: str) -> List[str]:
        out: List[str] = []
        current: List[str] = []
        tokens = 0
        for line in text.split("\n"):
            lt = count_tokens(line)
            if current and tokens + lt > self.config.target_tokens:
                out.append("\n".join(current))
                current = []
                tokens = 0
            current.append(line)
            tokens += lt
        if current:
            out.append("\n".join(current))
        return out

    def _split_list(self, blocks: List[ParsedBlock]) -> List[str]:
        out: List[str] = []
        current: List[str] = []
        tokens = 0
        for block in blocks:
            bt = count_tokens(block.text)
            if current and tokens + bt > self.config.target_tokens:
                out.append("\n".join(current))
                current = []
                tokens = 0
            current.append(block.text)
            tokens += bt
        if current:
            out.append("\n".join(current))
        return out

    # ------------------------------------------------------------ short merge
    def _merge_short(self, packed: List[_PackedChunk]) -> List[_PackedChunk]:
        if not self.config.merge_short_chunks or len(packed) < 2:
            return packed
        merged: List[_PackedChunk] = []
        i = 0
        while i < len(packed):
            current = packed[i]
            while (
                i + 1 < len(packed)
                and count_tokens(current.text) < self.config.min_tokens
                and _mergeable(current.kind, packed[i + 1].kind)
                and count_tokens(current.text) + count_tokens(packed[i + 1].text)
                <= self.config.max_tokens
            ):
                nxt = packed[i + 1]
                current = _PackedChunk(
                    current.text + "\n\n" + nxt.text,
                    current.blocks + nxt.blocks,
                    current.kind,
                    current.is_code and nxt.is_code,
                    "merged compatible short units in same section",
                )
                i += 1
            merged.append(current)
            i += 1
        return merged

    def _enforce_max(self, packed: List[_PackedChunk]) -> List[_PackedChunk]:
        out: List[_PackedChunk] = []
        for piece in packed:
            if count_tokens(piece.text) <= self.config.max_tokens:
                out.append(piece)
                continue
            for text in recursive_split(piece.text, self.config.max_tokens):
                if text.strip():
                    out.append(
                        _PackedChunk(
                            text,
                            list(piece.blocks),
                            piece.kind,
                            piece.is_code,
                            "hard maximum enforced",
                        )
                    )
        return out


def _mergeable(kind_a: str, kind_b: str) -> bool:
    if kind_a in _ATOMIC_KINDS or kind_b in _ATOMIC_KINDS:
        return kind_a == kind_b and kind_a in _ATOMIC_KINDS
    # prose-like families (heading, prose, list, faq, record) may mix.
    return True


def _same_atomic_kind(kind_a: str, kind_b: str) -> bool:
    return kind_a == kind_b and kind_a in _ATOMIC_KINDS


def _join_units(units: List[_Unit]) -> str:
    parts: List[str] = []
    for unit in units:
        if unit.text.strip():
            parts.append(unit.text)
    # Lists read better single-spaced; everything else double-spaced.
    if all(u.kind == "list" for u in units):
        return "\n".join(parts)
    return "\n\n".join(parts)
