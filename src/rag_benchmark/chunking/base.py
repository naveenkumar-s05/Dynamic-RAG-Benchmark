"""Chunk data model, configuration and shared chunking helpers.

All six strategies share :class:`Chunk`, :class:`ChunkingConfig` and the
:class:`BaseChunker` interface, so they can be compared fairly on the same
parsed blocks with the same configuration semantics.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..parsers.models import ContentType, ParsedBlock
from ..utils.tokens import count_tokens, decode_text


@dataclass
class ChunkingConfig:
    """Configuration shared by every chunking strategy."""

    target_tokens: int = 400
    max_tokens: int = 700
    min_tokens: int = 60
    overlap_tokens: int = 60
    semantic_threshold: float = 0.55  # cosine similarity; below => boundary
    sentences_per_chunk: int = 4
    merge_short_chunks: bool = True
    respect_sections: bool = True

    def validate(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens must not exceed max_tokens.")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative.")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens.")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be non-negative.")
        if not (-1.0 <= self.semantic_threshold <= 1.0):
            raise ValueError("semantic_threshold must be between -1 and 1.")
        if self.sentences_per_chunk <= 0:
            raise ValueError("sentences_per_chunk must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """A retrievable unit produced by a chunking strategy."""

    chunk_id: str
    source_id: str
    source_name: str
    text: str
    content_type: str
    chunking_strategy: str
    chunk_index: int
    token_count: int
    source_type: str = ""
    document_version: str = "1"
    page_number: Optional[int] = None
    page_end: Optional[int] = None
    section_path: List[str] = field(default_factory=list)
    parent_chunk_id: Optional[str] = None
    block_ids: List[str] = field(default_factory=list)
    table_id: Optional[str] = None
    row_number: Optional[int] = None
    is_code: bool = False
    boundary_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_range(self) -> Optional[str]:
        if self.page_number is None:
            return None
        if self.page_end is None or self.page_end == self.page_number:
            return str(self.page_number)
        return f"{self.page_number}-{self.page_end}"

    @property
    def section_path_str(self) -> str:
        return " > ".join(self.section_path)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["page_range"] = self.page_range
        data["section_path_str"] = self.section_path_str
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)

    def retrieval_text(self) -> str:
        """Text used for embedding/lexical search.

        A short provenance header is prepended so that the heading context of a
        passage is available to the retriever without polluting the raw text.
        """
        header_parts = [self.source_name]
        if self.section_path:
            header_parts.append(" > ".join(self.section_path))
        header = " | ".join(p for p in header_parts if p)
        if self.content_type == ContentType.TABLE:
            header = f"{header} | table"
        if self.is_code:
            header = f"{header} | code"
        return f"[{header}]\n{self.text}" if header else self.text


def make_chunk(
    *,
    strategy: str,
    source_id: str,
    source_name: str,
    source_type: str,
    document_version: str,
    text: str,
    content_type: str,
    chunk_index: int,
    blocks: Sequence[ParsedBlock],
    parent_chunk_id: Optional[str] = None,
    table_id: Optional[str] = None,
    row_number: Optional[int] = None,
    is_code: bool = False,
    boundary_reason: str = "",
    warnings: Optional[List[str]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Chunk:
    """Create a :class:`Chunk` with deterministic id and aggregated provenance."""
    pages = [b.page_number for b in blocks if b.page_number is not None]
    section_path: List[str] = []
    for block in blocks:
        if block.section_path:
            section_path = list(block.section_path)
            break

    if table_id is None:
        for block in blocks:
            if getattr(block, "table_id", None):
                table_id = block.table_id
                break
    if row_number is None:
        for block in blocks:
            if getattr(block, "row_number", None) is not None:
                row_number = block.row_number
                break

    content_types = {b.content_type for b in blocks}
    if len(content_types) == 1:
        ctype = next(iter(content_types))
    else:
        ctype = content_type or "mixed"

    block_ids = [b.block_id for b in blocks]
    meta: Dict[str, Any] = {"block_ids": block_ids}
    if blocks and blocks[0].meta:
        for key in ("headers", "columns", "row_count"):
            if key in blocks[0].meta:
                meta[key] = blocks[0].meta[key]
    if extra_meta:
        meta.update(extra_meta)

    return Chunk(
        chunk_id=f"{strategy}:{source_id}:c{chunk_index:05d}",
        source_id=source_id,
        source_name=source_name,
        text=text,
        content_type=ctype,
        chunking_strategy=strategy,
        chunk_index=chunk_index,
        token_count=count_tokens(text),
        source_type=source_type,
        document_version=document_version,
        page_number=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        section_path=section_path,
        parent_chunk_id=parent_chunk_id,
        block_ids=block_ids,
        table_id=table_id,
        row_number=row_number,
        is_code=is_code,
        boundary_reason=boundary_reason,
        warnings=list(warnings or []),
        meta=meta,
    )


def build_text_from_token_units(units: Iterable[tuple]) -> str:
    """Join ``(block, token)`` units into text, keeping block boundaries as blanks."""
    parts: List[List] = []
    current: List = []
    current_block_id: Optional[str] = None
    for block, token in units:
        if block.block_id != current_block_id and current:
            parts.append(current)
            current = []
        current_block_id = block.block_id
        current.append(token)
    if current:
        parts.append(current)
    return "\n\n".join(decode_text(p) for p in parts if p)


class BaseChunker(abc.ABC):
    """Abstract interface implemented by every chunking strategy."""

    strategy_name: str = "base"

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()
        self.config.validate()

    @abc.abstractmethod
    def chunk(
        self,
        blocks: Sequence[ParsedBlock],
        *,
        source_id: str,
        source_name: str,
        source_type: str,
        document_version: str = "1",
    ) -> List[Chunk]:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}({self.config.to_dict()})"
