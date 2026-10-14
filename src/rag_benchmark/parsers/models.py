"""Structured data models produced by the parsing layer.

Parsed documents are represented as an ordered list of :class:`ParsedBlock`
objects rather than a single flat string. This preserves headings, paragraphs,
lists, tables and code blocks, together with provenance (page numbers, section
paths, table rows) that downstream chunking and retrieval depend on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class ContentType:
    """Canonical content-type labels used throughout the pipeline."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    CODE = "code"
    RECORD = "record"  # a single CSV row serialized with its column headers
    CAPTION = "caption"
    PAGE = "page"
    OTHER = "other"


@dataclass
class ParsedBlock:
    """A single structured content block extracted from a document.

    Fields that do not apply to a given format are left as ``None`` rather than
    being filled with invented values.
    """

    block_id: str
    source_id: str
    source_name: str
    source_type: str
    content_type: str
    text: str
    page_number: Optional[int] = None
    section_path: List[str] = field(default_factory=list)
    heading_level: Optional[int] = None
    table_id: Optional[str] = None
    row_number: Optional[int] = None
    document_version: str = "1"
    parser_name: str = ""
    parser_version: str = ""
    extraction_status: str = "ok"
    warnings: List[str] = field(default_factory=list)
    # Extra, format-specific metadata (table headers, list level, order, ...).
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedBlock":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)

    @property
    def section_path_str(self) -> str:
        return " > ".join(self.section_path) if self.section_path else ""


@dataclass
class ParsedDocument:
    """A parsed source document and its extraction diagnostics."""

    source_id: str
    source_name: str
    source_type: str
    blocks: List[ParsedBlock] = field(default_factory=list)
    parser_name: str = ""
    parser_version: str = ""
    document_version: str = "1"
    source_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def page_count(self) -> Optional[int]:
        pages = [b.page_number for b in self.blocks if b.page_number is not None]
        return max(pages) if pages else None

    def to_dict(self, include_blocks: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "document_version": self.document_version,
            "source_path": self.source_path,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
            "block_count": self.block_count,
            "page_count": self.page_count,
        }
        if include_blocks:
            data["blocks"] = [b.to_dict() for b in self.blocks]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedDocument":
        blocks = [ParsedBlock.from_dict(b) for b in data.get("blocks", [])]
        return cls(
            source_id=data["source_id"],
            source_name=data["source_name"],
            source_type=data["source_type"],
            blocks=blocks,
            parser_name=data.get("parser_name", ""),
            parser_version=data.get("parser_version", ""),
            document_version=data.get("document_version", "1"),
            source_path=data.get("source_path"),
            warnings=list(data.get("warnings", [])),
            metadata=dict(data.get("metadata", {})),
        )
