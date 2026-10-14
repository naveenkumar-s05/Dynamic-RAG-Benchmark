"""Shared helpers for the format-specific parsers."""

from __future__ import annotations

import hashlib
from typing import List, Optional

from .models import ContentType, ParsedBlock

PARSER_VERSION = "1.0.0"


def stable_source_id(source_name: str, extra: str = "") -> str:
    """Deterministic, collision-resistant source identifier.

    Combines a sanitized name fragment with a short content hash so the same
    filename in different corpora does not silently collide.
    """
    key = f"{source_name}::{extra}".encode("utf-8", "ignore")
    digest = hashlib.sha1(key).hexdigest()[:10]
    stem = "".join(c if c.isalnum() else "_" for c in source_name)[:32].strip("_")
    return f"{stem or 'doc'}-{digest}"


def make_block_id(source_id: str, index: int) -> str:
    """Block identifiers are deterministic and unique within a source."""
    return f"{source_id}:b{index:05d}"


def new_block(
    *,
    source_id: str,
    source_name: str,
    source_type: str,
    content_type: str,
    text: str,
    index: int,
    parser_name: str,
    page_number: Optional[int] = None,
    section_path: Optional[List[str]] = None,
    heading_level: Optional[int] = None,
    table_id: Optional[str] = None,
    row_number: Optional[int] = None,
    document_version: str = "1",
    extraction_status: str = "ok",
    warnings: Optional[List[str]] = None,
    meta: Optional[dict] = None,
) -> ParsedBlock:
    return ParsedBlock(
        block_id=make_block_id(source_id, index),
        source_id=source_id,
        source_name=source_name,
        source_type=source_type,
        content_type=content_type,
        text=text,
        page_number=page_number,
        section_path=list(section_path or []),
        heading_level=heading_level,
        table_id=table_id,
        row_number=row_number,
        document_version=document_version,
        parser_name=parser_name,
        parser_version=PARSER_VERSION,
        extraction_status=extraction_status,
        warnings=list(warnings or []),
        meta=dict(meta or {}),
    )


def normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines / trailing spaces without losing paragraphs."""
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out: List[str] = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                out.append("")
        else:
            blank_run = 0
            out.append(ln)
    return "\n".join(out).strip()


def is_heading_like(text: str) -> bool:
    """Heuristic: short, title-cased, non-terminal lines may be headings."""
    t = text.strip()
    if not t or len(t) > 80:
        return False
    if t.endswith((".", ",", ";", ":", "!", "?")):
        return False
    words = t.split()
    if len(words) > 12:
        return False
    return True
