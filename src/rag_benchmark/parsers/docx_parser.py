"""DOCX parsing using python-docx.

Walks the document body in true reading order so headings, paragraphs, lists and
tables keep their relative position. Content is associated with the nearest
preceding heading via ``section_path``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from docx import Document  # type: ignore
from docx.table import Table  # type: ignore
from docx.text.paragraph import Paragraph  # type: ignore

from .base import PARSER_VERSION, new_block, normalize_whitespace
from .models import ContentType, ParsedBlock, ParsedDocument

PARSER_NAME = "docx_parser"
_HEADING_RE = re.compile(r"heading\s*(\d+)", re.IGNORECASE)


def _iter_block_items(parent):
    """Yield Paragraph and Table objects in document order."""
    from docx.oxml.ns import qn  # type: ignore

    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def _heading_level(paragraph: Paragraph) -> int | None:
    style = (paragraph.style.name or "") if paragraph.style else ""
    if style.lower() == "title":
        return 1
    match = _HEADING_RE.search(style)
    if match:
        return int(match.group(1))
    return None


def _list_level(paragraph: Paragraph) -> int:
    try:
        pPr = paragraph._p.pPr
        if pPr is not None and pPr.numPr is not None and pPr.numPr.ilvl is not None:
            return int(pPr.numPr.ilvl.val)
    except Exception:
        pass
    return 0


def _is_list(paragraph: Paragraph) -> bool:
    style = (paragraph.style.name or "") if paragraph.style else ""
    return "list" in style.lower() or _list_level(paragraph) > 0


def parse(
    path: str | Path,
    *,
    source_id: str,
    source_name: str,
    document_version: str = "1",
    enable_ocr: bool = False,
) -> ParsedDocument:
    path = Path(path)
    document = Document(str(path))
    blocks: List[ParsedBlock] = []
    warnings: List[str] = []
    section_path: List[str] = []
    index = 0
    table_counter = 0

    for item in _iter_block_items(document):
        if isinstance(item, Paragraph):
            text = normalize_whitespace(item.text or "")
            if not text:
                continue
            level = _heading_level(item)
            if level is not None:
                section_path = section_path[: level - 1] + [text]
                blocks.append(
                    new_block(
                        source_id=source_id,
                        source_name=source_name,
                        source_type="docx",
                        content_type=ContentType.HEADING,
                        text=text,
                        index=index,
                        parser_name=PARSER_NAME,
                        section_path=list(section_path),
                        heading_level=level,
                        document_version=document_version,
                    )
                )
            elif _is_list(item):
                blocks.append(
                    new_block(
                        source_id=source_id,
                        source_name=source_name,
                        source_type="docx",
                        content_type=ContentType.LIST_ITEM,
                        text=text,
                        index=index,
                        parser_name=PARSER_NAME,
                        section_path=list(section_path),
                        document_version=document_version,
                        meta={"list_level": _list_level(item)},
                    )
                )
            else:
                blocks.append(
                    new_block(
                        source_id=source_id,
                        source_name=source_name,
                        source_type="docx",
                        content_type=ContentType.PARAGRAPH,
                        text=text,
                        index=index,
                        parser_name=PARSER_NAME,
                        section_path=list(section_path),
                        document_version=document_version,
                    )
                )
            index += 1
        elif isinstance(item, Table):
            table_id = f"tbl{table_counter}"
            table_counter += 1
            rows = [
                [cell.text.strip() for cell in row.cells] for row in item.rows
            ]
            if not rows:
                warnings.append(f"Empty table encountered ({table_id}).")
                continue
            headers = rows[0]
            body = rows[1:]
            lines = [" | ".join(headers), "-" * len(" | ".join(headers))]
            for r_idx, row in enumerate(body, start=1):
                lines.append(f"row {r_idx}: " + " | ".join(row))
            blocks.append(
                new_block(
                    source_id=source_id,
                    source_name=source_name,
                    source_type="docx",
                    content_type=ContentType.TABLE,
                    text="\n".join(lines),
                    index=index,
                    parser_name=PARSER_NAME,
                    section_path=list(section_path),
                    table_id=table_id,
                    document_version=document_version,
                    meta={"headers": headers, "row_count": len(body)},
                )
            )
            index += 1

    if not blocks:
        warnings.append("No text or tables were extracted from the DOCX document.")

    return ParsedDocument(
        source_id=source_id,
        source_name=source_name,
        source_type="docx",
        blocks=blocks,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        document_version=document_version,
        source_path=str(path),
        warnings=warnings,
        metadata={},
    )
