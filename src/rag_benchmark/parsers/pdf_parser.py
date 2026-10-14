"""PDF parsing using PyMuPDF.

Preserves page boundaries, reading order and page numbers. Basic heading
detection uses relative font size. Tables are extracted where PyMuPDF's table
finder supports them and serialized with their header row so header/value
relationships survive chunking. Pages with little or no extractable text are
flagged and, when OCR is enabled and available, routed to the OCR parser.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import List, Optional, Tuple

from ..utils.logging_config import get_logger
from .base import PARSER_VERSION, is_heading_like, new_block, normalize_whitespace
from .models import ContentType, ParsedBlock, ParsedDocument

logger = get_logger(__name__)

PARSER_NAME = "pdf_parser"
MIN_PAGE_CHARS = 25


def _serialize_table(rows: List[List[Optional[str]]], headers: Optional[List[str]]) -> str:
    """Render a table as text keeping headers aligned with each value."""
    norm_rows = [
        ["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row]
        for row in rows
    ]
    if not norm_rows:
        return ""
    if not headers or all(h is None for h in headers):
        headers = norm_rows[0]
        norm_rows = norm_rows[1:]
    header_line = " | ".join(headers)
    lines = [header_line, "-" * len(header_line)]
    for i, row in enumerate(norm_rows, start=1):
        cells = " | ".join(row)
        lines.append(f"row {i}: {cells}")
    return "\n".join(lines)


def _extract_tables(page) -> List[Tuple[str, List[str], List[List[str]]]]:
    tables = []
    try:
        finder = page.find_tables()
    except Exception:
        return tables
    for t_idx, table in enumerate(getattr(finder, "tables", []) or []):
        try:
            data = table.extract()
        except Exception:
            continue
        if not data:
            continue
        headers = [str(c) if c is not None else "" for c in data[0]]
        body = [[str(c) if c is not None else "" for c in row] for row in data[1:]]
        tables.append((f"t{t_idx}", headers, body))
    return tables


def parse(
    path: str | Path,
    *,
    source_id: str,
    source_name: str,
    document_version: str = "1",
    enable_ocr: bool = False,
) -> ParsedDocument:
    import fitz  # PyMuPDF

    path = Path(path)
    doc = fitz.open(path)
    blocks: List[ParsedBlock] = []
    warnings: List[str] = []
    low_text_pages: List[int] = []
    section_path: List[str] = []
    index = 0

    try:
        for page_number, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            spans_sizes: List[float] = []
            for blk in page_dict.get("blocks", []):
                if blk.get("type") != 0:
                    continue
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            spans_sizes.append(float(span.get("size", 0.0)))
            body_size = statistics.median(spans_sizes) if spans_sizes else 0.0

            page_chars = 0
            for blk in page_dict.get("blocks", []):
                if blk.get("type") != 0:
                    continue
                text = " ".join(
                    span.get("text", "")
                    for line in blk.get("lines", [])
                    for span in line.get("spans", [])
                ).strip()
                if not text:
                    continue
                page_chars += len(text)
                block_size = max(
                    (
                        float(span.get("size", 0.0))
                        for line in blk.get("lines", [])
                        for span in line.get("spans", [])
                        if span.get("text", "").strip()
                    ),
                    default=body_size,
                )
                is_heading = (
                    body_size > 0
                    and block_size >= body_size * 1.15
                    and is_heading_like(text)
                )
                if is_heading:
                    level = 1 if block_size >= body_size * 1.6 else 2
                    section_path = (
                        section_path[: level - 1] + [text]
                        if level > 1
                        else [text]
                    )
                    blocks.append(
                        new_block(
                            source_id=source_id,
                            source_name=source_name,
                            source_type="pdf",
                            content_type=ContentType.HEADING,
                            text=text,
                            index=index,
                            parser_name=PARSER_NAME,
                            page_number=page_number,
                            section_path=list(section_path),
                            heading_level=level,
                            document_version=document_version,
                        )
                    )
                else:
                    blocks.append(
                        new_block(
                            source_id=source_id,
                            source_name=source_name,
                            source_type="pdf",
                            content_type=ContentType.PARAGRAPH,
                            text=normalize_whitespace(text),
                            index=index,
                            parser_name=PARSER_NAME,
                            page_number=page_number,
                            section_path=list(section_path),
                            document_version=document_version,
                        )
                    )
                index += 1

            for table_id, headers, body in _extract_tables(page):
                blocks.append(
                    new_block(
                        source_id=source_id,
                        source_name=source_name,
                        source_type="pdf",
                        content_type=ContentType.TABLE,
                        text=_serialize_table(body, headers),
                        index=index,
                        parser_name=PARSER_NAME,
                        page_number=page_number,
                        section_path=list(section_path),
                        table_id=table_id,
                        document_version=document_version,
                        meta={"headers": headers, "row_count": len(body)},
                    )
                )
                index += 1

            if page_chars < MIN_PAGE_CHARS:
                low_text_pages.append(page_number)
                has_images = bool(page.get_images())
                note = (
                    f"Page {page_number} yielded little extractable text "
                    f"({page_chars} chars)"
                    + (" and contains images" if has_images else "")
                    + "."
                )
                warnings.append(note)

        page_count = doc.page_count
    finally:
        doc.close()

    if low_text_pages:
        ocr_warning = _maybe_ocr(
            path,
            blocks,
            low_text_pages,
            source_id=source_id,
            source_name=source_name,
            document_version=document_version,
            enable_ocr=enable_ocr,
            warnings=warnings,
        )
        if ocr_warning:
            warnings.append(ocr_warning)

    return ParsedDocument(
        source_id=source_id,
        source_name=source_name,
        source_type="pdf",
        blocks=blocks,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        document_version=document_version,
        source_path=str(path),
        warnings=warnings,
        metadata={"page_count": page_count, "low_text_pages": low_text_pages},
    )


def _maybe_ocr(
    path: Path,
    blocks: List[ParsedBlock],
    low_text_pages: List[int],
    *,
    source_id: str,
    source_name: str,
    document_version: str,
    enable_ocr: bool,
    warnings: List[str],
) -> Optional[str]:
    from .ocr_parser import ocr_available, ocr_pdf_page

    if not ocr_available():
        return (
            "OCR was requested (or would help) but the Tesseract dependency is "
            "unavailable; scanned pages remain text-less. Install pytesseract "
            "and the Tesseract binary to enable OCR."
        )
    if not enable_ocr:
        return (
            "Low-text pages detected. OCR is available but disabled; enable OCR "
            "to extract text from scanned pages."
        )

    import fitz

    doc = fitz.open(path)
    try:
        next_index = len(blocks)
        for page_number in low_text_pages:
            page = doc[page_number - 1]
            text = ocr_pdf_page(page)
            if text and text.strip():
                blocks.append(
                    new_block(
                        source_id=source_id,
                        source_name=source_name,
                        source_type="pdf",
                        content_type=ContentType.PARAGRAPH,
                        text=normalize_whitespace(text),
                        index=next_index,
                        parser_name="ocr_parser",
                        page_number=page_number,
                        document_version=document_version,
                        extraction_status="ocr",
                        meta={"ocr": True},
                    )
                )
                next_index += 1
    finally:
        doc.close()
    return None
