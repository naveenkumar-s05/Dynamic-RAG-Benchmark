"""Plain-text and Markdown parsing.

Markdown headings, paragraphs, lists, fenced code blocks and pipe tables are
preserved as distinct block types. Plain text is split into paragraphs on blank
lines with light heading detection; fenced code blocks present in a ``.txt`` file
are still recognized and kept identifiable as code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .base import PARSER_VERSION, is_heading_like, new_block, normalize_whitespace
from .models import ContentType, ParsedBlock, ParsedDocument

PARSER_NAME = "text_parser"

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$")
_SETEXT_UNDERLINE = re.compile(r"^(=+|-{2,})\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _read(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def parse(
    path: str | Path,
    *,
    source_id: str,
    source_name: str,
    source_type: str,
    document_version: str = "1",
    enable_ocr: bool = False,
) -> ParsedDocument:
    path = Path(path)
    raw = _read(path).replace("\r\n", "\n").replace("\r", "\n")
    is_markdown = source_type in ("md", "markdown")
    lines = raw.split("\n")

    blocks: List[ParsedBlock] = []
    warnings: List[str] = []
    section_path: List[str] = []
    index = 0
    paragraph_buf: List[str] = []
    list_level = 0
    table_counter = 0

    def add(
        content_type: str,
        text: str,
        *,
        heading_level: int | None = None,
        meta: dict | None = None,
    ) -> None:
        nonlocal index
        blocks.append(
            new_block(
                source_id=source_id,
                source_name=source_name,
                source_type=source_type,
                content_type=content_type,
                text=text,
                index=index,
                parser_name=PARSER_NAME,
                section_path=list(section_path),
                heading_level=heading_level,
                document_version=document_version,
                meta=meta or {},
            )
        )
        index += 1

    def flush_paragraph() -> None:
        nonlocal paragraph_buf, list_level
        text = normalize_whitespace("\n".join(paragraph_buf).strip())
        if text:
            add(ContentType.PARAGRAPH, text)
        paragraph_buf = []
        list_level = 0

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        fence = _FENCE.match(line)
        if fence:
            flush_paragraph()
            marker = fence.group(1)
            code_lines: List[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith(marker):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            add(ContentType.CODE, "\n".join(code_lines).strip(), meta={"is_code": True})
            continue

        heading = _ATX_HEADING.match(line) if is_markdown else None
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            section_path = section_path[: level - 1] + [text]
            add(ContentType.HEADING, text, heading_level=level)
            i += 1
            continue

        # Setext headings (Markdown): text followed by === or ---
        if is_markdown and i + 1 < n and line.strip() and _SETEXT_UNDERLINE.match(lines[i + 1]):
            flush_paragraph()
            level = 1 if lines[i + 1].strip().startswith("=") else 2
            text = line.strip()
            section_path = section_path[: level - 1] + [text]
            add(ContentType.HEADING, text, heading_level=level)
            i += 2
            continue

        if is_markdown and line.strip().startswith("|") and "|" in line.strip()[1:]:
            # Pipe table: gather contiguous rows starting with '|'.
            flush_paragraph()
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = [
                [c.strip() for c in ln.strip().strip("|").split("|")]
                for ln in table_lines
                if not set(ln.replace("|", "").strip()) <= {"-", ":", " "}
            ]
            if rows:
                header = rows[0]
                body = rows[1:]
                serialized = [" | ".join(header), "-" * len(" | ".join(header))]
                serialized += ["row %d: %s" % (r, " | ".join(row)) for r, row in enumerate(body, 1)]
                blocks.append(
                    new_block(
                        source_id=source_id,
                        source_name=source_name,
                        source_type=source_type,
                        content_type=ContentType.TABLE,
                        text="\n".join(serialized),
                        index=index,
                        parser_name=PARSER_NAME,
                        section_path=list(section_path),
                        table_id=f"tbl{table_counter}",
                        document_version=document_version,
                        meta={"headers": header, "row_count": len(body)},
                    )
                )
                index += 1
                table_counter += 1
            continue

        list_match = _LIST_ITEM.match(line)
        if list_match:
            flush_paragraph()
            add(ContentType.LIST_ITEM, list_match.group(1).strip(), meta={"list_level": 0})
            i += 1
            continue

        if line.strip() == "":
            flush_paragraph()
            i += 1
            continue

        # Light heading detection for plain text files.
        if (
            not is_markdown
            and not paragraph_buf
            and is_heading_like(line)
            and (i + 1 >= n or lines[i + 1].strip() == "")
        ):
            section_path = [line.strip()]
            add(ContentType.HEADING, line.strip(), heading_level=1)
            i += 1
            continue

        paragraph_buf.append(line)
        i += 1

    flush_paragraph()

    if not blocks:
        warnings.append("The text file contained no extractable content.")

    return ParsedDocument(
        source_id=source_id,
        source_name=source_name,
        source_type=source_type,
        blocks=blocks,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        document_version=document_version,
        source_path=str(path),
        warnings=warnings,
        metadata={"encoding_read_as": "auto", "line_count": n},
    )
