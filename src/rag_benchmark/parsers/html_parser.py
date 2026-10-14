"""HTML parsing using BeautifulSoup.

Removes scripts, styles, navigation, footers and other boilerplate, then walks
the main readable content in document order, preserving headings (with
hierarchy), lists, tables and code blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup  # type: ignore

from .base import PARSER_VERSION, new_block, normalize_whitespace
from .models import ContentType, ParsedBlock, ParsedDocument

PARSER_NAME = "html_parser"

_STRIP_TAGS = [
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    "svg",
]
_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "pre", "blockquote"}


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
    document_version: str = "1",
    enable_ocr: bool = False,
) -> ParsedDocument:
    path = Path(path)
    soup = BeautifulSoup(_read(path), "lxml")

    for tag in _STRIP_TAGS:
        for element in soup.find_all(tag):
            element.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup

    blocks: List[ParsedBlock] = []
    warnings: List[str] = []
    section_path: List[str] = []
    index = 0

    def add(content_type: str, text: str, **kwargs) -> None:
        nonlocal index
        blocks.append(
            new_block(
                source_id=source_id,
                source_name=source_name,
                source_type="html",
                content_type=content_type,
                text=text,
                index=index,
                parser_name=PARSER_NAME,
                section_path=list(section_path),
                document_version=document_version,
                **kwargs,
            )
        )
        index += 1

    def render_table(table) -> Optional[str]:
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if not rows:
            return None
        header = rows[0]
        body = rows[1:]
        lines = [" | ".join(header), "-" * len(" | ".join(header))]
        lines += ["row %d: %s" % (r, " | ".join(row)) for r, row in enumerate(body, 1)]
        return "\n".join(lines)

    for element in main.find_all(_BLOCK_TAGS):
        # Skip nested blocks already handled by their parent list/table.
        if element.find_parent(["ul", "ol", "table", "pre"]) is not None:
            continue
        name = element.name
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            section_path = section_path[: level - 1] + [text]
            add(ContentType.HEADING, text, heading_level=level)
        elif name in {"p", "blockquote"}:
            text = normalize_whitespace(element.get_text(" ", strip=True))
            if text:
                add(ContentType.PARAGRAPH, text)
        elif name in {"ul", "ol"}:
            for li in element.find_all("li", recursive=False):
                text = normalize_whitespace(li.get_text(" ", strip=True))
                if text:
                    add(ContentType.LIST_ITEM, text)
        elif name == "table":
            rendered = render_table(element)
            if rendered:
                add(ContentType.TABLE, rendered)
        elif name == "pre":
            text = element.get_text("\n", strip=True)
            if text:
                add(ContentType.CODE, text, meta={"is_code": True})

    if not blocks:
        warnings.append("No readable content was extracted from the HTML document.")

    return ParsedDocument(
        source_id=source_id,
        source_name=source_name,
        source_type="html",
        blocks=blocks,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        document_version=document_version,
        source_path=str(path),
        warnings=warnings,
        metadata={"title": soup.title.get_text(strip=True) if soup.title else None},
    )
