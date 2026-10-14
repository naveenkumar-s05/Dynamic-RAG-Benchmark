"""Parser router: choose an extraction path based on format and content.

The router validates the file, inspects its extension and (where useful) its raw
content, then dispatches to the appropriate parser. It also performs post-parse
validation and records warnings instead of discarding problematic content.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

from ..utils.logging_config import get_logger
from ..utils.validation import ValidationError, validate_extension
from .base import stable_source_id
from .models import ContentType, ParsedBlock, ParsedDocument

logger = get_logger(__name__)

PARSER_ROUTES = {
    ".pdf": "pdf_parser",
    ".docx": "docx_parser",
    ".txt": "text_parser",
    ".md": "text_parser",
    ".markdown": "text_parser",
    ".html": "html_parser",
    ".htm": "html_parser",
    ".csv": "csv_parser",
}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sniff_extension(path: Path, extension: str) -> str:
    """Refine the extension when content contradicts it (light sniffing)."""
    if extension != ".txt":
        return extension
    try:
        head = path.read_bytes()[:1024].lstrip().lower()
    except OSError:
        return extension
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        return ".html"
    return extension


def choose_route(path: str | Path) -> str:
    """Return the parser name that will handle ``path``."""
    extension = validate_extension(Path(path).name)
    extension = _sniff_extension(Path(path), extension)
    return PARSER_ROUTES[extension]


def parse_document(
    path: str | Path,
    *,
    source_name: Optional[str] = None,
    source_id: Optional[str] = None,
    document_version: str = "1",
    enable_ocr: bool = True,
) -> ParsedDocument:
    """Parse a single document, returning a validated :class:`ParsedDocument`."""
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    if path.is_dir():
        raise ValidationError(f"Expected a file but got a directory: {path}")

    extension = validate_extension(path.name)
    extension = _sniff_extension(path, extension)
    route = PARSER_ROUTES[extension]

    name = source_name or path.name
    if source_id is None:
        source_id = stable_source_id(name, extra=_file_digest(path))

    if route == "pdf_parser":
        from . import pdf_parser as parser

        doc = parser.parse(
            path,
            source_id=source_id,
            source_name=name,
            document_version=document_version,
            enable_ocr=enable_ocr,
        )
    elif route == "docx_parser":
        from . import docx_parser as parser

        doc = parser.parse(
            path, source_id=source_id, source_name=name, document_version=document_version
        )
    elif route == "text_parser":
        from . import text_parser as parser

        source_type = "md" if extension in (".md", ".markdown") else "txt"
        doc = parser.parse(
            path,
            source_id=source_id,
            source_name=name,
            source_type=source_type,
            document_version=document_version,
        )
    elif route == "html_parser":
        from . import html_parser as parser

        doc = parser.parse(
            path, source_id=source_id, source_name=name, document_version=document_version
        )
    elif route == "csv_parser":
        from . import csv_parser as parser

        doc = parser.parse(
            path, source_id=source_id, source_name=name, document_version=document_version
        )
    else:  # pragma: no cover - defensive
        raise ValidationError(f"No parser registered for extension '{extension}'.")

    doc.metadata.setdefault("route", route)
    doc.metadata.setdefault("extension", extension)
    doc.metadata.setdefault("content_digest", _file_digest(path))
    doc.warnings.extend(validate_parsed_document(doc))
    logger.info(
        "Parsed '%s' via %s: %d blocks, %d warnings",
        name,
        route,
        doc.block_count,
        len(doc.warnings),
    )
    return doc


def validate_parsed_document(doc: ParsedDocument) -> List[str]:
    """Post-extraction sanity checks. Returns a list of warning strings."""
    warnings: List[str] = []
    text_chars = sum(len(b.text) for b in doc.blocks if b.text)
    if text_chars == 0:
        warnings.append("No meaningful text was extracted from the document.")
        return warnings
    if text_chars < 20:
        warnings.append(
            f"Very little text extracted ({text_chars} characters); the source "
            "may be scanned, encrypted or empty."
        )

    empty_blocks = sum(1 for b in doc.blocks if not (b.text or "").strip())
    if empty_blocks:
        warnings.append(f"{empty_blocks} empty block(s) were produced.")

    # Verify table blocks retain header information.
    for block in doc.blocks:
        if block.content_type == ContentType.TABLE:
            headers = block.meta.get("headers")
            if not headers:
                warnings.append(
                    f"Table block {block.block_id} is missing header metadata."
                )

    # Page numbers, when present, should be monotonically non-decreasing.
    pages = [b.page_number for b in doc.blocks if b.page_number is not None]
    if pages and pages != sorted(pages):
        warnings.append("Reading order may be unstable: page numbers are out of order.")

    return warnings
