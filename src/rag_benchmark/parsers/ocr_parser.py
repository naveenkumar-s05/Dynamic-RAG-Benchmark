"""Optional OCR support for scanned PDF pages and images.

OCR requires two external pieces: the ``pytesseract`` Python package and the
Tesseract binary. Both are optional. When they are missing the router records an
explicit warning instead of silently pretending the scanned content was read.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from ..utils.logging_config import get_logger
from .base import PARSER_VERSION, new_block, normalize_whitespace
from .models import ContentType, ParsedBlock, ParsedDocument

logger = get_logger(__name__)
PARSER_NAME = "ocr_parser"


def _configure_tesseract() -> bool:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return False
    cmd = os.getenv("TESSERACT_CMD", "").strip()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        logger.warning("pytesseract is installed but the Tesseract binary was not found.")
        return False
    return True


def ocr_available() -> bool:
    """Return True when both pytesseract and the Tesseract binary are usable."""
    return _configure_tesseract()


def _image_to_text(image) -> str:
    import pytesseract  # type: ignore

    try:
        return pytesseract.image_to_string(image)
    except Exception as exc:  # pragma: no cover - depends on external binary
        logger.warning("OCR failed: %s", exc)
        return ""


def ocr_pdf_page(page) -> str:
    """Render a PyMuPDF page to an image and OCR it."""
    try:
        pix = page.get_pixmap(dpi=200)
        from PIL import Image  # type: ignore

        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not render PDF page for OCR: %s", exc)
        return ""
    return _image_to_text(image)


def parse(
    path: str | Path,
    *,
    source_id: str,
    source_name: str,
    document_version: str = "1",
    enable_ocr: bool = True,
) -> ParsedDocument:
    """OCR a standalone image file into a single text block."""
    path = Path(path)
    blocks: List[ParsedBlock] = []
    warnings: List[str] = []

    if not ocr_available():
        warnings.append(
            "OCR requested for an image file but Tesseract is unavailable; "
            "no text was extracted."
        )
        return ParsedDocument(
            source_id=source_id,
            source_name=source_name,
            source_type="image",
            blocks=blocks,
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
            document_version=document_version,
            source_path=str(path),
            warnings=warnings,
        )

    from PIL import Image  # type: ignore

    text = normalize_whitespace(_image_to_text(Image.open(path)))
    if text:
        blocks.append(
            new_block(
                source_id=source_id,
                source_name=source_name,
                source_type="image",
                content_type=ContentType.PARAGRAPH,
                text=text,
                index=0,
                parser_name=PARSER_NAME,
                document_version=document_version,
                extraction_status="ocr",
                meta={"ocr": True},
            )
        )
    else:
        warnings.append("OCR produced no text for the image.")

    return ParsedDocument(
        source_id=source_id,
        source_name=source_name,
        source_type="image",
        blocks=blocks,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        document_version=document_version,
        source_path=str(path),
        warnings=warnings,
    )
