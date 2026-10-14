"""Input validation and safety helpers for uploaded / on-disk documents."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".csv",
}

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class ValidationError(ValueError):
    """Raised when user-supplied input is invalid or unsafe."""


def validate_extension(filename: str, allowed: Iterable[str] | None = None) -> str:
    """Return the lower-cased extension or raise :class:`ValidationError`."""
    allowed = set(allowed) if allowed else SUPPORTED_EXTENSIONS
    ext = Path(filename).suffix.lower()
    if not ext:
        raise ValidationError(f"File '{filename}' has no extension.")
    if ext not in allowed:
        raise ValidationError(
            f"Unsupported file type '{ext}'. Supported: {sorted(allowed)}"
        )
    return ext


def safe_filename(filename: str) -> str:
    """Return a filesystem-safe base name that cannot traverse directories.

    Strips directory components (defeating ``../`` traversal), normalizes
    unicode, and replaces unsafe characters.
    """
    if not filename:
        raise ValidationError("Empty filename.")
    # Take only the final path component regardless of separator style.
    name = filename.replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = _UNSAFE_CHARS.sub("_", name).strip("._")
    if not name:
        raise ValidationError(f"Filename '{filename}' is not usable.")
    return name


def validate_upload_name(filename: str, allowed: Iterable[str] | None = None) -> str:
    """Validate an uploaded filename and return a safe version of it."""
    safe = safe_filename(filename)
    validate_extension(safe, allowed)
    return safe


def validate_file_size(size_bytes: int, max_bytes: int) -> None:
    """Raise if ``size_bytes`` exceeds ``max_bytes`` (0 disables the check)."""
    if max_bytes and size_bytes > max_bytes:
        raise ValidationError(
            f"File is {size_bytes / 1e6:.1f} MB which exceeds the "
            f"{max_bytes / 1e6:.1f} MB limit."
        )
