"""Sentence and paragraph splitting helpers shared by chunking strategies."""

from __future__ import annotations

import re
from typing import List

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Common abbreviations that should not end a sentence.
_ABBREVIATIONS = [
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Mt", "Inc", "Ltd",
    "Co", "Corp", "e.g", "i.e", "etc", "vs", "Fig", "No", "Vol", "Ed", "al",
    "approx", "min", "max", "sec", "hrs", "hr", "dept", "est",
]

_ABBR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.(?=\s)",
    re.IGNORECASE,
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences without over-fragmenting abbreviations.

    Returns a list of non-empty, stripped sentences. Text with no sentence
    punctuation is returned as a single sentence.
    """
    if not text or not text.strip():
        return []

    # Protect abbreviation periods with a placeholder.
    protected = _ABBR_PATTERN.sub(lambda m: m.group(0)[:-1] + "\x00", text)

    sentences: List[str] = []
    for line in protected.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Never merge separate bullet lines into one sentence.
        if _BULLET.match(line):
            sentences.append(line)
            continue
        parts = _SENTENCE_END.split(line)
        for part in parts:
            cleaned = part.replace("\x00", ".").strip()
            if cleaned:
                sentences.append(cleaned)
    return sentences


def split_paragraphs(text: str) -> List[str]:
    """Split ``text`` into paragraphs on blank lines."""
    if not text:
        return []
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


def recursive_split(text: str, max_tokens: int) -> List[str]:
    """Split ``text`` into pieces of at most ``max_tokens`` tokens.

    Tries progressively smaller separators before falling back to a hard token
    slice. Shared by the recursive and hybrid chunking strategies.
    """
    from ..utils.tokens import count_tokens, decode_text, encode_text

    if count_tokens(text) <= max_tokens:
        return [text]

    for i, sep in enumerate(_SEPARATORS):
        if sep and sep in text:
            parts = text.split(sep)
            parts = [
                p + sep if j < len(parts) - 1 else p for j, p in enumerate(parts)
            ]
            return _merge_parts(parts, max_tokens)
    ids = encode_text(text)
    return [
        decode_text(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)
    ]


def _merge_parts(parts: List[str], max_tokens: int) -> List[str]:
    from ..utils.tokens import count_tokens

    results: List[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if count_tokens(part) > max_tokens:
            if current.strip():
                results.append(current)
                current = ""
            results.extend(recursive_split(part, max_tokens))
            continue
        candidate = (current + part) if current else part
        if count_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current.strip():
                results.append(current)
            current = part
    if current.strip():
        results.append(current)
    return results
