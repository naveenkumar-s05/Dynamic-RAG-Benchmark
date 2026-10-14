"""Consistent token counting used by every chunking strategy.

A single tokenizer is used across the whole application so that chunk sizes are
comparable between strategies. ``tiktoken``'s ``cl100k_base`` encoding is used
when available; otherwise a deterministic word-based approximation is used so
the project still runs in constrained environments.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Optional

_WORD_RE = re.compile(r"\S+")

_ENCODER: Optional[Any] = None
_ENCODER_RESOLVED = False


def _get_encoder() -> Optional[Any]:
    global _ENCODER, _ENCODER_RESOLVED
    if _ENCODER_RESOLVED:
        return _ENCODER
    _ENCODER_RESOLVED = True
    try:  # pragma: no cover - depends on optional runtime availability
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - graceful fallback
        _ENCODER = None
    return _ENCODER


def tokenizer_name() -> str:
    """Human-readable description of the active tokenizer."""
    return "tiktoken:cl100k_base" if _get_encoder() is not None else "words:whitespace"


@functools.lru_cache(maxsize=8192)
def count_tokens(text: str) -> int:
    """Return the number of tokens in ``text``.

    Empty or ``None`` input yields ``0``. The result is cached because token
    counts are requested repeatedly during chunking and benchmarking.
    """
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return len(_WORD_RE.findall(text))


def encode_text(text: str) -> list:
    """Encode ``text`` into a list of tokens.

    With tiktoken the tokens are integer ids; the fallback returns word strings.
    Both forms are understood by :func:`decode_text`.
    """
    if not text:
        return []
    encoder = _get_encoder()
    if encoder is not None:
        return list(encoder.encode(text))
    return _WORD_RE.findall(text)


def decode_text(tokens: list) -> str:
    """Inverse of :func:`encode_text` for a (possibly partial) token slice."""
    if not tokens:
        return ""
    encoder = _get_encoder()
    if encoder is not None and isinstance(tokens[0], int):
        return encoder.decode(list(tokens))
    return " ".join(str(t) for t in tokens)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens (best effort)."""
    if max_tokens <= 0 or not text:
        return ""
    encoder = _get_encoder()
    if encoder is not None:
        ids = encoder.encode(text)
        if len(ids) <= max_tokens:
            return text
        return encoder.decode(ids[:max_tokens])
    words = _WORD_RE.findall(text)
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])
