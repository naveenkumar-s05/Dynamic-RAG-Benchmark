"""Central logging configuration.

Logging is configured once and reused. By default it never logs document
contents or secrets -- only counts, identifiers and status messages.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", force: bool = False) -> None:
    """Configure the root logger exactly once unless ``force`` is set."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    for handler in list(root.handlers):
        if force:
            root.removeHandler(handler)

    if force or not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)

    # Keep noisy third-party loggers quiet by default.
    for noisy in ("urllib3", "sentence_transformers", "transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(name if name else "rag_benchmark")
