"""Application-level configuration and path management.

This module is intentionally dependency-light. It loads environment variables
from a local ``.env`` (if present), exposes the project directories, and creates
runtime directories on demand. Package modules receive configuration explicitly
so that the library stays importable without side effects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR: Path = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR: Path = DATA_DIR / "sample_documents"
BENCHMARK_DIR: Path = DATA_DIR / "benchmark"
UPLOAD_DIR: Path = DATA_DIR / "uploads"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
INDEX_DIR: Path = PROJECT_ROOT / "indexes"
CACHE_DIR: Path = PROJECT_ROOT / ".cache"


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings resolved from environment variables."""

    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    default_top_k: int = field(default_factory=lambda: _get_int("DEFAULT_TOP_K", 5))
    default_retrieval_mode: str = field(
        default_factory=lambda: os.getenv("DEFAULT_RETRIEVAL_MODE", "dense")
    )

    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "none").strip().lower()
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2")
    )

    tesseract_cmd: str = field(
        default_factory=lambda: os.getenv("TESSERACT_CMD", "").strip()
    )

    max_upload_mb: float = field(
        default_factory=lambda: _get_float("MAX_UPLOAD_MB", 25.0)
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


SETTINGS = Settings()


def ensure_directories() -> None:
    """Create runtime directories. Safe to call repeatedly."""
    for directory in (
        DATA_DIR,
        SAMPLE_DOCS_DIR,
        BENCHMARK_DIR,
        UPLOAD_DIR,
        REPORTS_DIR,
        INDEX_DIR,
        CACHE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def redacted_settings_summary() -> dict:
    """Return a settings summary safe to display in the UI.

    Secret-like values are reduced to a boolean ``configured`` flag.
    """
    return {
        "embedding_model": SETTINGS.embedding_model,
        "default_top_k": SETTINGS.default_top_k,
        "default_retrieval_mode": SETTINGS.default_retrieval_mode,
        "llm_provider": SETTINGS.llm_provider,
        "openai_api_key_configured": bool(SETTINGS.openai_api_key),
        "openai_base_url": SETTINGS.openai_base_url,
        "openai_model": SETTINGS.openai_model,
        "ollama_base_url": SETTINGS.ollama_base_url,
        "ollama_model": SETTINGS.ollama_model,
        "tesseract_cmd_configured": bool(SETTINGS.tesseract_cmd),
        "max_upload_mb": SETTINGS.max_upload_mb,
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import json

    ensure_directories()
    print(json.dumps(redacted_settings_summary(), indent=2))
