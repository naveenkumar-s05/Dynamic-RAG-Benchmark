"""Shared pytest fixtures.

Unit tests use a deterministic, dependency-free fake embedder so the suite runs
quickly and without network access. The real sentence-transformers model is
exercised by an opt-in integration test (set ``RUN_EMBEDDING_TESTS=1``).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
for _p in (str(PROJECT_ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rag_benchmark.parsers.models import (  # noqa: E402
    ContentType,
    ParsedBlock,
    ParsedDocument,
)


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder for tests."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype="float32")
        for token in (text or "").lower().split():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode(self, texts, normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return np.vstack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dimension), "float32")

    @property
    def model_name(self) -> str:  # pragma: no cover - convenience
        return "fake-embedder"

    def __call__(self, texts) -> np.ndarray:
        return self.encode(texts)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def _block(i, text, content_type=ContentType.PARAGRAPH, section=None, page=None,
           table_id=None, row_number=None, meta=None, source_id="src1",
           source_name="doc.md", source_type="md"):
    return ParsedBlock(
        block_id=f"{source_id}:b{i:05d}",
        source_id=source_id,
        source_name=source_name,
        source_type=source_type,
        content_type=content_type,
        text=text,
        page_number=page,
        section_path=list(section or []),
        table_id=table_id,
        row_number=row_number,
        meta=dict(meta or {}),
    )


@pytest.fixture
def make_block():
    return _block


@pytest.fixture
def sample_document() -> ParsedDocument:
    blocks = [
        _block(0, "Security Overview", ContentType.HEADING, section=["Security Overview"], page=1),
        _block(
            1,
            "Northwind encrypts data at rest with AES-256 and in transit with TLS 1.3. "
            "Keys live in a hardware security module.",
            ContentType.PARAGRAPH,
            section=["Security Overview"],
            page=1,
        ),
        _block(2, "Plans and Limits", ContentType.HEADING, section=["Plans and Limits"], page=2),
        _block(
            3,
            "row 1: Basic | 100 GB | 30 days | $5\nrow 2: Enterprise | 10 TB | 365 days | $49",
            ContentType.TABLE,
            section=["Plans and Limits"],
            page=2,
            table_id="t0",
            meta={"headers": ["Plan", "Storage", "Retention", "Price"], "row_count": 2},
        ),
        _block(4, "step one: stop the service", ContentType.LIST_ITEM, section=["Restore"], page=3),
        _block(5, "step two: restore the snapshot", ContentType.LIST_ITEM, section=["Restore"], page=3),
        _block(6, "def backup():\n    return True", ContentType.CODE, section=["Restore"], page=3,
               meta={"is_code": True}),
    ]
    return ParsedDocument(
        source_id="src1",
        source_name="doc.md",
        source_type="md",
        blocks=blocks,
        parser_name="test",
        parser_version="1.0.0",
    )


@pytest.fixture
def markdown_file(tmp_path) -> Path:
    content = (
        "# Title\n\n"
        "Intro paragraph with a sentence. And another sentence here.\n\n"
        "## Section A\n\n"
        "| Plan | Price |\n| --- | --- |\n| Basic | $5 |\n| Enterprise | $49 |\n\n"
        "- first item\n- second item\n\n"
        "```python\nprint('hello')\n```\n"
    )
    path = tmp_path / "sample.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def csv_file(tmp_path) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(
        "plan,storage_gb,retention_days\nBasic,100,30\nEnterprise,10000,365\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def html_file(tmp_path) -> Path:
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>T</title><script>alert(1)</script>"
        "<style>.x{}</style></head><body>"
        "<nav>menu</nav><h1>Heading</h1><p>Some paragraph text.</p>"
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        "</body></html>",
        encoding="utf-8",
    )
    return path
