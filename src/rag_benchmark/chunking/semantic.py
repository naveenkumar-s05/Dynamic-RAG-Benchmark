"""Strategy E: semantic chunking.

Sentences are embedded and grouped by topic continuity: a boundary is placed
wherever the cosine similarity between consecutive sentence embeddings drops
below ``semantic_threshold``. Groups are then merged up to the target size so
chunks are not absurdly small. If an embedder is not supplied (or too few
sentences exist) the strategy falls back to sentence grouping and records why.
"""

from __future__ import annotations

from itertools import groupby
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..parsers.models import ParsedBlock
from ..utils.tokens import count_tokens
from ._textsplit import split_sentences
from .base import BaseChunker, Chunk, make_chunk


class SemanticChunker(BaseChunker):
    strategy_name = "semantic"

    def __init__(self, config=None, embedder: Optional[Callable] = None) -> None:
        super().__init__(config)
        self.embedder = embedder

    def _embed(self, texts: List[str]) -> Optional[np.ndarray]:
        if self.embedder is None or not texts:
            return None
        try:
            if callable(self.embedder):
                vectors = self.embedder(texts)
            else:
                vectors = self.embedder.encode(texts, normalize_embeddings=True)
        except Exception:
            return None
        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Normalize for cosine similarity.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def chunk(
        self,
        blocks: Sequence[ParsedBlock],
        *,
        source_id: str,
        source_name: str,
        source_type: str,
        document_version: str = "1",
    ) -> List[Chunk]:
        live = [b for b in blocks if (b.text or "").strip()]
        if not live:
            return []

        chunks: List[Chunk] = []
        state = {"index": 0}

        def emit(text, member_blocks, reason, extra=None):
            if not text.strip():
                return
            chunks.append(
                make_chunk(
                    strategy=self.strategy_name,
                    source_id=source_id,
                    source_name=source_name,
                    source_type=source_type,
                    document_version=document_version,
                    text=text,
                    content_type="mixed",
                    chunk_index=state["index"],
                    blocks=member_blocks,
                    boundary_reason=reason,
                    extra_meta=extra or {},
                )
            )
            state["index"] += 1

        # Work section by section so semantic groups never cross headings.
        if self.config.respect_sections:
            key = lambda b: tuple(b.section_path)  # noqa: E731
        else:
            key = lambda b: ()  # noqa: E731

        for _, section_iter in groupby(live, key=key):
            section_blocks = list(section_iter)
            sentences: List[Tuple[str, ParsedBlock]] = []
            for block in section_blocks:
                for sentence in split_sentences(block.text):
                    sentences.append((sentence, block))
            if not sentences:
                continue

            if len(sentences) < 3 or self.embedder is None:
                reason = (
                    "fallback: no embedder available"
                    if self.embedder is None
                    else "fallback: too few sentences for semantic detection"
                )
                for text, member_blocks in _greedy_group(sentences, self.config.target_tokens):
                    emit(text, member_blocks, reason)
                continue

            vectors = self._embed([s for s, _ in sentences])
            if vectors is None:
                for text, member_blocks in _greedy_group(sentences, self.config.target_tokens):
                    emit(text, member_blocks, "fallback: embedding failed")
                continue

            sims = np.sum(vectors[:-1] * vectors[1:], axis=1)
            groups: List[List[Tuple[str, ParsedBlock]]] = [[sentences[0]]]
            for i, sim in enumerate(sims):
                if sim < self.config.semantic_threshold:
                    groups.append([sentences[i + 1]])
                else:
                    groups[-1].append(sentences[i + 1])

            # Merge small groups up to target size, but never cross a detected
            # semantic boundary unless the group is below min_tokens.
            merged: List[List[Tuple[str, ParsedBlock]]] = []
            current: List[Tuple[str, ParsedBlock]] = []
            current_tokens = 0
            for group in groups:
                group_text = " ".join(s for s, _ in group)
                group_tokens = count_tokens(group_text)
                if current and current_tokens + group_tokens > self.config.target_tokens:
                    merged.append(current)
                    current = []
                    current_tokens = 0
                current.extend(group)
                current_tokens += group_tokens
            if current:
                merged.append(current)

            for group in merged:
                text = " ".join(s for s, _ in group)
                member_blocks = _unique_blocks(group)
                emit(
                    text,
                    member_blocks,
                    "semantic boundary: consecutive sentences below similarity threshold",
                    extra={"semantic_threshold": self.config.semantic_threshold,
                           "sentence_count": len(group)},
                )

        return chunks


def _greedy_group(sentences, target_tokens):
    groups = []
    current: List[Tuple[str, ParsedBlock]] = []
    tokens = 0
    for sentence, block in sentences:
        st = count_tokens(sentence)
        if current and tokens + st > target_tokens:
            groups.append((" ".join(s for s, _ in current), _unique_blocks(current)))
            current = []
            tokens = 0
        current.append((sentence, block))
        tokens += st
    if current:
        groups.append((" ".join(s for s, _ in current), _unique_blocks(current)))
    return groups


def _unique_blocks(pairs) -> List[ParsedBlock]:
    seen = set()
    result: List[ParsedBlock] = []
    for _, block in pairs:
        if block.block_id not in seen:
            seen.add(block.block_id)
            result.append(block)
    return result
