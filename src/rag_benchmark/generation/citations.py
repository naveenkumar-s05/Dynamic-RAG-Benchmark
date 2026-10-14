"""Citation formatting and parsing for grounded answers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from ..retrieval.retriever import RetrievalResult

_CITATION_RE = re.compile(r"\[(\d+)\]")


def build_citation(index: int, result: RetrievalResult) -> Dict[str, Any]:
    return {
        "index": index,
        "chunk_id": result.chunk_id,
        "source_name": result.source_name,
        "source_type": result.source_type,
        "page": result.page_number,
        "page_range": result.page_range,
        "section": " > ".join(result.section_path) if result.section_path else None,
        "content_type": result.content_type,
        "score": result.score,
    }


def build_citations(results: Sequence[RetrievalResult]) -> List[Dict[str, Any]]:
    return [build_citation(i, r) for i, r in enumerate(results, start=1)]


def build_context_block(results: Sequence[RetrievalResult]) -> str:
    """Render numbered evidence passages for the LLM prompt.

    Document text is explicitly wrapped and labelled as data so the model does
    not treat it as instructions.
    """
    blocks: List[str] = []
    for i, result in enumerate(results, start=1):
        location_bits = [result.source_name]
        if result.page_range:
            location_bits.append(f"page {result.page_range}")
        if result.section_path:
            location_bits.append("section: " + " > ".join(result.section_path))
        header = " | ".join(location_bits)
        blocks.append(f"[{i}] {header}\n<document_data>\n{result.text}\n</document_data>")
    return "\n\n".join(blocks)


def parse_citation_indices(answer_text: str) -> List[int]:
    return sorted({int(m) for m in _CITATION_RE.findall(answer_text or "")})


def extract_cited_sources(answer_text: str, results: Sequence[RetrievalResult]) -> List[Dict[str, Any]]:
    """Return citation records for every [n] referenced in the answer."""
    citations = build_citations(results)
    by_index = {c["index"]: c for c in citations}
    return [by_index[i] for i in parse_citation_indices(answer_text) if i in by_index]
