"""Retrieval and answer evaluation metrics.

Every function handles empty inputs and zero denominators safely by returning
``None`` (meaning "not defined for this query") rather than a misleading 0.0.
Aggregation ignores undefined per-query values and reports how many queries
contributed.

Formulas
--------
Recall@k
    ``|relevant ∩ retrieved[:k]| / |relevant|`` over the set of relevant chunks.
    Undefined when a query has no relevant chunks.

MRR
    Reciprocal rank of the first relevant result, averaged over queries:
    ``1 / rank_first_relevant`` (0.0 when none retrieved but relevant exists).

nDCG@k
    Graded relevance with gain ``2^rel - 1`` and discount ``log2(rank + 1)``:
    ``DCG@k / IDCG@k`` where IDCG is the DCG of the ideal ranking.

Context completeness
    Fraction of hand-written essential facts present in the concatenated
    retrieved context (1.0 if there are no essential facts).

Citation accuracy
    Fraction of predicted citations whose source (and section/page when known)
    matches the expected evidence. Reported alongside coverage and the count of
    expected evidence items that received no correct citation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


# --------------------------------------------------------------------- recall
def recall_at_k(
    relevant_ids: Set[str], retrieved_ids: Sequence[str], k: int
) -> Optional[float]:
    if not relevant_ids:
        return None
    if k <= 0:
        return 0.0
    retrieved = list(retrieved_ids)[:k]
    hits = len(relevant_ids.intersection(retrieved))
    return hits / len(relevant_ids)


# ------------------------------------------------------------------------ mrr
def reciprocal_rank(relevant_ids: Set[str], retrieved_ids: Sequence[str]) -> Optional[float]:
    if not relevant_ids:
        return None
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


# -------------------------------------------------------------------- nDCG@k
def dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(i + 1) for i, gain in enumerate(gains, start=1))


def ndcg_at_k(
    graded_relevance: Dict[str, int], retrieved_ids: Sequence[str], k: int
) -> Optional[float]:
    if not graded_relevance:
        return None
    if k <= 0:
        return 0.0
    retrieved = list(retrieved_ids)[:k]
    gains = [2 ** graded_relevance.get(cid, 0) - 1 for cid in retrieved]
    ideal_grades = sorted((2 ** g - 1 for g in graded_relevance.values()), reverse=True)[:k]
    idcg = dcg(ideal_grades)
    if idcg == 0:
        return 0.0
    return dcg(gains) / idcg


# ------------------------------------------------------- context completeness
def context_completeness(
    essential_facts: Sequence[str], retrieved_texts: Iterable[str]
) -> float:
    facts = [f for f in essential_facts if f]
    if not facts:
        return 1.0
    haystack = "\n".join(retrieved_texts).lower()
    present = sum(1 for fact in facts if fact.lower() in haystack)
    return present / len(facts)


def missing_facts(
    essential_facts: Sequence[str], retrieved_texts: Iterable[str]
) -> List[str]:
    haystack = "\n".join(retrieved_texts).lower()
    return [f for f in essential_facts if f and f.lower() not in haystack]


# --------------------------------------------------------- citation accuracy
def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def citation_accuracy(
    predicted_citations: Sequence[Dict[str, Any]],
    expected_evidence: Sequence[Any],
) -> Dict[str, Any]:
    """Compare predicted citations against expected evidence locations.

    ``predicted_citations`` items are dicts with ``source_name`` and optional
    ``section`` / ``page``. ``expected_evidence`` items are dicts or objects with
    the same attributes. Returns a dict including ``precision`` (correct /
    number of citations), ``correct``, ``total_citations``,
    ``uncovered_evidence`` and ``coverage``.
    """
    def field(item, name):
        return item.get(name) if isinstance(item, dict) else getattr(item, name, None)

    expected = list(expected_evidence)
    total_citations = len(predicted_citations)
    correct = 0
    covered: Set[int] = set()

    for citation in predicted_citations:
        c_source = _norm(field(citation, "source_name"))
        c_section = _norm(field(citation, "section"))
        c_page = field(citation, "page")
        for idx, exp in enumerate(expected):
            e_source = _norm(field(exp, "source_name"))
            e_section = _norm(field(exp, "section"))
            e_page = field(exp, "page")
            if c_source != e_source:
                continue
            if e_section and c_section != e_section:
                continue
            if e_page is not None and c_page is not None and c_page != e_page:
                continue
            correct += 1
            covered.add(idx)
            break

    precision = (correct / total_citations) if total_citations else None
    uncovered = len(expected) - len(covered)
    coverage = (len(covered) / len(expected)) if expected else None
    return {
        "precision": precision,
        "correct": correct,
        "total_citations": total_citations,
        "uncovered_evidence": uncovered,
        "coverage": coverage,
        "missing_citations": max(0, total_citations - correct),
    }


# ------------------------------------------------------------- chunk statistics
def duplicate_chunk_rate(texts: Sequence[str]) -> float:
    """Fraction of chunks whose normalized text duplicates an earlier chunk."""
    if not texts:
        return 0.0
    seen: Set[str] = set()
    duplicates = 0
    for text in texts:
        key = " ".join((text or "").lower().split())
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates / len(texts)


def chunk_size_stats(token_counts: Sequence[int]) -> Dict[str, Optional[float]]:
    values = [t for t in token_counts if t is not None]
    if not values:
        return {"count": 0, "avg": None, "min": None, "max": None, "median": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2 == 1
        else (ordered[mid - 1] + ordered[mid]) / 2
    )
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "median": median,
    }


# ----------------------------------------------------------------- aggregation
def mean_ignoring_none(values: Sequence[Optional[float]]) -> Optional[float]:
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def count_defined(values: Sequence[Optional[float]]) -> int:
    return sum(1 for v in values if v is not None)
