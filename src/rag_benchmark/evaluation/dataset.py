"""Benchmark dataset model and relevance grading.

Ground truth is authored by hand in ``data/benchmark/questions.json`` and is never
generated from the system's own retrieval output. Relevance is defined by
provenance (expected source) plus the presence of hand-written essential facts,
so the labels are independent of any particular chunking strategy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..parsers.models import ContentType


@dataclass
class EvidenceLocation:
    source_name: str
    section: Optional[str] = None
    page: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkQuestion:
    question_id: str
    question: str
    category: str
    expected_sources: List[str]
    essential_facts: List[str]
    primary_fact: Optional[str] = None
    evidence: List[EvidenceLocation] = field(default_factory=list)
    reference_answer: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkQuestion":
        evidence = [EvidenceLocation(**e) for e in data.get("evidence", [])]
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            category=data.get("category", "factual_lookup"),
            expected_sources=list(data.get("expected_sources", [])),
            essential_facts=list(data.get("essential_facts", [])),
            primary_fact=data.get("primary_fact"),
            evidence=evidence,
            reference_answer=data.get("reference_answer", ""),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "expected_sources": list(self.expected_sources),
            "essential_facts": list(self.essential_facts),
            "primary_fact": self.primary_fact,
            "evidence": [e.to_dict() for e in self.evidence],
            "reference_answer": self.reference_answer,
            "notes": self.notes,
        }


@dataclass
class BenchmarkDataset:
    corpus_id: str
    questions: List[BenchmarkQuestion]
    documents: List[str] = field(default_factory=list)
    description: str = ""
    relevance_scale: Dict[str, str] = field(default_factory=dict)
    relevance_rule: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "description": self.description,
            "relevance_scale": self.relevance_scale,
            "relevance_rule": self.relevance_rule,
            "documents": list(self.documents),
            "questions": [q.to_dict() for q in self.questions],
        }

    def document_paths(self, docs_dir: str | Path) -> List[Path]:
        docs_dir = Path(docs_dir)
        return [docs_dir / name for name in self.documents]


def load_dataset(path: str | Path) -> BenchmarkDataset:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = [BenchmarkQuestion.from_dict(q) for q in data.get("questions", [])]
    return BenchmarkDataset(
        corpus_id=data.get("corpus_id", "corpus"),
        questions=questions,
        documents=list(data.get("documents", [])),
        description=data.get("description", ""),
        relevance_scale=dict(data.get("relevance_scale", {})),
        relevance_rule=data.get("relevance_rule", ""),
    )


def relevance_grade(chunk, question: BenchmarkQuestion) -> int:
    """Grade a chunk for a question: 2 (primary), 1 (relevant), 0 (not relevant)."""
    if chunk.source_name not in question.expected_sources:
        return 0
    text = (chunk.text or "").lower()
    if question.primary_fact and question.primary_fact.lower() in text:
        return 2
    for fact in question.essential_facts:
        if fact.lower() in text:
            return 1
    return 0


def relevant_chunk_ids(chunks, question: BenchmarkQuestion) -> Dict[str, int]:
    """Map chunk_id -> graded relevance for every relevant chunk."""
    graded: Dict[str, int] = {}
    for chunk in chunks:
        grade = relevance_grade(chunk, question)
        if grade > 0:
            graded[chunk.chunk_id] = grade
    return graded


def expected_source_ids(chunks, question: BenchmarkQuestion) -> set:
    """Set of chunk ids belonging to expected sources (for source-level recall)."""
    return {c.chunk_id for c in chunks if c.source_name in question.expected_sources}


def content_type_weight(content_type: str) -> float:
    """Relative importance of a content type when computing chunk statistics."""
    if content_type in (ContentType.TABLE, ContentType.CODE):
        return 1.0
    return 0.5
