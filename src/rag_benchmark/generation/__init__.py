"""Optional retrieval-grounded answer generation."""

from .answer_generator import (
    AnswerGenerator,
    GeneratedAnswer,
    LLMProvider,
    NoOpProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    provider_from_settings,
)
from .citations import (
    build_citations,
    build_context_block,
    extract_cited_sources,
    parse_citation_indices,
)

__all__ = [
    "AnswerGenerator",
    "GeneratedAnswer",
    "LLMProvider",
    "NoOpProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "provider_from_settings",
    "build_citations",
    "build_context_block",
    "extract_cited_sources",
    "parse_citation_indices",
]
