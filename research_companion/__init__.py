"""Research Paper Companion project package."""

from .modes import (
    AnswerResult,
    ask,
    compare_papers,
    load_provider,
    summarize_section,
)
from .paper_companion import (
    PaperIndex,
    RetrievalConfig,
    build_index_from_corpus,
    ingest_corpus,
)

__all__ = [
    "AnswerResult",
    "PaperIndex",
    "RetrievalConfig",
    "ask",
    "build_index_from_corpus",
    "compare_papers",
    "ingest_corpus",
    "load_provider",
    "summarize_section",
]
