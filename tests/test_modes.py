"""Tests for the mode router (ask / summarize_section / compare_papers).

These tests exercise the extractive-fallback path (provider=None) so they
don't require network or LLM credentials. The LLM path is covered separately
by the eval harness when a provider is configured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_companion import (
    ask,
    build_index_from_corpus,
    compare_papers,
    summarize_section,
)
from research_companion.paper_companion import RetrievalConfig

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "research_companion" / "corpus"


@pytest.fixture(scope="module")
def index():
    return build_index_from_corpus(CORPUS, RetrievalConfig())


@pytest.fixture(scope="module")
def math_paper_id(index):
    """The 'Mathematics of Neural Networks' paper, used by content-specific tests."""
    for p in index.papers:
        if "math-neural-networks" in p.source_path or "2403.04807" in p.source_path:
            return p.paper_id
    return index.papers[0].paper_id


def test_ask_returns_grounded_answer_with_citations(index, math_paper_id):
    result = ask(index, "What is Xavier initialization?", paper_ids=[math_paper_id])
    assert not result.refused
    assert result.grounded
    assert result.retrieved
    assert result.answer
    # extractive fallback embeds citation tags via paper_companion.extractive_answer
    assert "[" in result.answer and "]" in result.answer


def test_ask_refuses_on_weak_retrieval(index):
    paper_id = index.papers[0].paper_id
    result = ask(
        index,
        "zxcvbnm qwerty asdfgh nonsense placeholder",
        paper_ids=[paper_id],
        min_score=0.9,  # forced high threshold
    )
    assert result.refused
    assert result.refusal_reason in {"weak_retrieval", "no_results"}


def test_summarize_section_returns_section_content(index, math_paper_id):
    result = summarize_section(index, math_paper_id, "initialization")
    assert not result.refused
    assert result.retrieved
    # all retrieved chunks must be from the requested section
    assert all(r["chunk"]["section"] == "initialization" for r in result.retrieved)
    assert result.metadata["section"] == "initialization"


def test_summarize_section_refuses_on_empty_section(index):
    paper_id = index.papers[0].paper_id
    result = summarize_section(index, paper_id, "nonexistent_section_xyz")
    assert result.refused
    assert result.refusal_reason == "empty_section"


def test_compare_papers_refuses_same_paper(index):
    paper_id = index.papers[0].paper_id
    result = compare_papers(index, paper_id, paper_id)
    assert result.refused
    assert result.refusal_reason == "same_paper"


def test_compare_papers_refuses_missing_second_paper(index):
    paper_id = index.papers[0].paper_id
    result = compare_papers(index, paper_id, "definitely-not-a-paper-id")
    assert result.refused
    assert result.refusal_reason == "missing_evidence"
