"""Mode router for the Research Paper Companion.

Three modes layered over `paper_companion.PaperIndex`:
- `ask`: Q&A with cited answers (optional paper/section filter).
- `summarize_section`: section-wise summary with citations.
- `compare_papers`: structured method/results/limitations diff between two papers.

LLM provider: standalone OpenAI-compatible client in `llm_provider`.
Default model: qwen2.5:3b-instruct (override via MODES_LLM_MODEL env or arg).
See `.env` for provider config blocks (Gemini / Ollama / FPT).

"I don't know" gate: if top-1 retrieval score is below `min_score` or no chunks
are returned, the router refuses instead of letting the LLM hallucinate.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .paper_companion import PaperIndex, extractive_answer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("MODES_LLM_MODEL", "qwen2.5:3b-instruct")
DEFAULT_MIN_SCORE = 0.30  # below this top-1 score, refuse instead of generate
DEFAULT_TOP_K = 6


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class AnswerResult:
    mode: str
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved: List[Dict[str, Any]]
    grounded: bool
    refused: bool
    refusal_reason: Optional[str]
    retrieval_latency_ms: float
    generation_latency_ms: float
    token_usage_est: int  # rough word-count estimate; precise counting needs tiktoken
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


# ---------------------------------------------------------------------------
# Provider loader (lazy)
# ---------------------------------------------------------------------------

def load_provider(model: Optional[str] = None):
    """Build the OpenAI-compatible chat provider. Returns None if no API key.

    Resolves the model name AFTER loading .env so a model set only in the .env
    file (not exported as a shell var) is honored. The module-level DEFAULT_MODEL
    is read at import time — before .env is loaded — so we must re-read here.
    """
    from .llm_provider import build_provider
    if model is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        model = os.getenv("MODES_LLM_MODEL", "qwen2.5:3b-instruct")
    return build_provider(model=model)


# ---------------------------------------------------------------------------
# Prompt builders (kept inline; small and tightly coupled to mode logic)
# ---------------------------------------------------------------------------

_CITATION_RULES = (
    "RULES:\n"
    "1. Answer ONLY using the provided context. If insufficient, say so.\n"
    "2. Every sentence MUST end with at least one inline citation tag of the form\n"
    "   [paper_id §section p.N #chunk_id]. No exceptions, including in summaries.\n"
    "3. Quote extractively when possible; do not invent numbers, names, or claims.\n"
    "4. If the question is vague, broad, or has no specific anchor in the context\n"
    "   (e.g. 'is bigger better?', 'how do I tune this?'), do NOT answer — instead\n"
    "   ask exactly one clarifying question that would let you give a grounded answer.\n"
)


def _format_context(records: Sequence[Dict[str, Any]]) -> str:
    blocks = []
    for r in records:
        c = r["chunk"]
        tag = f"[{c['paper_id']} §{c['section']} p.{c['page_start']} #{c['chunk_id']}]"
        blocks.append(f"{tag}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def _word_count(*texts: str) -> int:
    return sum(len(t.split()) for t in texts)


# ---------------------------------------------------------------------------
# Mode: ask
# ---------------------------------------------------------------------------

def ask(
    index: PaperIndex,
    query: str,
    *,
    paper_ids: Optional[List[str]] = None,
    sections: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    provider: Any = None,
) -> AnswerResult:
    t0 = time.perf_counter()
    records = index.retrieve(query, paper_ids=paper_ids, sections=sections, top_k=top_k)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)

    if not records or records[0]["score"] < min_score:
        return AnswerResult(
            mode="ask", query=query,
            answer="I could not find sufficient evidence in the provided papers to answer this question.",
            citations=[], retrieved=records, grounded=False, refused=True,
            refusal_reason="weak_retrieval" if records else "no_results",
            retrieval_latency_ms=retrieval_ms, generation_latency_ms=0.0,
            token_usage_est=_word_count(query),
        )

    context = _format_context(records)
    system = "You are a research-paper assistant. " + _CITATION_RULES
    user = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer with inline citations."

    t1 = time.perf_counter()
    answer = _generate_or_fallback(provider, system, user, query, records)
    gen_ms = round((time.perf_counter() - t1) * 1000, 2)

    citations = [r["citation"] for r in records if r["citation"]["chunk_id"] in answer]
    return AnswerResult(
        mode="ask", query=query, answer=answer, citations=citations, retrieved=records,
        grounded=bool(citations), refused=False, refusal_reason=None,
        retrieval_latency_ms=retrieval_ms, generation_latency_ms=gen_ms,
        token_usage_est=_word_count(system, user, answer),
    )


# ---------------------------------------------------------------------------
# Mode: summarize_section
# ---------------------------------------------------------------------------

def summarize_section(
    index: PaperIndex,
    paper_id: str,
    section: str,
    *,
    max_chunks: int = 12,
    provider: Any = None,
) -> AnswerResult:
    pseudo_query = f"Summarize the {section} section"
    t0 = time.perf_counter()
    records = index.retrieve(pseudo_query, paper_ids=[paper_id], sections=[section], top_k=max_chunks)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)

    if not records:
        return AnswerResult(
            mode="summarize_section", query=pseudo_query,
            answer=f"No content found for paper {paper_id!r} section {section!r}.",
            citations=[], retrieved=[], grounded=False, refused=True, refusal_reason="empty_section",
            retrieval_latency_ms=retrieval_ms, generation_latency_ms=0.0, token_usage_est=0,
            metadata={"paper_id": paper_id, "section": section},
        )

    context = _format_context(records)
    system = "You are a research-paper assistant producing a concise section summary. " + _CITATION_RULES
    user = (
        f"CONTEXT (all chunks come from paper {paper_id!r}, section {section!r}):\n{context}\n\n"
        "Write a 4-6 sentence summary of this section. Cite at least three distinct chunks."
    )

    t1 = time.perf_counter()
    answer = _generate_or_fallback(provider, system, user, pseudo_query, records)
    gen_ms = round((time.perf_counter() - t1) * 1000, 2)
    citations = [r["citation"] for r in records if r["citation"]["chunk_id"] in answer]

    return AnswerResult(
        mode="summarize_section", query=pseudo_query, answer=answer, citations=citations,
        retrieved=records, grounded=bool(citations), refused=False, refusal_reason=None,
        retrieval_latency_ms=retrieval_ms, generation_latency_ms=gen_ms,
        token_usage_est=_word_count(system, user, answer),
        metadata={"paper_id": paper_id, "section": section},
    )


# ---------------------------------------------------------------------------
# Mode: compare_papers
# ---------------------------------------------------------------------------

def compare_papers(
    index: PaperIndex,
    paper_a: str,
    paper_b: str,
    query: str = "Compare these two papers on method, results, and limitations.",
    *,
    top_k_per_side: int = 5,
    provider: Any = None,
) -> AnswerResult:
    if paper_a == paper_b:
        return AnswerResult(
            mode="compare_papers", query=query, answer="Cannot compare a paper to itself.",
            citations=[], retrieved=[], grounded=False, refused=True, refusal_reason="same_paper",
            retrieval_latency_ms=0.0, generation_latency_ms=0.0, token_usage_est=0,
        )

    t0 = time.perf_counter()
    rec_a = index.retrieve(query, paper_ids=[paper_a], top_k=top_k_per_side)
    rec_b = index.retrieve(query, paper_ids=[paper_b], top_k=top_k_per_side)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)

    if not rec_a or not rec_b:
        missing = paper_a if not rec_a else paper_b
        return AnswerResult(
            mode="compare_papers", query=query,
            answer=f"Insufficient evidence for paper {missing!r} to perform comparison.",
            citations=[], retrieved=rec_a + rec_b, grounded=False, refused=True,
            refusal_reason="missing_evidence",
            retrieval_latency_ms=retrieval_ms, generation_latency_ms=0.0,
            token_usage_est=_word_count(query),
        )

    ctx_a, ctx_b = _format_context(rec_a), _format_context(rec_b)
    system = "You are a research-paper assistant producing a structured paper comparison. " + _CITATION_RULES
    user = (
        f"PAPER A ({paper_a}) CONTEXT:\n{ctx_a}\n\n"
        f"PAPER B ({paper_b}) CONTEXT:\n{ctx_b}\n\n"
        f"QUERY: {query}\n\n"
        "Produce a markdown comparison with three sections: ## Method, ## Results, ## Limitations.\n"
        "Each section must have one paragraph per paper with inline citations."
    )

    t1 = time.perf_counter()
    answer = _generate_or_fallback(provider, system, user, query, rec_a + rec_b)
    gen_ms = round((time.perf_counter() - t1) * 1000, 2)
    all_recs = rec_a + rec_b
    citations = [r["citation"] for r in all_recs if r["citation"]["chunk_id"] in answer]

    return AnswerResult(
        mode="compare_papers", query=query, answer=answer, citations=citations, retrieved=all_recs,
        grounded=bool(citations), refused=False, refusal_reason=None,
        retrieval_latency_ms=retrieval_ms, generation_latency_ms=gen_ms,
        token_usage_est=_word_count(system, user, answer),
        metadata={"paper_a": paper_a, "paper_b": paper_b},
    )


# ---------------------------------------------------------------------------
# Shared LLM dispatch with extractive fallback
# ---------------------------------------------------------------------------

def _generate_or_fallback(provider: Any, system: str, user: str, query: str, records: List[Dict[str, Any]]) -> str:
    if provider is None:
        return extractive_answer(query, records)
    try:
        return provider.generate([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
    except Exception as exc:
        # Loud fallback: a swallowed API error here (e.g. a 404 on a misconfigured
        # model name) used to masquerade as a low-quality "extractive" answer.
        # Log it with full traceback so the real cause is visible in the app log.
        logger.error(
            "LLM generation failed (model=%s); falling back to extractive answer: %s",
            getattr(provider, "_model", "?"), exc, exc_info=True,
        )
        return extractive_answer(query, records)
