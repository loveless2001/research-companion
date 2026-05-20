"""Gradio app for the Research Paper Companion.

Four tabs:
    1. Corpus    : list ingested papers (read-only — drop new PDFs into corpus/)
    2. Chat      : Q&A with cited answers; optional paper filter
    3. Summary   : section-wise summary for a chosen paper + section
    4. Compare   : structured method/results/limitations comparison of two papers

Launch:
    ./.venv/bin/python app.py
    ./.venv/bin/python app.py --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_companion import (  # noqa: E402
    AnswerResult,
    ask,
    build_index_from_corpus,
    compare_papers,
    load_provider,
    summarize_section,
)
from research_companion.paper_companion import RetrievalConfig  # noqa: E402

CORPUS_DIR = PROJECT_ROOT / "corpus"


# ---------------------------------------------------------------------------
# Shared state (loaded once at startup)
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        print("Building index...", file=sys.stderr)
        self.index = build_index_from_corpus(CORPUS_DIR, RetrievalConfig())
        self.provider = load_provider()
        self.papers = self.index.papers
        self.paper_choices = [(f"{p.title} ({p.paper_id})", p.paper_id) for p in self.papers]
        self.section_choices_by_paper: Dict[str, List[str]] = {
            p.paper_id: [s.normalized_heading for s in p.sections] for p in self.papers
        }
        print(f"  papers={len(self.papers)}, chunks={len(self.index.chunks)}, "
              f"provider={'LLM' if self.provider else 'extractive-fallback'}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Citation rendering (shared across tabs)
# ---------------------------------------------------------------------------

def _render_citations(result: AnswerResult) -> str:
    if result.refused:
        return f"_Refused: {result.refusal_reason or 'no reason'}_"
    if not result.citations:
        return "_No citations._"
    lines = ["| # | Paper | Section | Page | Snippet |", "|---|---|---|---|---|"]
    for i, c in enumerate(result.citations[:8], start=1):
        snippet = (c.get("quote_snippet") or "").replace("|", "\\|")
        page = c.get("page_start", c.get("page", ""))
        lines.append(f"| {i} | `{c['paper_id']}` | {c['section']} | {page} | {snippet} |")
    return "\n".join(lines)


def _render_metrics(result: AnswerResult) -> str:
    return (
        f"**Retrieval:** {result.retrieval_latency_ms} ms  |  "
        f"**Generation:** {result.generation_latency_ms} ms  |  "
        f"**Tokens (est):** {result.token_usage_est}  |  "
        f"**Grounded:** {result.grounded}"
    )


# ---------------------------------------------------------------------------
# Tab handlers
# ---------------------------------------------------------------------------

def handle_chat(state: AppState, query: str, paper_id: Optional[str]) -> tuple[str, str, str]:
    if not query.strip():
        return "Please enter a question.", "", ""
    paper_ids = [paper_id] if paper_id else None
    result = ask(state.index, query, paper_ids=paper_ids, provider=state.provider)
    return result.answer, _render_citations(result), _render_metrics(result)


def handle_summary(state: AppState, paper_id: str, section: str) -> tuple[str, str, str]:
    if not paper_id or not section:
        return "Please pick a paper and a section.", "", ""
    result = summarize_section(state.index, paper_id, section, provider=state.provider)
    return result.answer, _render_citations(result), _render_metrics(result)


def handle_compare(state: AppState, paper_a: str, paper_b: str, query: str) -> tuple[str, str, str]:
    if not paper_a or not paper_b:
        return "Please pick two papers.", "", ""
    query = query.strip() or "Compare these two papers on method, results, and limitations."
    result = compare_papers(state.index, paper_a, paper_b, query, provider=state.provider)
    return result.answer, _render_citations(result), _render_metrics(result)


def update_section_dropdown(state: AppState, paper_id: str):
    return gr.update(choices=state.section_choices_by_paper.get(paper_id, []), value=None)


# ---------------------------------------------------------------------------
# Build Blocks UI
# ---------------------------------------------------------------------------

def build_ui(state: AppState) -> gr.Blocks:
    with gr.Blocks(title="Research Paper Companion") as demo:
        gr.Markdown("# Research Paper Companion\nAsk, summarize, and compare research PDFs with cited answers.")

        # --- Tab 1: Corpus ---
        with gr.Tab("Corpus"):
            corpus_rows = [
                [p.paper_id, p.title, p.page_count, len(p.sections), Path(p.source_path).name]
                for p in state.papers
            ]
            gr.Dataframe(
                value=corpus_rows,
                headers=["paper_id", "title", "pages", "sections", "filename"],
                interactive=False,
                wrap=True,
            )
            gr.Markdown(f"_Drop new PDFs into `{CORPUS_DIR.relative_to(PROJECT_ROOT)}/` and restart the app to ingest them._")

        # --- Tab 2: Chat ---
        with gr.Tab("Chat"):
            with gr.Row():
                chat_q = gr.Textbox(label="Question", placeholder="What is Xavier initialization?", lines=2, scale=4)
                chat_paper = gr.Dropdown(label="Paper filter (optional)", choices=[("All papers", None), *state.paper_choices], value=None, scale=2)
            chat_btn = gr.Button("Ask", variant="primary")
            chat_answer = gr.Markdown(label="Answer")
            chat_cites = gr.Markdown(label="Citations")
            chat_metrics = gr.Markdown(label="Metrics")
            chat_btn.click(lambda q, p: handle_chat(state, q, p), [chat_q, chat_paper], [chat_answer, chat_cites, chat_metrics])

        # --- Tab 3: Summary ---
        with gr.Tab("Section Summary"):
            with gr.Row():
                sum_paper = gr.Dropdown(label="Paper", choices=state.paper_choices)
                sum_section = gr.Dropdown(label="Section", choices=[])
            sum_btn = gr.Button("Summarize", variant="primary")
            sum_answer = gr.Markdown(label="Summary")
            sum_cites = gr.Markdown(label="Citations")
            sum_metrics = gr.Markdown(label="Metrics")
            sum_paper.change(lambda p: update_section_dropdown(state, p), sum_paper, sum_section)
            sum_btn.click(lambda p, s: handle_summary(state, p, s), [sum_paper, sum_section], [sum_answer, sum_cites, sum_metrics])

        # --- Tab 4: Compare ---
        with gr.Tab("Compare"):
            with gr.Row():
                cmp_a = gr.Dropdown(label="Paper A", choices=state.paper_choices)
                cmp_b = gr.Dropdown(label="Paper B", choices=state.paper_choices)
            cmp_q = gr.Textbox(label="Comparison query (optional)", placeholder="Compare method, results, limitations.", lines=2)
            cmp_btn = gr.Button("Compare", variant="primary")
            cmp_answer = gr.Markdown(label="Comparison")
            cmp_cites = gr.Markdown(label="Citations")
            cmp_metrics = gr.Markdown(label="Metrics")
            cmp_btn.click(lambda a, b, q: handle_compare(state, a, b, q), [cmp_a, cmp_b, cmp_q], [cmp_answer, cmp_cites, cmp_metrics])

    return demo


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the Research Paper Companion Gradio app.")
    parser.add_argument("--host", default=os.getenv("APP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("APP_PORT", "7860")))
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args(argv)

    state = AppState()
    demo = build_ui(state)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
