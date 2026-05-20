# Project 2 Plan - Research Paper Companion

## Goal

Build a separate research-paper companion project: upload research PDFs, ask
questions against them, generate section-wise summaries, compare two papers, and
return citations for every substantive claim.

This is a new project, not Assignment 1/2/3 continuation work. It may reuse the
earlier document corpus and RAG utilities as seed material, but its deliverables
should live under `research_companion/`.

## Source Requirements

Chosen project: QuanSkill K6 Final Projects, Project 2, "Research Paper
Companion".

Required features:

- PDF ingestion with clean text extraction.
- Section-aware chunking with metadata such as `paper_id`, `section`, and page.
- Q&A mode with cited answers.
- Section-summary mode with cited chunks.
- Compare-two-papers mode focused on methods, results, and limitations.
- Evaluation prompts with factual and "where is this stated?" questions.
- Metrics for citation correctness proxy, latency, and token usage.
- Minimal quote snippets, capped at 20 words per citation.
- Gradio UI with upload, chat, and section-summary tabs.
- Final report and demo.

## Reuse Boundary

Reuse:

- `assignment/corpus/2403.04807.pdf` as the first seed paper.
- Existing PDF parsing patterns from `assignment/run_rag.py` and
  `hologram/parsers/pdf.py`.
- Existing FAISS + sentence-transformer retrieval pattern from the assignment
  runners.
- Existing provider wrapper behavior where practical, especially local Ollama
  compatibility.

Do not reuse as-is:

- A1/A2/A3 report framing, because this project has a different rubric.
- A3 multi-agent orchestration, unless it becomes a stretch goal.
- Assignment result tables, because Project 2 needs its own eval and metrics.

## Proposed Layout

```text
research_companion/
├── README.md
├── app.py
├── paper_companion.py
├── notebook.ipynb
├── REPORT.md
├── architecture.md
├── corpus/
│   └── README.md
├── results/
│   ├── eval_questions.json
│   ├── qa_outputs.json
│   ├── section_summaries.json
│   ├── compare_outputs.json
│   └── metrics.csv
└── tests/
    └── test_paper_companion.py
```

## System Design

Pipeline:

```text
PDF upload/import
  -> text extraction by page
  -> section detection
  -> section-aware chunks
  -> embeddings
  -> FAISS index
  -> Q&A, section summary, or paper comparison
  -> cited answer + bounded quote snippets + metrics log
```

Core data structures:

- `Paper`: `paper_id`, title, path, pages, sections.
- `Section`: `paper_id`, heading, normalized heading, page_start, page_end.
- `Chunk`: `chunk_id`, `paper_id`, section, page, text, embedding row.
- `Citation`: `paper_id`, section, page, chunk_id, quote_snippet`.

Citation contract:

- Every factual answer paragraph needs at least one citation.
- Citation snippets should be short and extractive, max 20 words.
- If retrieval evidence is weak, answer with uncertainty or ask for a narrower
  question instead of inventing a claim.

## Two-Week Implementation Plan

Day 1-2: Build ingestion and index.

- Create `paper_companion.py` with PDF loading, section detection, chunking,
  embeddings, FAISS indexing, and retrieval.
- Seed with `assignment/corpus/2403.04807.pdf`.
- Add room for multiple PDFs through `research_companion/corpus/`.

Day 3: Add section splitter.

- Detect common headings such as Abstract, Introduction, Methods, Results,
  Discussion, Limitations, Conclusion, References.
- Store section metadata on every chunk.
- Add a small section-map export for debugging.

Day 4: Add Q&A and section-summary modes.

- `ask(query, paper_filter=None, section_filter=None)` returns cited answers.
- `summarize_section(paper_id, section)` returns a short summary with citations.

Day 5: Add compare-two-papers mode.

- Compare method, results, and limitations for Paper A vs Paper B.
- If only one seed paper exists, create the mode with a clear "needs two papers"
  behavior and test it with a copied or second sample paper later.

Day 6: Build evaluation prompts.

- Create 50 prompts split across factual lookup, where-stated, section summary,
  comparison, ambiguous, and out-of-scope cases.
- Keep expected evidence references where easy to identify.

Day 7: Add metrics.

- Citation coverage.
- Citation correctness proxy: overlap between answer sentence and cited snippet.
- Retrieval hit counts by section.
- Latency and approximate token usage.

Day 8: Add minimal quote snippets.

- Enforce max 20 words per citation snippet.
- Store snippets in outputs for auditing.

Day 9: Build Gradio UI.

- Upload PDFs.
- Chat tab with answer and citations.
- Section summary tab.
- Compare papers tab.
- Results download area.

Day 10: Final report and demo package.

- Write `REPORT.md` from the actual metrics and outputs.
- Add `architecture.md` diagram and reproduction steps.
- Package only after running smoke tests and verifying the archive.

## Evaluation Set Shape

Target: 50 prompts.

Suggested split:

- 15 factual questions grounded in one section.
- 10 "where is this stated?" citation-location questions.
- 8 section summary prompts.
- 7 compare-two-papers prompts.
- 5 ambiguous questions requiring clarification.
- 5 out-of-scope questions requiring refusal.

The comparison bucket can start as placeholders until a second real paper is
added. Do not report comparison metrics as complete until two distinct papers
are present.

## Acceptance Criteria

- A fresh clone can run the notebook or CLI from documented setup steps.
- At least one real research PDF is ingested from disk, not hard-coded.
- Chunks include section and page metadata.
- Q&A returns citations with page, section, and bounded snippets.
- Section-summary mode works for at least Abstract, Introduction, Method, and
  Conclusion-like sections when present.
- Compare mode handles two papers, or clearly refuses when fewer than two papers
  are available.
- Evaluation artifacts are saved under `research_companion/results/`.
- Metrics and report claims match the saved artifacts.
- Gradio UI runs locally and exposes upload, chat, summary, and compare flows.

## First Build Step

Create the `research_companion/` scaffold and implement the ingestion/index path
first. The first smoke test should load `assignment/corpus/2403.04807.pdf`, build
section-aware chunks, and answer one cited factual question without requiring the
UI.

## Current Implementation Status

Implemented:

- `paper_companion.py` data layer with PDF/TXT/MD ingestion.
- Layered section detection using PyMuPDF font/layout signals plus regex
  fallback and an `unknown` bucket.
- Chunk metadata with `paper_id`, section, page range, chunk id, and
  `char_offset_start/end`.
- FAISS retrieval with optional paper and section filters.
- Deterministic extractive citation snippets capped at 20 words.
- `inspect` and `smoke` CLI commands.
- Focused tests for heading detection, unknown-section preservation, snippet
  bounds, and text-file offset generation.

Verified:

```bash
./.venv/bin/python -m pytest research_companion/tests/test_paper_companion.py -q
./.venv/bin/python research_companion/paper_companion.py inspect --corpus-dir assignment/corpus
./.venv/bin/python research_companion/paper_companion.py smoke --corpus-dir assignment/corpus --query "What is Xavier initialization?" --top-k 5
```

Seed-PDF smoke result:

- Loaded 1 paper: `assignment/corpus/2403.04807.pdf`.
- Detected 80 pages, 21 section spans, and 1,997 chunks.
- Retrieved cited initialization evidence on pages 28-29.
- Wrote the smoke output to `research_companion/results/smoke.json`.

## Corpus Expansion Status

After user approval, the Project 2 corpus was expanded to five local PDFs under
`research_companion/corpus/`:

- `2005.11401-rag.pdf`
- `2112.04426-retro.pdf`
- `2208.03299-atlas.pdf`
- `2310.11511-self-rag.pdf`
- `2403.04807-math-neural-networks.pdf`

`README.md` files are ignored by corpus discovery, so corpus notes do not become
spurious papers.

Verified:

```bash
./.venv/bin/python research_companion/paper_companion.py inspect --corpus-dir research_companion/corpus
```

Full-corpus inspect result:

- Loaded 5 papers.
- Built 4,359 chunks.
- Detected section spans for all five PDFs.

Multi-paper filter smoke:

- Built a full `PaperIndex` from `research_companion/corpus/`.
- Queried the RAG paper with `paper_ids=[2005.11401...]`.
- Queried the Self-RAG paper with `paper_ids=[2310.11511...]`.
- Both filtered searches returned only chunks from the requested paper.
- Top-1 chunks were distinct.
