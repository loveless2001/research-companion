# Research Paper Companion

Ask, summarize, and compare research PDFs with cited answers. Built for QuanSkill K6 Final Projects — Project 2.

- **5 papers, 4,359 chunks** in the seed corpus (4 RAG-area arXiv + math-of-NN seed)
- **3 modes**: Q&A, section summary, two-paper compare
- **OpenAI-compat LLM** (Gemini / Ollama / FPT — switch via `.env`)
- **50-question eval set** with retrieval-validated `(section, page)` ground truth
- **Gradio 4-tab UI**

See `REPORT.md` for full metrics and `architecture.md` for the pipeline diagram.

---

## Install

```bash
cd research-companion
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

Configure the LLM provider in `.env` (one block active at a time):

```bash
# --- Google Gemini (free tier; current default) ---
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_API_KEY=<your gemini key>
MODES_LLM_MODEL=gemini-2.5-flash-lite

# --- Local Ollama ---
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_API_KEY=ollama
# MODES_LLM_MODEL=qwen2.5:3b-instruct

# --- FPT Cloud ---
# OPENAI_BASE_URL=https://mkp-api.fptcloud.com/v1
# OPENAI_API_KEY=<your fpt key>
# MODES_LLM_MODEL=Qwen3-32B
```

If `OPENAI_API_KEY` is unset, the modes fall back to an extractive answer (top-2 chunks formatted as citations). If an LLM call fails, the app logs the API error before falling back.

---

## Run

### Gradio app (4 tabs)
```bash
python app.py --port 7860
```
First-startup is ~30–80 s (model warmup + ingestion). Then open http://127.0.0.1:7860.

### Notebook walkthrough
```bash
jupyter notebook notebook.ipynb
```

### Data-layer CLI
```bash
# Inspect corpus
python research_companion/paper_companion.py inspect --corpus-dir corpus

# Retrieval smoke test
python research_companion/paper_companion.py smoke --corpus-dir corpus \
  --query "What is Xavier initialization?" --top-k 5
```

### Eval harness
```bash
# Full 50-question eval (~3 min on Gemini)
python scripts/run-eval-harness-and-compute-metrics.py

# Quick 5-question smoke (~30 s, writes *-limit5 files)
python scripts/run-eval-harness-and-compute-metrics.py --limit 5

# Extractive baseline (no LLM)
python scripts/run-eval-harness-and-compute-metrics.py --no-llm
```

Full eval outputs land in `results/`: `metrics.csv`, `metrics_by_bucket.csv`,
`eval_runs.jsonl`. Limited runs use tagged filenames such as
`metrics-limit5.csv` so canonical results are not overwritten.

### Tests
```bash
python -m pytest tests/ -q
# 11 passing (no network, no LLM creds needed — uses extractive fallback)
```

---

## Python API

```python
from research_companion import (
    build_index_from_corpus, RetrievalConfig,
    ask, summarize_section, compare_papers, load_provider,
)
from pathlib import Path

index = build_index_from_corpus(Path("corpus"), RetrievalConfig())
provider = load_provider()  # None if no API key

# Q&A
r = ask(index, "What is Xavier initialization?", provider=provider)
print(r.answer)
for c in r.citations:
    print(f"  [{c['paper_id']} §{c['section']} p.{c['page']}] {c['quote_snippet']}")

# Section summary
r = summarize_section(index, paper_id="<some-id>", section="initialization", provider=provider)

# Compare two papers
r = compare_papers(index, paper_a="<id-a>", paper_b="<id-b>", provider=provider)
```

### Retrieval record shape
Each `index.retrieve(...)` entry has:
- `rank`, `score`
- `chunk`: `chunk_id`, `paper_id`, `section`, `page_start`, `page_end`, `text`, `char_offset_start/end`
- `citation`: same metadata plus `quote_snippet` (extractive, ≤20 words)

Heading-only, TOC-like, and reference chunks are preserved during ingestion but filtered from retrieval.

---

## Corpus

```
corpus/
├── 2005.11401-rag.pdf
├── 2112.04426-retro.pdf
├── 2208.03299-atlas.pdf
├── 2310.11511-self-rag.pdf
└── 2403.04807-math-neural-networks.pdf
```

Drop new PDFs into `corpus/` and restart the app — they're auto-discovered by `ingest_corpus()`.

---

## Final eval (Gemini 2.5 Flash Lite, 50 questions)

```
citation_coverage_pct: 97.5    retrieval_topk_hit_pct: 100.0
retrieval_top1_hit_pct: 88.0   correctly_refused_pct: 60.0
mean_latency_ms: 2127          p95_latency_ms: 5137
```

Three model runs are snapshotted in `results/` for comparison: `metrics-v1-llm.csv` (Qwen loose), `metrics-v2-qwen.csv` (Qwen tight), `metrics-v3-gemini.csv` (current). See `REPORT.md` §2.

---

## Project layout

```
research-companion/
├── research_companion/           importable Python package
│   ├── paper_companion.py        data layer (PDF → chunks → FAISS)
│   ├── retrieval_primitives.py   Embedder + VectorIndex + chunk splitter
│   ├── modes.py                  ask / summarize_section / compare_papers
│   └── llm_provider.py           OpenAI-compat client
├── app.py                        Gradio 4-tab UI
├── corpus/                       5 PDFs
├── results/                      eval_questions.json + metrics + run snapshots
├── scripts/                      eval-set author + harness CLIs
├── tests/                        pytest suite (11 tests)
├── pyproject.toml                editable install metadata
├── requirements.txt
├── REPORT.md                     full results + honest gaps
├── architecture.md               pipeline diagram + module map
└── notebook.ipynb                demo walkthrough
```

No dependency on any parent repo — the GitHub checkout is the project root.
