# Research Paper Companion — Final Report

**Project**: QuanSkill K6 Final Projects — Project 2 (Research Paper Companion)
**Stack**: PyMuPDF + sentence-transformers/all-MiniLM-L6-v2 + FAISS (cosine) + OpenAI-compat LLM
**LLM (final eval)**: `gemini-2.5-flash-lite` via Gemini OpenAI-compat endpoint
**Corpus**: 5 papers (4 RAG-area arXiv + 1 math-of-NN seed), 4,359 chunks

---

## 1. What it does (rubric-mapped)

| Feature | Rubric pts | Status |
|---|---|---|
| PDF ingestion + section-aware retrieval | 30 | ✅ PyMuPDF font-size heading detection + regex fallback + page metadata + `paper_id`/`section`/`page_start`/`char_offset_start` on every chunk |
| Q&A with citations consistently | 20 | ✅ `ask()` returns answer + ≤20-word extractive citation snippets per cited chunk |
| Section summarization mode | 15 | ✅ `summarize_section(paper_id, section)` retrieves all chunks for the section and produces a 4–6-sentence cited summary |
| Compare-two-papers feature | 15 | ✅ `compare_papers(a, b, query)` retrieves per-side, produces structured Method/Results/Limitations markdown with per-side citations |
| Evaluation + metrics saved | 10 | ✅ 50-question hand-authored eval set with retrieval-validated `(section, page)` GT; 6 buckets; full per-question JSONL + summary CSV + per-bucket CSV |
| UX + clarity + reproducibility | 10 | ✅ Gradio 4-tab app (Corpus / Chat / Summary / Compare); HTTP 200 verified; tests 11/11 pass; standalone install via `requirements.txt` |

---

## 2. Headline metrics (final eval — Gemini 2.5 Flash Lite)

```
total_questions: 50            skipped: 0
citation_coverage_pct: 97.5    retrieval_top1_hit_pct: 88.0
retrieval_topk_hit_pct: 100.0  correctly_refused_pct: 60.0
mean_latency_ms: 2126.98       p95_latency_ms: 5137.37
mean_token_usage_est: 494.8
```

### Per-bucket
| bucket | n | cite% | refused% | mean ms |
|---|---|---|---|---|
| factual | 15 | 93.3 | 0 | 2406 |
| where_stated | 10 | 100.0 | 0 | 1589 |
| section_summary | 8 | 100.0 | 0 | 2922 |
| compare | 7 | 100.0 | 0 | 3245 |
| ambiguous | 5 | 100.0 | 0 | 1637 |
| out_of_scope | 5 | n/a | 100 (gate) | 18 |

### Model comparison (50 questions, identical prompts & eval set)
| metric | Qwen2.5:3b loose | Qwen2.5:3b tight | **Gemini 2.5 Flash Lite** |
|---|---|---|---|
| citation_coverage_pct | 95.0 | 90.0 | **97.5** |
| compare-bucket cite% | 100.0 | 57.1 | **100.0** |
| correctly_refused_pct | 50.0 | **80.0** | 60.0 |
| mean_latency_ms | 3680 | 3678 | **2127** |

Qwen2.5:3b under tightened prompts gained refusal compliance (+30pt) but lost compare-mode formatting (-43pt) — token-attention trade-off the smaller model couldn't avoid. Gemini 2.5 Flash Lite handles both rules together at lower latency.

---

## 3. Eval set design (50 prompts, 6 buckets)

- **15 factual** — single-fact questions grounded in one section of the seed paper, paired with `(expected_section, expected_page)` GT validated by running retrieval.
- **10 where_stated** — citation-location questions ("which section covers X?"), same GT structure.
- **8 section_summary** — one summary prompt per major section.
- **7 compare** — paper-pair prompts cycling through the 5 papers' distinct pairs.
- **5 ambiguous** — vague queries that should trigger a clarifying question.
- **5 out_of_scope** — unrelated questions (stock prices, sports, recipes) that should be refused.

GT honesty: 22/25 grounded prompts had top-1 retrieval matching expected section. The 3 mismatches (`f02`, `w20`, `w22`) are kept in the eval set with `match: false` rather than masked — they are real retriever signal, not broken questions.

---

## 4. How the citation contract works

Every chunk carries `(paper_id, section, page_start, chunk_id, char_offset_start/end)`. Citation tags use the form:

```
[paper_id §section p.N #chunk_id]
```

The `_CITATION_RULES` block in `modes.py` instructs the LLM to:
1. answer only from supplied context,
2. end every sentence with at least one inline tag,
3. quote extractively (no invented numbers),
4. ask one clarifying question (not answer) for vague queries.

The eval harness counts a citation as "matched" when the chunk_id substring appears in the answer text. Extractive snippets are capped at 20 words per `paper_companion.MAX_QUOTE_WORDS`.

---

## 5. "I don't know" gate

Two layers:
- **Retrieval gate** (`min_score = 0.30` cosine): if top-1 score is below threshold, the router refuses *before* calling the LLM. Confirmed by the 100% OOS refusal at ~18 ms — the LLM was never invoked for stock-price/sports/recipe questions.
- **LLM gate** (system prompt rule 4): for vague-but-on-topic queries, the LLM is instructed to ask a clarifying question. Gemini's stronger "be helpful" prior makes it fall back to grounded answers on ambiguous queries about 4 out of 5 times (refusal 60% headline vs Qwen2.5:3b's 80% under the same rule).

---

## 6. Standalone footprint

```
research-companion/                   ~1860 LOC total
├── research_companion/               importable package
│   ├── paper_companion.py  ~640      data layer: ingest, chunk, FAISS, snippets
│   ├── retrieval_primitives.py ~140  vendored Embedder/VectorIndex/splitter
│   ├── modes.py            ~265      ask / summarize_section / compare_papers
│   └── llm_provider.py     ~80       OpenAI-compat client (any provider)
├── app.py                  ~190      Gradio 4-tab UI
├── corpus/                          5 PDFs (committed)
├── results/                         eval_questions.json + v1/v2/v3 snapshots
├── scripts/                         eval author + harness CLIs
├── tests/                           pytest suite (11 tests)
├── pyproject.toml                    editable install metadata
├── requirements.txt
├── README.md
├── REPORT.md  (this file)
├── architecture.md
└── notebook.ipynb
```

No dependency on the parent repository. A dependency grep for the old assignment
modules and app provider returns zero matches in the Python source.

---

## 7. Reproducing the eval

```bash
cd research-companion
pip install -e ".[test]"

# Set provider (Gemini, Ollama, or FPT — see .env for blocks)
export OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export OPENAI_API_KEY=<your gemini key>
export MODES_LLM_MODEL=gemini-2.5-flash-lite

# Full eval (~3 min on Gemini)
python scripts/run-eval-harness-and-compute-metrics.py

# Quick smoke (5 questions, writes results/*-limit5.* so canonical metrics stay intact)
python scripts/run-eval-harness-and-compute-metrics.py --limit 5

# No-LLM baseline (extractive fallback only, <1 min)
python scripts/run-eval-harness-and-compute-metrics.py --no-llm
```

Full-run outputs: `results/metrics.csv`, `results/metrics_by_bucket.csv`,
`results/eval_runs.jsonl`. Limited runs write tagged files such as
`results/metrics-limit5.csv`.

---

## 8. Honest gaps / known limitations

1. **Ambiguous-refusal compliance** drops with stronger models — Gemini answers vague queries 4/5 times instead of asking for clarification. The rule is in the system prompt but a stronger model's "be helpful" prior overrides it. Fix would require stricter output format constraints or a separate classifier.
2. **Retrieval top-1 hit = 88%** — three eval prompts retrieve the "expected" topic but from a different section than the human-authored expectation (e.g., backprop mentioned in `introduction` before its dedicated section). Not broken retrieval, but worth flagging.
3. **Compare-mode under small LLMs** — Qwen2.5:3b dropped citation tags when forced to also produce structural headings. Anything <7B is unreliable here without further prompt engineering.
4. **Vendored provider wrapper** — implemented as `llm_provider.py`. If the
   original upstream helper changes, this copy will not auto-update. Intentional
   trade-off for true standalone execution.

---

## 9. Unresolved questions

- Should ambiguous-refusal be enforced via post-processing (regex check on answer text → re-prompt if not a question)? Adds 1 LLM call per ambiguous query.
- Page-anchor citations are tied to PyMuPDF's page-extraction quirks; arXiv preprints with reflowed two-column layouts occasionally have page boundaries that disagree with section boundaries. Detected in the corpus but not measured.
