#!/usr/bin/env python3
"""Eval harness for the Research Paper Companion.

Loads `results/eval_questions.json`, dispatches each prompt to the right mode
(ask / summarize_section / compare_papers), and computes:

- citation_coverage: % grounded answers with >= 1 citation
- retrieval_hit_rate: for factual+where_stated, did top-k include the GT section?
- retrieval_top1_match: for factual+where_stated, did top-1 hit the GT section?
- refusal_rate: % of (ambiguous + out_of_scope) that were correctly refused
- mean_latency_ms / p95_latency_ms (retrieval + generation)
- mean_token_usage_est

Outputs:
- results/eval_runs.jsonl (one line per question, full record)
- results/metrics.csv (one row, summary)
- results/metrics_by_bucket.csv (one row per bucket)

Usage:
    ./.venv/bin/python scripts/run-eval-harness-and-compute-metrics.py
    ./.venv/bin/python scripts/run-eval-harness-and-compute-metrics.py --no-llm
    ./.venv/bin/python scripts/run-eval-harness-and-compute-metrics.py --no-llm --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research_companion import (  # noqa: E402
    ask,
    build_index_from_corpus,
    compare_papers,
    load_provider,
    summarize_section,
)
from research_companion.paper_companion import RetrievalConfig  # noqa: E402

EVAL_PATH = PROJECT_ROOT / "results" / "eval_questions.json"
CORPUS_DIR = PROJECT_ROOT / "corpus"
RESULTS_DIR = PROJECT_ROOT / "results"


def _output_paths(tag: Optional[str]) -> tuple[Path, Path, Path]:
    suffix = f"-{tag}" if tag else ""
    return (
        RESULTS_DIR / f"eval_runs{suffix}.jsonl",
        RESULTS_DIR / f"metrics{suffix}.csv",
        RESULTS_DIR / f"metrics_by_bucket{suffix}.csv",
    )


def _dispatch(index, q: Dict[str, Any], provider) -> Dict[str, Any]:
    """Route a question to the correct mode and return a serialisable record."""
    bucket = q["bucket"]
    qid = q["id"]
    started = time.perf_counter()

    if bucket in ("factual", "where_stated", "ambiguous", "out_of_scope"):
        result = ask(
            index,
            q["question"],
            paper_ids=q.get("paper_filter"),
            provider=provider,
        )
    elif bucket == "section_summary":
        # Use seed paper for summary prompts (only seed paper has the math sections)
        seed = q.get("paper_filter", [None])[0]
        section = q["ground_truth"]["section"]
        result = summarize_section(index, seed, section, provider=provider)
    elif bucket == "compare":
        pair = q.get("paper_pair") or []
        if len(pair) < 2:
            return {
                "id": qid, "bucket": bucket, "skipped": True,
                "reason": "no_paper_pair", "duration_ms": 0,
            }
        result = compare_papers(index, pair[0], pair[1], q["question"], provider=provider)
    else:
        return {"id": qid, "bucket": bucket, "skipped": True, "reason": "unknown_bucket"}

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    record = result.to_dict()
    record.update({"id": qid, "bucket": bucket, "duration_ms": duration_ms, "question": q["question"]})
    record["expected"] = q.get("ground_truth") or {"behavior": q.get("expected_behavior")}
    return record


def _metric_top1_hit(record: Dict[str, Any]) -> Optional[bool]:
    """True if top-1 retrieved chunk's section matches GT section."""
    if record.get("skipped") or record["bucket"] not in ("factual", "where_stated"):
        return None
    gt_section = record["expected"].get("section")
    retrieved = record.get("retrieved") or []
    if not gt_section or not retrieved:
        return False
    return retrieved[0]["chunk"]["section"] == gt_section


def _metric_topk_hit(record: Dict[str, Any]) -> Optional[bool]:
    """True if any retrieved chunk's section matches GT section."""
    if record.get("skipped") or record["bucket"] not in ("factual", "where_stated"):
        return None
    gt_section = record["expected"].get("section")
    retrieved = record.get("retrieved") or []
    if not gt_section or not retrieved:
        return False
    return any(r["chunk"]["section"] == gt_section for r in retrieved)


def _metric_correctly_refused(record: Dict[str, Any]) -> Optional[bool]:
    """For OOS: refusal is the correct behaviour.
    For ambiguous: either refusal OR a short clarifying question counts as correct."""
    if record.get("skipped"):
        return None
    bucket = record["bucket"]
    if bucket == "out_of_scope":
        return bool(record.get("refused"))
    if bucket == "ambiguous":
        if record.get("refused"):
            return True
        # Heuristic: a clarifying question = answer contains '?' AND is short
        # (catches questions even when they trail a citation tag at the end).
        ans = (record.get("answer") or "").strip()
        return "?" in ans and len(ans.split()) <= 100
    return None


def _percent(values: List[bool]) -> float:
    if not values:
        return 0.0
    return round(100.0 * sum(1 for v in values if v) / len(values), 2)


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grounded = [r for r in records if not r.get("skipped") and r["bucket"] not in ("ambiguous", "out_of_scope")]
    latencies = [(r.get("retrieval_latency_ms", 0) + r.get("generation_latency_ms", 0)) for r in records if not r.get("skipped")]
    tokens = [r.get("token_usage_est", 0) for r in records if not r.get("skipped")]

    return {
        "total_questions": len(records),
        "skipped": sum(1 for r in records if r.get("skipped")),
        "citation_coverage_pct": _percent([bool(r.get("citations")) for r in grounded]),
        "retrieval_top1_hit_pct": _percent([h for r in records if (h := _metric_top1_hit(r)) is not None]),
        "retrieval_topk_hit_pct": _percent([h for r in records if (h := _metric_topk_hit(r)) is not None]),
        "correctly_refused_pct": _percent([h for r in records if (h := _metric_correctly_refused(r)) is not None]),
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[-1], 2) if len(latencies) >= 20 else 0.0,
        "mean_token_usage_est": round(statistics.mean(tokens), 1) if tokens else 0.0,
    }


def summarise_by_bucket(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_b: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_b[r["bucket"]].append(r)
    rows = []
    for bucket, recs in by_b.items():
        latencies = [(r.get("retrieval_latency_ms", 0) + r.get("generation_latency_ms", 0)) for r in recs if not r.get("skipped")]
        rows.append({
            "bucket": bucket,
            "count": len(recs),
            "skipped": sum(1 for r in recs if r.get("skipped")),
            "citation_coverage_pct": _percent([bool(r.get("citations")) for r in recs if not r.get("skipped")]),
            "refused_pct": _percent([bool(r.get("refused")) for r in recs if not r.get("skipped")]),
            "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        })
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-llm", action="store_true", help="Use extractive fallback only (no LLM calls)")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N questions (0 = all)")
    parser.add_argument("--model", default=None, help="Override LLM model name")
    parser.add_argument(
        "--output-tag",
        default=None,
        help=(
            "Write tagged output filenames, e.g. --output-tag smoke writes "
            "metrics-smoke.csv. Defaults to limitN when --limit is used."
        ),
    )
    args = parser.parse_args(argv)
    output_tag = args.output_tag or (f"limit{args.limit}" if args.limit else None)
    runs_path, metrics_path, bucket_path = _output_paths(output_tag)

    print("Loading eval set...", file=sys.stderr)
    payload = json.loads(EVAL_PATH.read_text())
    questions = payload["questions"]
    if args.limit:
        questions = questions[: args.limit]

    print("Building index...", file=sys.stderr)
    index = build_index_from_corpus(CORPUS_DIR, RetrievalConfig())
    print(f"  papers={len(index.papers)}, chunks={len(index.chunks)}", file=sys.stderr)

    provider = None if args.no_llm else (load_provider(args.model) if args.model else load_provider())
    if provider is None and not args.no_llm:
        print("WARN: no LLM provider available (set OPENAI_API_KEY / OPENAI_BASE_URL); using extractive fallback.", file=sys.stderr)

    records: List[Dict[str, Any]] = []
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("w") as f:
        for i, q in enumerate(questions, start=1):
            rec = _dispatch(index, q, provider)
            records.append(rec)
            f.write(json.dumps(rec, default=str) + "\n")
            if i % 5 == 0 or i == len(questions):
                print(f"  [{i}/{len(questions)}] {q['bucket']:18} done", file=sys.stderr)

    summary = summarise(records)
    print("\n=== Summary ===", file=sys.stderr)
    for k, v in summary.items():
        print(f"  {k}: {v}", file=sys.stderr)

    _write_csv(metrics_path, [summary])
    _write_csv(bucket_path, summarise_by_bucket(records))
    print(f"\nWrote: {runs_path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
    print(f"Wrote: {metrics_path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
    print(f"Wrote: {bucket_path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
