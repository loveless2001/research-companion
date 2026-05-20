#!/usr/bin/env python3
"""Author the 50-prompt eval set with retrieval-validated ground truth.

Strategy: candidate questions are drafted by topic (matched to the seed paper's
sections). For each grounded question we run a retrieval pass, capture the
top-1 (section, page) as the proposed ground truth, and flag mismatches for
manual review. Output: results/eval_questions.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research_companion.paper_companion import (
    RetrievalConfig,
    build_index_from_corpus,
)

CORPUS_DIR = PROJECT_ROOT / "corpus"
OUT_PATH = PROJECT_ROOT / "results" / "eval_questions.json"

# Each entry: (question, expected_section, bucket)
# expected_section is the section we *expect* retrieval to land on; if top-1
# matches we promote it (plus its page) to ground truth, otherwise we record
# the mismatch for review.

FACTUAL = [
    ("What activation function is described as very similar to the sigmoid?", "artificial_neurons_activation_functions"),
    ("What property must a loss function have so that gradient descent is applicable?", "supervised_learning"),
    ("What does SGD stand for and how does it differ from full gradient descent?", "stochastic_gradient_descent"),
    ("Why is initialization important for training deep neural networks?", "initialization"),
    ("What is the role of the convolution operation in a CNN?", "convolutional_neural_networks"),
    ("What is automatic differentiation and how does it relate to backpropagation?", "automatic_differentiation_backpropagation"),
    ("What is the Adam optimizer and what hyperparameters does it use?", "adaptive_learning_rate_algorithms"),
    ("What is the definition of a smooth manifold used in the paper?", "manifolds"),
    ("What is a Lie group and give one example from the paper?", "lie_groups"),
    ("What is meant by a rotation-translation equivariant CNN?", "building_a_rotation_translation_equivariant_cnn"),
    ("What are tropical operators in the context of this paper?", "tropical_operators"),
    ("What is a shallow neural network as defined in the paper?", "shallow_networks"),
    ("What is the universal approximation property mentioned for shallow networks?", "shallow_networks"),
    ("How is Xavier initialization defined?", "initialization"),
    ("What is the difference between a deep and a shallow network?", "deep_neural_networks"),
]

WHERE_STATED = [
    ("Where in the paper is the hyperbolic tangent activation function introduced?", "artificial_neurons_activation_functions"),
    ("Which section discusses adaptive learning rate methods like Adam?", "adaptive_learning_rate_algorithms"),
    ("Where is the concept of a Lie subgroup defined?", "lie_groups"),
    ("Which section covers convolutional neural networks?", "convolutional_neural_networks"),
    ("Where does the paper introduce backpropagation?", "automatic_differentiation_backpropagation"),
    ("Where is stochastic gradient descent first defined?", "stochastic_gradient_descent"),
    ("Which section motivates the importance of weight initialization?", "initialization"),
    ("Where are tropical operators introduced?", "tropical_operators"),
    ("Where does the paper define manifolds?", "manifolds"),
    ("Which section builds the rotation-translation equivariant CNN?", "building_a_rotation_translation_equivariant_cnn"),
]

# Section-summary prompts (the answer must summarize a whole section).
SUMMARY = [
    ("Summarize the section on supervised learning.", "supervised_learning"),
    ("Summarize the section on stochastic gradient descent.", "stochastic_gradient_descent"),
    ("Summarize the section on initialization.", "initialization"),
    ("Summarize the section on convolutional neural networks.", "convolutional_neural_networks"),
    ("Summarize the section on automatic differentiation and backpropagation.", "automatic_differentiation_backpropagation"),
    ("Summarize the section on adaptive learning rate algorithms.", "adaptive_learning_rate_algorithms"),
    ("Summarize the section on Lie groups.", "lie_groups"),
    ("Summarize the section on tropical operators.", "tropical_operators"),
]

# Compare-two-papers placeholders: need a second paper before they can be run.
COMPARE = [
    ("Compare the method sections of Paper A and Paper B.", "method"),
    ("Compare the experimental results of Paper A and Paper B.", "results"),
    ("Compare the limitations discussed in Paper A and Paper B.", "limitations"),
    ("What datasets are used in Paper A vs Paper B?", "experiments"),
    ("Compare the proposed loss functions in Paper A and Paper B.", "method"),
    ("Compare the evaluation metrics in Paper A and Paper B.", "results"),
    ("Compare the architectural choices in Paper A and Paper B.", "method"),
]

# Ambiguous: should trigger a clarification request, not a confident answer.
AMBIGUOUS = [
    "What is the best learning rate?",
    "How do I tune the network?",
    "Which is better, this or the other one?",
    "What about regularization?",
    "Is bigger better?",
]

# Out-of-scope: should be refused (not present in seed paper).
OUT_OF_SCOPE = [
    "What was the closing price of NVIDIA stock yesterday?",
    "Who won the 2024 FIFA World Cup?",
    "What is the recipe for Vietnamese pho?",
    "When is the next SpaceX Starship launch?",
    "What is the capital of Mongolia?",
]


def _seed_paper_id(papers) -> str:
    """Find the math-neural-networks paper id (grounded questions target it)."""
    for p in papers:
        if "2403.04807" in p.source_path or "math-neural-networks" in p.source_path:
            return p.paper_id
    return papers[0].paper_id  # fallback


def _pick_compare_pairs(papers, n: int = 7):
    """Pick n paper pairs for the compare bucket. Cycles through distinct (a,b)."""
    ids = [p.paper_id for p in papers]
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append((ids[i], ids[j]))
    # repeat if needed; truncate to n
    while len(pairs) < n:
        pairs.extend(pairs)
    return pairs[:n]


def main() -> int:
    print("Building index...", file=sys.stderr)
    idx = build_index_from_corpus(CORPUS_DIR, RetrievalConfig())
    print(f"  papers={len(idx.papers)}, chunks={len(idx.chunks)}", file=sys.stderr)

    paper_id = _seed_paper_id(idx.papers)
    compare_pairs = _pick_compare_pairs(idx.papers, n=len(COMPARE))
    out = {
        "version": 2,
        "seed_paper_id": paper_id,
        "all_paper_ids": [p.paper_id for p in idx.papers],
        "questions": [],
    }

    def ground_truth(question: str, expected_section: str) -> dict:
        """Run retrieval, return ground-truth label + match status.
        Filters to the seed paper since grounded questions are seed-specific."""
        records = idx.retrieve(question, paper_ids=[paper_id], top_k=3)
        if not records:
            return {"section": expected_section, "page": None, "retrieval_top1_section": None, "match": False}
        top = records[0]["chunk"]
        match = (top["section"] == expected_section)
        return {
            "section": expected_section,
            "page": top["page_start"] if match else None,
            "retrieval_top1_section": top["section"],
            "retrieval_top1_page": top["page_start"],
            "match": match,
        }

    qid = 0

    for q, expected in FACTUAL:
        qid += 1
        out["questions"].append({
            "id": f"f{qid:02d}",
            "bucket": "factual",
            "question": q,
            "paper_filter": [paper_id],
            "ground_truth": ground_truth(q, expected),
        })

    for q, expected in WHERE_STATED:
        qid += 1
        out["questions"].append({
            "id": f"w{qid:02d}",
            "bucket": "where_stated",
            "question": q,
            "paper_filter": [paper_id],
            "ground_truth": ground_truth(q, expected),
        })

    for q, expected in SUMMARY:
        qid += 1
        out["questions"].append({
            "id": f"s{qid:02d}",
            "bucket": "section_summary",
            "question": q,
            "paper_filter": [paper_id],
            "ground_truth": {"section": expected},
        })

    for (q, expected), (pa, pb) in zip(COMPARE, compare_pairs):
        qid += 1
        out["questions"].append({
            "id": f"c{qid:02d}",
            "bucket": "compare",
            "question": q,
            "paper_pair": [pa, pb],
            "ground_truth": {"section": expected},
            "needs_two_papers": True,
        })

    for q in AMBIGUOUS:
        qid += 1
        out["questions"].append({
            "id": f"a{qid:02d}",
            "bucket": "ambiguous",
            "question": q,
            "paper_filter": [paper_id],
            "expected_behavior": "clarify",
        })

    for q in OUT_OF_SCOPE:
        qid += 1
        out["questions"].append({
            "id": f"o{qid:02d}",
            "bucket": "out_of_scope",
            "question": q,
            "paper_filter": [paper_id],
            "expected_behavior": "refuse",
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    # Summary
    total = len(out["questions"])
    grounded = [q for q in out["questions"] if q["bucket"] in ("factual", "where_stated")]
    matches = sum(1 for q in grounded if q["ground_truth"].get("match"))
    print(f"\nWrote {total} questions to {OUT_PATH.relative_to(PROJECT_ROOT)}", file=sys.stderr)
    print(f"Grounded questions: {len(grounded)}; retrieval-validated: {matches}/{len(grounded)}", file=sys.stderr)
    print("\nMismatches (need manual review):", file=sys.stderr)
    for q in grounded:
        gt = q["ground_truth"]
        if not gt.get("match"):
            print(f"  {q['id']}: expected={gt['section']!r} top1={gt.get('retrieval_top1_section')!r} -- {q['question']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
