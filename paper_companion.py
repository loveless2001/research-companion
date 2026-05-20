#!/usr/bin/env python3
"""Data layer for the Research Paper Companion project.

This module owns PDF/text ingestion, section-aware chunking, FAISS retrieval with
paper/section filters, and deterministic citation snippets. It is deliberately
kept separate from the earlier assignment runners while reusing the stable
embedding/index primitives from Assignment 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


try:
    from .retrieval_primitives import (
        BASELINE_EMBED_MODEL,
        Embedder,
        VectorIndex,
        split_text_recursive,
    )
except ImportError:  # pragma: no cover - direct `python paper_companion.py`
    from retrieval_primitives import (  # type: ignore
        BASELINE_EMBED_MODEL,
        Embedder,
        VectorIndex,
        split_text_recursive,
    )


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "corpus"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
MAX_QUOTE_WORDS = 20
TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}


SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "background",
    "related work": "related_work",
    "preliminaries": "preliminaries",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "model": "method",
    "experiments": "experiments",
    "experiment": "experiments",
    "results": "results",
    "evaluation": "results",
    "analysis": "analysis",
    "discussion": "discussion",
    "limitations": "limitations",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "references": "references",
    "bibliography": "references",
    "appendix": "appendix",
}


HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?("
    + "|".join(re.escape(key) for key in sorted(SECTION_ALIASES, key=len, reverse=True))
    + r")\b[:.\-\s]*(.*)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PageBlock:
    paper_id: str
    source_path: str
    page: int
    text: str
    char_start: int
    char_end: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    max_font_size: Optional[float] = None
    is_heading_candidate: bool = False


@dataclass(frozen=True)
class Section:
    paper_id: str
    section_id: str
    heading: str
    normalized_heading: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    source_path: str
    page_count: int
    char_count: int
    sections: List[Section]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    source_path: str
    section: str
    section_heading: str
    page_start: int
    page_end: int
    chunk_index: int
    char_offset_start: int
    char_offset_end: int
    text: str


@dataclass(frozen=True)
class Citation:
    paper_id: str
    section: str
    page_start: int
    page_end: int
    chunk_id: str
    quote_snippet: str
    char_offset_start: int
    char_offset_end: int


@dataclass(frozen=True)
class RetrievalConfig:
    chunk_size: int = 650
    overlap: int = 120
    top_k: int = 5
    metric: str = "cosine"
    embed_model: str = BASELINE_EMBED_MODEL
    search_multiplier: int = 8


def stable_paper_id(path: Path) -> str:
    digest = hashlib.blake2b(str(path.resolve()).encode("utf-8"), digest_size=5).hexdigest()
    stem = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:40]
    return f"{stem or 'paper'}-{digest}"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_heading(raw: str) -> str:
    cleaned = normalize_space(raw).strip(" .:-").lower()
    cleaned = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", cleaned)
    for label, normalized in SECTION_ALIASES.items():
        if cleaned == label or cleaned.startswith(label + " "):
            return normalized
    return re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_") or "unknown"


def looks_like_heading(text: str, *, is_font_candidate: bool = False) -> bool:
    line = normalize_space(text)
    if not line or len(line) > 120:
        return False
    if HEADING_RE.match(line):
        return True
    if is_font_candidate and len(line.split()) <= 12:
        alpha = re.sub(r"[^A-Za-z]", "", line)
        if alpha and sum(1 for char in alpha if char.isupper()) / len(alpha) > 0.45:
            return True
        if re.match(r"^\d+(?:\.\d+)*\.?\s+\S+", line):
            return True
    return False


def iter_corpus_files(corpus_dir: Path) -> List[Path]:
    supported = PDF_EXTENSIONS | TEXT_EXTENSIONS
    return sorted(
        path
        for path in corpus_dir.expanduser().resolve().rglob("*")
        if path.is_file() and path.suffix.lower() in supported
        and path.name.lower() != "readme.md"
    )


def _title_from_blocks(blocks: Sequence[PageBlock], fallback: Path) -> str:
    first_page = [block for block in blocks if block.page == 1 and block.text]
    if not first_page:
        return fallback.stem
    candidates = [
        block
        for block in first_page[:12]
        if 5 <= len(normalize_space(block.text)) <= 180
        and "arxiv" not in block.text.lower()
    ]
    if not candidates:
        return fallback.stem
    candidates = sorted(
        candidates,
        key=lambda block: (block.max_font_size or 0.0, len(block.text)),
        reverse=True,
    )
    return normalize_space(candidates[0].text)


def _load_pdf_blocks(path: Path, paper_id: str) -> List[PageBlock]:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-only path
        raise RuntimeError("PDF ingestion requires PyMuPDF. Install it with `pip install pymupdf`.") from exc

    doc = fitz.open(str(path))
    font_sizes: List[float] = []
    raw: List[Dict[str, Any]] = []
    cursor = 0
    for page_index, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines: List[str] = []
            block_sizes: List[float] = []
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(str(span.get("text", "")) for span in spans).strip()
                if text:
                    lines.append(text)
                for span in spans:
                    size = float(span.get("size", 0.0) or 0.0)
                    if size > 0:
                        block_sizes.append(size)
                        font_sizes.append(size)
            block_text = normalize_space(" ".join(lines))
            if not block_text:
                continue
            raw.append(
                {
                    "page": page_index,
                    "text": block_text,
                    "bbox": tuple(float(x) for x in block.get("bbox", (0, 0, 0, 0))),
                    "max_font_size": max(block_sizes) if block_sizes else None,
                    "char_start": cursor,
                    "char_end": cursor + len(block_text),
                }
            )
            cursor += len(block_text) + 2

    median_font = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 0.0
    blocks: List[PageBlock] = []
    for item in raw:
        max_font = item["max_font_size"]
        font_candidate = bool(max_font and median_font and max_font >= median_font * 1.18)
        text = item["text"]
        blocks.append(
            PageBlock(
                paper_id=paper_id,
                source_path=str(path),
                page=item["page"],
                text=text,
                char_start=item["char_start"],
                char_end=item["char_end"],
                bbox=item["bbox"],
                max_font_size=max_font,
                is_heading_candidate=looks_like_heading(text, is_font_candidate=font_candidate),
            )
        )
    return blocks


def _load_text_blocks(path: Path, paper_id: str) -> List[PageBlock]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks: List[PageBlock] = []
    cursor = 0
    for raw in re.split(r"\n\s*\n", text):
        block_text = normalize_space(raw)
        if not block_text:
            cursor += len(raw) + 2
            continue
        blocks.append(
            PageBlock(
                paper_id=paper_id,
                source_path=str(path),
                page=1,
                text=block_text,
                char_start=cursor,
                char_end=cursor + len(block_text),
                is_heading_candidate=looks_like_heading(block_text, is_font_candidate=True),
            )
        )
        cursor += len(raw) + 2
    return blocks


def load_blocks(path: Path, paper_id: Optional[str] = None) -> List[PageBlock]:
    resolved = path.expanduser().resolve()
    actual_id = paper_id or stable_paper_id(resolved)
    if resolved.suffix.lower() in PDF_EXTENSIONS:
        return _load_pdf_blocks(resolved, actual_id)
    if resolved.suffix.lower() in TEXT_EXTENSIONS:
        return _load_text_blocks(resolved, actual_id)
    raise ValueError(f"Unsupported file type: {resolved}")


def detect_sections(blocks: Sequence[PageBlock]) -> List[Section]:
    if not blocks:
        return []
    starts: List[Tuple[int, str, str]] = []
    for idx, block in enumerate(blocks):
        if not block.is_heading_candidate:
            continue
        heading = normalize_space(block.text)
        normalized = normalize_heading(heading)
        if normalized == "unknown" and not re.match(r"^\d+(?:\.\d+)*\.?\s+\S+", heading):
            continue
        starts.append((idx, heading, normalized))

    if not starts or starts[0][0] != 0:
        starts.insert(0, (0, "Unknown", "unknown"))

    sections: List[Section] = []
    for section_index, (start_idx, heading, normalized) in enumerate(starts):
        end_idx = starts[section_index + 1][0] if section_index + 1 < len(starts) else len(blocks)
        section_blocks = blocks[start_idx:end_idx]
        if not section_blocks:
            continue
        first = section_blocks[0]
        last = section_blocks[-1]
        sections.append(
            Section(
                paper_id=first.paper_id,
                section_id=f"{first.paper_id}:s{section_index:03d}",
                heading=heading,
                normalized_heading=normalized,
                page_start=first.page,
                page_end=last.page,
                char_start=first.char_start,
                char_end=last.char_end,
            )
        )
    return sections


def _section_for_block(block: PageBlock, sections: Sequence[Section]) -> Section:
    for section in sections:
        if section.char_start <= block.char_start <= section.char_end:
            return section
    return sections[0]


def ingest_file(path: Path, *, chunk_size: int = 650, overlap: int = 120) -> Tuple[Paper, List[Chunk]]:
    paper_id = stable_paper_id(path)
    blocks = load_blocks(path, paper_id=paper_id)
    sections = detect_sections(blocks)
    if not sections and blocks:
        first, last = blocks[0], blocks[-1]
        sections = [
            Section(
                paper_id=paper_id,
                section_id=f"{paper_id}:s000",
                heading="Unknown",
                normalized_heading="unknown",
                page_start=first.page,
                page_end=last.page,
                char_start=first.char_start,
                char_end=last.char_end,
            )
        ]

    chunks: List[Chunk] = []
    chunk_index = 0
    for block in blocks:
        section = _section_for_block(block, sections)
        local_parts = split_text_recursive(block.text, chunk_size=chunk_size, overlap=overlap)
        local_cursor = 0
        for part in local_parts:
            rel_start = block.text.find(part[:80], local_cursor)
            if rel_start < 0:
                rel_start = local_cursor
            rel_end = rel_start + len(part)
            local_cursor = max(rel_start + 1, rel_end - 120)
            chunks.append(
                Chunk(
                    chunk_id=f"{paper_id}:c{chunk_index:05d}",
                    paper_id=paper_id,
                    source_path=str(path.expanduser().resolve()),
                    section=section.normalized_heading,
                    section_heading=section.heading,
                    page_start=block.page,
                    page_end=block.page,
                    chunk_index=chunk_index,
                    char_offset_start=block.char_start + rel_start,
                    char_offset_end=min(block.char_end, block.char_start + rel_end),
                    text=part,
                )
            )
            chunk_index += 1

    paper = Paper(
        paper_id=paper_id,
        title=_title_from_blocks(blocks, path),
        source_path=str(path.expanduser().resolve()),
        page_count=max((block.page for block in blocks), default=0),
        char_count=max((block.char_end for block in blocks), default=0),
        sections=sections,
    )
    return paper, chunks


def ingest_corpus(
    corpus_dir: Path,
    *,
    chunk_size: int = 650,
    overlap: int = 120,
) -> Tuple[List[Paper], List[Chunk]]:
    files = iter_corpus_files(corpus_dir)
    papers: List[Paper] = []
    chunks: List[Chunk] = []
    for path in files:
        paper, paper_chunks = ingest_file(path, chunk_size=chunk_size, overlap=overlap)
        papers.append(paper)
        chunks.extend(paper_chunks)
    if not papers:
        raise RuntimeError(f"No PDF/TXT/MD corpus files found under {corpus_dir}")
    return papers, chunks


def quote_snippet(text: str, query: str, max_words: int = MAX_QUOTE_WORDS) -> str:
    words = re.findall(r"\S+", normalize_space(text))
    if not words:
        return ""
    query_terms = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]*", query.lower())
        if len(token) > 2
    }
    best_start = 0
    best_score = -1
    for idx in range(0, len(words), max(1, max_words // 2)):
        window = words[idx : idx + max_words]
        score = len(query_terms & {re.sub(r"[^a-z0-9\-]", "", word.lower()) for word in window})
        if score > best_score:
            best_start = idx
            best_score = score
    snippet = " ".join(words[best_start : best_start + max_words]).strip()
    return snippet


def citation_for_chunk(chunk: Chunk, query: str) -> Citation:
    return Citation(
        paper_id=chunk.paper_id,
        section=chunk.section,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        chunk_id=chunk.chunk_id,
        quote_snippet=quote_snippet(chunk.text, query),
        char_offset_start=chunk.char_offset_start,
        char_offset_end=chunk.char_offset_end,
    )


def _content_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", text))


def _is_retrievable_chunk(chunk: Chunk) -> bool:
    text = normalize_space(chunk.text)
    if _content_token_count(text) < 8:
        return False
    if text.count(".") > max(8, len(text) // 8):
        return False
    if chunk.section == "references":
        return False
    return True


class PaperIndex:
    def __init__(self, papers: Sequence[Paper], chunks: Sequence[Chunk], config: RetrievalConfig) -> None:
        if not chunks:
            raise RuntimeError("Cannot build an index without chunks.")
        self.papers = list(papers)
        self.chunks = list(chunks)
        self.config = config
        self.embedder = Embedder(config.embed_model)
        embeddings = self.embedder.encode([chunk.text for chunk in self.chunks], is_query=False)
        self.index = VectorIndex(embeddings, config.metric)

    def retrieve(
        self,
        query: str,
        *,
        paper_ids: Optional[Iterable[str]] = None,
        sections: Optional[Iterable[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        wanted_papers = set(paper_ids or [])
        wanted_sections = {normalize_heading(section) for section in (sections or [])}
        limit = top_k or self.config.top_k
        candidate_k = min(len(self.chunks), max(limit, limit * self.config.search_multiplier, 50))
        query_vec = self.embedder.encode([query], is_query=True)
        scores, indices = self.index.search(query_vec, candidate_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if wanted_papers and chunk.paper_id not in wanted_papers:
                continue
            if wanted_sections and chunk.section not in wanted_sections:
                continue
            if not _is_retrievable_chunk(chunk):
                continue
            citation = citation_for_chunk(chunk, query)
            results.append(
                {
                    "rank": len(results) + 1,
                    "score": round(float(score), 6),
                    "chunk": asdict(chunk),
                    "citation": asdict(citation),
                }
            )
            if len(results) >= limit:
                break
        return results


def build_index_from_corpus(corpus_dir: Path, config: RetrievalConfig) -> PaperIndex:
    papers, chunks = ingest_corpus(
        corpus_dir,
        chunk_size=config.chunk_size,
        overlap=config.overlap,
    )
    return PaperIndex(papers, chunks, config)


def extractive_answer(query: str, retrieved: Sequence[Dict[str, Any]]) -> str:
    if not retrieved:
        return "I could not find enough evidence in the provided documents."
    bullets = []
    for item in retrieved[:2]:
        citation = item["citation"]
        tag = (
            f"[{citation['paper_id']} p.{citation['page_start']} "
            f"section={citation['section']} chunk={citation['chunk_id']}]"
        )
        bullets.append(f"- {citation['quote_snippet']} {tag}")
    return "\n".join(bullets)


def corpus_summary(papers: Sequence[Paper], chunks: Sequence[Chunk]) -> Dict[str, Any]:
    return {
        "paper_count": len(papers),
        "chunk_count": len(chunks),
        "papers": [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "page_count": paper.page_count,
                "section_count": len(paper.sections),
                "sections": [section.normalized_heading for section in paper.sections[:20]],
                "source_path": paper.source_path,
            }
            for paper in papers
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run_inspect(args: argparse.Namespace) -> int:
    papers, chunks = ingest_corpus(Path(args.corpus_dir))
    print(json.dumps(corpus_summary(papers, chunks), indent=2, ensure_ascii=True))
    if args.out:
        write_json(Path(args.out), {"papers": [asdict(paper) for paper in papers], "chunks": [asdict(c) for c in chunks]})
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    config = RetrievalConfig(top_k=args.top_k, embed_model=args.embed_model)
    index = build_index_from_corpus(Path(args.corpus_dir), config)
    retrieved = index.retrieve(
        args.query,
        paper_ids=args.paper_id,
        sections=args.section,
        top_k=args.top_k,
    )
    payload = {
        "query": args.query,
        "latency_ms": round((time.perf_counter() - start) * 1000, 3),
        "summary": corpus_summary(index.papers, index.chunks),
        "answer": extractive_answer(args.query, retrieved),
        "retrieved": retrieved,
    }
    if args.out:
        write_json(Path(args.out), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research Paper Companion data layer")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Ingest a corpus and print paper/section stats")
    inspect.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    inspect.add_argument("--out")
    inspect.set_defaults(func=run_inspect)

    smoke = sub.add_parser("smoke", help="Build an index and run one cited retrieval smoke test")
    smoke.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    smoke.add_argument("--query", default="What is Xavier initialization?")
    smoke.add_argument("--top-k", type=int, default=5)
    smoke.add_argument("--paper-id", action="append")
    smoke.add_argument("--section", action="append")
    smoke.add_argument("--embed-model", default=BASELINE_EMBED_MODEL)
    smoke.add_argument("--out", default=str(DEFAULT_RESULTS_DIR / "smoke.json"))
    smoke.set_defaults(func=run_smoke)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
