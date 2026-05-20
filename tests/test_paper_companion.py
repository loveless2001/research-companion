from pathlib import Path

from research_companion.paper_companion import (
    PageBlock,
    citation_for_chunk,
    detect_sections,
    ingest_file,
    iter_corpus_files,
    looks_like_heading,
    quote_snippet,
    stable_paper_id,
)


def test_heading_detection_covers_numbered_research_sections():
    assert looks_like_heading("1 Introduction")
    assert looks_like_heading("2. Methodology")
    assert looks_like_heading("Results")
    assert not looks_like_heading("This is a normal sentence in the body with many details.")


def test_detect_sections_keeps_unknown_bucket_before_first_heading():
    blocks = [
        PageBlock("p1", "paper.txt", 1, "Title text", 0, 10),
        PageBlock("p1", "paper.txt", 1, "Introduction", 11, 23, is_heading_candidate=True),
        PageBlock("p1", "paper.txt", 1, "Body", 24, 28),
    ]
    sections = detect_sections(blocks)
    assert [section.normalized_heading for section in sections] == ["unknown", "introduction"]


def test_quote_snippet_is_bounded():
    text = " ".join(f"word{i}" for i in range(40))
    snippet = quote_snippet(text, "word30", max_words=20)
    assert len(snippet.split()) <= 20


def test_ingest_text_file_produces_offsets(tmp_path):
    path = tmp_path / "paper.md"
    path.write_text(
        "# Title\n\nAbstract\n\nThis paper studies retrieval.\n\n1 Introduction\n\nRetrieval uses chunks.",
        encoding="utf-8",
    )
    paper, chunks = ingest_file(path)
    assert paper.paper_id
    assert chunks
    assert all(chunk.char_offset_start <= chunk.char_offset_end for chunk in chunks)

    citation = citation_for_chunk(chunks[0], "retrieval")
    assert citation.quote_snippet
    assert len(citation.quote_snippet.split()) <= 20


def test_corpus_discovery_ignores_readme_notes(tmp_path):
    (tmp_path / "README.md").write_text("Notes, not a paper.", encoding="utf-8")
    paper = tmp_path / "paper.md"
    paper.write_text("Abstract\n\nReal corpus text.", encoding="utf-8")

    assert iter_corpus_files(tmp_path) == [paper]


def test_stable_paper_id_ignores_parent_directory():
    left = stable_paper_id(Path("/tmp/one/2403.04807-math-neural-networks.pdf"))
    right = stable_paper_id(Path("/tmp/two/2403.04807-math-neural-networks.pdf"))

    assert left == right
