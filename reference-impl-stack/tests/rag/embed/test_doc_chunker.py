from __future__ import annotations

from codescribe_train.rag.embed.doc_chunker import DocChunk, chunk_document


def test_markdown_splits_on_headings() -> None:
    md = "# Title\n\nintro\n\n## Setup\n\nrun make up\n\n## Usage\n\ndo things\n"
    chunks = chunk_document("README.md", md)
    headings = [c.heading for c in chunks]
    assert "Title" in headings
    assert "Setup" in headings
    assert "Usage" in headings
    setup = next(c for c in chunks if c.heading == "Setup")
    assert "run make up" in setup.text
    assert all(isinstance(c, DocChunk) for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_non_markdown_is_single_chunk() -> None:
    makefile = "up:\n\tdocker compose up\ndown:\n\tdocker compose down\n"
    chunks = chunk_document("Makefile", makefile)
    assert len(chunks) == 1
    assert chunks[0].heading == ""
    assert "docker compose up" in chunks[0].text


def test_oversized_chunk_is_hard_split() -> None:
    body = "x" * 5000
    md = f"# Big\n\n{body}\n"
    chunks = chunk_document("docs/BIG.md", md, max_chars=2000)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 2000 for c in chunks)


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document("docs/EMPTY.md", "") == []
    assert chunk_document("docs/EMPTY.md", "   \n  \n") == []
