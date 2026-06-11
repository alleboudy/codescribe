from __future__ import annotations

from pathlib import Path

from codescribe_train.rag.sources.worktree_docs_source import Doc, WorktreeDocsSource


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# R\n\n## Run\n\nmake up\n")
    (tmp_path / "Makefile").write_text("up:\n\tdocker compose up\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/DEV.md").write_text("# Dev\n\nstuff\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.py").write_text("print('x')\n")  # not a doc
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules/README.md").write_text("# excluded\n")
    return tmp_path


def test_yields_docs_with_chunks(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    docs = list(WorktreeDocsSource(repo).iter_docs())
    paths = {d.path for d in docs}
    assert "README.md" in paths
    assert "Makefile" in paths
    assert "docs/DEV.md" in paths
    assert "src/main.py" not in paths
    assert "node_modules/README.md" not in paths
    assert all(isinstance(d, Doc) and d.chunks for d in docs)


def test_readme_is_heading_chunked(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    readme = next(d for d in WorktreeDocsSource(repo).iter_docs() if d.path == "README.md")
    headings = {c.heading for c in readme.chunks}
    assert "Run" in headings


def test_empty_repo_yields_nothing(tmp_path: Path) -> None:
    assert list(WorktreeDocsSource(tmp_path).iter_docs()) == []
