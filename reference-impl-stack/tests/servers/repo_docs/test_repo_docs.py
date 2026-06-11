"""Tests for the repo-docs MCP server.

Uses real tmp_path filesystems (no mocking of Path) and patches
``REPO_DIR`` to point at the fixture. This is a closer mirror of
real-world usage than mocking Path/glob, and the test set-up cost is
microseconds.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from codescribe_train.servers.repo_docs import server as docs_server


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a small repo tree that exercises the doc patterns + exclusions.

    Layout:

        <tmp>/
          README.md
          README.dev.md
          Makefile
          docker-compose.yml
          docker-compose.prod.yaml
          Dockerfile
          backend/Dockerfile
          backend/Dockerfile.prod
          restart.sh
          setup.sh
          docs/
            DEVELOPMENT.md
            ARCHITECTURE.md
            subdir/
              nested.md
          src/
            main.py                # not a doc, must NOT appear
          node_modules/
            pkg/README.md          # excluded dir, must NOT appear
          .git/
            HEAD                   # excluded dir, must NOT appear
    """
    (tmp_path / "README.md").write_text("# README\n\nGetting started\n")
    (tmp_path / "README.dev.md").write_text("# Dev README\n")
    (tmp_path / "Makefile").write_text("up:\n\tdocker compose up\n")
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / "docker-compose.prod.yaml").write_text("services: {prod: {}}\n")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")
    (tmp_path / "restart.sh").write_text("#!/bin/sh\necho restart\n")
    (tmp_path / "setup.sh").write_text("#!/bin/sh\necho setup\n")

    (tmp_path / "backend").mkdir()
    (tmp_path / "backend/Dockerfile").write_text("FROM python:3.12\n")
    (tmp_path / "backend/Dockerfile.prod").write_text("FROM python:3.12\n")

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/DEVELOPMENT.md").write_text("# Development\n")
    (tmp_path / "docs/ARCHITECTURE.md").write_text("# Architecture\n")
    (tmp_path / "docs/subdir").mkdir()
    (tmp_path / "docs/subdir/nested.md").write_text("# Nested\n")

    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.py").write_text("print('hi')\n")

    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules/pkg").mkdir()
    (tmp_path / "node_modules/pkg/README.md").write_text("# excluded\n")

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/HEAD").write_text("ref: refs/heads/main\n")

    # Point the server module at this fixture for the duration of the test.
    monkeypatch.setattr(docs_server, "REPO_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------- #
# list_docs
# ---------------------------------------------------------------------- #


class TestListDocs:
    def test_returns_expected_files(self, fake_repo: Path) -> None:
        result = docs_server.list_docs()
        assert result["repo"] == str(fake_repo)
        names = set(result["docs"])
        # Required to appear:
        for required in (
            "README.md",
            "README.dev.md",
            "Makefile",
            "docker-compose.yml",
            "docker-compose.prod.yaml",
            "Dockerfile",
            "backend/Dockerfile",
            "backend/Dockerfile.prod",
            "restart.sh",
            "setup.sh",
            "docs/DEVELOPMENT.md",
            "docs/ARCHITECTURE.md",
            "docs/subdir/nested.md",
        ):
            assert required in names, f"missing: {required}; got {sorted(names)}"
        # Required to be absent:
        for forbidden in ("src/main.py", "node_modules/pkg/README.md", ".git/HEAD"):
            assert forbidden not in names, f"leaked: {forbidden}"

    def test_excludes_node_modules_and_dot_git(self, fake_repo: Path) -> None:
        # Even though README.md inside node_modules would technically match
        # the "**/README*.md" pattern (we don't ship that pattern, but be
        # defensive), the EXCLUDED_DIR_PARTS gate keeps it out.
        result = docs_server.list_docs()
        for p in result["docs"]:
            parts = Path(p).parts
            assert ".git" not in parts
            assert "node_modules" not in parts

    def test_returns_sorted_list(self, fake_repo: Path) -> None:
        result = docs_server.list_docs()
        assert result["docs"] == sorted(result["docs"])

    def test_subdir_scopes_listing(self, fake_repo: Path) -> None:
        result = docs_server.list_docs(subdir="docs")
        # docs/DEVELOPMENT.md, docs/ARCHITECTURE.md, docs/subdir/nested.md
        # should appear; the root-level README/Makefile should NOT.
        names = set(result["docs"])
        assert "docs/DEVELOPMENT.md" in names
        assert "docs/ARCHITECTURE.md" in names
        assert "docs/subdir/nested.md" in names
        assert "README.md" not in names
        assert "Makefile" not in names

    def test_pattern_filters_substring(self, fake_repo: Path) -> None:
        result = docs_server.list_docs(pattern="docker")
        names = set(result["docs"])
        # All docker-themed files match (case-insensitive substring on path).
        assert "docker-compose.yml" in names
        assert "docker-compose.prod.yaml" in names
        assert "Dockerfile" in names
        assert "backend/Dockerfile" in names
        assert "backend/Dockerfile.prod" in names
        # Things without "docker" anywhere don't.
        assert "README.md" not in names
        assert "Makefile" not in names

    def test_pattern_is_case_insensitive(self, fake_repo: Path) -> None:
        assert (
            set(docs_server.list_docs(pattern="DOCKER")["docs"])
            == set(docs_server.list_docs(pattern="docker")["docs"])
        )

    def test_rejects_absolute_subdir(self, fake_repo: Path) -> None:
        result = docs_server.list_docs(subdir="/etc")
        assert "error" in result
        assert "absolute" in result["error"].lower()

    def test_rejects_subdir_escaping_repo(self, fake_repo: Path) -> None:
        result = docs_server.list_docs(subdir="../somewhere")
        assert "error" in result
        assert "escape" in result["error"].lower()

    def test_rejects_missing_subdir(self, fake_repo: Path) -> None:
        result = docs_server.list_docs(subdir="nonexistent")
        assert "error" in result


# ---------------------------------------------------------------------- #
# read_doc
# ---------------------------------------------------------------------- #


class TestReadDoc:
    def test_reads_relative_file(self, fake_repo: Path) -> None:
        result = docs_server.read_doc("README.md")
        assert result["path"] == "README.md"
        assert "Getting started" in result["content"]
        assert result["truncated"] is False
        assert result["size_bytes"] > 0

    def test_reads_nested_file(self, fake_repo: Path) -> None:
        result = docs_server.read_doc("docs/DEVELOPMENT.md")
        assert result["path"] == "docs/DEVELOPMENT.md"
        assert "# Development" in result["content"]

    def test_rejects_empty_path(self, fake_repo: Path) -> None:
        result = docs_server.read_doc("")
        assert "error" in result
        assert "required" in result["error"].lower()

    def test_rejects_absolute_path(self, fake_repo: Path) -> None:
        result = docs_server.read_doc("/etc/passwd")
        assert "error" in result
        assert "absolute" in result["error"].lower()

    def test_rejects_traversal_outside_repo(self, fake_repo: Path) -> None:
        # The "../" prefix resolves to the parent of REPO_DIR (which is the
        # tmp_path's parent in this test). Server must reject.
        result = docs_server.read_doc("../some-other-file")
        assert "error" in result
        assert "escape" in result["error"].lower()

    def test_rejects_missing_file(self, fake_repo: Path) -> None:
        result = docs_server.read_doc("does-not-exist.md")
        assert "error" in result
        assert "not a file" in result["error"].lower()

    def test_rejects_directory(self, fake_repo: Path) -> None:
        result = docs_server.read_doc("docs")
        assert "error" in result
        assert "not a file" in result["error"].lower()

    def test_truncates_oversized_content(self, fake_repo: Path) -> None:
        # Write a big file and read with a tight cap.
        big = fake_repo / "BIG.md"
        big.write_text("x" * 5_000)
        result = docs_server.read_doc("BIG.md", max_chars=100)
        assert result["truncated"] is True
        assert "TRUNCATED" in result["content"]
        # The content prefix is exactly the cap (then the marker is appended).
        # The marker is appended AFTER the cap so total length > cap.
        assert result["content"].startswith("x" * 100)

    def test_respects_hard_cap_above_request(self, fake_repo: Path) -> None:
        # Asking for more than MAX_READ_CHARS should clamp to MAX_READ_CHARS.
        big = fake_repo / "HUGE.md"
        big.write_text("y" * (docs_server.MAX_READ_CHARS + 10_000))
        result = docs_server.read_doc("HUGE.md", max_chars=10_000_000)
        assert result["truncated"] is True
        # The effective cap is MAX_READ_CHARS; the content (sans marker)
        # should be that many y's.
        assert result["content"].startswith("y" * docs_server.MAX_READ_CHARS)


# ---------------------------------------------------------------------- #
# Module-level sanity: REPO_DIR has a reasonable default
# ---------------------------------------------------------------------- #


def test_repo_dir_is_resolved_path() -> None:
    """REPO_DIR is always an absolute, resolved Path (no relative ./.."""
    # Reload the module's REPO_DIR computation by importing again into a
    # fresh namespace is overkill; just assert the live REPO_DIR is absolute.
    with mock.patch.dict("os.environ", {"TARGET_REPO": "../some-relative"}):
        # The runtime resolves at module import — we can't easily
        # re-execute that here without reload, but we can verify the
        # currently-loaded REPO_DIR is absolute regardless of where the
        # test was launched from.
        assert docs_server.REPO_DIR.is_absolute()
