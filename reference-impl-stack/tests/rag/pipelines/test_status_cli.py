"""``python -m codescribe_train.rag status`` prints counts + link distribution.

The status command is wired to a few SELECT COUNT(*) queries plus a
GROUP BY against ``issue_pr_links``. The spec just says human-readable
output, so we assert on substrings.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_cli(args: list[str], env: dict[str, str], cwd: Path):
    return subprocess.run(
        [sys.executable, "-m", "codescribe_train.rag", *args],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, **env},
        cwd=str(cwd),
        check=False,
    )


_DEPS_FACTORY_SRC = """
from pathlib import Path
from codescribe_train.rag.pipelines._common import IndexerDeps
from tests.rag.pipelines.fakes import (
    FakeEmbedder, FakeGitHubSource, FakeGitSource,
    make_issue, make_pull, make_commit,
)


def build(db_path: Path) -> IndexerDeps:
    git = FakeGitSource(
        commits=[
            make_commit("a" * 40, message="m1", authored_at="2026-05-01T10:00:00Z"),
            make_commit("b" * 40, message="m2", authored_at="2026-05-02T10:00:00Z"),
        ]
    )
    gh = FakeGitHubSource(
        issues=[make_issue(1), make_issue(2), make_issue(3)],
        pulls=[make_pull(10), make_pull(11)],
        closing_refs=[(10, 1)],
    )
    return IndexerDeps(
        git=git, github=gh, embedder=FakeEmbedder(), store_path=db_path,
    )
"""


def test_status_prints_counts(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    db_path = tmp_path / "rag.db"
    cfg = tmp_path / "rag.yaml"
    cfg.write_text(
        "sources:\n  git:\n    repo_path: ../your-repo\n  github:\n"
        "    owner: example-org\n    repo: sample\n"
        f"store:\n  db_path: {db_path}\n"
    )
    fixture = tmp_path / "fixture_deps.py"
    fixture.write_text(_DEPS_FACTORY_SRC)
    env = {
        "RAG_CONFIG": str(cfg),
        "RAG_TEST_DEPS_FACTORY": "fixture_deps:build",
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{repo_root}",
    }
    _run_cli(["index", "--bootstrap"], env, cwd=tmp_path)
    proc = _run_cli(["status"], env, cwd=tmp_path)

    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    out = proc.stdout.lower()
    # Counts (the exact format is not part of the contract — substring checks
    # are sufficient).
    assert "issues" in out
    assert "pulls" in out or "prs" in out
    assert "commits" in out
    assert "links" in out


def test_status_against_empty_db(tmp_path) -> None:
    """Status against a freshly-created (empty) DB exits 0; counts are 0."""
    repo_root = Path(__file__).resolve().parents[3]
    db_path = tmp_path / "rag.db"
    cfg = tmp_path / "rag.yaml"
    cfg.write_text(
        "sources:\n  git:\n    repo_path: ../your-repo\n  github:\n"
        "    owner: example-org\n    repo: sample\n"
        f"store:\n  db_path: {db_path}\n"
    )
    env = {
        "RAG_CONFIG": str(cfg),
        "PYTHONPATH": str(repo_root),
    }
    proc = _run_cli(["status"], env, cwd=tmp_path)
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    # The string "0" should appear somewhere (every count is 0).
    assert "0" in proc.stdout
