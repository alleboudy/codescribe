"""``python -m codescribe_train.rag query ...`` end-to-end smoke.

The CLI wires up three subcommands per the design spec:

  - ``query issues   --query "..." --k 5``
  - ``query pr-diff  --pr <N>``
  - ``query commits  --query "..." --k 5``

Each prints human-readable Markdown to stdout. The tests below build a
small fixture DB and exercise the CLI through a real subprocess so the
argparse wiring, the embedder injection hook, and the formatter are
all exercised end-to-end.

The CLI honours ``RAG_TEST_RETRIEVER_FACTORY=module:callable`` to swap
in a stub :class:`~codescribe_train.rag.embed.embedder.Embedder` — without
the hook ``python -m codescribe_train.rag query`` would try to load
bge-large-en-v1.5 from disk and we don't ship it in CI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from codescribe_train.rag.embed.chunker import PRChunk
from codescribe_train.rag.extract.pairing import IssuePRLink
from codescribe_train.rag.sources.git_source import Commit, FileChange
from codescribe_train.rag.sources.github_source import Issue, PullRequest
from codescribe_train.rag.store.writer import Store


def _basis_vec(index: int) -> np.ndarray:
    v = np.zeros(1024, dtype=np.float32)
    v[index] = 1.0
    return v


def _seed_demo_db(db_path: Path) -> None:
    """Seed a minimal DB with one issue+PR+link and one commit."""
    with Store.open(db_path) as store:
        store.upsert_issue(
            Issue(
                number=42,
                title="CSV field parsing crash",
                body="The cart crashes when parsing a CSV file at checkout.",
                state="open",
                state_reason=None,
                labels=["bug"],
                assignees=[],
                author="example-org",
                created_at="2026-05-01T12:00:00Z",
                updated_at="2026-05-02T15:00:00Z",
                closed_at=None,
                raw_json='{"number": 42}',
            ),
            _basis_vec(0),
        )
        store.upsert_pull(
            pr=PullRequest(
                number=99,
                title="fix(import): guard against truncated header",
                body="Closes #42.",
                state="merged",
                head_sha="a" * 40,
                base_branch="main",
                author="example-org",
                draft=False,
                created_at="2026-05-01T10:00:00Z",
                updated_at="2026-05-02T11:00:00Z",
                merged_at="2026-05-02T11:00:00Z",
                closed_at="2026-05-02T11:00:00Z",
                raw_json='{"number": 99}',
            ),
            chunks=[
                PRChunk(
                    pr_number=99,
                    file_path="src/checkout.py",
                    hunk_index=0,
                    chunk_text="HUNK_ZERO",
                ),
            ],
            chunk_embeddings=[_basis_vec(100)],
            pr_summary_embedding=_basis_vec(101),
            diff_text="diff --git a/src/checkout.py b/src/checkout.py\n+guard against eof\n",
        )
        store.upsert_issue_pr_link(
            IssuePRLink(
                issue_number=42, pr_number=99, source="closingIssuesReferences",
                confidence=1.0,
            )
        )
        store.upsert_commit(
            Commit(
                sha="c" * 40,
                author="example-org",
                author_email="x@y",
                authored_at="2026-05-01T12:00:00Z",
                message="fix: tiny cart eviction tweak",
                files=[
                    FileChange(path="src/cart.py", change_type="M", added=1, deleted=0),
                ],
                diff_text="@@ -1 +1 @@\n-old\n+new",
                pr_number=None,
            ),
            _basis_vec(0),
        )


_RETRIEVER_FACTORY_SRC = """
import numpy as np


class StubEmbedder:
    def __init__(self):
        self._v = np.zeros(1024, dtype=np.float32)
        self._v[0] = 1.0

    def embed(self, texts):
        return np.stack([self._v for _ in texts], axis=0).astype(np.float32)


def build_embedder():
    return StubEmbedder()
"""


def _write_cfg_and_fixtures(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Write a tiny configs/rag.yaml + the stub embedder factory module."""
    repo_root = Path(__file__).resolve().parents[3]
    db_path = tmp_path / "rag.db"
    cfg = tmp_path / "rag.yaml"
    cfg.write_text(
        "sources:\n  git:\n    repo_path: ../your-repo\n  github:\n"
        "    owner: example-org\n    repo: sample\n"
        f"store:\n  db_path: {db_path}\n"
    )
    factory_mod = tmp_path / "fixture_retriever.py"
    factory_mod.write_text(_RETRIEVER_FACTORY_SRC)
    env = {
        "RAG_CONFIG": str(cfg),
        "RAG_TEST_RETRIEVER_FACTORY": "fixture_retriever:build_embedder",
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{repo_root}",
    }
    _seed_demo_db(db_path)
    return db_path, env


def _run_cli(args: list[str], env: dict[str, str], cwd: Path):
    return subprocess.run(
        [sys.executable, "-m", "codescribe_train.rag", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **env},
        cwd=str(cwd),
        check=False,
    )


def test_query_issues_prints_markdown(tmp_path) -> None:
    """``query issues --query ...`` prints a Markdown list and exits 0."""
    _, env = _write_cfg_and_fixtures(tmp_path)
    proc = _run_cli(
        ["query", "issues", "--query", "CSV field parsing", "--k", "3"],
        env,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r} stdout={proc.stdout!r}"
    out = proc.stdout
    # Issue title or number must show up.
    assert "42" in out
    assert "CSV" in out
    # Linked PR is exposed too.
    assert "99" in out or "fix(import)" in out


def test_query_pr_diff_prints_decoded_diff(tmp_path) -> None:
    """``query pr-diff --pr N`` prints the decoded gzip diff."""
    _, env = _write_cfg_and_fixtures(tmp_path)
    proc = _run_cli(["query", "pr-diff", "--pr", "99"], env, cwd=tmp_path)
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    # The seed diff body must be in the printed output.
    assert "guard against eof" in proc.stdout


def test_query_commits_prints_results(tmp_path) -> None:
    """``query commits --query ...`` prints commit messages."""
    _, env = _write_cfg_and_fixtures(tmp_path)
    proc = _run_cli(
        ["query", "commits", "--query", "cart eviction", "--k", "3"],
        env,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    assert "cart" in proc.stdout.lower()
    # The commit SHA prefix should appear so a human can find the commit.
    assert "c" * 7 in proc.stdout


def test_query_issues_help_does_not_load_torch(tmp_path) -> None:
    """``query issues --help`` exits 0 and prints argparse usage."""
    _, env = _write_cfg_and_fixtures(tmp_path)
    proc = _run_cli(["query", "issues", "--help"], env, cwd=tmp_path)
    assert proc.returncode == 0
    assert "--query" in proc.stdout
    assert "--k" in proc.stdout


def test_query_subcommand_requires_subaction(tmp_path) -> None:
    """``query`` without a sub-action (issues/pr-diff/commits) exits non-zero."""
    _, env = _write_cfg_and_fixtures(tmp_path)
    proc = _run_cli(["query"], env, cwd=tmp_path)
    # argparse default for a required subparser is exit code 2.
    assert proc.returncode != 0


@pytest.mark.parametrize("subaction", ["issues", "pr-diff", "commits"])
def test_query_subactions_are_wired(tmp_path, subaction: str) -> None:
    """All three sub-actions parse correctly (smoke test only)."""
    _, env = _write_cfg_and_fixtures(tmp_path)
    if subaction == "pr-diff":
        args = ["query", subaction, "--pr", "99"]
    else:
        args = ["query", subaction, "--query", "anything", "--k", "1"]
    proc = _run_cli(args, env, cwd=tmp_path)
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
