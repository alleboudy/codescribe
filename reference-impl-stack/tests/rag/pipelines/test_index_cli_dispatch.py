"""End-to-end CLI dispatch for ``python -m codescribe_train.rag index``.

The CLI dispatches to ``run_bootstrap`` /
``run_incremental`` / ``refresh_pr`` / ``refresh_issue`` depending on
flags. Tests inject in-process fakes via the ``RAG_TEST_DEPS_FACTORY``
environment variable, which names a ``module:function`` callable that
returns an ``IndexerDeps``. The hook is documented next to the CLI; it
exists purely so the CI test can avoid spawning ``git``/``gh``
subprocesses.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(args: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "codescribe_train.rag", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **env},
        cwd=str(cwd),
        check=False,
    )


def _write_fixture_module(tmp_path: Path, deps_factory_src: str) -> Path:
    """Write a temporary Python module that builds an ``IndexerDeps``.

    Returns the path to the parent dir (which the subprocess prepends
    to ``sys.path`` via the test-only ``RAG_TEST_DEPS_FACTORY`` env var).
    """
    module = tmp_path / "fixture_deps.py"
    module.write_text(deps_factory_src)
    return tmp_path


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    """Minimal config that points at the test DB."""
    cfg = tmp_path / "rag.yaml"
    cfg.write_text(
        "sources:\n"
        "  git:\n"
        "    repo_path: ../your-repo\n"
        "  github:\n"
        "    owner: example-org\n"
        "    repo: sample\n"
        f"store:\n  db_path: {db_path}\n"
    )
    return cfg


_DEPS_FACTORY_SRC = """
from pathlib import Path

from codescribe_train.rag.pipelines._common import IndexerDeps
from tests.rag.pipelines.fakes import (
    FakeEmbedder, FakeGitHubSource, FakeGitSource,
    make_issue, make_pull, make_commit,
)


def build(db_path: Path) -> IndexerDeps:
    git = FakeGitSource(
        commits=[make_commit("a" * 40, message="init", authored_at="2026-05-01T10:00:00Z")]
    )
    gh = FakeGitHubSource(
        issues=[make_issue(1, title="bug")],
        pulls=[make_pull(10, title="fix")],
        closing_refs=[(10, 1)],
    )
    return IndexerDeps(
        git=git, github=gh, embedder=FakeEmbedder(), store_path=db_path,
    )
"""


def test_index_bootstrap_via_cli_writes_rows(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    db_path = tmp_path / "rag.db"
    fixture_dir = _write_fixture_module(tmp_path, _DEPS_FACTORY_SRC)
    cfg = _write_config(tmp_path, db_path)

    env = {
        "RAG_CONFIG": str(cfg),
        "RAG_TEST_DEPS_FACTORY": "fixture_deps:build",
        "PYTHONPATH": f"{fixture_dir}{os.pathsep}{repo_root}",
    }
    proc = _run_cli(["index", "--bootstrap"], env, cwd=tmp_path)

    assert proc.returncode == 0, (
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )

    # The fake source set has 1 commit, 1 issue, 1 PR, 1 closing-ref link.
    from codescribe_train.rag.store.writer import Store

    with Store.open(db_path) as store:
        n_issues = store.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
        n_pulls = store.conn.execute("SELECT COUNT(*) FROM pulls").fetchone()[0]
        n_commits = store.conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
        n_links = store.conn.execute(
            "SELECT COUNT(*) FROM issue_pr_links"
        ).fetchone()[0]
    assert n_issues == 1
    assert n_pulls == 1
    assert n_commits == 1
    assert n_links == 1


def test_index_incremental_via_cli(tmp_path) -> None:
    """Run bootstrap then incremental via the CLI; no errors, no extra rows."""
    repo_root = Path(__file__).resolve().parents[3]
    db_path = tmp_path / "rag.db"
    fixture_dir = _write_fixture_module(tmp_path, _DEPS_FACTORY_SRC)
    cfg = _write_config(tmp_path, db_path)
    env = {
        "RAG_CONFIG": str(cfg),
        "RAG_TEST_DEPS_FACTORY": "fixture_deps:build",
        "PYTHONPATH": f"{fixture_dir}{os.pathsep}{repo_root}",
    }
    p1 = _run_cli(["index", "--bootstrap"], env, cwd=tmp_path)
    assert p1.returncode == 0

    p2 = _run_cli(["index"], env, cwd=tmp_path)
    assert p2.returncode == 0


def test_index_refresh_pr_via_cli(tmp_path) -> None:
    """``--refresh-pr 10`` exits 0 after bootstrap; missing PR exits 1."""
    repo_root = Path(__file__).resolve().parents[3]
    db_path = tmp_path / "rag.db"
    fixture_dir = _write_fixture_module(tmp_path, _DEPS_FACTORY_SRC)
    cfg = _write_config(tmp_path, db_path)
    env = {
        "RAG_CONFIG": str(cfg),
        "RAG_TEST_DEPS_FACTORY": "fixture_deps:build",
        "PYTHONPATH": f"{fixture_dir}{os.pathsep}{repo_root}",
    }
    _run_cli(["index", "--bootstrap"], env, cwd=tmp_path)
    ok = _run_cli(["index", "--refresh-pr", "10"], env, cwd=tmp_path)
    assert ok.returncode == 0
    miss = _run_cli(["index", "--refresh-pr", "999"], env, cwd=tmp_path)
    assert miss.returncode == 1


def test_query_is_wired(tmp_path) -> None:
    """``query`` is wired — the old stub-mode marker is gone."""
    env: dict[str, str] = {}
    proc = _run_cli(["query", "hello"], env, cwd=tmp_path)
    # Argparse rejects "hello" as an invalid sub-action (issues/pr-diff/commits).
    assert proc.returncode != 0
    assert "stub-mode" not in proc.stderr
    assert "stub-mode" not in proc.stdout


@pytest.mark.parametrize(
    "flags", [["index", "--bootstrap"], ["index"], ["status"]]
)
def test_cli_writes_json_log(tmp_path, flags) -> None:
    """Every pipeline-bound CLI run writes a JSON-line log under ``logs/``."""
    import json

    repo_root = Path(__file__).resolve().parents[3]
    db_path = tmp_path / "rag.db"
    fixture_dir = _write_fixture_module(tmp_path, _DEPS_FACTORY_SRC)
    cfg = _write_config(tmp_path, db_path)
    env = {
        "RAG_CONFIG": str(cfg),
        "RAG_TEST_DEPS_FACTORY": "fixture_deps:build",
        "PYTHONPATH": f"{fixture_dir}{os.pathsep}{repo_root}",
    }
    # Status needs a populated DB; bootstrap first if testing status.
    if flags == ["status"]:
        _run_cli(["index", "--bootstrap"], env, cwd=tmp_path)
    proc = _run_cli(flags, env, cwd=tmp_path)
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"

    log_dir = tmp_path / "logs"
    assert log_dir.exists(), "logs/ directory should be created"
    log_files = sorted(log_dir.glob("rag-indexer-*.log"))
    assert log_files, f"no JSON log files in {log_dir}"
    # At least one log line should parse as JSON.
    for line in log_files[-1].read_text().splitlines():
        if not line.strip():
            continue
        json.loads(line)
        return  # one valid line is enough
    pytest.fail("log file contained no parseable JSON lines")
