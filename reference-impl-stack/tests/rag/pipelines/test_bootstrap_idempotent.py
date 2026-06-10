"""Bootstrap idempotency — re-running adds zero rows.

Acceptance: spec test #1 from the design spec.
"""

from __future__ import annotations

import asyncio

from codescribe_train.rag.pipelines.bootstrap import BootstrapStats, run_bootstrap
from codescribe_train.rag.store.writer import Store
from tests.rag.pipelines.fakes import (
    FakeEmbedder,
    FakeGitHubSource,
    FakeGitSource,
    make_commit,
    make_issue,
    make_pull,
)


def _populated_sources():
    """Build a small but non-trivial source set."""
    git = FakeGitSource(
        commits=[
            make_commit("a" * 40, message="initial", authored_at="2026-05-01T10:00:00Z"),
            make_commit("b" * 40, message="follow-up", authored_at="2026-05-02T10:00:00Z"),
        ]
    )
    gh = FakeGitHubSource(
        issues=[
            make_issue(1, title="bug", updated_at="2026-05-01T11:00:00Z"),
            make_issue(2, title="feature", updated_at="2026-05-02T11:00:00Z"),
        ],
        pulls=[
            make_pull(10, title="fix bug", updated_at="2026-05-02T12:00:00Z"),
        ],
        pr_diffs={
            10: (
                "diff --git a/src/foo.py b/src/foo.py\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            ),
        },
        closing_refs=[(10, 1)],
    )
    return git, gh


def _run(deps_factory) -> BootstrapStats:
    """Drive ``run_bootstrap`` with an injected deps factory."""
    return asyncio.run(_run_async(deps_factory))


async def _run_async(deps_factory) -> BootstrapStats:
    return await run_bootstrap(config=None, deps=deps_factory())


def test_bootstrap_empty_sources_writes_nothing(tmp_path) -> None:
    """An empty source set produces zero counts and an empty store."""
    from codescribe_train.rag.pipelines._common import IndexerDeps

    db_path = tmp_path / "rag.db"

    def deps_factory() -> IndexerDeps:
        return IndexerDeps(
            git=FakeGitSource(commits=[]),
            github=FakeGitHubSource(),
            embedder=FakeEmbedder(),
            store_path=db_path,
        )

    stats = _run(deps_factory)

    assert stats.commits == 0
    assert stats.issues == 0
    assert stats.pulls == 0
    assert stats.links == 0

    with Store.open(db_path) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0] == 0
        assert store.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0] == 0
        assert store.conn.execute("SELECT COUNT(*) FROM pulls").fetchone()[0] == 0
        assert store.conn.execute("SELECT COUNT(*) FROM issue_pr_links").fetchone()[0] == 0


def test_bootstrap_indexes_all_records(tmp_path) -> None:
    """First bootstrap on a populated source set persists all records."""
    from codescribe_train.rag.pipelines._common import IndexerDeps

    db_path = tmp_path / "rag.db"
    git, gh = _populated_sources()

    def deps_factory() -> IndexerDeps:
        return IndexerDeps(
            git=git, github=gh, embedder=FakeEmbedder(), store_path=db_path
        )

    stats = _run(deps_factory)

    assert stats.commits == 2
    assert stats.issues == 2
    assert stats.pulls == 1
    # One closing-ref → exactly one link.
    assert stats.links >= 1

    with Store.open(db_path) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0] == 2
        assert store.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0] == 2
        assert store.conn.execute("SELECT COUNT(*) FROM pulls").fetchone()[0] == 1
        # The closing reference (PR 10 → issue 1) lands at confidence 1.0.
        link = store.conn.execute(
            "SELECT issue_number, pr_number, confidence FROM issue_pr_links"
        ).fetchall()
        assert (1, 10, 1.0) in link


def test_bootstrap_twice_is_idempotent(tmp_path) -> None:
    """Re-running bootstrap on the same source set adds zero new rows."""
    from codescribe_train.rag.pipelines._common import IndexerDeps

    db_path = tmp_path / "rag.db"
    git, gh = _populated_sources()

    def deps_factory() -> IndexerDeps:
        return IndexerDeps(
            git=git, github=gh, embedder=FakeEmbedder(), store_path=db_path
        )

    _run(deps_factory)
    # Snapshot row counts after the first run.
    with Store.open(db_path) as store:
        before = {
            t: store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("commits", "issues", "pulls", "issue_pr_links", "pr_chunks")
        }

    _run(deps_factory)

    with Store.open(db_path) as store:
        after = {
            t: store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("commits", "issues", "pulls", "issue_pr_links", "pr_chunks")
        }

    assert before == after, f"second bootstrap added rows: before={before} after={after}"
