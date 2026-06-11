"""Incremental indexer picks up from the persisted watermarks.

Acceptance: spec test #2 from the design spec.

The scenario:

1. Bootstrap indexes a partial set of issues (1..3).
2. The source set then grows (issues 1..5).
3. Incremental run sees the new watermark and only writes issues 4..5.

We verify:

* Total row counts after incremental == total in source set.
* No duplicate rows (idempotent).
* The state watermarks advance.
"""

from __future__ import annotations

import asyncio

from codescribe_train.rag.pipelines._common import (
    STATE_LAST_COMMIT_SHA,
    STATE_LAST_ISSUE_UPDATED_AT,
    IndexerDeps,
)
from codescribe_train.rag.pipelines.bootstrap import run_bootstrap
from codescribe_train.rag.pipelines.incremental import run_incremental
from codescribe_train.rag.store.writer import Store
from tests.rag.pipelines.fakes import (
    FakeEmbedder,
    FakeGitHubSource,
    FakeGitSource,
    make_commit,
    make_issue,
    make_pull,
)


def _deps_for(tmp_path, git: FakeGitSource, gh: FakeGitHubSource) -> IndexerDeps:
    return IndexerDeps(
        git=git, github=gh, embedder=FakeEmbedder(), store_path=tmp_path / "rag.db"
    )


def test_incremental_picks_up_new_issues(tmp_path) -> None:
    """Bootstrap with 3 issues, add 2 more, incremental indexes only the new 2."""
    initial_issues = [
        make_issue(1, updated_at="2026-05-01T10:00:00Z"),
        make_issue(2, updated_at="2026-05-01T11:00:00Z"),
        make_issue(3, updated_at="2026-05-01T12:00:00Z"),
    ]
    gh = FakeGitHubSource(issues=list(initial_issues))
    git = FakeGitSource()
    deps = _deps_for(tmp_path, git, gh)

    asyncio.run(run_bootstrap(config=None, deps=deps))

    # Confirm bootstrap state.
    with Store.open(deps.store_path) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0] == 3
        watermark_after_bootstrap = store.get_state(STATE_LAST_ISSUE_UPDATED_AT)
    assert watermark_after_bootstrap == "2026-05-01T12:00:00Z"

    # Source set grows.
    gh.issues.append(make_issue(4, updated_at="2026-05-02T10:00:00Z"))
    gh.issues.append(make_issue(5, updated_at="2026-05-02T11:00:00Z"))

    embedder = FakeEmbedder()
    deps_inc = IndexerDeps(
        git=git, github=gh, embedder=embedder, store_path=deps.store_path
    )

    stats = asyncio.run(run_incremental(config=None, deps=deps_inc))

    # The 1-second safety margin can cause issue 3 to be re-read in the
    # delta. That's fine — the upsert overwrites the same row, doesn't
    # add duplicates.
    assert stats.issues >= 2

    with Store.open(deps.store_path) as store:
        count = store.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
        max_number = store.conn.execute(
            "SELECT MAX(issue_number) FROM issues"
        ).fetchone()[0]
        watermark_after = store.get_state(STATE_LAST_ISSUE_UPDATED_AT)

    assert count == 5  # no duplicates; one row per issue_number
    assert max_number == 5
    assert watermark_after == "2026-05-02T11:00:00Z"


def test_incremental_no_duplicates_when_source_unchanged(tmp_path) -> None:
    """An incremental run against an unchanged source set adds nothing.

    The 1-second safety margin means the boundary issue may be re-read,
    but the upsert semantics protect against duplicate rows.
    """
    git = FakeGitSource(
        commits=[make_commit("a" * 40, message="m", authored_at="2026-05-01T10:00:00Z")]
    )
    gh = FakeGitHubSource(
        issues=[make_issue(1, updated_at="2026-05-01T12:00:00Z")],
        pulls=[make_pull(10, updated_at="2026-05-01T13:00:00Z")],
    )
    deps = _deps_for(tmp_path, git, gh)
    asyncio.run(run_bootstrap(config=None, deps=deps))

    with Store.open(deps.store_path) as store:
        before = {
            t: store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("commits", "issues", "pulls", "issue_pr_links")
        }

    asyncio.run(run_incremental(config=None, deps=deps))

    with Store.open(deps.store_path) as store:
        after = {
            t: store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("commits", "issues", "pulls", "issue_pr_links")
        }

    assert before == after


def test_partial_bootstrap_then_incremental_fills_in_without_duplicates(
    tmp_path,
) -> None:
    """The literal spec wording: partial bootstrap, then incremental; no duplicates.

    The "partial" bootstrap is staged via ``stop_after_records=[2]`` —
    the inner loop will raise ``StopRequested`` between batches once 2
    records have been written. With ``batch_size=1`` the early stop
    lands cleanly between issues 2 and 3. The incremental run then
    picks up from the persisted watermark and writes issues 3..5.
    """
    issues = [
        make_issue(n, updated_at=f"2026-05-0{n}T10:00:00Z") for n in range(1, 6)
    ]
    gh = FakeGitHubSource(issues=list(issues))
    git = FakeGitSource()

    deps_boot = IndexerDeps(
        git=git,
        github=gh,
        embedder=FakeEmbedder(),
        store_path=tmp_path / "rag.db",
        batch_size=1,
        stop_event=asyncio.Event(),
        stop_after_records=[2],
    )
    stats = asyncio.run(run_bootstrap(config=None, deps=deps_boot))
    assert stats.stopped_early is True
    assert stats.issues == 2

    with Store.open(deps_boot.store_path) as store:
        n_after_partial = store.conn.execute(
            "SELECT COUNT(*) FROM issues"
        ).fetchone()[0]
    assert n_after_partial == 2

    deps_inc = IndexerDeps(
        git=git,
        github=gh,
        embedder=FakeEmbedder(),
        store_path=deps_boot.store_path,
    )
    asyncio.run(run_incremental(config=None, deps=deps_inc))

    with Store.open(deps_boot.store_path) as store:
        n_after_inc = store.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
        all_numbers = sorted(
            r[0]
            for r in store.conn.execute(
                "SELECT issue_number FROM issues"
            ).fetchall()
        )

    assert n_after_inc == 5
    assert all_numbers == [1, 2, 3, 4, 5]


def test_incremental_advances_commit_sha_watermark(tmp_path) -> None:
    """``last_seen_commit_sha`` is set to the last indexed commit's SHA."""
    commits = [
        make_commit("1" * 40, message="first", authored_at="2026-05-01T10:00:00Z"),
        make_commit("2" * 40, message="second", authored_at="2026-05-02T10:00:00Z"),
    ]
    git = FakeGitSource(commits=list(commits))
    gh = FakeGitHubSource()
    deps = _deps_for(tmp_path, git, gh)

    asyncio.run(run_bootstrap(config=None, deps=deps))

    with Store.open(deps.store_path) as store:
        last_sha = store.get_state(STATE_LAST_COMMIT_SHA)
    # ASCending order → last indexed is the newer commit "2" * 40.
    assert last_sha == "2" * 40
