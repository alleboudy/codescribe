"""Single-PR refresh path.

Acceptance: spec test #4 from the design spec.

``--refresh-pr 42`` is the operator-facing escape hatch for after-merge
manual refresh. It must re-pull just PR 42 and leave the rest of the
store untouched.

The mirroring test for ``--refresh-issue N`` lives in the same file.
"""

from __future__ import annotations

import asyncio

from codescribe_train.rag.pipelines._common import IndexerDeps, refresh_issue, refresh_pr
from codescribe_train.rag.pipelines.bootstrap import run_bootstrap
from codescribe_train.rag.store.writer import Store
from tests.rag.pipelines.fakes import (
    FakeEmbedder,
    FakeGitHubSource,
    FakeGitSource,
    make_issue,
    make_pull,
)


def _bootstrap_world(tmp_path):
    """Index two PRs + two issues so the refresh tests can compare deltas."""
    gh = FakeGitHubSource(
        issues=[
            make_issue(1, title="bug", updated_at="2026-05-01T10:00:00Z"),
            make_issue(2, title="feature", updated_at="2026-05-01T11:00:00Z"),
        ],
        pulls=[
            make_pull(41, title="fix bug v1", updated_at="2026-05-02T10:00:00Z"),
            make_pull(42, title="add feature v1", updated_at="2026-05-02T11:00:00Z"),
        ],
    )
    git = FakeGitSource()
    deps = IndexerDeps(
        git=git, github=gh, embedder=FakeEmbedder(), store_path=tmp_path / "rag.db"
    )
    asyncio.run(run_bootstrap(config=None, deps=deps))
    return deps, gh


def test_refresh_pr_updates_only_target_pr(tmp_path) -> None:
    deps, gh = _bootstrap_world(tmp_path)

    # Snapshot every record's title (so we can prove only PR 42 changed).
    with Store.open(deps.store_path) as store:
        before = dict(
            store.conn.execute(
                "SELECT pr_number, title FROM pulls ORDER BY pr_number"
            ).fetchall()
        )
        issue_titles_before = dict(
            store.conn.execute(
                "SELECT issue_number, title FROM issues ORDER BY issue_number"
            ).fetchall()
        )

    # Source mutates PR 42's title; PR 41 + issues stay identical.
    for i, pr in enumerate(gh.pulls):
        if pr.number == 42:
            gh.pulls[i] = make_pull(
                42, title="add feature v2 - REFRESHED", updated_at="2026-05-03T10:00:00Z"
            )

    found = asyncio.run(refresh_pr(deps, 42))
    assert found is True

    with Store.open(deps.store_path) as store:
        after = dict(
            store.conn.execute(
                "SELECT pr_number, title FROM pulls ORDER BY pr_number"
            ).fetchall()
        )
        issue_titles_after = dict(
            store.conn.execute(
                "SELECT issue_number, title FROM issues ORDER BY issue_number"
            ).fetchall()
        )

    assert after[41] == before[41]  # unchanged
    assert after[42] == "add feature v2 - REFRESHED"
    assert issue_titles_after == issue_titles_before  # unchanged
    # No phantom rows added.
    assert set(after.keys()) == set(before.keys())


def test_refresh_pr_missing_returns_false(tmp_path) -> None:
    deps, _ = _bootstrap_world(tmp_path)
    found = asyncio.run(refresh_pr(deps, 999))
    assert found is False


def test_refresh_issue_updates_only_target_issue(tmp_path) -> None:
    deps, gh = _bootstrap_world(tmp_path)

    with Store.open(deps.store_path) as store:
        before_titles = dict(
            store.conn.execute(
                "SELECT issue_number, title FROM issues ORDER BY issue_number"
            ).fetchall()
        )
        pr_titles_before = dict(
            store.conn.execute(
                "SELECT pr_number, title FROM pulls ORDER BY pr_number"
            ).fetchall()
        )

    for i, issue in enumerate(gh.issues):
        if issue.number == 2:
            gh.issues[i] = make_issue(
                2, title="feature REFRESHED", updated_at="2026-05-03T10:00:00Z"
            )

    found = asyncio.run(refresh_issue(deps, 2))
    assert found is True

    with Store.open(deps.store_path) as store:
        after_titles = dict(
            store.conn.execute(
                "SELECT issue_number, title FROM issues ORDER BY issue_number"
            ).fetchall()
        )
        pr_titles_after = dict(
            store.conn.execute(
                "SELECT pr_number, title FROM pulls ORDER BY pr_number"
            ).fetchall()
        )

    assert after_titles[1] == before_titles[1]
    assert after_titles[2] == "feature REFRESHED"
    assert pr_titles_after == pr_titles_before
