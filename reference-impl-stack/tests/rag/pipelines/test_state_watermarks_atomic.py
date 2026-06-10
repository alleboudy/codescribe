"""State watermarks are written inside the same transaction as the batch.

Per ``codescribe_train/rag/AGENTS.md``: "State checkpoints
(last_seen_commit_sha, last_seen_issue_updated_at) are atomic with the
data upserts they correspond to."

This file asserts the post-conditions; the atomicity guarantee itself
comes from the ``Store.batch()`` context manager wrapping both the
upsert calls and the ``set_state`` call inside the per-batch helpers.
"""

from __future__ import annotations

import asyncio

from codescribe_train.rag.pipelines._common import (
    STATE_LAST_COMMIT_SHA,
    STATE_LAST_ISSUE_UPDATED_AT,
    IndexerDeps,
)
from codescribe_train.rag.pipelines.bootstrap import run_bootstrap
from codescribe_train.rag.store.writer import Store
from tests.rag.pipelines.fakes import (
    FakeEmbedder,
    FakeGitHubSource,
    FakeGitSource,
    make_commit,
    make_issue,
)


def test_bootstrap_advances_both_watermarks(tmp_path) -> None:
    """Both watermarks are set to the latest indexed values after bootstrap."""
    git = FakeGitSource(
        commits=[
            make_commit("1" * 40, message="m1", authored_at="2026-05-01T10:00:00Z"),
            make_commit("2" * 40, message="m2", authored_at="2026-05-02T10:00:00Z"),
        ]
    )
    gh = FakeGitHubSource(
        issues=[
            make_issue(1, updated_at="2026-05-01T10:00:00Z"),
            make_issue(2, updated_at="2026-05-02T10:00:00Z"),
            make_issue(3, updated_at="2026-05-03T10:00:00Z"),
        ]
    )
    deps = IndexerDeps(
        git=git, github=gh, embedder=FakeEmbedder(), store_path=tmp_path / "rag.db"
    )
    asyncio.run(run_bootstrap(config=None, deps=deps))

    with Store.open(deps.store_path) as store:
        sha = store.get_state(STATE_LAST_COMMIT_SHA)
        issue_at = store.get_state(STATE_LAST_ISSUE_UPDATED_AT)

    assert sha == "2" * 40
    assert issue_at == "2026-05-03T10:00:00Z"


def test_no_watermark_when_nothing_to_index(tmp_path) -> None:
    """Empty source set leaves the watermarks unset."""
    deps = IndexerDeps(
        git=FakeGitSource(),
        github=FakeGitHubSource(),
        embedder=FakeEmbedder(),
        store_path=tmp_path / "rag.db",
    )
    asyncio.run(run_bootstrap(config=None, deps=deps))
    with Store.open(deps.store_path) as store:
        assert store.get_state(STATE_LAST_COMMIT_SHA) is None
        assert store.get_state(STATE_LAST_ISSUE_UPDATED_AT) is None
