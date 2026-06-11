"""Issue indexed before its PR; later PR + closing-ref pairs them up.

Acceptance: spec test #3 from the design spec.

Stages two runs against the same store:

1. First run: only the issue exists in the source set. The pairing pass
   has no PRs and no closing refs to chew on → zero links.
2. Second run: the PR shows up and the GraphQL closing-ref iterator
   yields ``(pr_number, issue_number)``. The pairing post-pass upserts
   the link at confidence 1.0.

The link must reference the *original* issue number — i.e. the issue
that was indexed in run 1 — so the pairing pass needs to see the
already-stored issue, not just the freshly indexed ones.
"""

from __future__ import annotations

import asyncio

from codescribe_train.rag.pipelines._common import IndexerDeps
from codescribe_train.rag.pipelines.bootstrap import run_bootstrap
from codescribe_train.rag.pipelines.incremental import run_incremental
from codescribe_train.rag.store.writer import Store
from tests.rag.pipelines.fakes import (
    FakeEmbedder,
    FakeGitHubSource,
    FakeGitSource,
    make_issue,
    make_pull,
)


def test_late_arriving_closing_ref_creates_link(tmp_path) -> None:
    """Issue indexed in run 1; PR + closing-ref arrives in run 2."""
    gh = FakeGitHubSource(
        issues=[make_issue(7, title="bug", updated_at="2026-05-01T10:00:00Z")],
        pulls=[],
        closing_refs=[],  # GraphQL pass yields nothing yet
    )
    git = FakeGitSource()
    deps = IndexerDeps(
        git=git, github=gh, embedder=FakeEmbedder(), store_path=tmp_path / "rag.db"
    )

    asyncio.run(run_bootstrap(config=None, deps=deps))

    with Store.open(deps.store_path) as store:
        n_links = store.conn.execute(
            "SELECT COUNT(*) FROM issue_pr_links"
        ).fetchone()[0]
    assert n_links == 0

    # Stage 2: PR arrives, with a closing reference back to issue 7.
    gh.pulls.append(
        make_pull(
            42,
            title="fix bug",
            body="implements the fix",
            updated_at="2026-05-02T10:00:00Z",
            merged_at="2026-05-02T10:00:00Z",
        )
    )
    gh.closing_refs.append((42, 7))

    asyncio.run(run_incremental(config=None, deps=deps))

    with Store.open(deps.store_path) as store:
        link_rows = store.conn.execute(
            "SELECT issue_number, pr_number, confidence, source "
            "FROM issue_pr_links"
        ).fetchall()
    assert (7, 42, 1.0, "closingIssuesReferences") in link_rows


def test_late_closing_ref_overrides_lower_confidence_body_link(tmp_path) -> None:
    """A PR that *first* has only a body-keyword link gains a 1.0 link later.

    Stage 1: PR has a body that says ``Closes #7`` but no GraphQL closing
    ref yet. The body-keyword pass produces a 0.85 link.
    Stage 2: GraphQL closing ref arrives. Higher-confidence-wins on
    :meth:`Store.upsert_issue_pr_link` overrides the 0.85 link with the
    1.0 closingIssuesReferences row.
    """
    gh = FakeGitHubSource(
        issues=[make_issue(7, updated_at="2026-05-01T10:00:00Z")],
        pulls=[
            make_pull(
                42,
                body="Closes #7. body keyword pairing should fire.",
                updated_at="2026-05-02T10:00:00Z",
            )
        ],
        closing_refs=[],
    )
    git = FakeGitSource()
    deps = IndexerDeps(
        git=git, github=gh, embedder=FakeEmbedder(), store_path=tmp_path / "rag.db"
    )
    asyncio.run(run_bootstrap(config=None, deps=deps))

    with Store.open(deps.store_path) as store:
        row = store.conn.execute(
            "SELECT confidence, source FROM issue_pr_links "
            "WHERE issue_number = 7 AND pr_number = 42"
        ).fetchone()
    assert row == (0.85, "body_closes_keyword")

    # GraphQL closing ref arrives.
    gh.closing_refs.append((42, 7))
    asyncio.run(run_incremental(config=None, deps=deps))

    with Store.open(deps.store_path) as store:
        row = store.conn.execute(
            "SELECT confidence, source FROM issue_pr_links "
            "WHERE issue_number = 7 AND pr_number = 42"
        ).fetchone()
    # Confidence promoted to 1.0; source field carries the comma-joined
    # set of contributing signals (merge_links is deterministic in
    # alphabetical order). Both signals fire on the second run.
    assert row[0] == 1.0
    sources = set(row[1].split(","))
    assert "closingIssuesReferences" in sources
