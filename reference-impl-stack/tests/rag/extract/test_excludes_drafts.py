"""Drafts (`PullRequest.draft == True`) must never produce a pair.

Per `codescribe_train/rag/sources/AGENTS.md` anti-pattern #5: "Do NOT include
drafts (PRs with `draft: true`) in the high-confidence corpus."

This sits across both `pair_via_body_keyword` and `pair_via_temporal` —
the two paths that consume `PullRequest` directly.
(`pair_via_closing_refs` works off tuples emitted by the GraphQL probe;
the probe itself already filters drafts, so there's no draft
signal to test here.)
"""

from __future__ import annotations

from codescribe_train.rag.extract.pairing import pair_via_body_keyword, pair_via_temporal
from codescribe_train.rag.sources.github_source import Issue, PullRequest


def _draft_pr(body: str | None, *, merged_at: str | None) -> PullRequest:
    return PullRequest(
        number=42,
        title="draft",
        body=body,
        state="open",  # drafts are by definition not yet merged
        head_sha="d" * 40,
        base_branch="main",
        author="alice",
        draft=True,
        created_at="2026-04-30T12:00:00Z",
        updated_at="2026-05-01T12:00:00Z",
        merged_at=merged_at,
        closed_at=merged_at,
        raw_json="{}",
    )


def _issue() -> Issue:
    return Issue(
        number=1,
        title="t",
        body=None,
        state="closed",
        state_reason="completed",
        labels=[],
        assignees=["alice"],
        author="reporter",
        created_at="2026-04-30T12:00:00Z",
        updated_at="2026-05-01T12:00:00Z",
        closed_at="2026-05-01T12:00:00Z",
        raw_json="{}",
    )


def test_draft_pr_excluded_from_body_keyword_pairing() -> None:
    """Even with `Closes #1` in the body, a draft PR yields no link."""
    pr = _draft_pr("Closes #1", merged_at=None)
    assert pair_via_body_keyword([pr]) == []


def test_draft_pr_excluded_from_temporal_pairing() -> None:
    """A draft PR matching every other temporal criterion yields no link."""
    pr = _draft_pr(None, merged_at="2026-05-01T18:00:00Z")
    assert pair_via_temporal([_issue()], [pr]) == []
