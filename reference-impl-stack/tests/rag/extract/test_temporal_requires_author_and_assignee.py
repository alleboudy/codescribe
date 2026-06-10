"""`pair_via_temporal` must not pair when PR author or issue assignees are missing.

Hard constraint from the design spec: "pair_via_temporal must not pair if
the PR author or the issue assignee is missing (None / empty)".
"""

from __future__ import annotations

from codescribe_train.rag.extract.pairing import pair_via_temporal
from codescribe_train.rag.sources.github_source import Issue, PullRequest


def _issue(assignees: list[str]) -> Issue:
    return Issue(
        number=1,
        title="t",
        body=None,
        state="closed",
        state_reason="completed",
        labels=[],
        assignees=assignees,
        author="reporter",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-05-01T12:00:00Z",
        closed_at="2026-05-01T12:00:00Z",
        raw_json="{}",
    )


def _pr(author: str | None) -> PullRequest:
    return PullRequest(
        number=100,
        title="t",
        body=None,
        state="merged",
        head_sha="d" * 40,
        base_branch="main",
        author=author,
        draft=False,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-05-03T12:00:00Z",
        merged_at="2026-05-03T12:00:00Z",
        closed_at="2026-05-03T12:00:00Z",
        raw_json="{}",
    )


def test_missing_pr_author_blocks_pair() -> None:
    """PR.author == None → no temporal pair."""
    assert pair_via_temporal([_issue(["alice"])], [_pr(None)]) == []


def test_empty_assignees_blocks_pair() -> None:
    """Issue.assignees == [] → no temporal pair."""
    assert pair_via_temporal([_issue([])], [_pr("alice")]) == []


def test_assignee_login_empty_string_does_not_match_any_author() -> None:
    """An assignee literal of '' must not match a None or '' author."""
    assert pair_via_temporal([_issue([""])], [_pr(None)]) == []
    assert pair_via_temporal([_issue([""])], [_pr("")]) == []
