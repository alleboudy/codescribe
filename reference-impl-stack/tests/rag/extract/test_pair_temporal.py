"""`pair_via_temporal` — confidence 0.5 path (temporal + author match).

Per the design spec: for PRs merged within 7 days of an issue close, where the PR
author is one of the issue's assignees, emit a link with confidence 0.5,
source="temporal_author_match", signals={"temporal_author_match": 0.5}.

Drafts and rows missing the PR author or all issue assignees must NOT
produce a link (drafts: anti-pattern #5; missing author/assignee: a
hard constraint in the design spec).
"""

from __future__ import annotations

from codescribe_train.rag.extract.pairing import IssuePRLink, pair_via_temporal
from codescribe_train.rag.sources.github_source import Issue, PullRequest


def _issue(
    number: int,
    *,
    closed_at: str | None,
    assignees: list[str] | None = None,
    state: str = "closed",
) -> Issue:
    return Issue(
        number=number,
        title=f"issue #{number}",
        body=None,
        state=state,
        state_reason="completed" if state == "closed" else None,
        labels=[],
        assignees=assignees or [],
        author="reporter",
        created_at="2026-01-01T00:00:00Z",
        updated_at=closed_at or "2026-01-01T00:00:00Z",
        closed_at=closed_at,
        raw_json="{}",
    )


def _pr(
    number: int,
    *,
    merged_at: str | None,
    author: str | None = "alice",
    draft: bool = False,
    body: str | None = None,
) -> PullRequest:
    return PullRequest(
        number=number,
        title=f"PR #{number}",
        body=body,
        state="merged" if merged_at else "open",
        head_sha="deadbeef" * 5,
        base_branch="main",
        author=author,
        draft=draft,
        created_at="2026-01-01T00:00:00Z",
        updated_at=merged_at or "2026-01-01T00:00:00Z",
        merged_at=merged_at,
        closed_at=merged_at,
        raw_json="{}",
    )


# --- spec fixtures -----------------------------------------------------


def test_pr_merged_within_window_with_assignee_match_yields_link() -> None:
    """Spec: issue closed T, PR merged T+5d by the assignee → conf 0.5."""
    issue = _issue(1, closed_at="2026-05-01T12:00:00Z", assignees=["alice"])
    pr = _pr(100, merged_at="2026-05-06T12:00:00Z", author="alice")

    [link] = pair_via_temporal([issue], [pr])

    assert isinstance(link, IssuePRLink)
    assert link.issue_number == 1
    assert link.pr_number == 100
    assert link.confidence == 0.5
    assert link.source == "temporal_author_match"
    assert link.signals == {"temporal_author_match": 0.5}


def test_pr_merged_outside_window_yields_no_link() -> None:
    """Spec: T+30d is well outside the 7-day window → no link."""
    issue = _issue(1, closed_at="2026-05-01T12:00:00Z", assignees=["alice"])
    pr = _pr(100, merged_at="2026-05-31T12:00:00Z", author="alice")

    assert pair_via_temporal([issue], [pr]) == []


# --- window edges ------------------------------------------------------


def test_pr_merged_just_inside_window_is_paired() -> None:
    """Exactly 7 days (within tolerance) should still pair."""
    issue = _issue(1, closed_at="2026-05-01T00:00:00Z", assignees=["alice"])
    pr = _pr(100, merged_at="2026-05-08T00:00:00Z", author="alice")
    links = pair_via_temporal([issue], [pr])
    assert len(links) == 1


def test_pr_merged_before_issue_close_within_window_is_paired() -> None:
    """The window is bidirectional — PR can be merged before issue closes."""
    issue = _issue(1, closed_at="2026-05-08T00:00:00Z", assignees=["alice"])
    pr = _pr(100, merged_at="2026-05-01T00:00:00Z", author="alice")
    links = pair_via_temporal([issue], [pr])
    assert len(links) == 1


# --- exclusions --------------------------------------------------------


def test_open_issue_yields_no_link() -> None:
    """An issue without `closed_at` cannot anchor a temporal pair."""
    issue = _issue(1, closed_at=None, assignees=["alice"], state="open")
    pr = _pr(100, merged_at="2026-05-06T12:00:00Z", author="alice")
    assert pair_via_temporal([issue], [pr]) == []


def test_unmerged_pr_yields_no_link() -> None:
    """A PR without `merged_at` cannot anchor a temporal pair."""
    issue = _issue(1, closed_at="2026-05-01T12:00:00Z", assignees=["alice"])
    pr = _pr(100, merged_at=None, author="alice")
    assert pair_via_temporal([issue], [pr]) == []


def test_draft_pr_yields_no_link_even_when_temporally_matched() -> None:
    """Anti-pattern #5: drafts excluded from pairing."""
    issue = _issue(1, closed_at="2026-05-01T12:00:00Z", assignees=["alice"])
    pr = _pr(100, merged_at="2026-05-06T12:00:00Z", author="alice", draft=True)
    assert pair_via_temporal([issue], [pr]) == []


def test_author_must_be_one_of_the_assignees() -> None:
    """Author 'bob' is not an assignee — no link."""
    issue = _issue(1, closed_at="2026-05-01T12:00:00Z", assignees=["alice"])
    pr = _pr(100, merged_at="2026-05-06T12:00:00Z", author="bob")
    assert pair_via_temporal([issue], [pr]) == []


def test_custom_window_days_is_respected() -> None:
    """Override window_days=2 → a 5-day gap is now out of window."""
    issue = _issue(1, closed_at="2026-05-01T12:00:00Z", assignees=["alice"])
    pr = _pr(100, merged_at="2026-05-06T12:00:00Z", author="alice")
    assert pair_via_temporal([issue], [pr], window_days=2) == []


def test_multiple_assignees_one_matching_yields_link() -> None:
    issue = _issue(
        1, closed_at="2026-05-01T12:00:00Z", assignees=["bob", "alice", "carol"]
    )
    pr = _pr(100, merged_at="2026-05-03T12:00:00Z", author="alice")
    assert len(pair_via_temporal([issue], [pr])) == 1
