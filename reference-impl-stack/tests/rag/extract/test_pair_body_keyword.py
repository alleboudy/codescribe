"""`pair_via_body_keyword` — confidence 0.85 path (PR-body regex).

Per the design spec: source="body_closes_keyword", confidence=0.85,
signals={"body_closes_keyword": 0.85}.

Draft PRs are excluded per `codescribe_train/rag/sources/AGENTS.md` anti-pattern
#5 ("Do NOT include drafts (PRs with `draft: true`) in the high-confidence
corpus.").
"""

from __future__ import annotations

from codescribe_train.rag.extract.pairing import IssuePRLink, pair_via_body_keyword
from codescribe_train.rag.sources.github_source import PullRequest


def _pr(
    number: int,
    body: str | None,
    *,
    draft: bool = False,
    author: str | None = "alice",
    merged_at: str | None = "2026-05-01T12:00:00Z",
) -> PullRequest:
    """Build a `PullRequest` with the fields pairing actually reads."""
    return PullRequest(
        number=number,
        title=f"PR #{number}",
        body=body,
        state="merged" if merged_at else "open",
        head_sha="deadbeef" * 5,
        base_branch="main",
        author=author,
        draft=draft,
        created_at="2026-04-30T12:00:00Z",
        updated_at="2026-05-01T12:00:00Z",
        merged_at=merged_at,
        closed_at=merged_at,
        raw_json="{}",
    )


def test_single_pr_with_one_closes_link_conf_085() -> None:
    """Spec fixture: PR body 'Closes #5' → one link, confidence 0.85."""
    pr = _pr(42, "Closes #5")

    [link] = pair_via_body_keyword([pr])

    assert isinstance(link, IssuePRLink)
    assert link.issue_number == 5
    assert link.pr_number == 42
    assert link.confidence == 0.85
    assert link.source == "body_closes_keyword"
    assert link.signals == {"body_closes_keyword": 0.85}


def test_pr_with_multiple_keyword_refs_yields_one_link_per_issue() -> None:
    pr = _pr(7, "Closes #1\nFixes #2\nresolves #3\n")
    links = pair_via_body_keyword([pr])
    pairs = sorted((link.issue_number, link.pr_number) for link in links)
    assert pairs == [(1, 7), (2, 7), (3, 7)]
    assert {link.confidence for link in links} == {0.85}


def test_pr_with_no_keyword_refs_yields_no_link() -> None:
    pr = _pr(7, "Just refactoring some things.")
    assert pair_via_body_keyword([pr]) == []


def test_pr_with_none_body_yields_no_link() -> None:
    pr = _pr(7, None)
    assert pair_via_body_keyword([pr]) == []


def test_pr_with_empty_body_yields_no_link() -> None:
    pr = _pr(7, "")
    assert pair_via_body_keyword([pr]) == []


def test_duplicate_keyword_refs_in_one_body_dedupe_to_one_link() -> None:
    pr = _pr(7, "Closes #1\nFixes #1\n")
    links = pair_via_body_keyword([pr])
    assert len(links) == 1
    assert (links[0].issue_number, links[0].pr_number) == (1, 7)


def test_multiple_prs_yield_links_per_pr() -> None:
    prs = [_pr(10, "Closes #100"), _pr(20, "Fixes #200")]
    links = pair_via_body_keyword(prs)
    pairs = sorted((link.issue_number, link.pr_number) for link in links)
    assert pairs == [(100, 10), (200, 20)]
