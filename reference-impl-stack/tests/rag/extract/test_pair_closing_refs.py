"""`pair_via_closing_refs` — gold-confidence path (GraphQL `closingIssuesReferences`).

Per the design spec's pairing table: source="closingIssuesReferences", confidence=1.0,
signals={"closing_issues_reference": 1.0}.

The function consumes the iterable yielded by
`GitHubSource.iter_closing_references()`, which emits ``(pr_number,
issue_number)`` tuples.
"""

from __future__ import annotations

from codescribe_train.rag.extract.pairing import IssuePRLink, pair_via_closing_refs


def test_pair_via_closing_refs_two_links_for_one_pr() -> None:
    """Spec fixture: (pr=12, issue=100) and (pr=12, issue=101) → two links."""
    links = pair_via_closing_refs([(12, 100), (12, 101)])

    assert len(links) == 2
    assert all(isinstance(link, IssuePRLink) for link in links)
    assert all(link.confidence == 1.0 for link in links)
    assert all(link.source == "closingIssuesReferences" for link in links)

    pairs = {(link.issue_number, link.pr_number) for link in links}
    assert pairs == {(100, 12), (101, 12)}


def test_pair_via_closing_refs_signals_recorded() -> None:
    [link] = pair_via_closing_refs([(7, 1)])
    assert link.signals == {"closing_issues_reference": 1.0}


def test_pair_via_closing_refs_empty_iterable_returns_empty_list() -> None:
    assert pair_via_closing_refs([]) == []


def test_pair_via_closing_refs_dedupes_identical_tuples() -> None:
    """Two identical (pr, issue) tuples must collapse to one IssuePRLink.

    The GraphQL endpoint can theoretically emit the same pair twice across
    pagination boundaries; the pairing step normalises that out so the
    pipeline doesn't double-write.
    """
    links = pair_via_closing_refs([(12, 100), (12, 100)])
    assert len(links) == 1
    assert links[0].issue_number == 100
    assert links[0].pr_number == 12
