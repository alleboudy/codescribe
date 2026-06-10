"""`filter_by_confidence` — strict-threshold filter for the pairing table.

Per the design spec spec test "test_threshold_strict": running `merge_links` +
filter with threshold 0.8 keeps 1.0 + 0.85, drops 0.5.
"""

from __future__ import annotations

from codescribe_train.rag.extract.pairing import (
    IssuePRLink,
    filter_by_confidence,
    merge_links,
)


def _link(issue: int, pr: int, source: str, conf: float) -> IssuePRLink:
    signal_key = {
        "closingIssuesReferences": "closing_issues_reference",
        "body_closes_keyword": "body_closes_keyword",
        "temporal_author_match": "temporal_author_match",
    }[source]
    return IssuePRLink(
        issue_number=issue,
        pr_number=pr,
        source=source,
        confidence=conf,
        signals={signal_key: conf},
    )


def test_strict_threshold_keeps_1_0_and_0_85_drops_0_5() -> None:
    """Spec fixture: 3 distinct pairs at 1.0, 0.85, 0.5 → threshold 0.8 keeps 2."""
    refs = [_link(1, 10, "closingIssuesReferences", 1.0)]
    body = [_link(2, 20, "body_closes_keyword", 0.85)]
    temporal = [_link(3, 30, "temporal_author_match", 0.5)]

    merged = merge_links(refs, body, temporal)
    kept = filter_by_confidence(merged, threshold=0.8)

    confidences = sorted(link.confidence for link in kept)
    assert confidences == [0.85, 1.0]
    kept_pairs = {(link.issue_number, link.pr_number) for link in kept}
    assert kept_pairs == {(1, 10), (2, 20)}


def test_threshold_at_exact_boundary_keeps_equal_confidence() -> None:
    """A confidence equal to the threshold passes (>= semantics)."""
    links = [_link(1, 10, "body_closes_keyword", 0.85)]
    assert filter_by_confidence(links, threshold=0.85) == links


def test_threshold_above_max_drops_everything() -> None:
    links = [_link(1, 10, "closingIssuesReferences", 1.0)]
    assert filter_by_confidence(links, threshold=1.01) == []


def test_empty_input_returns_empty() -> None:
    assert filter_by_confidence([], threshold=0.8) == []
