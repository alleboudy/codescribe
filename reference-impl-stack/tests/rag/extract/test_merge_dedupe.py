"""`merge_links` — collapse duplicates by (issue, PR); keep MAX confidence.

Per the design spec: "collapses duplicates by (issue_number, pr_number), keeping
the HIGHEST confidence + a comma-separated `source` field."
"""

from __future__ import annotations

from codescribe_train.rag.extract.pairing import IssuePRLink, merge_links

# The signals dict uses *signal keys* (not source strings) — these match
# the keys that `pair_via_*` actually emit. Spelled out here so the helper
# stays a thin wrapper that does not silently substitute a wrong key.
_SIGNAL_KEY_BY_SOURCE = {
    "closingIssuesReferences": "closing_issues_reference",
    "body_closes_keyword": "body_closes_keyword",
    "temporal_author_match": "temporal_author_match",
}


def _link(
    issue: int,
    pr: int,
    *,
    source: str,
    confidence: float,
    signals: dict[str, float] | None = None,
) -> IssuePRLink:
    if signals is None:
        signals = {_SIGNAL_KEY_BY_SOURCE[source]: confidence}
    return IssuePRLink(
        issue_number=issue,
        pr_number=pr,
        source=source,
        confidence=confidence,
        signals=signals,
    )


# --- spec fixture: three sources point at the same (issue, PR) --------


def test_three_sources_collapse_to_one_row_with_max_confidence() -> None:
    """Spec: three sources at the same (issue, PR) → one row, max conf, joined source."""
    a = _link(5, 100, source="closingIssuesReferences", confidence=1.0)
    b = _link(5, 100, source="body_closes_keyword", confidence=0.85)
    c = _link(5, 100, source="temporal_author_match", confidence=0.5)

    merged = merge_links([a], [b], [c])

    assert len(merged) == 1
    row = merged[0]
    assert (row.issue_number, row.pr_number) == (5, 100)
    assert row.confidence == 1.0  # MAX
    # Sources are joined with commas in sorted (deterministic) order.
    parts = row.source.split(",")
    assert sorted(parts) == [
        "body_closes_keyword",
        "closingIssuesReferences",
        "temporal_author_match",
    ]
    # Signals merge — each contributing path contributes its key.
    assert row.signals == {
        "closing_issues_reference": 1.0,
        "body_closes_keyword": 0.85,
        "temporal_author_match": 0.5,
    }


# --- edges -------------------------------------------------------------


def test_no_sources_returns_empty() -> None:
    assert merge_links() == []


def test_single_source_passes_through_unmodified() -> None:
    a = _link(5, 100, source="closingIssuesReferences", confidence=1.0)
    merged = merge_links([a])
    assert len(merged) == 1
    assert merged[0].confidence == 1.0
    assert merged[0].source == "closingIssuesReferences"


def test_distinct_pairs_are_preserved_independently() -> None:
    a = _link(1, 10, source="body_closes_keyword", confidence=0.85)
    b = _link(2, 10, source="body_closes_keyword", confidence=0.85)
    merged = merge_links([a, b])
    assert len(merged) == 2
    pairs = {(m.issue_number, m.pr_number) for m in merged}
    assert pairs == {(1, 10), (2, 10)}


def test_intra_group_duplicates_collapse_to_one() -> None:
    a = _link(5, 100, source="closingIssuesReferences", confidence=1.0)
    b = _link(5, 100, source="closingIssuesReferences", confidence=1.0)
    merged = merge_links([a, b])
    assert len(merged) == 1
    assert merged[0].source == "closingIssuesReferences"


def test_source_field_dedupes_within_join() -> None:
    """If the same source appears twice, it should appear only once in the joined string."""
    a = _link(5, 100, source="body_closes_keyword", confidence=0.85)
    b = _link(5, 100, source="body_closes_keyword", confidence=0.85)
    [row] = merge_links([a], [b])
    assert row.source == "body_closes_keyword"


def test_max_confidence_picked_when_keyword_beats_temporal() -> None:
    a = _link(5, 100, source="body_closes_keyword", confidence=0.85)
    b = _link(5, 100, source="temporal_author_match", confidence=0.5)
    [row] = merge_links([a], [b])
    assert row.confidence == 0.85
