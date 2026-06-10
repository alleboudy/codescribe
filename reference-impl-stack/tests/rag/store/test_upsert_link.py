"""Issue↔PR link upsert; conflict on (issue, PR) keeps MAX confidence.

Acceptance: spec test #5 from the design spec — "link upsert; conflict on
(issue, PR) updates to higher confidence".
"""

from __future__ import annotations

from codescribe_train.rag.extract.pairing import IssuePRLink
from codescribe_train.rag.store.writer import Store


def _link(
    issue_number: int = 7,
    pr_number: int = 42,
    confidence: float = 0.5,
    source: str = "temporal_author_match",
) -> IssuePRLink:
    return IssuePRLink(
        issue_number=issue_number,
        pr_number=pr_number,
        source=source,
        confidence=confidence,
        signals={source: confidence},
    )


def test_upsert_link_first_write(tmp_path) -> None:
    link = _link(issue_number=7, pr_number=42, confidence=0.85, source="body_closes_keyword")
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue_pr_link(link)
        row = store.conn.execute(
            "SELECT issue_number, pr_number, confidence, source "
            "FROM issue_pr_links WHERE issue_number = 7 AND pr_number = 42"
        ).fetchone()

    assert row == (7, 42, 0.85, "body_closes_keyword")


def test_upsert_link_higher_confidence_overwrites(tmp_path) -> None:
    weak = _link(issue_number=7, pr_number=42, confidence=0.5, source="temporal_author_match")
    strong = _link(
        issue_number=7,
        pr_number=42,
        confidence=1.0,
        source="closingIssuesReferences",
    )
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue_pr_link(weak)
        store.upsert_issue_pr_link(strong)
        row = store.conn.execute(
            "SELECT confidence, source FROM issue_pr_links "
            "WHERE issue_number = 7 AND pr_number = 42"
        ).fetchone()

    assert row == (1.0, "closingIssuesReferences")


def test_upsert_link_lower_confidence_does_not_downgrade(tmp_path) -> None:
    """A later, lower-confidence write must NOT overwrite a stored higher one.

    This is the monotonicity guarantee called out in the implementation
    notes ("higher confidence wins"). Implementation pattern:
    ``ON CONFLICT DO UPDATE SET ... WHERE excluded.confidence > confidence``.
    """
    strong = _link(
        issue_number=7,
        pr_number=42,
        confidence=1.0,
        source="closingIssuesReferences",
    )
    weak = _link(issue_number=7, pr_number=42, confidence=0.5, source="temporal_author_match")
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue_pr_link(strong)
        store.upsert_issue_pr_link(weak)
        row = store.conn.execute(
            "SELECT confidence, source FROM issue_pr_links "
            "WHERE issue_number = 7 AND pr_number = 42"
        ).fetchone()

    assert row == (1.0, "closingIssuesReferences")


def test_upsert_link_distinct_pairs_coexist(tmp_path) -> None:
    a = _link(issue_number=7, pr_number=42, confidence=1.0, source="closingIssuesReferences")
    b = _link(issue_number=7, pr_number=43, confidence=0.85, source="body_closes_keyword")
    c = _link(issue_number=8, pr_number=42, confidence=0.5, source="temporal_author_match")
    with Store.open(tmp_path / "rag.db") as store:
        for link in (a, b, c):
            store.upsert_issue_pr_link(link)
        rows = store.conn.execute(
            "SELECT issue_number, pr_number FROM issue_pr_links "
            "ORDER BY issue_number, pr_number"
        ).fetchall()

    assert rows == [(7, 42), (7, 43), (8, 42)]
