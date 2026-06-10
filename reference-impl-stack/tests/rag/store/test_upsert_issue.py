"""Issue insert/update + vector + FTS roundtrip.

Acceptance: spec test #2 from the design spec.
"""

from __future__ import annotations

import json

import numpy as np

from codescribe_train.rag.sources.github_source import Issue
from codescribe_train.rag.store.writer import Store


def _make_issue(number: int = 1, title: str = "Cart crash") -> Issue:
    return Issue(
        number=number,
        title=title,
        body="App crashes when cart loads item X near checkout.",
        state="open",
        state_reason=None,
        labels=["bug", "mobile"],
        assignees=["example-org"],
        author="example-org",
        created_at="2026-05-01T12:00:00Z",
        updated_at="2026-05-02T15:00:00Z",
        closed_at=None,
        raw_json='{"number": 1}',
    )


def _embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def test_upsert_issue_writes_metadata(tmp_path) -> None:
    issue = _make_issue(number=7, title="Item not loading")
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue(issue, _embedding())

        row = store.conn.execute(
            "SELECT issue_number, title, body, state, labels, assignees "
            "FROM issues WHERE issue_number = ?",
            (7,),
        ).fetchone()

    assert row is not None
    issue_number, title, body, state, labels_json, assignees_json = row
    assert issue_number == 7
    assert title == "Item not loading"
    assert state == "open"
    assert json.loads(labels_json) == ["bug", "mobile"]
    assert json.loads(assignees_json) == ["example-org"]
    assert "cart" in body.lower()


def test_upsert_issue_writes_vector(tmp_path) -> None:
    issue = _make_issue(number=11)
    emb = _embedding(seed=11)
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue(issue, emb)
        rows = store.conn.execute(
            "SELECT rowid FROM issue_vectors WHERE rowid = ?", (11,)
        ).fetchall()
    assert rows == [(11,)]


def test_upsert_issue_writes_fts(tmp_path) -> None:
    issue = _make_issue(number=13, title="Cart crash")
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue(issue, _embedding(seed=13))
        rows = store.conn.execute(
            "SELECT rowid FROM issue_fts WHERE issue_fts MATCH 'cart'"
        ).fetchall()
    assert rows == [(13,)]


def test_upsert_issue_is_idempotent(tmp_path) -> None:
    issue = _make_issue(number=21, title="initial")
    updated = _make_issue(number=21, title="updated title")
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue(issue, _embedding(seed=1))
        store.upsert_issue(updated, _embedding(seed=2))

        title = store.conn.execute(
            "SELECT title FROM issues WHERE issue_number = ?", (21,)
        ).fetchone()[0]
        vec_rows = store.conn.execute(
            "SELECT rowid FROM issue_vectors WHERE rowid = ?", (21,)
        ).fetchall()
        fts_rows = store.conn.execute(
            "SELECT rowid FROM issue_fts WHERE issue_fts MATCH 'updated'"
        ).fetchall()

    assert title == "updated title"
    assert vec_rows == [(21,)]  # one vector row, not two
    assert fts_rows == [(21,)]
