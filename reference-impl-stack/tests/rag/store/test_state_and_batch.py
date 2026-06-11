"""state K/V table + batch() context manager.

Spec deliverables 5 & 6 — ``get_state`` / ``set_state`` and a batch
context manager wrapping multi-record writes in a single transaction.
"""

from __future__ import annotations

import numpy as np

from codescribe_train.rag.sources.github_source import Issue
from codescribe_train.rag.store.writer import Store


def _issue(n: int) -> Issue:
    return Issue(
        number=n,
        title=f"issue {n}",
        body="body",
        state="open",
        state_reason=None,
        labels=[],
        assignees=[],
        author="example-org",
        created_at="2026-05-01T12:00:00Z",
        updated_at="2026-05-02T15:00:00Z",
        closed_at=None,
        raw_json="{}",
    )


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def test_set_state_then_get_state(tmp_path) -> None:
    with Store.open(tmp_path / "rag.db") as store:
        store.set_state("last_seen_commit_sha", "a" * 40)
        store.set_state("last_seen_issue_updated_at", "2026-05-21T00:00:00Z")

        assert store.get_state("last_seen_commit_sha") == "a" * 40
        assert store.get_state("last_seen_issue_updated_at") == "2026-05-21T00:00:00Z"


def test_get_state_missing_returns_none(tmp_path) -> None:
    with Store.open(tmp_path / "rag.db") as store:
        assert store.get_state("never_set") is None


def test_set_state_overwrites(tmp_path) -> None:
    with Store.open(tmp_path / "rag.db") as store:
        store.set_state("k", "v1")
        store.set_state("k", "v2")
        assert store.get_state("k") == "v2"


def test_batch_writes_multiple_records_in_one_transaction(tmp_path) -> None:
    """A bulk insert in batch mode commits exactly once.

    Validated indirectly: after the batch block, all records are visible
    and ``conn.in_transaction`` is False (the batch commit ran).
    """
    with Store.open(tmp_path / "rag.db") as store:
        with store.batch():
            for i in range(5):
                store.upsert_issue(_issue(100 + i), _emb(i))
            # Inside the batch the connection is mid-transaction.
            assert store.conn.in_transaction is True

        assert store.conn.in_transaction is False
        count = store.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
    assert count == 5


def test_batch_rolls_back_on_exception(tmp_path) -> None:
    class _Boom(RuntimeError):
        pass

    with Store.open(tmp_path / "rag.db") as store:
        try:
            with store.batch():
                store.upsert_issue(_issue(1), _emb(1))
                raise _Boom("explode mid-batch")
        except _Boom:
            pass

        count = store.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
    assert count == 0
