from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from codescribe_rag.rag.store.writer import Store

UTC = timezone.utc


def unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeBug:
    def __init__(self, i):
        self.id = i
        self.summary = f"summary {i}"
        self.description = f"desc {i} npe"
        self.component = "core"
        self.severity = "major"
        self.status = "RESOLVED"
        self.resolution = "FIXED"
        self.creation_time = datetime(2026, 1, 1, tzinfo=UTC)
        self.last_change_time = datetime(2026, 1, 2, tzinfo=UTC)
        self.assigned_to = "alice"
        self.raw_json = {"id": i}


def test_open_creates_schema(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    store._conn.execute("SELECT * FROM bug_vectors LIMIT 0")  # raises if vec0 absent
    store.close()


def test_open_idempotent(tmp_path):
    p = tmp_path / "rag.db"
    Store.open(p).close()
    Store.open(p).close()


def test_upsert_bug_roundtrip_and_idempotent(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    with store.batch():
        store.upsert_bug(FakeBug(1001), unit(1))
        store.upsert_bug(FakeBug(1001), unit(1))
    assert store._conn.execute("SELECT COUNT(*) FROM bugs").fetchone()[0] == 1
    store.close()


def test_fts_bm25_returns_match(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    with store.batch():
        for i in range(1001, 1006):
            store.upsert_bug(FakeBug(i), unit(i))
    rows = store._conn.execute(
        "SELECT bug_id FROM bug_fts WHERE bug_fts MATCH 'npe' LIMIT 3").fetchall()
    assert rows
    store.close()


def test_vec_query_nearest(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    with store.batch():
        for i in range(1001, 1011):
            store.upsert_bug(FakeBug(i), unit(i))
    q = unit(1005).astype(np.float32).tobytes()
    rows = store._conn.execute(
        "SELECT rowid, distance FROM bug_vectors WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
        (q,)).fetchall()
    assert rows[0][0] == 1005 and rows[0][1] < 1e-4
    store.close()


def test_diff_gzip_roundtrip(tmp_path):
    store = Store.open(tmp_path / "rag.db")

    class CL:
        cl = 50001
        author = "alice"
        submitted_at = datetime(2026, 1, 3, tzinfo=UTC)
        description = "fix"
        files = ()
        diff_text = "@@ -1 +1 @@\n-a\n+b\n" * 5000

    with store.batch():
        store.upsert_cl(CL(), [], [], unit(7))
    raw = store._conn.execute(
        "SELECT diff_text FROM changes WHERE cl_number=50001").fetchone()[0]
    assert raw[:2] == b"\x1f\x8b"  # gzip magic
    assert store.get_diff_text(50001) == CL.diff_text
    store.close()


def test_batch_rollback(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    with pytest.raises(RuntimeError):
        with store.batch():
            store.upsert_bug(FakeBug(1001), unit(1))
            raise RuntimeError("boom")
    assert store._conn.execute("SELECT COUNT(*) FROM bugs").fetchone()[0] == 0
    store.close()
