from __future__ import annotations

import time

import numpy as np
import pytest

from codescribe_rag.rag.store.retrieve import Retriever


def test_hybrid_finds_relevant_bug(populated_store, embedder):
    r = Retriever(populated_store._conn, embedder)
    results = r.find_similar_bugs("NPE on startup in ConfigLoader", k=3)
    assert results[0].bug_id == 1001


def test_confidence_filter(populated_store, embedder):
    r = Retriever(populated_store._conn, embedder)
    res = {x.bug_id: x for x in r.find_similar_bugs("startup crash", k=3)}
    assert res[1001].fix_cl == 12345          # 0.9 >= 0.8
    if 1002 in res:
        assert res[1002].fix_cl is None       # 0.7 filtered out


def test_no_unlinked_bug_returns_none(populated_store, embedder):
    r = Retriever(populated_store._conn, embedder)
    res = {x.bug_id: x for x in r.find_similar_bugs("scheduler deadlock", k=3)}
    assert res[1003].fix_cl is None and res[1003].fix_diff_excerpt is None


def test_query_normalisation_required(populated_store):
    class BadEmbedder:
        def embed_one(self, t):
            return np.full(1024, 5.0, dtype=np.float32)

        def embed(self, ts):
            return np.full((len(ts), 1024), 5.0, dtype=np.float32)

    r = Retriever(populated_store._conn, BadEmbedder())
    with pytest.raises(RuntimeError):
        r.find_similar_bugs("x")


def test_get_fix_diff_roundtrip(populated_store, embedder):
    r = Retriever(populated_store._conn, embedder)
    diff = r.get_fix_diff(12345)
    joined = "".join(f.diff for f in diff.files)
    assert "+good" in joined


def test_sanitise_fts_drops_punctuation(populated_store, embedder):
    r = Retriever(populated_store._conn, embedder)
    r.find_similar_bugs("foo:bar* (startup)")  # must not raise FTS5 syntax error


@pytest.mark.slow
def test_latency_under_200ms(tmp_path, embedder):
    from codescribe_rag.rag.store.writer import Store

    store = Store.open(tmp_path / "big.db")
    rng = np.random.default_rng(0)
    with store.batch():
        for i in range(1001, 101001):
            v = rng.standard_normal(1024).astype(np.float32)
            v /= np.linalg.norm(v)
            store._conn.execute(
                "INSERT INTO bug_vectors (rowid, embedding) VALUES (?, ?)", (i, v.tobytes()))
            store._conn.execute(
                "INSERT INTO bugs (bug_id, summary) VALUES (?, ?)", (i, f"bug {i}"))
    r = Retriever(store._conn, embedder)
    lat = []
    for _ in range(100):
        t = time.perf_counter()
        r.find_similar_bugs("startup npe", k=5)
        lat.append(time.perf_counter() - t)
    lat.sort()
    assert lat[98] < 0.2  # p99
    store.close()
