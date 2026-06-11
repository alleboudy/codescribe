from __future__ import annotations

from pathlib import Path

import numpy as np

from codescribe_train.rag.embed.doc_chunker import DocChunk
from codescribe_train.rag.store.writer import Store


def _vec() -> np.ndarray:
    return np.ones(1024, dtype=np.float32)


def test_upsert_doc_writes_chunks_vectors_fts(tmp_path: Path) -> None:
    db = tmp_path / "rag.db"
    chunks = [DocChunk("Run", "make up", 0), DocChunk("Usage", "do things", 1)]
    with Store.open(db) as store:
        store.upsert_doc("README.md", chunks, [_vec(), _vec()])
        n_chunks = store.conn.execute(
            "SELECT COUNT(*) FROM doc_chunks WHERE doc_path = ?", ("README.md",)
        ).fetchone()[0]
        n_vecs = store.conn.execute("SELECT COUNT(*) FROM doc_chunk_vectors").fetchone()[0]
        n_fts = store.conn.execute("SELECT COUNT(*) FROM doc_chunk_fts").fetchone()[0]
    assert n_chunks == 2
    assert n_vecs == 2
    assert n_fts == 2


def test_upsert_doc_is_idempotent_by_path(tmp_path: Path) -> None:
    db = tmp_path / "rag.db"
    with Store.open(db) as store:
        store.upsert_doc("README.md", [DocChunk("A", "old", 0)], [_vec()])
        store.upsert_doc(
            "README.md",
            [DocChunk("A", "new", 0), DocChunk("B", "more", 1)],
            [_vec(), _vec()],
        )
        rows = store.conn.execute(
            "SELECT chunk_text FROM doc_chunks WHERE doc_path = ? ORDER BY chunk_index",
            ("README.md",),
        ).fetchall()
    assert [r[0] for r in rows] == ["new", "more"]


def test_upsert_doc_rejects_length_mismatch(tmp_path: Path) -> None:
    db = tmp_path / "rag.db"
    with Store.open(db) as store:
        try:
            store.upsert_doc("README.md", [DocChunk("A", "x", 0)], [_vec(), _vec()])
        except ValueError as e:
            assert "mismatch" in str(e)
        else:
            raise AssertionError("expected ValueError on chunk/embedding mismatch")
