"""The doc_chunks tables exist in a freshly-created store."""
from __future__ import annotations

from pathlib import Path

from codescribe_train.rag.store.writer import Store


def test_doc_tables_exist_in_fresh_store(tmp_path: Path) -> None:
    db = tmp_path / "rag.db"
    with Store.open(db) as store:
        names = {
            r[0]
            for r in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            ).fetchall()
        }
    assert "doc_chunks" in names
    assert "doc_chunk_vectors" in names
    assert "doc_chunk_fts" in names


def test_doc_chunks_columns(tmp_path: Path) -> None:
    db = tmp_path / "rag.db"
    with Store.open(db) as store:
        cols = {
            r[1]  # name column of PRAGMA table_info
            for r in store.conn.execute("PRAGMA table_info(doc_chunks)").fetchall()
        }
    assert cols == {"chunk_id", "doc_path", "heading", "chunk_index", "chunk_text"}
