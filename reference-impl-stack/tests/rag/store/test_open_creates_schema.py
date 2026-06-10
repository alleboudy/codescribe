"""Fresh DB has every table + virtual table the schema declares.

Acceptance: spec test #1 from the design spec.
"""

from __future__ import annotations

import sqlite3

from codescribe_train.rag.store.writer import Store


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r[0] for r in rows}


def test_open_creates_all_tables(tmp_path) -> None:
    db_path = tmp_path / "rag.db"

    with Store.open(db_path) as store:
        names = _table_names(store.conn)

    expected = {
        # plain tables
        "issues",
        "pulls",
        "commits",
        "issue_pr_links",
        "pr_chunks",
        "state",
        # vector virtual tables (vec0)
        "issue_vectors",
        "pr_vectors",
        "commit_vectors",
        "pr_chunk_vectors",
        # FTS5 virtual tables
        "issue_fts",
        "pr_fts",
        "commit_fts",
    }
    missing = expected - names
    assert not missing, f"missing tables after Store.open: {sorted(missing)}"


def test_open_sets_pragmas(tmp_path) -> None:
    db_path = tmp_path / "rag.db"
    with Store.open(db_path) as store:
        journal_mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        fk = store.conn.execute("PRAGMA foreign_keys").fetchone()[0]

    assert journal_mode.lower() == "wal"
    assert fk == 1


def test_open_is_idempotent(tmp_path) -> None:
    """Re-opening an existing DB must not re-apply the schema (would error).

    The schema applies CREATE TABLE / CREATE VIRTUAL TABLE without IF NOT
    EXISTS clauses (verbatim from spec). Open must detect a populated DB.
    """
    db_path = tmp_path / "rag.db"
    with Store.open(db_path):
        pass
    # Second open should not raise.
    with Store.open(db_path) as store:
        names = _table_names(store.conn)
    assert "issues" in names
