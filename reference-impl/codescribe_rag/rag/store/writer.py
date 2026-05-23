from __future__ import annotations

import gzip
import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import sqlite_vec

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Store:
    """sqlite-vec + FTS5 wrapper. One DB file. WAL journal. Foreign keys ON."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def open(cls, db_path: Path, schema_path: Path = SCHEMA_PATH) -> "Store":
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit-able
        # CRITICAL: load the vec extension BEFORE any vec0 virtual-table creation.
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(schema_path.read_text())
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------
    @contextmanager
    def batch(self) -> Iterator[None]:
        """Wrap multiple upserts in one transaction (10x faster than per-row)."""
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # State (resume watermarks)
    # ------------------------------------------------------------------
    def get_state(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, value))

    # ------------------------------------------------------------------
    # Upserts
    # ------------------------------------------------------------------
    def upsert_bug(self, bug, embedding: np.ndarray) -> None:
        if embedding.shape != (1024,):
            raise ValueError(f"expected (1024,), got {embedding.shape}")
        if abs(float(np.linalg.norm(embedding)) - 1.0) > 0.01:
            raise ValueError("embedding must be L2-normalised")
        emb_blob = embedding.astype(np.float32).tobytes()
        self._conn.execute(
            """INSERT INTO bugs (bug_id, summary, description, component, severity,
                                  status, resolution, creation_time, last_change_time,
                                  assigned_to, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(bug_id) DO UPDATE SET
                 summary = excluded.summary, description = excluded.description,
                 component = excluded.component, severity = excluded.severity,
                 status = excluded.status, resolution = excluded.resolution,
                 last_change_time = excluded.last_change_time,
                 assigned_to = excluded.assigned_to, raw_json = excluded.raw_json
            """,
            (bug.id, bug.summary, bug.description, bug.component, bug.severity,
             bug.status, bug.resolution, bug.creation_time.isoformat(),
             bug.last_change_time.isoformat(), bug.assigned_to, json.dumps(bug.raw_json)),
        )
        # vec0 supports DELETE + INSERT but not UPSERT.
        self._conn.execute("DELETE FROM bug_vectors WHERE rowid = ?", (bug.id,))
        self._conn.execute(
            "INSERT INTO bug_vectors (rowid, embedding) VALUES (?, ?)", (bug.id, emb_blob))
        # Standalone FTS5: replace this bug's row.
        self._conn.execute("DELETE FROM bug_fts WHERE bug_id = ?", (bug.id,))
        self._conn.execute(
            "INSERT INTO bug_fts (bug_id, summary, description) VALUES (?, ?, ?)",
            (bug.id, bug.summary, bug.description))

    def upsert_cl(self, cl, chunks, chunk_embeddings, cl_summary_embedding) -> None:
        gzipped = gzip.compress(cl.diff_text.encode("utf-8"))
        self._conn.execute(
            """INSERT INTO changes (cl_number, author, submitted_at, description,
                                     file_count, diff_text)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(cl_number) DO UPDATE SET
                 description = excluded.description, file_count = excluded.file_count,
                 diff_text = excluded.diff_text
            """,
            (cl.cl, cl.author, cl.submitted_at.isoformat(), cl.description,
             len(cl.files), gzipped),
        )
        # Idempotent re-index: clear this CL's old chunks + their vectors + FTS rows.
        old_ids = [r[0] for r in self._conn.execute(
            "SELECT chunk_id FROM cl_chunks WHERE cl_number = ?", (cl.cl,)).fetchall()]
        for cid in old_ids:
            self._conn.execute("DELETE FROM cl_chunk_vectors WHERE rowid = ?", (cid,))
        self._conn.execute("DELETE FROM cl_fts WHERE cl_number = ?", (cl.cl,))
        self._conn.execute("DELETE FROM cl_chunks WHERE cl_number = ?", (cl.cl,))

        offset = self._conn.execute(
            "SELECT COALESCE(MAX(chunk_id), 0) FROM cl_chunks").fetchone()[0]
        for i, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings, strict=True)):
            chunk_id = offset + 1 + i
            self._conn.execute(
                """INSERT INTO cl_chunks (chunk_id, cl_number, file_path, hunk_index, chunk_text)
                   VALUES (?, ?, ?, ?, ?)""",
                (chunk_id, cl.cl, chunk.file_path, chunk.hunk_index, chunk.text))
            self._conn.execute(
                "INSERT INTO cl_chunk_vectors (rowid, embedding) VALUES (?, ?)",
                (chunk_id, np.asarray(emb).astype(np.float32).tobytes()))
            self._conn.execute(
                "INSERT INTO cl_fts (cl_number, chunk_id, chunk_text) VALUES (?, ?, ?)",
                (cl.cl, chunk_id, chunk.text))

    def upsert_fix_link(self, link) -> None:
        self._conn.execute(
            """INSERT INTO fix_links (bug_id, cl_number, confidence, signals, extracted_at)
               VALUES (?, ?, ?, ?, datetime('now'))
               ON CONFLICT(bug_id, cl_number) DO UPDATE SET
                 confidence = MAX(fix_links.confidence, excluded.confidence),
                 signals = excluded.signals
            """,
            (link.bug_id, link.cl_number, link.confidence, json.dumps(link.signals)))

    # ------------------------------------------------------------------
    # Read helper (used by the retriever / get_fix_diff tool)
    # ------------------------------------------------------------------
    def get_diff_text(self, cl_number: int) -> str:
        row = self._conn.execute(
            "SELECT diff_text FROM changes WHERE cl_number = ?", (cl_number,)).fetchone()
        if row is None:
            raise KeyError(f"no such CL: {cl_number}")
        return gzip.decompress(row[0]).decode("utf-8", errors="replace")
