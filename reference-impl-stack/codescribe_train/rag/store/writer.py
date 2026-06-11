"""sqlite-vec + FTS5 store writer.

Single-file store under ``indices/rag.db``. The schema lives in
``schema.sql`` next to this module; this module loads and applies it.

Rules (from ``codescribe_train/rag/store/AGENTS.md``):

* one DB file; never split.
* sqlite-vec extension loaded explicitly via
  ``conn.enable_load_extension(True); sqlite_vec.load(conn)`` — never
  relies on a globally-loaded vec0.
* ``PRAGMA journal_mode=WAL`` and ``PRAGMA foreign_keys=ON`` on every
  connection.
* ``diff_text`` columns (pulls AND commits) are gzip-compressed on disk;
  decompression happens at read time.
* All upserts run inside a single transaction across metadata + vec0 +
  FTS5 so partial rows never escape.
* No unsafe binary-serialisation modules; JSON + gzip only.

Public surface:

* :class:`Store` with :meth:`open`, :meth:`upsert_issue`,
  :meth:`upsert_pull`, :meth:`upsert_commit`,
  :meth:`upsert_issue_pr_link`, :meth:`get_state`, :meth:`set_state`,
  :meth:`batch`.

The read-path helpers live in ``retrieve.py``.
"""

from __future__ import annotations

import gzip
import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import sqlite_vec

if TYPE_CHECKING:
    import numpy as np

    from codescribe_train.rag.embed.chunker import PRChunk
    from codescribe_train.rag.embed.doc_chunker import DocChunk
    from codescribe_train.rag.extract.pairing import IssuePRLink
    from codescribe_train.rag.sources.git_source import Commit
    from codescribe_train.rag.sources.github_source import Issue, PullRequest

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _schema_sql() -> str:
    """Load the schema SQL from ``schema.sql``."""
    return _SCHEMA_PATH.read_text(encoding="utf-8")


def _compress(text: str | None) -> bytes:
    """gzip-compress ``text`` for on-disk storage of large diffs.

    ``None`` and the empty string both round-trip via gzip — we never
    skip the gzip even on an empty body (the AGENTS.md rule: "Do NOT
    skip the gzip on diffs to save dev time"). Decompression in
    ``retrieve.py`` always assumes a gzip-shaped blob.
    """
    return gzip.compress((text or "").encode("utf-8"))


def _embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Pack a 1024-dim float32 vector into the byte layout sqlite-vec expects."""
    import numpy as np

    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 1024:
        raise ValueError(
            f"embedding must be 1024-dim, got {arr.shape[0]}"
        )
    return arr.tobytes()


class Store:
    """Context-managed handle to the rag sqlite-vec + FTS5 store."""

    def __init__(self, conn: sqlite3.Connection, db_path: Path) -> None:
        self.conn = conn
        self.db_path = db_path
        self._in_batch = False

    # --- lifecycle -----------------------------------------------------

    @classmethod
    @contextmanager
    def open(cls, db_path: Path | str) -> Iterator[Store]:
        """Open (or create) the store at ``db_path``.

        Loads the sqlite-vec extension, applies the schema on a fresh DB,
        and sets the WAL + foreign-keys pragmas. The connection is closed
        on context exit; any in-flight transaction is committed.
        """
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not db_path.exists()

        conn = sqlite3.connect(str(db_path))
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            # Disable extension loading post-load so library code that runs
            # later can't pull in additional extensions silently.
            conn.enable_load_extension(False)

            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

            if is_new:
                conn.executescript(_schema_sql())
                conn.commit()

            yield cls(conn=conn, db_path=db_path)
            conn.commit()
        finally:
            conn.close()

    # --- upsert helpers ------------------------------------------------

    def upsert_issue(self, issue: Issue, embedding: np.ndarray) -> None:
        """Insert or update one issue, its vector, and its FTS row.

        Idempotent: re-running with the same ``issue.number`` overwrites
        all three columns. The whole upsert runs inside a single
        transaction so partial rows never escape on crash.
        """
        vec = _embedding_to_bytes(embedding)
        labels_json = json.dumps(issue.labels)
        assignees_json = json.dumps(issue.assignees)

        with self._transaction():
            self.conn.execute(
                """
                INSERT INTO issues (
                    issue_number, title, body, state, state_reason,
                    labels, assignees, created_at, updated_at, closed_at,
                    raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(issue_number) DO UPDATE SET
                    title = excluded.title,
                    body = excluded.body,
                    state = excluded.state,
                    state_reason = excluded.state_reason,
                    labels = excluded.labels,
                    assignees = excluded.assignees,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    closed_at = excluded.closed_at,
                    raw_json = excluded.raw_json
                """,
                (
                    issue.number,
                    issue.title,
                    issue.body,
                    issue.state,
                    issue.state_reason,
                    labels_json,
                    assignees_json,
                    issue.created_at,
                    issue.updated_at,
                    issue.closed_at,
                    issue.raw_json,
                ),
            )
            # vec0 has no upsert; delete + insert keeps it idempotent.
            self.conn.execute(
                "DELETE FROM issue_vectors WHERE rowid = ?", (issue.number,)
            )
            self.conn.execute(
                "INSERT INTO issue_vectors(rowid, embedding) VALUES (?, ?)",
                (issue.number, vec),
            )
            # FTS5 external-content table: explicit mirror.
            self.conn.execute(
                "INSERT OR REPLACE INTO issue_fts(rowid, title, body) VALUES (?, ?, ?)",
                (issue.number, issue.title, issue.body),
            )

    def upsert_pull(
        self,
        pr: PullRequest,
        chunks: list[PRChunk],
        chunk_embeddings: list[np.ndarray],
        pr_summary_embedding: np.ndarray,
        diff_text: str = "",
    ) -> None:
        """Insert or update one PR, its summary vector, and its chunks.

        ``diff_text`` (kwarg, defaults to "") is gzip-compressed before
        being written to ``pulls.diff_text``. The chunk count and chunk
        embedding count must match; mismatch raises before any write.
        ``file_count`` is derived from the distinct ``file_path`` values
        in ``chunks`` so the schema column stays populated for retrieval
        without the writer reaching back into the source object for it.
        """
        if len(chunks) != len(chunk_embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: "
                f"{len(chunks)} chunks vs {len(chunk_embeddings)} embeddings"
            )

        summary_vec = _embedding_to_bytes(pr_summary_embedding)
        diff_blob = _compress(diff_text)
        file_count = len({c.file_path for c in chunks})

        with self._transaction():
            self.conn.execute(
                """
                INSERT INTO pulls (
                    pr_number, title, body, state, head_sha, base_branch,
                    merged_at, author, file_count, diff_text, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pr_number) DO UPDATE SET
                    title = excluded.title,
                    body = excluded.body,
                    state = excluded.state,
                    head_sha = excluded.head_sha,
                    base_branch = excluded.base_branch,
                    merged_at = excluded.merged_at,
                    author = excluded.author,
                    file_count = excluded.file_count,
                    diff_text = excluded.diff_text,
                    raw_json = excluded.raw_json
                """,
                (
                    pr.number,
                    pr.title,
                    pr.body,
                    pr.state,
                    pr.head_sha,
                    pr.base_branch,
                    pr.merged_at,
                    pr.author,
                    file_count,
                    diff_blob,
                    pr.raw_json,
                ),
            )
            # Summary vector — vec0 has no upsert; delete + insert.
            self.conn.execute(
                "DELETE FROM pr_vectors WHERE rowid = ?", (pr.number,)
            )
            self.conn.execute(
                "INSERT INTO pr_vectors(rowid, embedding) VALUES (?, ?)",
                (pr.number, summary_vec),
            )
            # FTS5 mirror.
            self.conn.execute(
                "INSERT OR REPLACE INTO pr_fts(rowid, title, body) VALUES (?, ?, ?)",
                (pr.number, pr.title, pr.body),
            )
            # Replace chunks + their vectors. Drop the old chunk_ids first
            # so the chunk_id ↔ pr_chunk_vectors.rowid invariant holds.
            old_chunk_ids = [
                r[0]
                for r in self.conn.execute(
                    "SELECT chunk_id FROM pr_chunks WHERE pr_number = ?",
                    (pr.number,),
                ).fetchall()
            ]
            for chunk_id in old_chunk_ids:
                self.conn.execute(
                    "DELETE FROM pr_chunk_vectors WHERE rowid = ?", (chunk_id,)
                )
            self.conn.execute(
                "DELETE FROM pr_chunks WHERE pr_number = ?", (pr.number,)
            )
            for chunk, embedding in zip(chunks, chunk_embeddings, strict=True):
                cursor = self.conn.execute(
                    """
                    INSERT INTO pr_chunks (pr_number, file_path, hunk_index, chunk_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (chunk.pr_number, chunk.file_path, chunk.hunk_index, chunk.chunk_text),
                )
                chunk_id = cursor.lastrowid
                vec_bytes = _embedding_to_bytes(embedding)
                self.conn.execute(
                    "INSERT INTO pr_chunk_vectors(rowid, embedding) VALUES (?, ?)",
                    (chunk_id, vec_bytes),
                )

    def upsert_commit(self, commit: Commit, embedding: np.ndarray) -> None:
        """Insert or update one commit, its vector, and its FTS row.

        ``commit.pr_number`` is allowed to be ``None``. The schema's
        foreign key on ``commits.pr_number → pulls.pr_number`` is
        enforced only when the column is non-null and the parent row
        exists; pipelines that ingest commits before the related PR is
        seen should pass ``pr_number=None`` and back-fill later.

        The vec0 + FTS5 mirrors are keyed by the commits row's implicit
        sqlite rowid (not the SHA), per
        ``content_rowid='rowid'`` on ``commit_fts``.
        """
        vec = _embedding_to_bytes(embedding)
        diff_blob = _compress(commit.diff_text)
        file_count = len(commit.files)

        with self._transaction():
            # On a re-upsert by sha, the existing rowid is preserved by
            # the conflict path — so we resolve it after the metadata
            # write to mirror vec0 + FTS5 against the same key.
            self.conn.execute(
                """
                INSERT INTO commits (
                    sha, author, author_email, authored_at, message,
                    file_count, diff_text, pr_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sha) DO UPDATE SET
                    author = excluded.author,
                    author_email = excluded.author_email,
                    authored_at = excluded.authored_at,
                    message = excluded.message,
                    file_count = excluded.file_count,
                    diff_text = excluded.diff_text,
                    pr_number = excluded.pr_number
                """,
                (
                    commit.sha,
                    commit.author,
                    commit.author_email,
                    commit.authored_at,
                    commit.message,
                    file_count,
                    diff_blob,
                    commit.pr_number,
                ),
            )
            rowid = self.conn.execute(
                "SELECT rowid FROM commits WHERE sha = ?", (commit.sha,)
            ).fetchone()[0]
            self.conn.execute(
                "DELETE FROM commit_vectors WHERE rowid = ?", (rowid,)
            )
            self.conn.execute(
                "INSERT INTO commit_vectors(rowid, embedding) VALUES (?, ?)",
                (rowid, vec),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO commit_fts(rowid, message) VALUES (?, ?)",
                (rowid, commit.message),
            )

    def upsert_doc(
        self,
        doc_path: str,
        chunks: list[DocChunk],
        chunk_embeddings: list[np.ndarray],
    ) -> None:
        """Replace all chunks for ``doc_path`` with ``chunks`` + their vectors.

        Idempotent by ``doc_path``: the path's existing chunks (and their
        vec0 + FTS5 mirrors) are deleted first, then the new ones inserted,
        all inside one transaction. ``chunks`` and ``chunk_embeddings`` must
        be the same length.
        """
        if len(chunks) != len(chunk_embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: "
                f"{len(chunks)} chunks vs {len(chunk_embeddings)} embeddings"
            )
        with self._transaction():
            old_chunk_ids = [
                r[0]
                for r in self.conn.execute(
                    "SELECT chunk_id FROM doc_chunks WHERE doc_path = ?", (doc_path,)
                ).fetchall()
            ]
            for chunk_id in old_chunk_ids:
                self.conn.execute(
                    "DELETE FROM doc_chunk_vectors WHERE rowid = ?", (chunk_id,)
                )
                self.conn.execute(
                    "DELETE FROM doc_chunk_fts WHERE rowid = ?", (chunk_id,)
                )
            self.conn.execute(
                "DELETE FROM doc_chunks WHERE doc_path = ?", (doc_path,)
            )
            for chunk, embedding in zip(chunks, chunk_embeddings, strict=True):
                cursor = self.conn.execute(
                    """
                    INSERT INTO doc_chunks (doc_path, heading, chunk_index, chunk_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (doc_path, chunk.heading, chunk.chunk_index, chunk.text),
                )
                chunk_id = cursor.lastrowid
                self.conn.execute(
                    "INSERT INTO doc_chunk_vectors(rowid, embedding) VALUES (?, ?)",
                    (chunk_id, _embedding_to_bytes(embedding)),
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO doc_chunk_fts(rowid, heading, chunk_text) "
                    "VALUES (?, ?, ?)",
                    (chunk_id, chunk.heading, chunk.text),
                )

    def upsert_issue_pr_link(self, link: IssuePRLink) -> None:
        """Insert or update one issue↔PR link with higher-confidence-wins.

        ``(issue_number, pr_number)`` is the composite primary key. On
        conflict the row is overwritten ONLY if the incoming
        ``confidence`` is strictly greater than the stored one — so a
        late, low-confidence signal (e.g. temporal_author_match arriving
        after a closingIssuesReferences row) cannot downgrade the link.

        ``extracted_at`` is filled with the current UTC timestamp so the
        retrieval path can age out very stale links if desired.
        """
        from datetime import UTC, datetime

        extracted_at = datetime.now(tz=UTC).isoformat()
        with self._transaction():
            self.conn.execute(
                """
                INSERT INTO issue_pr_links (
                    issue_number, pr_number, confidence, source, extracted_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(issue_number, pr_number) DO UPDATE SET
                    confidence = excluded.confidence,
                    source = excluded.source,
                    extracted_at = excluded.extracted_at
                WHERE excluded.confidence > issue_pr_links.confidence
                """,
                (
                    link.issue_number,
                    link.pr_number,
                    link.confidence,
                    link.source,
                    extracted_at,
                ),
            )

    # --- state K/V table -----------------------------------------------

    def get_state(self, key: str) -> str | None:
        """Return the stored value for ``key`` or ``None`` if unset."""
        row = self.conn.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else row[0]

    def set_state(self, key: str, value: str) -> None:
        """Overwrite ``state[key]`` with ``value`` atomically.

        The pipeline checkpoints (``last_seen_commit_sha``,
        ``last_seen_issue_updated_at``, ...) are written through this
        method; in the pipeline they share a transaction with the
        corresponding data upserts so a crash never leaves the watermark
        ahead of the data.
        """
        with self._transaction():
            self.conn.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                (key, value),
            )

    # --- batch ---------------------------------------------------------

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Run multiple upserts inside one outer transaction.

        Nested ``_transaction()`` calls inside the block become no-ops so
        the batch commits exactly once on exit. On exception the batch
        rolls back.
        """
        if self._in_batch:
            # Reentrant batches are not part of the contract; this would
            # surprise the caller silently. Loudly refuse.
            raise RuntimeError("Store.batch() is not reentrant")
        self.conn.execute("BEGIN")
        self._in_batch = True
        try:
            yield
        except Exception:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()
        finally:
            self._in_batch = False

    # --- transaction helpers -------------------------------------------

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Run a block inside an explicit BEGIN/COMMIT, rolling back on error.

        sqlite3's default "autocommit-by-default" hides the boundary; the
        store's hard rule from AGENTS.md is "single transaction across
        metadata + vec0 + FTS5 per record". When a caller has already
        opened a :meth:`batch` block, this nested call is a no-op so the
        caller's outer transaction wins.
        """
        if self._in_batch:
            yield
            return
        self.conn.execute("BEGIN")
        try:
            yield
        except Exception:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()
