"""Commit roundtrip; pr_number column is nullable.

Acceptance: spec test #4 from the design spec.
"""

from __future__ import annotations

import numpy as np

from codescribe_train.rag.sources.git_source import Commit, FileChange
from codescribe_train.rag.store.writer import Store


def _commit(
    sha: str = "a" * 40,
    pr_number: int | None = None,
    message: str = "fix: cart item cache eviction",
) -> Commit:
    return Commit(
        sha=sha,
        author="Alice",
        author_email="alice@example.com",
        authored_at="2026-05-01T12:00:00Z",
        message=message,
        files=[FileChange(path="src/cart.py", change_type="M", added=3, deleted=1)],
        diff_text="@@ -1,3 +1,3 @@\n-old\n+new\n",
        pr_number=pr_number,
    )


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_parent_pull(number: int = 42):
    from codescribe_train.rag.sources.github_source import PullRequest

    return PullRequest(
        number=number,
        title="parent PR",
        body=None,
        state="merged",
        head_sha="0" * 40,
        base_branch="main",
        author="example-org",
        draft=False,
        created_at="2026-05-01T10:00:00Z",
        updated_at="2026-05-01T10:00:00Z",
        merged_at="2026-05-01T10:00:00Z",
        closed_at="2026-05-01T10:00:00Z",
        raw_json='{"number": 42}',
    )


def test_upsert_commit_writes_metadata(tmp_path) -> None:
    commit = _commit(sha="b" * 40, pr_number=42)
    with Store.open(tmp_path / "rag.db") as store:
        # FK on commits.pr_number → pulls.pr_number; insert the parent
        # first so a fully-linked commit can be written.
        store.upsert_pull(
            pr=_make_parent_pull(42),
            chunks=[],
            chunk_embeddings=[],
            pr_summary_embedding=_emb(99),
        )
        store.upsert_commit(commit, _emb(0))

        row = store.conn.execute(
            "SELECT sha, author, author_email, authored_at, message, "
            "file_count, pr_number FROM commits WHERE sha = ?",
            ("b" * 40,),
        ).fetchone()

    assert row is not None
    sha, author, email, ts, message, file_count, pr_number = row
    assert sha == "b" * 40
    assert author == "Alice"
    assert email == "alice@example.com"
    assert ts == "2026-05-01T12:00:00Z"
    assert message == "fix: cart item cache eviction"
    assert file_count == 1
    assert pr_number == 42


def test_upsert_commit_pr_number_is_nullable(tmp_path) -> None:
    commit = _commit(sha="c" * 40, pr_number=None)
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_commit(commit, _emb(1))
        row = store.conn.execute(
            "SELECT pr_number FROM commits WHERE sha = ?", ("c" * 40,)
        ).fetchone()
    assert row == (None,)


def test_upsert_commit_writes_vector_and_fts(tmp_path) -> None:
    commit = _commit(sha="d" * 40, message="fix: improve cart eviction")
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_commit(commit, _emb(2))

        # commit_vectors keyed by rowid; commits has no integer PK, so the
        # store assigns its own rowid. Confirm at least one row exists.
        vec_count = store.conn.execute(
            "SELECT COUNT(*) FROM commit_vectors"
        ).fetchone()[0]
        # FTS5 mirror is keyed off commits.rowid (content_rowid='rowid'),
        # so we look up the rowid of the inserted commit and confirm match.
        commit_rowid = store.conn.execute(
            "SELECT rowid FROM commits WHERE sha = ?", ("d" * 40,)
        ).fetchone()[0]
        fts_rows = store.conn.execute(
            "SELECT rowid FROM commit_fts WHERE commit_fts MATCH 'cart'"
        ).fetchall()

    assert vec_count == 1
    assert (commit_rowid,) in fts_rows


def test_upsert_commit_is_idempotent(tmp_path) -> None:
    first = _commit(sha="e" * 40, message="initial subject")
    updated = _commit(sha="e" * 40, message="updated subject")
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_commit(first, _emb(3))
        store.upsert_commit(updated, _emb(4))

        rows = store.conn.execute(
            "SELECT message FROM commits WHERE sha = ?", ("e" * 40,)
        ).fetchall()
        vec_count = store.conn.execute(
            "SELECT COUNT(*) FROM commit_vectors"
        ).fetchone()[0]
    assert rows == [("updated subject",)]
    assert vec_count == 1
