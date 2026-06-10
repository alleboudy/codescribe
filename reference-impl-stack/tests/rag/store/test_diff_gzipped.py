"""Diff text columns on pulls AND commits are gzip-compressed on disk.

Acceptance: spec test #6 from the design spec — "assert ``diff_text`` starts
with ``\\x1f\\x8b`` magic". The rule is from
``codescribe_train/rag/store/AGENTS.md`` ("Diff text is GZIP-compressed"); both
the pulls and commits tables carry the same BLOB column type.
"""

from __future__ import annotations

import gzip

import numpy as np

from codescribe_train.rag.sources.git_source import Commit, FileChange
from codescribe_train.rag.sources.github_source import PullRequest
from codescribe_train.rag.store.writer import Store

GZIP_MAGIC = b"\x1f\x8b"


def _pull(number: int = 42) -> PullRequest:
    return PullRequest(
        number=number,
        title="t",
        body="b",
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


def _commit(sha: str = "a" * 40) -> Commit:
    return Commit(
        sha=sha,
        author="Alice",
        author_email="alice@example.com",
        authored_at="2026-05-01T12:00:00Z",
        message="fix",
        files=[FileChange(path="src/a.py", change_type="M", added=1, deleted=0)],
        diff_text="@@ -1 +1,2 @@\n line\n+added\n" * 50,  # non-trivial size
        pr_number=None,
    )


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def test_pull_diff_text_starts_with_gzip_magic(tmp_path) -> None:
    long_diff = "@@ -1,3 +1,3 @@\n-old line\n+new line\n" * 200
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_pull(
            pr=_pull(7),
            chunks=[],
            chunk_embeddings=[],
            pr_summary_embedding=_emb(0),
            diff_text=long_diff,
        )
        blob = store.conn.execute(
            "SELECT diff_text FROM pulls WHERE pr_number = 7"
        ).fetchone()[0]

    assert isinstance(blob, bytes)
    assert blob[:2] == GZIP_MAGIC
    # Round-trip — the magic without working gzip would be a regression.
    assert gzip.decompress(blob).decode("utf-8") == long_diff


def test_commit_diff_text_starts_with_gzip_magic(tmp_path) -> None:
    commit = _commit(sha="f" * 40)
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_commit(commit, _emb(1))
        blob = store.conn.execute(
            "SELECT diff_text FROM commits WHERE sha = ?", ("f" * 40,)
        ).fetchone()[0]
    assert isinstance(blob, bytes)
    assert blob[:2] == GZIP_MAGIC
    assert gzip.decompress(blob).decode("utf-8") == commit.diff_text
