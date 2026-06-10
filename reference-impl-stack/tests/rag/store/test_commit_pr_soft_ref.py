"""Regression: a commit referencing an unindexed PR must still upsert.

The indexer streams commits and pulls concurrently (per the spec).
With the original schema's `FOREIGN KEY (pr_number)
REFERENCES pulls (pr_number)`, a commit batch can raise
`sqlite3.IntegrityError: FOREIGN KEY constraint failed` because git
commits that mention "Merges PR #N" or "(#N)" may arrive before the
corresponding PR rows when both streams run at once. The FK constraint was
removed; the relationship is now informational only.

This test pins the new contract: commits can land before, after, or
without their PR.
"""

from __future__ import annotations

from pathlib import Path

from codescribe_train.rag.sources.git_source import Commit, FileChange
from codescribe_train.rag.store.writer import Store


def _make_commit(sha: str, pr_number: int | None) -> Commit:
    return Commit(
        sha=sha,
        author="Alice",
        author_email="example-org@example.test",
        authored_at="2026-05-21T00:00:00+00:00",
        message=f"feat: change ({sha[:7]})",
        files=[FileChange(path="src/a.py", change_type="M", added=1, deleted=0)],
        diff_text="--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+new\n",
        pr_number=pr_number,
    )


def test_commit_with_unindexed_pr_number_upserts_without_fk_error(tmp_path: Path) -> None:
    db = tmp_path / "rag.db"
    with Store.open(db) as store, store.batch():
        # PR row never inserted, but the commit references PR #4242.
        store.upsert_commit(_make_commit("a" * 40, pr_number=4242), embedding=_one_vector())

    with Store.open(db) as store:
        cur = store.conn.execute(
            "SELECT sha, pr_number FROM commits WHERE sha=?", ("a" * 40,)
        )
        sha, pr_number = cur.fetchone()
        assert sha == "a" * 40
        assert pr_number == 4242


def test_commit_with_null_pr_number_upserts(tmp_path: Path) -> None:
    db = tmp_path / "rag.db"
    with Store.open(db) as store, store.batch():
        store.upsert_commit(_make_commit("b" * 40, pr_number=None), embedding=_one_vector())

    with Store.open(db) as store:
        cur = store.conn.execute(
            "SELECT sha, pr_number FROM commits WHERE sha=?", ("b" * 40,)
        )
        sha, pr_number = cur.fetchone()
        assert sha == "b" * 40
        assert pr_number is None


def _one_vector():
    import numpy as np

    rng = np.random.default_rng(seed=0)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / max(float((v @ v) ** 0.5), 1e-12)
