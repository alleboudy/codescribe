"""Pull-with-chunks fan-out into pulls + pr_vectors + pr_chunks + pr_chunk_vectors.

Acceptance: spec test #3 from the design spec — "pull with 3 chunks → 3 chunk
rows + 3 chunk vectors".
"""

from __future__ import annotations

import numpy as np

from codescribe_train.rag.embed.chunker import PRChunk
from codescribe_train.rag.sources.github_source import PullRequest
from codescribe_train.rag.store.writer import Store


def _make_pull(number: int = 42) -> PullRequest:
    return PullRequest(
        number=number,
        title="fix: cart item loader",
        body="Closes #7. Patches the LRU eviction order.",
        state="merged",
        head_sha="a" * 40,
        base_branch="main",
        author="example-org",
        draft=False,
        created_at="2026-05-01T10:00:00Z",
        updated_at="2026-05-02T11:00:00Z",
        merged_at="2026-05-02T11:00:00Z",
        closed_at="2026-05-02T11:00:00Z",
        raw_json='{"number": 42}',
    )


def _chunks(pr_number: int = 42) -> list[PRChunk]:
    return [
        PRChunk(
            pr_number=pr_number,
            file_path="src/cart.py",
            hunk_index=0,
            chunk_text="@@ -1,3 +1,3 @@\n-old\n+new",
        ),
        PRChunk(
            pr_number=pr_number,
            file_path="src/cart.py",
            hunk_index=1,
            chunk_text="@@ -10,3 +10,3 @@\n-x\n+y",
        ),
        PRChunk(
            pr_number=pr_number,
            file_path="tests/test_cart.py",
            hunk_index=0,
            chunk_text="@@ -5,1 +5,3 @@\n+assert_loaded()\n+assert_evicted()",
        ),
    ]


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def test_upsert_pull_writes_metadata_and_summary_vector(tmp_path) -> None:
    pr = _make_pull(number=42)
    chunks = _chunks(pr_number=42)
    chunk_embs = [_emb(1), _emb(2), _emb(3)]
    summary_emb = _emb(100)

    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_pull(
            pr=pr,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            pr_summary_embedding=summary_emb,
        )

        pull_row = store.conn.execute(
            "SELECT pr_number, title, state, head_sha, base_branch, file_count "
            "FROM pulls WHERE pr_number = ?",
            (42,),
        ).fetchone()
        summary_rows = store.conn.execute(
            "SELECT rowid FROM pr_vectors WHERE rowid = ?", (42,)
        ).fetchall()

    assert pull_row is not None
    assert pull_row[0] == 42
    assert "cart" in pull_row[1].lower()
    assert pull_row[2] == "merged"
    assert summary_rows == [(42,)]


def test_upsert_pull_creates_three_chunks_and_three_chunk_vectors(tmp_path) -> None:
    pr = _make_pull(number=42)
    chunks = _chunks(pr_number=42)
    chunk_embs = [_emb(i) for i in range(3)]

    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_pull(
            pr=pr,
            chunks=chunks,
            chunk_embeddings=chunk_embs,
            pr_summary_embedding=_emb(99),
        )

        chunk_rows = store.conn.execute(
            "SELECT pr_number, file_path, hunk_index FROM pr_chunks "
            "WHERE pr_number = ? ORDER BY chunk_id",
            (42,),
        ).fetchall()
        chunk_vec_count = store.conn.execute(
            "SELECT COUNT(*) FROM pr_chunk_vectors"
        ).fetchone()[0]

    assert len(chunk_rows) == 3
    assert chunk_vec_count == 3
    # Order preserved (file_path + hunk_index match the input list).
    assert chunk_rows[0] == (42, "src/cart.py", 0)
    assert chunk_rows[1] == (42, "src/cart.py", 1)
    assert chunk_rows[2] == (42, "tests/test_cart.py", 0)


def test_upsert_pull_chunk_vector_rowids_match_chunk_ids(tmp_path) -> None:
    """Each pr_chunk_vectors row's rowid must match the pr_chunks.chunk_id.

    Without this invariant, the retrieval path (vec0 returns rowids; the join
    happens in Python on chunk_id) cannot stitch chunks to their text.
    """
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_pull(
            pr=_make_pull(number=5),
            chunks=_chunks(pr_number=5),
            chunk_embeddings=[_emb(i) for i in range(3)],
            pr_summary_embedding=_emb(99),
        )

        chunk_ids = {
            r[0]
            for r in store.conn.execute(
                "SELECT chunk_id FROM pr_chunks WHERE pr_number = 5"
            ).fetchall()
        }
        vec_rowids = {
            r[0]
            for r in store.conn.execute(
                "SELECT rowid FROM pr_chunk_vectors"
            ).fetchall()
        }
    assert chunk_ids == vec_rowids
    assert len(chunk_ids) == 3


def test_upsert_pull_replaces_existing_chunks_on_reupsert(tmp_path) -> None:
    pr = _make_pull(number=9)
    first = _chunks(pr_number=9)
    second = first[:2]  # fewer chunks the second time

    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_pull(
            pr=pr,
            chunks=first,
            chunk_embeddings=[_emb(i) for i in range(3)],
            pr_summary_embedding=_emb(1),
        )
        store.upsert_pull(
            pr=pr,
            chunks=second,
            chunk_embeddings=[_emb(i) for i in range(2)],
            pr_summary_embedding=_emb(2),
        )

        chunk_count = store.conn.execute(
            "SELECT COUNT(*) FROM pr_chunks WHERE pr_number = 9"
        ).fetchone()[0]
        chunk_vec_count = store.conn.execute(
            "SELECT COUNT(*) FROM pr_chunk_vectors"
        ).fetchone()[0]

    assert chunk_count == 2
    assert chunk_vec_count == 2


def test_upsert_pull_writes_fts(tmp_path) -> None:
    pr = _make_pull(number=12)
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_pull(
            pr=pr,
            chunks=[],
            chunk_embeddings=[],
            pr_summary_embedding=_emb(0),
        )
        matches = store.conn.execute(
            "SELECT rowid FROM pr_fts WHERE pr_fts MATCH 'cart'"
        ).fetchall()
    assert matches == [(12,)]
