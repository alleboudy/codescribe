"""``find_similar_issues`` attaches the linked PR's first 3 chunks.

Spec test from the design spec:

  > test_retrieve_attaches_pr_excerpt.py — issue with linked PR at
  > conf 1.0 → returns first 3 chunks.

Chunks are ordered by ``chunk_id`` (the auto-incrementing PK), so
"first 3" is the earliest 3 ``pr_chunks`` rows for that PR — i.e. the
first 3 hunks the chunker emitted from the diff (which the chunker
walks per-file in order).
"""

from __future__ import annotations

import numpy as np

from codescribe_train.rag.embed.chunker import PRChunk
from codescribe_train.rag.extract.pairing import IssuePRLink
from codescribe_train.rag.sources.github_source import Issue, PullRequest
from codescribe_train.rag.store.writer import Store


class _StubEmbedder:
    def __init__(self, query_vector: np.ndarray) -> None:
        self._q = query_vector

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._q for _ in texts], axis=0).astype(np.float32)


def _basis_vec(index: int) -> np.ndarray:
    v = np.zeros(1024, dtype=np.float32)
    v[index] = 1.0
    return v


def _make_issue(number: int) -> Issue:
    return Issue(
        number=number,
        title="crash",
        body="cart body",
        state="open",
        state_reason=None,
        labels=[],
        assignees=[],
        author="example-org",
        created_at="2026-05-01T12:00:00Z",
        updated_at="2026-05-02T15:00:00Z",
        closed_at=None,
        raw_json=f'{{"number": {number}}}',
    )


def _make_pull(number: int) -> PullRequest:
    return PullRequest(
        number=number,
        title="fix",
        body=None,
        state="merged",
        head_sha="a" * 40,
        base_branch="main",
        author="example-org",
        draft=False,
        created_at="2026-05-01T10:00:00Z",
        updated_at="2026-05-02T11:00:00Z",
        merged_at="2026-05-02T11:00:00Z",
        closed_at="2026-05-02T11:00:00Z",
        raw_json=f'{{"number": {number}}}',
    )


def test_excerpt_contains_first_three_chunks(tmp_path) -> None:
    """A 5-chunk PR linked at conf 1.0 returns just its first 3 chunks."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    chunks = [
        PRChunk(
            pr_number=42,
            file_path=f"f{i}.py",
            hunk_index=i,
            chunk_text=f"CHUNK_{i}_BODY",
        )
        for i in range(5)
    ]
    query_vec = _basis_vec(0)

    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue(_make_issue(7), query_vec)
        store.upsert_pull(
            pr=_make_pull(42),
            chunks=chunks,
            chunk_embeddings=[_basis_vec(100 + i) for i in range(5)],
            pr_summary_embedding=_basis_vec(99),
        )
        store.upsert_issue_pr_link(
            IssuePRLink(
                issue_number=7, pr_number=42, source="closingIssuesReferences",
                confidence=1.0,
            )
        )

    embedder = _StubEmbedder(query_vec)
    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
        hits = retriever.find_similar_issues(query="cart", k=5)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.fix_pr_number == 42
    assert hit.fix_excerpt is not None
    # First three chunks present; chunks 3 and 4 NOT present.
    assert "CHUNK_0_BODY" in hit.fix_excerpt
    assert "CHUNK_1_BODY" in hit.fix_excerpt
    assert "CHUNK_2_BODY" in hit.fix_excerpt
    assert "CHUNK_3_BODY" not in hit.fix_excerpt
    assert "CHUNK_4_BODY" not in hit.fix_excerpt


def test_pr_with_no_chunks_yields_empty_excerpt(tmp_path) -> None:
    """A linked PR that has zero chunks → fix_excerpt is None (still a hit)."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue(_make_issue(7), _basis_vec(0))
        store.upsert_pull(
            pr=_make_pull(42),
            chunks=[],  # empty PR, no diff hunks
            chunk_embeddings=[],
            pr_summary_embedding=_basis_vec(99),
        )
        store.upsert_issue_pr_link(
            IssuePRLink(
                issue_number=7, pr_number=42, source="closingIssuesReferences",
                confidence=1.0,
            )
        )

    embedder = _StubEmbedder(_basis_vec(0))
    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
        hits = retriever.find_similar_issues(query="cart", k=5)

    assert hits[0].fix_pr_number == 42
    assert hits[0].fix_excerpt is None  # no chunks → no excerpt


def test_pr_with_fewer_than_three_chunks_returns_all(tmp_path) -> None:
    """A 2-chunk PR returns both chunks (LIMIT 3 is a max, not a requirement)."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    chunks = [
        PRChunk(pr_number=42, file_path="a.py", hunk_index=0, chunk_text="ALPHA"),
        PRChunk(pr_number=42, file_path="b.py", hunk_index=0, chunk_text="BETA"),
    ]
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue(_make_issue(7), _basis_vec(0))
        store.upsert_pull(
            pr=_make_pull(42),
            chunks=chunks,
            chunk_embeddings=[_basis_vec(100), _basis_vec(101)],
            pr_summary_embedding=_basis_vec(99),
        )
        store.upsert_issue_pr_link(
            IssuePRLink(
                issue_number=7, pr_number=42, source="closingIssuesReferences",
                confidence=1.0,
            )
        )

    embedder = _StubEmbedder(_basis_vec(0))
    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
        hits = retriever.find_similar_issues(query="cart", k=5)

    assert "ALPHA" in hits[0].fix_excerpt
    assert "BETA" in hits[0].fix_excerpt
