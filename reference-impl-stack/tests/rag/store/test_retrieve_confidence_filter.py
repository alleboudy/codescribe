"""``find_similar_issues`` filters linked PRs by ``min_confidence``.

Spec test from the design spec:

  > test_retrieve_confidence_filter.py — issue with linked PR at
  > conf 0.7 → fix_pr_number=None at default threshold 0.8.

The pairing layer attaches confidences from 0.3 (temporal author
match) to 1.0 (GraphQL closing reference). The retriever's default
``min_confidence=0.8`` is the strict threshold from the design spec: anything
under it is too noisy to surface as a "fix" suggestion.
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
        body="closes",
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


def _seed_issue_with_link(tmp_path, *, confidence: float) -> None:
    query_vec = _basis_vec(0)
    issue = _make_issue(7)
    pr = _make_pull(42)
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue(issue, query_vec)
        store.upsert_pull(
            pr=pr,
            chunks=[PRChunk(pr_number=42, file_path="x", hunk_index=0, chunk_text="x")],
            chunk_embeddings=[_basis_vec(100)],
            pr_summary_embedding=_basis_vec(101),
        )
        store.upsert_issue_pr_link(
            IssuePRLink(
                issue_number=7,
                pr_number=42,
                source="custom",
                confidence=confidence,
            )
        )


def test_link_below_default_threshold_yields_none(tmp_path) -> None:
    """Conf 0.7 < default 0.8 → ``fix_pr_number`` is None and excerpt is None."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    _seed_issue_with_link(tmp_path, confidence=0.7)

    embedder = _StubEmbedder(_basis_vec(0))
    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
        # default min_confidence is 0.8
        hits = retriever.find_similar_issues(query="cart", k=5)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.issue_number == 7
    assert hit.fix_pr_number is None
    assert hit.fix_excerpt is None
    assert hit.confidence == 0.0


def test_link_at_threshold_is_included(tmp_path) -> None:
    """Conf == 0.8 at default threshold → fix_pr_number populated."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    _seed_issue_with_link(tmp_path, confidence=0.8)

    embedder = _StubEmbedder(_basis_vec(0))
    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
        hits = retriever.find_similar_issues(query="cart", k=5)

    assert hits[0].fix_pr_number == 42
    assert hits[0].confidence == 0.8


def test_custom_min_confidence_can_relax_threshold(tmp_path) -> None:
    """Caller can drop ``min_confidence`` to admit lower-confidence links."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    _seed_issue_with_link(tmp_path, confidence=0.5)

    embedder = _StubEmbedder(_basis_vec(0))
    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
        hits = retriever.find_similar_issues(
            query="cart", k=5, min_confidence=0.4,
        )

    assert hits[0].fix_pr_number == 42
    assert hits[0].confidence == 0.5


def test_highest_confidence_link_wins(tmp_path) -> None:
    """When an issue has multiple linked PRs, the highest-conf one is attached."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    query_vec = _basis_vec(0)
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue(_make_issue(7), query_vec)
        for pr_number, conf in [(42, 0.85), (43, 1.0), (44, 0.9)]:
            store.upsert_pull(
                pr=_make_pull(pr_number),
                chunks=[
                    PRChunk(
                        pr_number=pr_number, file_path="x", hunk_index=0, chunk_text="x",
                    )
                ],
                chunk_embeddings=[_basis_vec(100 + pr_number)],
                pr_summary_embedding=_basis_vec(200 + pr_number),
            )
            store.upsert_issue_pr_link(
                IssuePRLink(
                    issue_number=7, pr_number=pr_number, source="x", confidence=conf,
                )
            )

    embedder = _StubEmbedder(query_vec)
    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
        hits = retriever.find_similar_issues(query="cart", k=5)

    assert hits[0].fix_pr_number == 43  # conf 1.0 wins
    assert hits[0].confidence == 1.0
