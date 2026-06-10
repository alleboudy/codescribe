"""RRF fusion picks the right document when neither signal alone does.

Spec test from the design spec:

  > test_retrieve_hybrid_ranks.py — fixture corpus where the right
  > answer ranks #1 by vector but #5 by BM25 (or vice versa); RRF puts
  > it #1.

The test crafts a small corpus that defeats either retrieval arm in
isolation:

* **T** is the correct answer. Its vector matches the query exactly
  (vector rank 1). Its body has only one "cart" hit — three other
  issues outrank it on BM25 by repetition (BM25 rank ~5).
* **V** is the vector-only winner. Its vector is close to the query
  (vector rank 2) but its body shares no terms with the query
  (absent from the BM25 list).
* **B** is the BM25-only winner. Its body crams in "cart crash"
  repeats (BM25 rank 1) but its vector is orthogonal to the query,
  and we cap ``k_vec=2`` so B never enters the vector list.

The RRF defaults (``rrf_k=60``) yield:

  * T's score = ``1/(60+1) + 1/(60+5)`` ≈ 0.0318
  * B's score = ``1/(60+1)``              ≈ 0.0164
  * V's score = ``1/(60+2)``              ≈ 0.0161

so T (in both lists) beats either single-list winner — exactly the
fusion behaviour the design spec asks us to lock in.
"""

from __future__ import annotations

import numpy as np

from codescribe_train.rag.sources.github_source import Issue
from codescribe_train.rag.store.writer import Store


class _StubEmbedder:
    """In-test embedder: returns a fixed query vector regardless of input."""

    def __init__(self, query_vector: np.ndarray) -> None:
        self._q = query_vector

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._q for _ in texts], axis=0).astype(np.float32)


def _basis_vec(index: int) -> np.ndarray:
    """A 1024-d L2-normalised vector with all weight on dimension ``index``."""
    v = np.zeros(1024, dtype=np.float32)
    v[index] = 1.0
    return v


def _mix(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """L2-normalised mixture of basis vectors ``a`` and ``b`` at weight ``w``."""
    out = (1 - w) * a + w * b
    return out / np.linalg.norm(out)


def _make_issue(number: int, title: str, body: str) -> Issue:
    return Issue(
        number=number,
        title=title,
        body=body,
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


def test_rrf_beats_either_signal_alone(tmp_path) -> None:
    """T (vector #1, BM25 #5) outranks the pure-vector and pure-BM25 winners."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    query_vec = _basis_vec(0)  # query direction
    far_vec = _basis_vec(900)  # orthogonal to query

    # Tight caps so the corpus stays small and B is forced out of the
    # vector list while staying in the BM25 list. The k_vec/k_bm25 cap
    # is the mechanism that lets RRF beat either arm in isolation.
    config = RetrieveConfig(k_vec=2, k_bm25=20, rrf_k=60)

    # Issue 10 = T: vector rank 1, body has one "cart crash" hit.
    # Issue 20 = V: vector rank 2 (close cosine), body has no overlap.
    # Issue 30 = B: BM25 rank 1 (10 cart-crashes), vector orthogonal.
    # Issues 41..49 = noise: bodies that out-rank T on BM25 by repetition.
    # The numbering uses gaps so an off-by-one in the IN-clause query
    # would surface as a wrong issue_number instead of a near-miss.
    issues: list[tuple[Issue, np.ndarray]] = [
        (
            _make_issue(10, "T (correct)", "cart crash once"),
            query_vec,
        ),
        (
            _make_issue(20, "V (vector-only)", "quilts recipes baking"),
            _mix(query_vec, far_vec, 0.05),  # cosine ≈ 0.999 → rank 2
        ),
        (
            _make_issue(
                30,
                "B (BM25-only)",
                # 10 cart + 10 crash → towers over T on BM25 alone.
                "cart crash " * 10,
            ),
            far_vec,  # cosine = 0 → ranks below k_vec=2
        ),
    ]
    # Noise issues that out-rank T on BM25 (three with 2-3 hits, so T
    # lands at BM25 rank ~5 not rank 2).
    noise_specs = [
        (41, "N1", "cart crash cart crash"),
        (42, "N2", "cart crash cart"),
        (43, "N3", "cart crash"),
    ]
    for num, title, body in noise_specs:
        # Distinct orthogonal directions so noise doesn't sneak into
        # the top-2 vector list.
        issues.append((_make_issue(num, title, body), _basis_vec(num + 100)))

    with Store.open(tmp_path / "rag.db") as store:
        for issue, vec in issues:
            store.upsert_issue(issue, vec)

    embedder = _StubEmbedder(query_vec)
    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=embedder, config=config)
        hits = retriever.find_similar_issues(
            query="cart crash",
            k=5,
            min_confidence=0.8,
        )

    assert hits, "retriever returned no results"
    top = hits[0]
    assert top.issue_number == 10, (
        f"expected issue 10 (T) at rank 1; got #{top.issue_number} "
        f"with full ordering {[h.issue_number for h in hits]}"
    )


def test_top_result_carries_score_and_excerpt(tmp_path) -> None:
    """Returned :class:`RetrievedIssue` has a non-zero score and a body excerpt."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    query_vec = _basis_vec(0)
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_issue(
            _make_issue(7, "Lone match", "cart body content for FTS to grab"),
            query_vec,
        )

    embedder = _StubEmbedder(query_vec)
    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
        hits = retriever.find_similar_issues(query="cart", k=5)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.issue_number == 7
    assert hit.title == "Lone match"
    assert "cart" in hit.body_excerpt.lower()
    assert hit.state == "open"
    assert hit.score > 0.0
    # No PR linked → fix_pr_number and fix_excerpt are None.
    assert hit.fix_pr_number is None
    assert hit.fix_excerpt is None
    # No PR linkage → no confidence to report; spec dataclass uses 0.0 sentinel.
    assert hit.confidence == 0.0
