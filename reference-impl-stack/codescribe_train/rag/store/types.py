"""Read-side dataclasses for the hybrid retriever.

Lives in its own module (separate from ``writer.py``) so the read path
and the write path don't share import-time state — and so callers can
``from codescribe_train.rag.store.types import RetrievedIssue`` without
pulling in the gzip / vec0 plumbing from ``writer``.

The dataclasses match the shape of the rows the
:class:`~codescribe_train.rag.store.retrieve.Retriever` returns:

* :class:`RetrieveConfig` — knobs for the RRF aggregation (k_vec /
  k_bm25 / rrf_k). Defaults reconcile the design spec.
* :class:`RetrievedIssue` — one row from ``find_similar_issues``,
  optionally enriched with the highest-confidence linked PR and the
  first 3 of that PR's chunks.
* :class:`PRDiff` — the decoded PR diff + metadata returned by
  ``get_pr_diff``, with a ``truncated`` flag set when ``max_chars`` cut
  the diff short.
* :class:`RetrievedCommit` — one row from ``search_commits``; no PR
  linkage required.

All dataclasses are ``frozen=True`` — the retriever is read-only and
returning immutable rows makes it trivial for downstream consumers
(the MCP server and the agent wiring) to cache or hash them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrieveConfig:
    """Knobs for the hybrid retriever's RRF aggregation.

    Defaults: ``k_vec=20`` and ``k_bm25=20`` cap the per-list
    candidate pool that goes into Reciprocal Rank Fusion, and
    ``rrf_k=60`` is the textbook Cormack et al. 2009 default that
    smooths the per-rank reciprocal so rank 1 doesn't completely
    dominate rank 2.
    """

    k_vec: int = 20
    k_bm25: int = 20
    rrf_k: int = 60


@dataclass(frozen=True)
class RetrievedIssue:
    """One issue + its (optional) fix PR + that PR's first 3 chunks.

    ``fix_pr_number`` is ``None`` when no linked PR meets the
    ``min_confidence`` threshold passed to ``find_similar_issues``;
    ``fix_excerpt`` is correspondingly ``None`` in that case. When a
    fix PR is attached, ``confidence`` is the confidence of that
    issue↔PR link.

    ``score`` is the RRF fusion score over (vector rank, BM25 rank);
    higher is better, callers use it to display "why this came back".
    """

    issue_number: int
    title: str
    body_excerpt: str
    state: str
    confidence: float
    fix_pr_number: int | None
    fix_excerpt: str | None
    score: float


@dataclass(frozen=True)
class PRDiff:
    """The decoded PR diff + the row's identifying metadata.

    ``diff_text`` is the gzip-decoded ``pulls.diff_text`` blob.
    ``truncated`` is True iff ``max_chars`` cut the diff short *and*
    the truncation marker was appended to ``diff_text``.
    """

    pr_number: int
    title: str
    author: str | None
    merged_at: str | None
    files: int
    diff_text: str
    truncated: bool


@dataclass(frozen=True)
class RetrievedCommit:
    """One commit row from ``search_commits``.

    ``diff_excerpt`` is the gzip-decoded ``commits.diff_text`` blob,
    optionally truncated (the retriever caps the excerpt at a small
    constant to keep MCP payloads small). ``files`` is the
    cached ``file_count`` column.

    ``score`` is the RRF fusion score (higher is better) over
    (vector rank, BM25 rank).
    """

    sha: str
    message: str
    author: str | None
    authored_at: str | None
    files: int
    diff_excerpt: str
    score: float


@dataclass(frozen=True)
class RetrievedDocChunk:
    """One doc chunk returned by Retriever.search_docs."""

    doc_path: str
    heading: str
    text_excerpt: str
    score: float
