"""Hybrid retriever (semantic + BM25 + RRF) over the rag store.

The :class:`Retriever` is the only read path through which downstream
consumers (the MCP server, the Rust harness) reach the
sqlite-vec + FTS5 store. The store layer writes; this layer
reads. The split keeps query semantics in one place — without it,
each consumer would re-invent the RRF aggregation slightly
differently.

Algorithm (per the design spec):

1. Embed the query → 1024-d L2-normalised vector.
2. Run a vector kNN against ``<table>_vectors`` (capped at
   ``config.k_vec`` candidates) — sqlite-vec returns rowids + cosine
   distances.
3. Run an FTS5 BM25 query against ``<table>_fts`` (capped at
   ``config.k_bm25`` candidates) — FTS5 returns rowids + negative
   BM25 ranks (lower is better).
4. Aggregate via Reciprocal Rank Fusion (RRF). For each rowid, the
   fused score is ``sum(1 / (config.rrf_k + rank_i))`` across each
   list it appears in. Higher score is better.
5. Sort by score desc, take top-k, then issue a single
   ``WHERE pk IN (?, ?, ...)`` against the metadata table to hydrate
   the rows. Never JOIN vec0 with regular tables — that's the rule
   from ``codescribe_train/rag/store/AGENTS.md``.

The retriever is **read-only**. It opens no transactions, runs no
``INSERT``/``UPDATE``/``DELETE``, and treats the :class:`Store` handle
solely as a borrowed sqlite connection.
"""

from __future__ import annotations

import gzip
import logging
from typing import TYPE_CHECKING

from codescribe_train.rag.store.types import (
    PRDiff,
    RetrieveConfig,
    RetrievedCommit,
    RetrievedDocChunk,
    RetrievedIssue,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from codescribe_train.rag.embed.embedder import Embedder
    from codescribe_train.rag.store.writer import Store

logger = logging.getLogger(__name__)

# Commit diff_excerpt cap. Commits ship with the full unified diff
# inlined; capping the response keeps MCP/agent payloads small.
_COMMIT_DIFF_EXCERPT_CHARS = 2000

# Issue body_excerpt cap. The body is the issue's original description;
# capping it keeps the RetrievedIssue payload small for the agent.
_ISSUE_BODY_EXCERPT_CHARS = 500

# Number of fix-PR chunks attached to a RetrievedIssue when a linked PR
# clears the min_confidence threshold. Locked by spec (the design spec).
_FIX_EXCERPT_CHUNKS = 3

# Doc chunk excerpt cap — docs can be long; keep the agent payload small.
_DOC_CHUNK_EXCERPT_CHARS = 1500


def _truncate_with_marker(text: str, max_chars: int) -> tuple[str, bool]:
    """Clip ``text`` to ``max_chars`` and append a deterministic marker.

    Returns ``(clipped_text, was_truncated)``. The marker shape is
    ``\\n\\n[...truncated; full diff is <N> chars]`` — stable so tests
    and the MCP layer can detect truncation by substring.
    """
    if len(text) <= max_chars:
        return text, False
    head = text[:max_chars]
    marker = f"\n\n[...truncated; full diff is {len(text)} chars]"
    return head + marker, True


def _embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Pack a 1024-dim float32 vector into sqlite-vec's byte layout."""
    import numpy as np

    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 1024:
        raise ValueError(f"query embedding must be 1024-dim, got {arr.shape[0]}")
    return arr.tobytes()


def _rrf_fuse(
    *ranked_lists: Sequence[int],
    rrf_k: int,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over any number of ranked rowid lists.

    Each input is a sequence of rowids ordered best-first (rank 1, 2,
    3, ...). For each unique rowid the fused score is the sum of
    ``1 / (rrf_k + rank_i)`` over each list it appears in. Returns a
    list of ``(rowid, score)`` tuples sorted by score descending.

    A rowid that appears in only one list still gets a real (non-zero)
    score so the retriever can return useful results when one signal
    silently underperforms (e.g. an empty FTS query on a non-English
    string).
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, rowid in enumerate(ranked, start=1):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (rrf_k + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused


class Retriever:
    """Hybrid (semantic + BM25 + RRF) read-only retriever over the rag store.

    Holds a borrowed :class:`Store` handle and an :class:`Embedder`. The
    store provides the sqlite-vec connection; the embedder converts the
    query string into the same 1024-d vector space the write path used.

    The retriever issues queries only — never writes — so callers may
    share one ``Store`` between the indexer and the retriever (under
    WAL the read does not block the write).
    """

    def __init__(
        self,
        store: Store,
        embedder: Embedder | None,
        config: RetrieveConfig,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config

    # --- internal: hybrid candidate generators -------------------------

    def _vector_candidates(
        self,
        table: str,
        query_vec_bytes: bytes,
        k: int,
    ) -> list[int]:
        """Return rowids from a vec0 kNN query, ordered best-first.

        sqlite-vec 0.1.9 requires the ``k = ?`` clause inside the
        ``WHERE`` (it isn't a hint — omit it and you get a syntax
        error). Distance is cosine for normalised vectors; we don't
        return it because RRF only needs the rank order.
        """
        rows = self.store.conn.execute(
            f"""
            SELECT rowid
            FROM {table}
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,  # noqa: S608 — table name is module-internal, not user input
            (query_vec_bytes, k),
        ).fetchall()
        return [r[0] for r in rows]

    def _fts_candidates(self, table: str, query: str, k: int) -> list[int]:
        """Return rowids from an FTS5 BM25 query, ordered best-first.

        FTS5 ``MATCH`` raises ``OperationalError`` on syntactically
        invalid queries (e.g. a query that's all punctuation). The
        retriever swallows those: the BM25 list comes back empty and
        the vector list alone drives the result. This matches the
        spirit of RRF — it degrades gracefully when one signal goes
        silent.
        """
        import sqlite3

        try:
            rows = self.store.conn.execute(
                f"""
                SELECT rowid
                FROM {table}
                WHERE {table} MATCH ?
                ORDER BY rank
                LIMIT ?
                """,  # noqa: S608 — table name is module-internal, not user input
                (query, k),
            ).fetchall()
        except sqlite3.OperationalError:
            logger.warning("fts5 query rejected; falling back to vector-only", exc_info=True)
            return []
        return [r[0] for r in rows]

    def _hybrid_rowids(
        self,
        vec_table: str,
        fts_table: str,
        query: str,
        k: int,
    ) -> list[tuple[int, float]]:
        """Run both retrieval arms + RRF; return ``(rowid, score)`` top-k.

        Embeds the query exactly once and reuses the result for both
        the vector kNN and (indirectly, via the FTS5 query string) the
        BM25 search.
        """
        if self.embedder is None:
            raise RuntimeError(
                "Retriever requires an embedder for hybrid queries; "
                "only get_pr_diff() may be called without one"
            )
        query_vec = self.embedder.embed([query])[0]
        query_vec_bytes = _embedding_to_bytes(query_vec)
        vec_rowids = self._vector_candidates(vec_table, query_vec_bytes, self.config.k_vec)
        fts_rowids = self._fts_candidates(fts_table, query, self.config.k_bm25)
        fused = _rrf_fuse(vec_rowids, fts_rowids, rrf_k=self.config.rrf_k)
        return fused[:k]

    # --- find_similar_issues ------------------------------------------

    def _fetch_fix_excerpt(self, pr_number: int) -> str | None:
        """Return the first ``_FIX_EXCERPT_CHUNKS`` chunks of a PR, joined.

        The spec says "first 3 chunks". Chunks are ordered by
        ``chunk_id`` (the auto-incrementing PK) so the first 3 are the
        earliest-inserted — which matches the chunker's
        per-file/per-hunk traversal order. Returns ``None`` only if the
        PR has no chunks at all (e.g. an empty-diff PR).
        """
        rows = self.store.conn.execute(
            """
            SELECT chunk_text FROM pr_chunks
            WHERE pr_number = ?
            ORDER BY chunk_id
            LIMIT ?
            """,
            (pr_number, _FIX_EXCERPT_CHUNKS),
        ).fetchall()
        if not rows:
            return None
        return "\n\n".join(r[0] for r in rows)

    def _best_linked_pr(
        self,
        issue_number: int,
        min_confidence: float,
    ) -> tuple[int, float] | None:
        """Return ``(pr_number, confidence)`` of the highest-conf linked PR.

        Filters by ``confidence >= min_confidence``. Ties broken by the
        smaller ``pr_number`` for stability — irrelevant in production
        (real confidences won't tie at float precision) but deterministic
        for tests.
        """
        row = self.store.conn.execute(
            """
            SELECT pr_number, confidence
            FROM issue_pr_links
            WHERE issue_number = ? AND confidence >= ?
            ORDER BY confidence DESC, pr_number ASC
            LIMIT 1
            """,
            (issue_number, min_confidence),
        ).fetchone()
        if row is None:
            return None
        return int(row[0]), float(row[1])

    def find_similar_issues(
        self,
        query: str,
        k: int = 5,
        min_confidence: float = 0.8,
    ) -> list[RetrievedIssue]:
        """Top-k issues by RRF, each enriched with its best-fit fix PR.

        For each retrieved issue, the highest-confidence linked PR
        whose confidence meets ``min_confidence`` is attached as
        ``fix_pr_number`` and that PR's first 3 chunks (joined) become
        ``fix_excerpt``. When no link clears the threshold both fields
        are ``None`` and ``confidence`` is 0.0.

        The retriever doesn't re-rank by ``confidence`` — RRF already
        decided the ordering by topical relevance; confidence is purely
        a quality signal for the PR enrichment.
        """
        fused = self._hybrid_rowids(
            vec_table="issue_vectors",
            fts_table="issue_fts",
            query=query,
            k=k,
        )
        if not fused:
            return []
        issue_numbers = [rowid for rowid, _ in fused]
        score_by_issue = dict(fused)
        placeholders = ",".join(["?"] * len(issue_numbers))
        rows = self.store.conn.execute(
            f"""
            SELECT issue_number, title, body, state
            FROM issues WHERE issue_number IN ({placeholders})
            """,  # noqa: S608 — placeholders only, no user input in the SQL string
            issue_numbers,
        ).fetchall()
        by_number: dict[int, RetrievedIssue] = {}
        for issue_number, title, body, state in rows:
            link = self._best_linked_pr(issue_number, min_confidence=min_confidence)
            if link is None:
                fix_pr_number: int | None = None
                fix_excerpt: str | None = None
                confidence = 0.0
            else:
                fix_pr_number, confidence = link
                fix_excerpt = self._fetch_fix_excerpt(fix_pr_number)
            body_excerpt = (body or "")[:_ISSUE_BODY_EXCERPT_CHARS]
            by_number[issue_number] = RetrievedIssue(
                issue_number=issue_number,
                title=title,
                body_excerpt=body_excerpt,
                state=state,
                confidence=confidence,
                fix_pr_number=fix_pr_number,
                fix_excerpt=fix_excerpt,
                score=score_by_issue[issue_number],
            )
        # Re-order by the RRF score (WHERE IN doesn't preserve it).
        return [by_number[n] for n in issue_numbers if n in by_number]

    # --- search_commits ------------------------------------------------

    def search_commits(self, query: str, k: int = 5) -> list[RetrievedCommit]:
        """Return the top-k commits by RRF-fused (vector + BM25) score.

        No PR linkage is applied — commits with ``pr_number IS NULL``
        are first-class results. The caller (the MCP tool or the agent)
        is free to filter by ``commit.pr_number`` in post.
        """
        fused = self._hybrid_rowids(
            vec_table="commit_vectors",
            fts_table="commit_fts",
            query=query,
            k=k,
        )
        if not fused:
            return []
        rowids = [rowid for rowid, _ in fused]
        score_by_rowid = dict(fused)
        placeholders = ",".join(["?"] * len(rowids))
        rows = self.store.conn.execute(
            f"""
            SELECT rowid, sha, message, author, authored_at, file_count, diff_text
            FROM commits WHERE rowid IN ({placeholders})
            """,  # noqa: S608 — placeholders only, no user input in the SQL string
            rowids,
        ).fetchall()
        by_rowid: dict[int, RetrievedCommit] = {}
        for rowid, sha, message, author, authored_at, file_count, diff_blob in rows:
            diff_text = gzip.decompress(diff_blob).decode("utf-8") if diff_blob else ""
            excerpt, _ = _truncate_with_marker(diff_text, _COMMIT_DIFF_EXCERPT_CHARS)
            by_rowid[rowid] = RetrievedCommit(
                sha=sha,
                message=message,
                author=author,
                authored_at=authored_at,
                files=file_count or 0,
                diff_excerpt=excerpt,
                score=score_by_rowid[rowid],
            )
        # Re-order by the RRF score (the WHERE IN doesn't preserve it).
        return [by_rowid[rowid] for rowid in rowids if rowid in by_rowid]

    # --- search_docs --------------------------------------------------

    def search_docs(self, query: str, k: int = 5) -> list[RetrievedDocChunk]:
        """Return the top-k worktree-doc chunks by RRF-fused score.

        Hybrid (vector + BM25 + RRF) over ``doc_chunk_vectors`` /
        ``doc_chunk_fts`` — same algorithm as find_similar_issues /
        search_commits, no entity-specific enrichment. Each result is a
        single chunk with its source path + nearest heading so the agent
        can cite the file and (optionally) follow up with the repo-docs
        ``read_doc`` tool for the full file.
        """
        fused = self._hybrid_rowids(
            vec_table="doc_chunk_vectors",
            fts_table="doc_chunk_fts",
            query=query,
            k=k,
        )
        if not fused:
            return []
        chunk_ids = [rowid for rowid, _ in fused]
        score_by_id = dict(fused)
        placeholders = ",".join(["?"] * len(chunk_ids))
        rows = self.store.conn.execute(
            f"""
            SELECT chunk_id, doc_path, heading, chunk_text
            FROM doc_chunks WHERE chunk_id IN ({placeholders})
            """,  # noqa: S608 — placeholders only, no user input in the SQL string
            chunk_ids,
        ).fetchall()
        by_id: dict[int, RetrievedDocChunk] = {}
        for chunk_id, doc_path, heading, chunk_text in rows:
            excerpt, _ = _truncate_with_marker(chunk_text or "", _DOC_CHUNK_EXCERPT_CHARS)
            by_id[chunk_id] = RetrievedDocChunk(
                doc_path=doc_path,
                heading=heading or "",
                text_excerpt=excerpt,
                score=score_by_id[chunk_id],
            )
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    # --- get_pr_diff ---------------------------------------------------

    def get_pr_diff(self, pr_number: int, max_chars: int | None = None) -> PRDiff:
        """Return the gzip-decoded PR diff + metadata, optionally truncated.

        ``max_chars=None`` returns the full decoded text and
        ``truncated=False``. With ``max_chars`` set the text is
        clipped to the requested length and the truncation marker
        is appended (so callers can substring-match it without
        re-reading the row).
        """
        row = self.store.conn.execute(
            """
            SELECT title, author, merged_at, file_count, diff_text
            FROM pulls WHERE pr_number = ?
            """,
            (pr_number,),
        ).fetchone()
        if row is None:
            raise KeyError(f"PR #{pr_number} not found in store")
        title, author, merged_at, file_count, diff_blob = row
        diff_text = gzip.decompress(diff_blob).decode("utf-8") if diff_blob else ""
        truncated = False
        if max_chars is not None:
            diff_text, truncated = _truncate_with_marker(diff_text, max_chars)
        return PRDiff(
            pr_number=pr_number,
            title=title,
            author=author,
            merged_at=merged_at,
            files=file_count or 0,
            diff_text=diff_text,
            truncated=truncated,
        )
