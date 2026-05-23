from __future__ import annotations

import gzip
import logging
import sqlite3

import numpy as np

from ..embed.chunker import _split_by_file
from .types import CLDiff, FileDiff, RetrievedBugFix

logger = logging.getLogger(__name__)

RRF_CONSTANT = 60   # textbook default; tune only after measuring lift


class Retriever:
    """Hybrid retrieval: vector (sqlite-vec) + BM25 (FTS5), merged via RRF.

    vec0 cannot JOIN regular tables, so we fetch ids from the indices, then look
    up metadata + fix links in Python.
    """

    def __init__(self, conn: sqlite3.Connection, embedder) -> None:
        self._conn = conn
        self._embedder = embedder

    def find_similar_bugs(
        self,
        query: str,
        k: int = 5,
        k_vec: int = 20,
        k_bm25: int = 20,
        confidence_threshold: float = 0.8,
    ) -> list[RetrievedBugFix]:
        q_emb = self._embedder.embed_one(query)
        if abs(float(np.linalg.norm(q_emb)) - 1.0) > 0.01:
            raise RuntimeError("embedder must return L2-normalised vectors")

        merged = self._rrf_merge(self._vec_ranks(q_emb, k_vec), self._bm25_ranks(query, k_bm25))
        top = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:k]

        results: list[RetrievedBugFix] = []
        for bug_id, rrf_score in top:
            row = self._conn.execute(
                "SELECT summary, severity, status FROM bugs WHERE bug_id = ?",
                (bug_id,)).fetchone()
            if row is None:
                continue
            summary, severity, status = row
            fix_cl, confidence, excerpt = self._resolve_fix(bug_id, confidence_threshold)
            results.append(RetrievedBugFix(
                bug_id=bug_id, summary=summary, severity=severity, status=status,
                score=rrf_score, fix_cl=fix_cl, fix_diff_excerpt=excerpt, confidence=confidence))
        return results

    def get_fix_diff(self, cl_number: int, max_chars: int | None = None) -> CLDiff:
        row = self._conn.execute(
            "SELECT author, submitted_at, description, diff_text FROM changes WHERE cl_number = ?",
            (cl_number,)).fetchone()
        if row is None:
            raise KeyError(f"no such CL: {cl_number}")
        author, submitted_at, description, blob = row
        full = gzip.decompress(blob).decode("utf-8", "replace")
        if max_chars is not None and len(full) > max_chars:
            full = full[:max_chars] + "\n... (truncated)"
        blocks = _split_by_file(full) or [("", full)]
        files = tuple(FileDiff(fp, d) for fp, d in blocks)
        return CLDiff(cl_number=cl_number, author=author or "", submitted_at=submitted_at or "",
                      description=description or "", files=files)

    # ------------------------------------------------------------------
    def _vec_ranks(self, q_emb: np.ndarray, k: int) -> dict[int, int]:
        # k is an int we control; inline it (sqlite-vec KNN wants a concrete LIMIT).
        rows = self._conn.execute(
            f"SELECT rowid, distance FROM bug_vectors WHERE embedding MATCH ? "
            f"ORDER BY distance LIMIT {int(k)}",
            (q_emb.astype(np.float32).tobytes(),)).fetchall()
        return {row[0]: rank + 1 for rank, row in enumerate(rows)}

    def _bm25_ranks(self, query: str, k: int) -> dict[int, int]:
        terms = self._sanitise_fts_query(query)
        if not terms:
            return {}
        rows = self._conn.execute(
            "SELECT bug_id, bm25(bug_fts) AS score FROM bug_fts "
            "WHERE bug_fts MATCH ? ORDER BY score LIMIT ?",
            (terms, k)).fetchall()
        # FTS5 bm25() is negative (lower = better); ORDER BY ASC ranks best first.
        return {row[0]: rank + 1 for rank, row in enumerate(rows)}

    @staticmethod
    def _sanitise_fts_query(query: str) -> str:
        """Drop characters that break FTS5 query syntax; OR the surviving terms."""
        cleaned = "".join(c if c.isalnum() or c in " -_" else " " for c in query)
        tokens = [t for t in cleaned.split() if t]
        return " OR ".join(f'"{t}"' for t in tokens)

    @staticmethod
    def _rrf_merge(*rankings: dict[int, int], k: int = RRF_CONSTANT) -> dict[int, float]:
        merged: dict[int, float] = {}
        for ranking in rankings:
            for doc_id, rank in ranking.items():
                merged[doc_id] = merged.get(doc_id, 0.0) + 1.0 / (k + rank)
        return merged

    def _resolve_fix(self, bug_id: int, threshold: float) -> tuple[int | None, float, str | None]:
        row = self._conn.execute(
            """SELECT cl_number, confidence FROM fix_links
               WHERE bug_id = ? AND confidence >= ?
               ORDER BY confidence DESC LIMIT 1""",
            (bug_id, threshold)).fetchone()
        if row is None:
            return None, float("nan"), None
        cl_number, conf = row
        return cl_number, conf, self._get_diff_excerpt(cl_number, max_chars=800)

    def _get_diff_excerpt(self, cl_number: int, max_chars: int) -> str:
        row = self._conn.execute(
            "SELECT diff_text FROM changes WHERE cl_number = ?", (cl_number,)).fetchone()
        if row is None:
            return ""
        full = gzip.decompress(row[0]).decode("utf-8", "replace")
        if len(full) <= max_chars:
            return full
        return full[:max_chars] + "\n... (truncated)"
