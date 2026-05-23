"""Streaming bootstrap indexer.

Synchronous and idempotent: re-running over the same data is a no-op (upserts +
a 'seen' guard). The #5 SS10 sketch uses asyncio producers/consumers; that's an
optional throughput optimisation, not required for correctness, so v1 streams
synchronously and stays fully testable without an event loop.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field

from ..config import RagConfig
from ..embed.chunker import chunk_diff
from ..extract.pairing import BugView, CLView, pair_bugs_to_cls

logger = logging.getLogger(__name__)


@dataclass
class BootstrapStats:
    bugs_added: int = 0
    cls_added: int = 0
    links_added: int = 0
    total_seconds: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)


def _existing(conn, table: str, col: str) -> set:
    return {r[0] for r in conn.execute(f"SELECT {col} FROM {table}").fetchall()}


def run_bootstrap(cfg: RagConfig, sources, store, embedder, *, batch_size: int = 256) -> BootstrapStats:
    started = time.perf_counter()
    stats = BootstrapStats()
    seen_bugs = _existing(store._conn, "bugs", "bug_id")
    seen_cls = _existing(store._conn, "changes", "cl_number")

    bug_views: list[BugView] = []
    cl_views: list[CLView] = []

    # --- bugs ---
    batch: list = []

    def flush_bugs() -> None:
        if not batch:
            return
        embs = embedder.embed([f"{b.summary} {b.description}" for b in batch])
        with store.batch():
            for bug, emb in zip(batch, embs):
                store.upsert_bug(bug, emb)
        stats.batch_sizes.append(len(batch))
        batch.clear()

    for bug in sources.iter_bugs():
        comments = sources.comments_for(bug.id)
        comment_text = "\n".join(c.text for c in comments)
        desc = comments[0].text if comments else ""          # comment 0 = description
        bug = dataclasses.replace(bug, description=desc)
        bug_views.append(BugView(bug.id, bug.summary, bug.status, bug.resolution,
                                 bug.assigned_to, bug.last_change_time, comment_text))
        if bug.id not in seen_bugs:
            batch.append(bug)
            stats.bugs_added += 1
            if len(batch) >= batch_size:
                flush_bugs()
    flush_bugs()

    # --- changelists ---
    clbatch: list = []

    def flush_cls() -> None:
        if not clbatch:
            return
        with store.batch():
            for cl in clbatch:
                chunks = chunk_diff(cl)
                chunk_embs = list(embedder.embed([c.text for c in chunks])) if chunks else []
                summary_emb = embedder.embed_one(cl.description or "")
                store.upsert_cl(cl, chunks, chunk_embs, summary_emb)
        stats.batch_sizes.append(len(clbatch))
        clbatch.clear()

    for cl in sources.iter_cls():
        cl_views.append(CLView(cl.cl, cl.author, cl.description, cl.submitted_at))
        if cl.cl not in seen_cls:
            clbatch.append(cl)
            stats.cls_added += 1
            if len(clbatch) >= batch_size:
                flush_cls()
    flush_cls()

    # --- final pairing pass over the temporal candidate window ---
    # (catches the case where a bug was indexed before its fixing CL arrived).
    links = pair_bugs_to_cls(bug_views, cl_views, cfg.pairing)
    with store.batch():
        for link in links:
            exists = store._conn.execute(
                "SELECT 1 FROM fix_links WHERE bug_id=? AND cl_number=?",
                (link.bug_id, link.cl_number)).fetchone()
            store.upsert_fix_link(link)
            if not exists:
                stats.links_added += 1
        if cl_views:
            store.set_state("last_seen_cl", str(max(c.cl for c in cl_views)))
        if bug_views:
            store.set_state("last_seen_bug_modtime",
                            max(b.last_change_time for b in bug_views).isoformat())

    stats.total_seconds = time.perf_counter() - started
    logger.info("bootstrap done: %s", stats)
    return stats
