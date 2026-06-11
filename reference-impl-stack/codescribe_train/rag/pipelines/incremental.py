"""Incremental indexer — delta indexer driven by the state watermarks.

Reads ``state.last_seen_commit_sha`` and ``state.last_seen_issue_updated_at``
from the store and runs the same inner loop as the bootstrap pipeline
over the smaller delta. Watermarks are bumped atomically inside the same
transactions as the batches they correspond to (per
``codescribe_train/rag/AGENTS.md``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from codescribe_train.rag.config import RagConfig
from codescribe_train.rag.pipelines._common import (
    STATE_LAST_COMMIT_SHA,
    STATE_LAST_ISSUE_UPDATED_AT,
    IndexerDeps,
    apply_safety_margin,
    epoch_datetime,
    parse_watermark,
    run_inner_loop,
)
from codescribe_train.rag.store.writer import Store

logger = logging.getLogger(__name__)


@dataclass
class IncrementalStats:
    """Per-run counts returned by :func:`run_incremental`."""

    commits: int = 0
    issues: int = 0
    pulls: int = 0
    links: int = 0
    last_commit_sha: str | None = None
    last_issue_updated_at: str | None = None
    stopped_early: bool = False


async def run_incremental(
    config: RagConfig | None,
    *,
    deps: IndexerDeps,
) -> IncrementalStats:
    """Run a delta indexing pass driven by the store's watermarks.

    Reads ``state.last_seen_commit_sha`` (exclusive: that commit is not
    re-indexed) and ``state.last_seen_issue_updated_at`` (inclusive,
    after applying a 1-second safety margin) and runs the shared inner
    loop over the resulting delta. Atomic state bumps happen inside each
    batch's transaction so a crash mid-run never leaves the watermark
    ahead of the data.
    """
    _ = config  # reserved; see bootstrap docstring

    with Store.open(deps.store_path) as store:
        last_sha = store.get_state(STATE_LAST_COMMIT_SHA)
        last_issue_at_raw = store.get_state(STATE_LAST_ISSUE_UPDATED_AT)

    last_issue_at = parse_watermark(last_issue_at_raw)
    # Apply the 1-second safety margin so a record updated within the
    # same second as the prior watermark isn't missed at the boundary.
    since_issues = (
        apply_safety_margin(last_issue_at)
        if last_issue_at_raw
        else epoch_datetime()
    )

    logger.info(
        "incremental start since_sha=%s since_issue_updated_at=%s",
        last_sha,
        since_issues.isoformat(),
    )

    result = await run_inner_loop(
        deps,
        since_sha=last_sha,
        since_issue_updated_at=since_issues,
        since_pull_updated_at=since_issues,
    )

    logger.info(
        "incremental done commits=%d issues=%d pulls=%d links=%d stopped_early=%s",
        result.commits,
        result.issues,
        result.pulls,
        result.links,
        result.stopped_early,
    )

    return IncrementalStats(
        commits=result.commits,
        issues=result.issues,
        pulls=result.pulls,
        links=result.links,
        last_commit_sha=result.last_commit_sha,
        last_issue_updated_at=result.last_issue_updated_at,
        stopped_early=result.stopped_early,
    )
