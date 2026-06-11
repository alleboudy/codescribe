"""Bootstrap indexer — full-history indexer for the rag sidecar.

See ``codescribe_train/rag/AGENTS.md`` for the rules this module obeys
(idempotency, atomic state watermarks, batch ≤256, no stdout, no
network beyond the source modules' allowed `gh` egress).

The bootstrap and incremental pipelines share an inner loop (see
``codescribe_train.rag.pipelines._common``); this module only differs in the
*initial* watermarks (epoch by default, or caller-supplied to support
"resume from a known checkpoint" reruns) and the resulting stats shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from codescribe_train.rag.config import RagConfig
from codescribe_train.rag.pipelines._common import (
    IndexerDeps,
    epoch_datetime,
    run_inner_loop,
)

logger = logging.getLogger(__name__)


@dataclass
class BootstrapStats:
    """Per-run counts returned by :func:`run_bootstrap`."""

    commits: int = 0
    issues: int = 0
    pulls: int = 0
    links: int = 0
    last_commit_sha: str | None = None
    last_issue_updated_at: str | None = None
    stopped_early: bool = False


async def run_bootstrap(
    config: RagConfig | None,
    *,
    deps: IndexerDeps,
    since_sha: str | None = None,
    since_issue_updated_at: datetime | None = None,
) -> BootstrapStats:
    """Run a full-history indexing pass.

    ``since_sha`` / ``since_issue_updated_at`` are optional resume hints —
    a clean bootstrap leaves both ``None`` and walks every commit + every
    GitHub-issue/PR since the epoch. The two arguments mirror the
    signature in the design spec verbatim.

    The pipeline is idempotent: a second call against the same source set
    produces the same store state (upsert semantics on every write).

    ``config`` is currently unused by the bootstrap body itself — the
    state path / embedder / sources are passed via ``deps`` so tests can
    inject fakes — but it stays in the signature to match the spec and
    so the CLI dispatcher can pass it through.
    """
    _ = config  # reserved for future use; kept for spec parity
    since_issues = since_issue_updated_at or epoch_datetime()
    logger.info(
        "bootstrap start since_sha=%s since_issue_updated_at=%s",
        since_sha,
        since_issues.isoformat(),
    )
    result = await run_inner_loop(
        deps,
        since_sha=since_sha,
        since_issue_updated_at=since_issues,
        since_pull_updated_at=since_issues,
    )
    logger.info(
        "bootstrap done commits=%d issues=%d pulls=%d links=%d stopped_early=%s",
        result.commits,
        result.issues,
        result.pulls,
        result.links,
        result.stopped_early,
    )
    return BootstrapStats(
        commits=result.commits,
        issues=result.issues,
        pulls=result.pulls,
        links=result.links,
        last_commit_sha=result.last_commit_sha,
        last_issue_updated_at=result.last_issue_updated_at,
        stopped_early=result.stopped_early,
    )
