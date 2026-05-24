from __future__ import annotations

import logging
from datetime import datetime, timezone

from .bootstrap import BootstrapStats, run_bootstrap

logger = logging.getLogger(__name__)


def read_watermarks(store) -> tuple[int, datetime]:
    last_cl = int(store.get_state("last_seen_cl") or 0)
    raw = store.get_state("last_seen_bug_modtime")
    since = datetime.fromisoformat(raw) if raw else datetime(1970, 1, 1, tzinfo=timezone.utc)
    return last_cl, since


def run_incremental(cfg, sources, store, embedder, *, batch_size: int = 256) -> BootstrapStats:
    # The streaming bootstrap is idempotent and upsert-based, so an incremental
    # run is the same inner loop over a sources object already scoped to deltas.
    return run_bootstrap(cfg, sources, store, embedder, batch_size=batch_size)
