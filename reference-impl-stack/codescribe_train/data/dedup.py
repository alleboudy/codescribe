"""Content-based deduplication.

Stage 3 of the data pipeline. Hashes file content with SHA-256 and rejects
records whose hash has already been seen. MinHash near-dup is intentionally
deferred — exact-content dedup catches what matters for a single repo (vendored
files, generated boilerplate, accidental copies).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Iterator

from codescribe_train.data.walker import FileRecord

logger = logging.getLogger(__name__)


def dedup_records(records: Iterable[FileRecord]) -> Iterator[FileRecord]:
    """Yield records whose file content has not been seen before in this iteration.

    Each surviving file is read fully into memory once. The upstream filter
    should bound ``max_bytes`` so this never explodes; we don't second-guess it
    here.
    """
    seen: set[bytes] = set()
    for record in records:
        try:
            content = record.abspath.read_bytes()
        except OSError as e:
            logger.warning("dedup: cannot read %s: %s — skipping", record.relpath, e)
            continue
        digest = hashlib.sha256(content).digest()
        if digest in seen:
            continue
        seen.add(digest)
        yield record
