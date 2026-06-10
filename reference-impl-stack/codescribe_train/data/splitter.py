"""Train / val / test splitter — by file path, not by sample.

Splits a :class:`FileRecord` stream into deterministic train/val/test buckets
keyed on each record's ``relpath``. All samples later derived from the same
file land in the same bucket; this prevents in-file leakage across splits.

Determinism: the bucket assignment is a stable hash of ``salt + ":" + relpath``,
so re-runs produce identical splits and adding new files only redistributes a
small fraction of records (no whole-set reshuffle).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from codescribe_train.data.walker import FileRecord


@dataclass(frozen=True, slots=True)
class SplitWeights:
    """Train / val / test weights. Must sum to 1.0."""

    train: float = 0.90
    val: float = 0.05
    test: float = 0.05

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split weights must sum to 1.0; got {total}")
        for name, w in (("train", self.train), ("val", self.val), ("test", self.test)):
            if w < 0:
                raise ValueError(f"{name} weight must be >= 0; got {w}")


_DEFAULT_WEIGHTS = SplitWeights()


def split_records(
    records: Iterable[FileRecord],
    *,
    weights: SplitWeights = _DEFAULT_WEIGHTS,
    salt: str = "",
) -> dict[str, list[FileRecord]]:
    """Bucket records into ``"train"`` / ``"val"`` / ``"test"`` deterministically.

    The bucket for a record depends only on ``salt`` and ``record.relpath`` —
    not on the iteration order or the surrounding records. Same input set,
    same salt → same buckets every time.
    """
    train_t = weights.train
    val_t = weights.train + weights.val
    buckets: dict[str, list[FileRecord]] = {"train": [], "val": [], "test": []}
    for record in records:
        h = hashlib.sha256((salt + ":" + record.relpath).encode("utf-8")).digest()
        # First 8 bytes -> uint64 -> [0, 1).
        u = int.from_bytes(h[:8], "big") / (1 << 64)
        if u < train_t:
            buckets["train"].append(record)
        elif u < val_t:
            buckets["val"].append(record)
        else:
            buckets["test"].append(record)
    return buckets
