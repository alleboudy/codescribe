"""Tests for :mod:`codescribe_train.data.splitter`."""

from __future__ import annotations

from pathlib import Path

import pytest

from codescribe_train.data.splitter import SplitWeights, split_records
from codescribe_train.data.walker import FileRecord


def _rec(relpath: str) -> FileRecord:
    return FileRecord(relpath=relpath, abspath=Path(f"/tmp/{relpath}"), size_bytes=100, ext="py")


def test_total_count_preserved() -> None:
    records = [_rec(f"file_{i}.py") for i in range(1000)]
    buckets = split_records(records)
    assert sum(len(v) for v in buckets.values()) == 1000


def test_buckets_have_expected_keys() -> None:
    records = [_rec(f"file_{i}.py") for i in range(10)]
    buckets = split_records(records)
    assert set(buckets.keys()) == {"train", "val", "test"}


def test_deterministic_under_same_salt() -> None:
    records = [_rec(f"file_{i}.py") for i in range(500)]
    a = split_records(records, salt="myrepo")
    b = split_records(records, salt="myrepo")
    assert {k: [r.relpath for r in v] for k, v in a.items()} == {
        k: [r.relpath for r in v] for k, v in b.items()
    }


def test_different_salts_produce_different_splits() -> None:
    records = [_rec(f"file_{i}.py") for i in range(500)]
    a = split_records(records, salt="salt-a")
    b = split_records(records, salt="salt-b")
    train_a = {r.relpath for r in a["train"]}
    train_b = {r.relpath for r in b["train"]}
    # Most records will agree (≈90% in both train), but the tails differ.
    assert train_a != train_b


def test_ratios_approximate_weights() -> None:
    records = [_rec(f"file_{i}.py") for i in range(10000)]
    weights = SplitWeights(train=0.80, val=0.10, test=0.10)
    buckets = split_records(records, weights=weights, salt="ratio-test")
    n = len(records)
    assert abs(len(buckets["train"]) / n - 0.80) < 0.02
    assert abs(len(buckets["val"]) / n - 0.10) < 0.02
    assert abs(len(buckets["test"]) / n - 0.10) < 0.02


def test_same_relpath_lands_in_same_bucket_across_runs() -> None:
    """Adding more records doesn't move existing ones across splits."""
    initial = [_rec(f"file_{i}.py") for i in range(100)]
    extended = initial + [_rec(f"new_{i}.py") for i in range(50)]
    initial_buckets = split_records(initial, salt="stable")
    extended_buckets = split_records(extended, salt="stable")
    initial_train = {r.relpath for r in initial_buckets["train"]}
    extended_train = {r.relpath for r in extended_buckets["train"]}
    # Every original train file must still be in train in the extended split.
    assert initial_train.issubset(extended_train)


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        SplitWeights(train=0.5, val=0.3, test=0.3)


def test_weights_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        SplitWeights(train=1.1, val=-0.05, test=-0.05)


def test_empty_input_yields_empty_buckets() -> None:
    buckets = split_records([])
    assert buckets == {"train": [], "val": [], "test": []}
