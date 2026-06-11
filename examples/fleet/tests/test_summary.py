"""Summary table tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codescribe_fleet.summary import best, format_table, load_state, summarise


def _write_sweep(dir_: Path) -> None:
    state = {
        "sweep": "s",
        "total": 3,
        "settled": 3,
        "jobs": {
            "combo-000-lr=1e-4": {
                "worker": "fleet-01", "status": "done",
                "task_mean": 0.55, "train_loss": 0.45, "attempts": 1,
            },
            "combo-001-lr=2e-4": {
                "worker": "fleet-02", "status": "done",
                "task_mean": 0.62, "train_loss": 0.40, "attempts": 1,
            },
            "combo-002-lr=4e-4": {
                "worker": "fleet-01", "status": "failed",
                "task_mean": None, "train_loss": None, "attempts": 3,
            },
        },
        "workers": {},
    }
    (dir_ / "state.json").write_text(json.dumps(state), encoding="utf-8")


def test_summarise_sorted_desc(tmp_path: Path) -> None:
    _write_sweep(tmp_path)
    rows = summarise(tmp_path)
    assert [r.job_id for r in rows][:2] == ["combo-001-lr=2e-4", "combo-000-lr=1e-4"]
    assert rows[-1].task_mean is None  # None sorts last


def test_best_is_top_done(tmp_path: Path) -> None:
    _write_sweep(tmp_path)
    win = best(summarise(tmp_path))
    assert win is not None and win.job_id == "combo-001-lr=2e-4"


def test_format_table_renders(tmp_path: Path) -> None:
    _write_sweep(tmp_path)
    table = format_table(summarise(tmp_path))
    assert "task_mean" in table
    assert "combo-001-lr=2e-4" in table
    assert "0.620" in table


def test_falls_back_to_report_file(tmp_path: Path) -> None:
    # state.json has task_mean=None but the pulled report has it.
    state = {
        "sweep": "s", "total": 1, "settled": 1,
        "jobs": {"combo-000": {"worker": "w", "status": "done", "task_mean": None,
                               "train_loss": 0.4, "attempts": 1}},
        "workers": {},
    }
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    job_dir = tmp_path / "combo-000"
    job_dir.mkdir()
    (job_dir / "eval-report.json").write_text(json.dumps({"task_mean": 0.7}), "utf-8")
    rows = summarise(tmp_path)
    assert rows[0].task_mean == pytest.approx(0.7)


def test_missing_state_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_state(tmp_path)
