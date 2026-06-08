"""Config loader + validation tests against the real example YAMLs."""

from __future__ import annotations

from pathlib import Path

import pytest

from codescribe_fleet.config import (
    ConfigError,
    TrainLauncher,
    load_fleet_config,
    load_sweep_config,
)

EXAMPLE = Path(__file__).resolve().parents[1]


def test_load_production_fleet() -> None:
    fleet = load_fleet_config(EXAMPLE / "configs" / "fleet.yaml")
    assert len(fleet.workers) == 4
    assert fleet.train.module == "codescribe_train.train"
    w = fleet.worker("fleet-01")
    assert w.gpu_compute_cap == "8.9"  # YAML 8.9 (float) coerced to string
    assert w.socket_ifname == "eno1"


def test_load_local_demo_fleet() -> None:
    fleet = load_fleet_config(EXAMPLE / "configs" / "fleet.local.yaml")
    assert fleet.train.script == "scripts/fake_train.py"
    assert all(w.ssh_host == "localhost" for w in fleet.workers)


def test_load_sweep() -> None:
    sweep = load_sweep_config(EXAMPLE / "sweeps" / "lr-rank.yaml")
    assert sweep.num_jobs == 9  # 3 lr x 3 r
    assert sweep.constraints.max_requeue == 2
    assert sweep.constraints.quarantine_after == 3


def test_train_launcher_requires_exactly_one() -> None:
    with pytest.raises(ConfigError):
        TrainLauncher(module="a", script="b")
    with pytest.raises(ConfigError):
        TrainLauncher()  # neither


def test_train_launcher_prefix_quotes_target() -> None:
    assert TrainLauncher(module="codescribe_train.train").prefix().endswith(
        "codescribe_train.train"
    )
    assert "-m " in TrainLauncher(module="m").prefix()


def test_missing_required_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("workers: []\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_fleet_config(bad)

    bad.write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_sweep_config(bad)


def test_non_string_grid_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "sweep.yaml"
    bad.write_text(
        "name: s\nbase_train_config: c\nbase_dataset: d/\ngrid:\n  8: [1, 2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_sweep_config(bad)


def test_empty_segment_grid_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "sweep.yaml"
    bad.write_text(
        "name: s\nbase_train_config: c\nbase_dataset: d/\ngrid:\n  optim..lr: [1, 2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_sweep_config(bad)


def test_duplicate_worker_name_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "workers:\n"
        "  - {name: a, ssh_host: h1, workspace: /w}\n"
        "  - {name: a, ssh_host: h2, workspace: /w}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_fleet_config(bad)
