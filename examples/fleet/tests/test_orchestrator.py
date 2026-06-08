"""Orchestrator behaviour tests using the in-memory FakeTransport.

These pin the §5.5 robustness rules: re-queue, quarantine, finite-loss abort, and
command quoting — without spawning any process.
"""

from __future__ import annotations

import json
from pathlib import Path

from codescribe_fleet.orchestrator import Orchestrator

from .conftest import FakeTransport, make_fleet, make_sweep


def _orch(fleet, sweep, out_dir, base_dir, factory) -> Orchestrator:
    return Orchestrator(
        fleet,
        sweep,
        out_dir=out_dir,
        transport_factory=factory,
        base_config_dir=base_dir,
        empty_poll_interval=0.01,
    )


async def test_happy_path_two_workers(base_config_dir, worker_specs, tmp_path) -> None:
    specs = worker_specs(2)
    fleet = make_fleet(specs)
    sweep = make_sweep({"optim.lr": [1e-4, 2e-4], "lora.r": [8, 16]})
    out = tmp_path / "out"

    transports = {s.name: FakeTransport(s.name) for s in specs}
    result = await _orch(
        fleet, sweep, out, base_config_dir, lambda s: transports[s.name]
    ).run()

    assert len(result.done) == 4
    assert not result.failed
    assert not result.aborted
    # results were pulled back + scored
    for j in result.done:
        assert (out / j.id / "eval-report.json").is_file()
        assert j.task_mean == 0.5
    # state.json reflects completion
    state = json.loads((out / "state.json").read_text())
    assert state["settled"] == 4


async def test_requeue_then_succeeds(base_config_dir, worker_specs, tmp_path) -> None:
    specs = worker_specs(1)
    fleet = make_fleet(specs)
    sweep = make_sweep({"optim.lr": [2e-4], "lora.r": [16]}, max_requeue=2)
    out = tmp_path / "out"

    # fail the first training, succeed on retry
    t = FakeTransport(specs[0].name, fail_train_times=1)
    result = await _orch(fleet, sweep, out, base_config_dir, lambda s: t).run()

    assert len(result.done) == 1
    assert result.done[0].attempts == 2  # first attempt failed, second succeeded
    assert not result.aborted


async def test_quarantine_aborts_when_only_worker_dies(
    base_config_dir, worker_specs, tmp_path
) -> None:
    specs = worker_specs(1)
    fleet = make_fleet(specs)
    sweep = make_sweep(
        {"optim.lr": [1e-4, 2e-4], "lora.r": [8, 16]},
        max_requeue=2,
        quarantine_after=3,
    )
    out = tmp_path / "out"

    t = FakeTransport(specs[0].name, fail_train_times=999)  # always fails
    result = await _orch(fleet, sweep, out, base_config_dir, lambda s: t).run()

    assert result.aborted
    assert len(result.failed) == 4  # all jobs end failed (none could complete)
    state = json.loads((out / "state.json").read_text())
    assert state["workers"][specs[0].name]["status"] == "quarantined"


async def test_finite_loss_fail_aborts(base_config_dir, worker_specs, tmp_path) -> None:
    specs = worker_specs(1)
    fleet = make_fleet(specs)
    sweep = make_sweep(
        {"optim.lr": [1e-4, 2e-4]}, abort_on_first_finite_loss_fail=True
    )
    out = tmp_path / "out"

    t = FakeTransport(specs[0].name, train_loss="nan")
    result = await _orch(fleet, sweep, out, base_config_dir, lambda s: t).run()

    assert result.aborted
    assert all(j.status == "failed" for j in result.jobs)


async def test_commands_are_quoted(base_config_dir, tmp_path) -> None:
    # a workspace path containing a space must be shlex-quoted in every command.
    from codescribe_fleet.config import (
        CoordinatorSpec,
        FleetConfig,
        TrainLauncher,
        WorkerSpec,
    )

    spacey = tmp_path / "ws with space"
    spacey.mkdir()
    spec = WorkerSpec(name="fleet-01", ssh_host="localhost", workspace=str(spacey))
    fleet = FleetConfig(
        coordinator=CoordinatorSpec(),
        workers=[spec],
        train=TrainLauncher(runner="python", script="scripts/fake_train.py"),
    )
    sweep = make_sweep({"lora.r": [16]})
    out = tmp_path / "out"

    t = FakeTransport(spec.name)
    await _orch(fleet, sweep, out, base_config_dir, lambda s: t).run()

    train_cmds = [c for c in t.calls if " run " in f" {c} "]
    assert train_cmds
    # the spacey workspace appears quoted, never bare
    assert any("'" + str(spacey) + "'" in c for c in train_cmds)
