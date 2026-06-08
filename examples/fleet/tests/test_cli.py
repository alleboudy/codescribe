"""CLI tests for `ddp-plan` guards and the torchrun-runner derivation."""

from __future__ import annotations

from pathlib import Path

import pytest

from codescribe_fleet.__main__ import _torchrun_runner, main

EXAMPLE = Path(__file__).resolve().parents[1]
FLEET = str(EXAMPLE / "configs" / "fleet.yaml")
FLEET_LOCAL = str(EXAMPLE / "configs" / "fleet.local.yaml")
SWEEP = str(EXAMPLE / "sweeps" / "lr-rank.yaml")


@pytest.mark.parametrize(
    "runner,expected",
    [("uv run python", "uv run"), ("python", ""), ("python3", ""), ("uv run", "uv run")],
)
def test_torchrun_runner_strips_trailing_python(runner: str, expected: str) -> None:
    assert _torchrun_runner(runner) == expected


def test_ddp_plan_ok(capsys) -> None:
    rc = main(["ddp-plan", "--fleet", FLEET, "--sweep", SWEEP, "--rdzv-host", "h1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "uv run torchrun" in out          # python stripped, not "uv run python torchrun"
    assert "--node-rank=0" in out and "--node-rank=3" in out
    assert "python torchrun" not in out      # never emit a non-runnable prefix


def test_ddp_plan_rejects_excess_nnodes() -> None:
    rc = main(["ddp-plan", "--fleet", FLEET, "--sweep", SWEEP, "--nnodes", "99"])
    assert rc == 2                            # 99 > 4 workers -> guarded


def test_ddp_plan_requires_module() -> None:
    # the local demo fleet is script-based; torchrun needs `-m <module>`
    rc = main(["ddp-plan", "--fleet", FLEET_LOCAL, "--sweep", SWEEP])
    assert rc == 2
