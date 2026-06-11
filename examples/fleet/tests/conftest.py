"""Shared fixtures + an in-memory FakeTransport for orchestrator unit tests.

The FakeTransport implements the same three-method contract as Local/SSH but never
spawns a process: it interprets the (quoted) command strings the orchestrator
builds, writes the artefacts a real worker would, and can be told to fail training
N times to exercise re-queue/quarantine paths. Because the test "remote" paths are
real tmp paths, push/pull are genuine filesystem copies.
"""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path

import pytest
import yaml

from codescribe_fleet.config import (
    CoordinatorSpec,
    FleetConfig,
    SweepConfig,
    SweepConstraints,
    TrainLauncher,
    WorkerSpec,
)
from codescribe_fleet.transport import CommandResult


class FakeTransport:
    def __init__(
        self,
        name: str,
        *,
        fail_train_times: int = 0,
        train_loss: str = "0.42",
        task_mean: float = 0.5,
    ) -> None:
        self.name = name
        self._fail_remaining = fail_train_times
        self.train_loss = train_loss
        self.task_mean = task_mean
        self.calls: list[str] = []

    async def run(self, command: str, *, timeout: float | None = None) -> CommandResult:
        self.calls.append(command)
        tokens = shlex.split(command)

        if tokens[:1] == ["mkdir"]:
            Path(tokens[-1]).mkdir(parents=True, exist_ok=True)
            return CommandResult(0, "", "")

        if tokens[:1] == ["test"] and "-f" in tokens:
            return CommandResult(0 if Path(tokens[-1]).exists() else 1, "", "")

        if "nvidia-smi" in command:
            return CommandResult(
                0, "NVIDIA 8 GB Ada-class laptop GPU, 72, 71.5, 4096, 8188, 95\n", ""
            )

        if "run" in tokens and "--out" in tokens:
            if self._fail_remaining > 0:
                self._fail_remaining -= 1
                return CommandResult(1, "", "boom: simulated training failure")
            out = Path(_arg(tokens, "--out"))
            out.mkdir(parents=True, exist_ok=True)
            (out / "adapter_model.safetensors").write_bytes(b"fake")
            return CommandResult(0, f"step 200 train_loss={self.train_loss}\n", "")

        if "eval" in tokens and "--report" in tokens:
            report = Path(_arg(tokens, "--report"))
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"task_mean": self.task_mean}), encoding="utf-8")
            return CommandResult(0, f"task_mean={self.task_mean}\n", "")

        return CommandResult(0, "", "")

    async def push(self, local, remote) -> None:
        _copy(Path(local), Path(remote))

    async def pull(self, remote, local) -> None:
        _copy(Path(remote), Path(local))

    async def close(self) -> None:
        return None


def _arg(tokens: list[str], flag: str) -> str:
    return tokens[tokens.index(flag) + 1]


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


@pytest.fixture
def base_config_dir(tmp_path: Path) -> Path:
    """A dir containing a minimal base train config the sweep references."""
    cfg_dir = tmp_path / "configs" / "train"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "qwen7b_qlora.yaml").write_text(
        yaml.safe_dump(
            {
                "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "seq_len": 1024,
                "lora": {"r": 16, "alpha": 16},
                "optim": {"lr": 2.0e-4, "name": "paged_adamw_8bit"},
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def worker_specs(tmp_path: Path):
    def make(n: int) -> list[WorkerSpec]:
        specs = []
        for i in range(1, n + 1):
            ws = tmp_path / f"ws{i}"
            ws.mkdir(parents=True, exist_ok=True)
            specs.append(
                WorkerSpec(name=f"fleet-0{i}", ssh_host="localhost", workspace=str(ws))
            )
        return specs

    return make


def make_sweep(grid: dict, **constraint_kw) -> SweepConfig:
    return SweepConfig(
        name="test-sweep",
        base_train_config="configs/train/qwen7b_qlora.yaml",
        base_dataset="datasets/none/",
        grid=grid,
        seed=1000,
        eval_tasks=None,
        constraints=SweepConstraints(**constraint_kw),
    )


def make_fleet(specs: list[WorkerSpec]) -> FleetConfig:
    return FleetConfig(
        coordinator=CoordinatorSpec(),
        workers=specs,
        train=TrainLauncher(runner="python", script="scripts/fake_train.py"),
    )
