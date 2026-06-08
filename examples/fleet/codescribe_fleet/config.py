"""Typed config models + validated loaders for ``fleet.yaml`` and ``sweep.yaml``.

Everything the orchestrator needs is parsed here, once, with clear error messages,
so the rest of the codebase deals in dataclasses rather than raw dicts. Mirrors
codescribe issue #3 §5.1 (fleet inventory) and §5.2 (sweep config).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a fleet/sweep YAML is missing required fields or malformed."""


# --------------------------------------------------------------------------- #
# Training launcher                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainLauncher:
    """How the orchestrator invokes the training CLI *on a worker*.

    Two mutually-exclusive shapes:

    * ``module`` — production: ``<runner> -m <module> <subcommand> ...``
      (e.g. ``uv run python -m codescribe_train.train run ...``).
    * ``script`` — demo/test: ``<runner> <script> <subcommand> ...``
      (e.g. ``python scripts/fake_train.py run ...``).
    """

    runner: str = "uv run python"
    module: str | None = None
    script: str | None = None

    def __post_init__(self) -> None:
        if bool(self.module) == bool(self.script):
            raise ConfigError(
                "train launcher must set exactly one of `module` or `script` "
                f"(got module={self.module!r}, script={self.script!r})"
            )

    def prefix(self) -> str:
        """The command prefix up to (but not including) the subcommand.

        The runner is intentionally NOT quoted (it is a trusted, multi-word
        operator-supplied string like ``uv run python``). The module/script IS
        quoted because it could in principle contain odd characters.
        """
        if self.module is not None:
            return f"{self.runner} -m {shlex.quote(self.module)}"
        return f"{self.runner} {shlex.quote(self.script)}"  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Fleet inventory                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkerSpec:
    name: str
    ssh_host: str
    workspace: str
    ssh_user: str | None = None
    ssh_port: int = 22
    identity_file: str | None = None
    hf_cache: str | None = None
    gpu_compute_cap: str | None = None
    vram_gb: int | None = None
    socket_ifname: str | None = None       # Path B NCCL interface (issue #3 §6.3)
    preflight_smoke_seconds: float | None = None


@dataclass(frozen=True)
class CoordinatorSpec:
    ssh_host: str = "localhost"
    workspace: str = "."


@dataclass(frozen=True)
class FleetConfig:
    coordinator: CoordinatorSpec
    workers: list[WorkerSpec]
    train: TrainLauncher

    def worker(self, name: str) -> WorkerSpec:
        for w in self.workers:
            if w.name == name:
                return w
        raise KeyError(name)


def load_fleet_config(path: str | Path) -> FleetConfig:
    raw = _read_yaml(path)
    if "workers" not in raw or not raw["workers"]:
        raise ConfigError(f"{path}: `workers` must be a non-empty list")

    coord_raw = raw.get("coordinator", {}) or {}
    coordinator = CoordinatorSpec(
        ssh_host=coord_raw.get("ssh_host", "localhost"),
        workspace=coord_raw.get("workspace", "."),
    )

    train_raw = raw.get("train", {}) or {}
    train = TrainLauncher(
        runner=train_raw.get("runner", "uv run python"),
        module=train_raw.get("module"),
        script=train_raw.get("script"),
    )

    workers: list[WorkerSpec] = []
    seen: set[str] = set()
    for i, w in enumerate(raw["workers"]):
        for required in ("name", "ssh_host", "workspace"):
            if required not in w:
                raise ConfigError(
                    f"{path}: workers[{i}] is missing required field `{required}`"
                )
        if w["name"] in seen:
            raise ConfigError(f"{path}: duplicate worker name {w['name']!r}")
        seen.add(w["name"])
        workers.append(
            WorkerSpec(
                name=w["name"],
                ssh_host=w["ssh_host"],
                workspace=w["workspace"],
                ssh_user=w.get("ssh_user"),
                ssh_port=int(w.get("ssh_port", 22)),
                identity_file=w.get("identity_file"),
                hf_cache=w.get("hf_cache"),
                gpu_compute_cap=_as_str(w.get("gpu_compute_cap")),
                vram_gb=w.get("vram_gb"),
                socket_ifname=w.get("socket_ifname"),
                preflight_smoke_seconds=w.get("preflight_smoke_seconds"),
            )
        )
    return FleetConfig(coordinator=coordinator, workers=workers, train=train)


# --------------------------------------------------------------------------- #
# Sweep config                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SweepConstraints:
    max_concurrent_per_worker: int = 1
    abort_on_first_finite_loss_fail: bool = True
    per_run_timeout_hours: float = 6.0
    max_requeue: int = 2
    quarantine_after: int = 3


@dataclass(frozen=True)
class SweepConfig:
    name: str
    base_train_config: str
    base_dataset: str
    grid: dict[str, list[Any]]
    seed: int = 0
    eval_tasks: str | None = None
    constraints: SweepConstraints = field(default_factory=SweepConstraints)

    @property
    def num_jobs(self) -> int:
        n = 1
        for values in self.grid.values():
            n *= len(values)
        return n


def load_sweep_config(path: str | Path) -> SweepConfig:
    raw = _read_yaml(path)
    for required in ("name", "base_train_config", "base_dataset", "grid"):
        if required not in raw:
            raise ConfigError(f"{path}: missing required field `{required}`")

    grid = raw["grid"]
    if not isinstance(grid, dict) or not grid:
        raise ConfigError(f"{path}: `grid` must be a non-empty mapping")
    for key, values in grid.items():
        # Keys are dotted config paths (e.g. "optim.lr"). YAML happily parses a
        # bare/numeric/boolean key as a non-string, which would later crash
        # expand_grid's str.split with an opaque traceback — surface it here.
        if not isinstance(key, str):
            raise ConfigError(
                f"{path}: grid keys must be dotted-path strings (got {key!r} of "
                f"type {type(key).__name__}); quote it in YAML"
            )
        if not key or any(part == "" for part in key.split(".")):
            raise ConfigError(
                f"{path}: malformed grid key {key!r}: empty path segment"
            )
        if not isinstance(values, list) or not values:
            raise ConfigError(
                f"{path}: grid[{key!r}] must be a non-empty list (got {values!r})"
            )

    c = raw.get("constraints", {}) or {}
    constraints = SweepConstraints(
        max_concurrent_per_worker=int(c.get("max_concurrent_per_worker", 1)),
        abort_on_first_finite_loss_fail=bool(
            c.get("abort_on_first_finite_loss_fail", True)
        ),
        per_run_timeout_hours=float(c.get("per_run_timeout_hours", 6.0)),
        max_requeue=int(c.get("max_requeue", 2)),
        quarantine_after=int(c.get("quarantine_after", 3)),
    )
    if constraints.max_concurrent_per_worker < 1:
        raise ConfigError(f"{path}: max_concurrent_per_worker must be >= 1")

    return SweepConfig(
        name=raw["name"],
        base_train_config=raw["base_train_config"],
        base_dataset=raw["base_dataset"],
        grid=grid,
        seed=int(raw.get("seed", 0)),
        eval_tasks=raw.get("eval_tasks"),
        constraints=constraints,
    )


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping, got {type(data).__name__}")
    return data


def _as_str(v: Any) -> str | None:
    """YAML parses ``8.9`` as a float; compute-cap is always compared as a string."""
    return None if v is None else str(v)
