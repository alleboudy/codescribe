"""Expand a sweep grid into concrete, rendered training jobs.

Cartesian product of ``sweep.grid`` -> one :class:`Job` per combination. Each job
carries a *rendered* train config (the base config deep-merged with that job's
dotted overrides) and a deterministic per-job seed so no two workers train an
identical run (issue #3 §8 anti-patterns).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import SweepConfig


@dataclass(frozen=True)
class Job:
    id: str
    index: int
    overrides: dict[str, Any]       # dotted-key -> value, e.g. {"optim.lr": 2e-4}
    seed: int
    rendered_config: dict[str, Any]  # full train config with overrides + seed applied


def expand_grid(sweep: SweepConfig, *, base_config_dir: str | Path = ".") -> list[Job]:
    """Return one :class:`Job` per point in the cartesian product of the grid.

    ``base_config_dir`` is the directory the sweep's ``base_train_config`` path is
    resolved against (normally the repo root / the dir you run the CLI from).
    """
    base = _load_base_config(Path(base_config_dir) / sweep.base_train_config)

    keys = sorted(sweep.grid)                 # sorted -> deterministic job ordering
    value_lists = [sweep.grid[k] for k in keys]

    jobs: list[Job] = []
    for index, combo in enumerate(itertools.product(*value_lists)):
        overrides = dict(zip(keys, combo, strict=True))
        seed = sweep.seed + index
        rendered = _render_config(base, overrides, seed)
        jobs.append(
            Job(
                id=_job_id(index, overrides),
                index=index,
                overrides=overrides,
                seed=seed,
                rendered_config=rendered,
            )
        )
    return jobs


def render_job_config_yaml(job: Job) -> str:
    """Serialise a job's rendered config to YAML (what gets pushed to a worker)."""
    return yaml.safe_dump(job.rendered_config, sort_keys=True)


# --------------------------------------------------------------------------- #
# internals                                                                    #
# --------------------------------------------------------------------------- #
def _load_base_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"base_train_config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: base train config must be a mapping")
    return data


def _render_config(
    base: dict[str, Any], overrides: dict[str, Any], seed: int
) -> dict[str, Any]:
    import copy

    cfg = copy.deepcopy(base)
    for dotted, value in overrides.items():
        _set_dotted(cfg, dotted, value)
    cfg["seed"] = seed
    return cfg


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``d["a"]["b"] = value`` given the dotted path ``"a.b"``.

    Creates intermediate dicts as needed; raises if an intermediate exists but is
    not a mapping (a config typo we want surfaced, not silently overwritten).
    """
    parts = dotted.split(".")
    cur: dict[str, Any] = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if nxt is None:
            nxt = {}
            cur[p] = nxt
        elif not isinstance(nxt, dict):
            raise ValueError(
                f"cannot apply override {dotted!r}: {p!r} is a "
                f"{type(nxt).__name__}, not a mapping"
            )
        cur = nxt
    cur[parts[-1]] = value


def _job_id(index: int, overrides: dict[str, Any]) -> str:
    """Stable, filesystem-safe job id, e.g. ``combo-001-lr=0.0002-r=16``."""
    parts = [f"combo-{index:03d}"]
    for key in sorted(overrides):
        leaf = key.split(".")[-1]
        parts.append(f"{leaf}={_fmt_value(overrides[key])}")
    return _sanitise("-".join(parts))


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._=+-")


def _sanitise(s: str) -> str:
    return "".join(c if c in _SAFE else "_" for c in s)
