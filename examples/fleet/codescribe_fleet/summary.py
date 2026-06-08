"""Summarise a finished (or in-flight) sweep into a sorted results table.

Reads ``state.json`` plus each job's pulled ``eval-report.json`` and renders a
``job_id / worker / status / task_mean / train_loss / tries`` table — a variant of
issue #3 §5.4's table — sorted by ``task_mean`` descending (the winning adapter
first). (Wall-clock is recorded by the trainer in ``run-meta.json`` but not
surfaced in this table.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JobSummary:
    job_id: str
    worker: str | None
    status: str
    task_mean: float | None
    train_loss: float | None
    attempts: int


def load_state(sweep_dir: str | Path) -> dict:
    path = Path(sweep_dir) / "state.json"
    if not path.is_file():
        raise FileNotFoundError(f"no state.json in {sweep_dir} (has the sweep started?)")
    return json.loads(path.read_text(encoding="utf-8"))


def summarise(sweep_dir: str | Path) -> list[JobSummary]:
    """Return per-job summaries, sorted best task_mean first (None last)."""
    state = load_state(sweep_dir)
    summaries: list[JobSummary] = []
    for jid, js in state.get("jobs", {}).items():
        task_mean = js.get("task_mean")
        if task_mean is None:
            # Fall back to the pulled report if state.json predates the pull.
            task_mean = _read_report_task_mean(Path(sweep_dir) / jid / "eval-report.json")
        summaries.append(
            JobSummary(
                job_id=jid,
                worker=js.get("worker"),
                status=js.get("status", "?"),
                task_mean=task_mean,
                train_loss=js.get("train_loss"),
                attempts=js.get("attempts", 0),
            )
        )
    summaries.sort(key=lambda s: (s.task_mean is None, -(s.task_mean or 0.0)))
    return summaries


def format_table(summaries: list[JobSummary]) -> str:
    header = f"{'job_id':<34} {'worker':<10} {'status':<8} {'task_mean':>9} {'train_loss':>10} {'tries':>5}"
    lines = [header, "-" * len(header)]
    for s in summaries:
        lines.append(
            f"{s.job_id:<34} "
            f"{(s.worker or '-'):<10} "
            f"{s.status:<8} "
            f"{_fmt(s.task_mean):>9} "
            f"{_fmt(s.train_loss):>10} "
            f"{s.attempts:>5}"
        )
    return "\n".join(lines)


def best(summaries: list[JobSummary]) -> JobSummary | None:
    scored = [s for s in summaries if s.task_mean is not None and s.status == "done"]
    return scored[0] if scored else None


def _read_report_task_mean(report: Path) -> float | None:
    if not report.is_file():
        return None
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    val = data.get("task_mean")
    return float(val) if isinstance(val, (int, float)) else None


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"
