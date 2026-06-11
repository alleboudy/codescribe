"""Per-worker operations, transport-agnostic.

A :class:`Worker` wraps a :class:`~codescribe_fleet.config.WorkerSpec` and a
:class:`~codescribe_fleet.transport.Transport`. It knows how to push a job's
config, launch ``train run`` / ``train eval`` on the worker, verify the adapter
landed, and pull results back. Every interpolated path/value is ``shlex.quote``d
before it becomes part of a command string.

Remote workspace paths should be absolute (e.g. ``/home/you/repos/...``) or, for
a localhost worker, ``.`` — SFTP/scp does not expand a leading ``~``.
"""

from __future__ import annotations

import logging
import posixpath
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from .config import TrainLauncher, WorkerSpec
from .grid import Job, render_job_config_yaml
from .monitor import SMI_QUERY_FIELDS, build_smi_command, parse_smi_csv
from .transport import Transport

logger = logging.getLogger(__name__)

ADAPTER_FILENAME = "adapter_model.safetensors"
_LOSS_RE = re.compile(r"train_loss\s*=\s*(nan|[+-]?inf|[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


@dataclass
class TrainOutcome:
    job_id: str
    returncode: int
    train_loss: float | None
    adapter_present: bool
    log_tail: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.adapter_present


class Worker:
    def __init__(
        self,
        spec: WorkerSpec,
        transport: Transport,
        launcher: TrainLauncher,
    ) -> None:
        self.spec = spec
        self.transport = transport
        self.launcher = launcher

    @property
    def name(self) -> str:
        return self.spec.name

    # ---- paths (remote, posix) ------------------------------------------- #
    def _ws(self, *parts: str) -> str:
        return posixpath.join(self.spec.workspace, *parts)

    def remote_config_path(self, job: Job) -> str:
        return self._ws("configs", "_sweep", f"{job.id}.yaml")

    def remote_out_dir(self, sweep_name: str, job: Job) -> str:
        return self._ws("checkpoints", sweep_name, job.id)

    # ---- operations ------------------------------------------------------ #
    async def send_config(self, job: Job, staging_dir: Path) -> str:
        """Write the rendered config to a coordinator-side staging file and push it.

        Returns the remote path the config was placed at.
        """
        staging_dir.mkdir(parents=True, exist_ok=True)
        local = staging_dir / f"{job.id}.yaml"
        local.write_text(render_job_config_yaml(job), encoding="utf-8")
        remote = self.remote_config_path(job)
        await self._ensure_remote_dir(posixpath.dirname(remote))
        await self.transport.push(local, remote)
        return remote

    async def send_dataset(self, local_dataset: Path, remote_dataset: str) -> None:
        # Create the PARENT of the destination, never the destination itself: the
        # transport contract is "children land directly under remote_dataset".
        # Pre-creating remote_dataset would make scp nest the source inside it
        # (and would turn a file destination into a directory). Works for both a
        # directory dataset and a single-file eval-tasks path.
        await self._ensure_remote_dir(posixpath.dirname(remote_dataset))
        await self.transport.push(local_dataset, remote_dataset)

    async def run_training(
        self,
        job: Job,
        *,
        remote_config: str,
        remote_dataset: str,
        sweep_name: str,
        timeout: float,
    ) -> TrainOutcome:
        out_dir = self.remote_out_dir(sweep_name, job)
        await self._ensure_remote_dir(out_dir)
        cmd = (
            f"cd {shlex.quote(self.spec.workspace)} && "
            f"{self.launcher.prefix()} run "
            f"--config {shlex.quote(remote_config)} "
            f"--dataset {shlex.quote(remote_dataset)} "
            f"--out {shlex.quote(out_dir)}"
        )
        logger.info("[%s] run %s", self.name, job.id)
        result = await self.transport.run(cmd, timeout=timeout)
        adapter_present = await self._remote_file_exists(
            posixpath.join(out_dir, ADAPTER_FILENAME)
        )
        loss = _parse_train_loss(result.stdout) or _parse_train_loss(result.stderr)
        if not result.ok:
            logger.warning(
                "[%s] %s training rc=%d: %s",
                self.name,
                job.id,
                result.returncode,
                _tail(result.stderr),
            )
        return TrainOutcome(
            job_id=job.id,
            returncode=result.returncode,
            train_loss=loss,
            adapter_present=adapter_present,
            log_tail=_tail(result.stdout + result.stderr),
        )

    async def score_job(
        self,
        job: Job,
        *,
        sweep_name: str,
        remote_tasks: str | None,
        timeout: float,
    ) -> int:
        """Run ``train eval`` on the worker, writing ``eval-report.json`` beside the
        adapter. Returns the eval command's return code; the actual task_mean is read
        from the pulled report by :mod:`codescribe_fleet.summary`."""
        out_dir = self.remote_out_dir(sweep_name, job)
        report = posixpath.join(out_dir, "eval-report.json")
        cmd = (
            f"cd {shlex.quote(self.spec.workspace)} && "
            f"{self.launcher.prefix()} eval "
            f"--adapter {shlex.quote(out_dir)} "
            f"--report {shlex.quote(report)}"
        )
        if remote_tasks is not None:
            cmd += f" --tasks {shlex.quote(remote_tasks)}"
        logger.info("[%s] eval %s", self.name, job.id)
        result = await self.transport.run(cmd, timeout=timeout)
        if not result.ok:
            logger.warning("[%s] %s eval rc=%d", self.name, job.id, result.returncode)
        return result.returncode

    async def pull_results(self, job: Job, *, sweep_name: str, local_dest: Path) -> None:
        out_dir = self.remote_out_dir(sweep_name, job)
        local_dest.mkdir(parents=True, exist_ok=True)
        await self.transport.pull(out_dir, str(local_dest))

    async def health_check(self, *, timeout: float = 30.0) -> dict[str, str] | None:
        """Return parsed ``nvidia-smi`` fields, or ``None`` if the query failed."""
        result = await self.transport.run(build_smi_command(), timeout=timeout)
        if not result.ok or not result.stdout.strip():
            return None
        first = result.stdout.strip().splitlines()[0]
        return parse_smi_csv(first, SMI_QUERY_FIELDS)

    # ---- helpers --------------------------------------------------------- #
    async def _ensure_remote_dir(self, remote_dir: str) -> None:
        await self.transport.run(f"mkdir -p {shlex.quote(remote_dir)}", timeout=30.0)

    async def _remote_file_exists(self, remote_path: str) -> bool:
        result = await self.transport.run(
            f"test -f {shlex.quote(remote_path)}", timeout=30.0
        )
        return result.ok


def _parse_train_loss(text: str) -> float | None:
    matches = _LOSS_RE.findall(text or "")
    if not matches:
        return None
    return float(matches[-1])


def _tail(text: str, n: int = 600) -> str:
    text = (text or "").strip()
    return text[-n:]
