"""Path A orchestrator — parallel hyperparameter sweep across the fleet.

Implements codescribe issue #3 §5.3 as runnable code: one independent training
job per worker, no gradient sync. The coordinator schedules jobs onto idle workers
and pulls back each adapter + ``eval-report.json``.

Robustness rules from §5.5 are first-class here:

* a failed job is re-queued onto another worker up to ``max_requeue`` times;
* a worker that fails ``quarantine_after`` jobs in a row is benched;
* a non-finite training loss aborts the whole sweep when
  ``abort_on_first_finite_loss_fail`` is set;
* progress is persisted to ``state.json`` after every transition so ``status`` can
  read it and a killed sweep is inspectable.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import FleetConfig, SweepConfig
from .grid import Job, expand_grid
from .transport import Transport
from .worker import Worker

logger = logging.getLogger(__name__)

TransportFactory = Callable[..., Transport]  # (WorkerSpec) -> Transport


@dataclass
class JobState:
    id: str
    overrides: dict
    seed: int
    status: str = "pending"          # pending|running|done|failed
    attempts: int = 0
    worker: str | None = None
    returncode: int | None = None
    train_loss: float | None = None
    task_mean: float | None = None
    error: str | None = None


@dataclass
class WorkerState:
    name: str
    status: str = "active"           # active|quarantined
    consecutive_failures: int = 0
    completed: int = 0


@dataclass
class SweepResult:
    name: str
    aborted: bool
    jobs: list[JobState] = field(default_factory=list)

    @property
    def done(self) -> list[JobState]:
        return [j for j in self.jobs if j.status == "done"]

    @property
    def failed(self) -> list[JobState]:
        return [j for j in self.jobs if j.status == "failed"]


class Orchestrator:
    def __init__(
        self,
        fleet: FleetConfig,
        sweep: SweepConfig,
        out_dir: str | Path,
        transport_factory: TransportFactory,
        *,
        base_config_dir: str | Path = ".",
        empty_poll_interval: float = 0.5,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.fleet = fleet
        self.sweep = sweep
        self.out_dir = Path(out_dir)
        self.transport_factory = transport_factory
        self.base_config_dir = Path(base_config_dir)
        self.empty_poll_interval = empty_poll_interval
        self.clock = clock

        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: dict[str, Job] = {}
        self._job_state: dict[str, JobState] = {}
        self._worker_state: dict[str, WorkerState] = {}
        self._settled: set[str] = set()
        self._abort = asyncio.Event()
        self._lock = asyncio.Lock()
        self._total = 0
        self._write_seq = itertools.count()

    async def run(self) -> SweepResult:
        jobs = expand_grid(self.sweep, base_config_dir=self.base_config_dir)
        self._total = len(jobs)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "sweep %s: %d jobs across %d workers",
            self.sweep.name,
            self._total,
            len(self.fleet.workers),
        )

        for job in jobs:
            self._jobs[job.id] = job
            self._job_state[job.id] = JobState(
                id=job.id, overrides=job.overrides, seed=job.seed
            )
            self._queue.put_nowait(job)

        # Build all Workers (and transports) up front and append BEFORE connecting,
        # so the finally below closes every transport opened — including those from
        # before a failing connect — and we never leak SSH connections.
        workers: list[Worker] = []
        for spec in self.fleet.workers:
            self._worker_state[spec.name] = WorkerState(name=spec.name)
            workers.append(Worker(spec, self.transport_factory(spec), self.fleet.train))

        await self._persist_state(aborted=False)

        try:
            # Connect; a worker that can't be reached is quarantined (not fatal)
            # unless NONE can be reached. Mirrors §5.5: one dead worker ≠ dead sweep.
            live: list[Worker] = []
            for w in workers:
                connect = getattr(w.transport, "connect", None)
                if connect is not None:
                    try:
                        await connect()
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "worker %s failed to connect (%s); quarantining", w.name, exc
                        )
                        self._worker_state[w.name].status = "quarantined"
                        continue
                live.append(w)

            if not live:
                self._abort.set()
                logger.error("ABORT: no worker could be reached")

            await self._distribute_inputs(live)
            concurrency = max(1, self.sweep.constraints.max_concurrent_per_worker)
            # NOTE: after an abort, a worker already inside _run_one finishes that
            # one job (or hits per_run_timeout_hours) before its loop exits — we do
            # not hard-cancel in-flight remote training.
            loops = [
                asyncio.create_task(self._worker_loop(w))
                for w in live
                for _ in range(concurrency)
            ]
            await asyncio.gather(*loops)
        finally:
            for w in workers:
                try:
                    await w.transport.close()
                except Exception:  # noqa: BLE001 - cleanup must not raise
                    pass

        # Any job neither done nor failed was stranded (e.g. all workers quarantined).
        async with self._lock:
            for st in self._job_state.values():
                if st.status not in ("done", "failed"):
                    st.status = "failed"
                    st.error = st.error or "stranded: no active worker available"
        await self._persist_state(aborted=self._abort.is_set())

        result = SweepResult(
            name=self.sweep.name,
            aborted=self._abort.is_set(),
            jobs=list(self._job_state.values()),
        )
        logger.info(
            "sweep %s complete: %d done, %d failed%s",
            self.sweep.name,
            len(result.done),
            len(result.failed),
            " (ABORTED)" if result.aborted else "",
        )
        return result

    # ---- worker loop ----------------------------------------------------- #
    async def _worker_loop(self, worker: Worker) -> None:
        ws = self._worker_state[worker.name]
        while not self._abort.is_set():
            if len(self._settled) >= self._total:
                return
            if ws.status == "quarantined":
                return
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                if len(self._settled) >= self._total:
                    return
                await asyncio.sleep(self.empty_poll_interval)
                continue

            await self._run_one(worker, job)

    async def _run_one(self, worker: Worker, job: Job) -> None:
        st = self._job_state[job.id]
        st.status = "running"
        st.worker = worker.name
        st.attempts += 1
        await self._persist_state(aborted=False)

        timeout = self.sweep.constraints.per_run_timeout_hours * 3600.0
        try:
            remote_config = await worker.send_config(job, self.out_dir / "_jobs")
            remote_dataset = worker._ws(self.sweep.base_dataset)  # noqa: SLF001 (same pkg)
            outcome = await worker.run_training(
                job,
                remote_config=remote_config,
                remote_dataset=remote_dataset,
                sweep_name=self.sweep.name,
                timeout=timeout,
            )
            st.returncode = outcome.returncode
            st.train_loss = outcome.train_loss

            if self._is_finite_loss_fail(outcome.train_loss):
                if self.sweep.constraints.abort_on_first_finite_loss_fail:
                    await self._mark_failed(
                        worker, st, "non-finite train_loss; aborting sweep", requeue=False
                    )
                    self._abort.set()
                    logger.error("ABORT: %s produced non-finite loss", job.id)
                    return
                await self._mark_failed(worker, st, "non-finite train_loss", requeue=True)
                return

            if not outcome.ok:
                await self._mark_failed(
                    worker,
                    st,
                    f"training rc={outcome.returncode}, adapter_present={outcome.adapter_present}",
                    requeue=True,
                )
                return

            remote_tasks = (
                worker._ws(self.sweep.eval_tasks)  # noqa: SLF001
                if self.sweep.eval_tasks
                else None
            )
            await worker.score_job(
                job, sweep_name=self.sweep.name, remote_tasks=remote_tasks, timeout=timeout
            )
            local_dest = self.out_dir / job.id
            await worker.pull_results(job, sweep_name=self.sweep.name, local_dest=local_dest)
            st.task_mean = _read_task_mean(local_dest / "eval-report.json")

            await self._mark_done(worker, st)
        except Exception as exc:  # noqa: BLE001 - one job's failure must not kill the loop
            logger.exception("[%s] job %s raised", worker.name, job.id)
            await self._mark_failed(worker, st, f"{type(exc).__name__}: {exc}", requeue=True)

    # ---- state transitions (all under the lock) -------------------------- #
    async def _mark_done(self, worker: Worker, st: JobState) -> None:
        async with self._lock:
            st.status = "done"
            st.error = None
            self._settled.add(st.id)
            wstate = self._worker_state[worker.name]
            wstate.consecutive_failures = 0
            wstate.completed += 1
        await self._persist_state(aborted=False)

    async def _mark_failed(
        self, worker: Worker, st: JobState, reason: str, *, requeue: bool
    ) -> None:
        async with self._lock:
            st.error = reason
            wstate = self._worker_state[worker.name]
            wstate.consecutive_failures += 1

            can_requeue = requeue and st.attempts <= self.sweep.constraints.max_requeue
            if can_requeue:
                st.status = "pending"
                st.worker = None
                self._queue.put_nowait(self._jobs[st.id])
                logger.warning(
                    "[%s] re-queueing %s (attempt %d/%d): %s",
                    worker.name,
                    st.id,
                    st.attempts,
                    self.sweep.constraints.max_requeue + 1,
                    reason,
                )
            else:
                st.status = "failed"
                self._settled.add(st.id)
                logger.error("[%s] %s permanently failed: %s", worker.name, st.id, reason)

            if wstate.consecutive_failures >= self.sweep.constraints.quarantine_after:
                wstate.status = "quarantined"
                logger.error(
                    "quarantining worker %s after %d consecutive failures",
                    worker.name,
                    wstate.consecutive_failures,
                )
                if self._no_active_workers():
                    self._abort.set()
                    logger.error("ABORT: all workers quarantined")
        await self._persist_state(aborted=self._abort.is_set())

    def _no_active_workers(self) -> bool:
        return all(w.status == "quarantined" for w in self._worker_state.values())

    @staticmethod
    def _is_finite_loss_fail(loss: float | None) -> bool:
        if loss is None:
            return False  # loss not parseable != non-finite; treat rc as the signal
        return not _is_finite(loss)

    # ---- dataset / tasks distribution ------------------------------------ #
    async def _distribute_inputs(self, workers: list[Worker]) -> None:
        local_dataset = self.base_config_dir / self.sweep.base_dataset
        local_tasks = (
            self.base_config_dir / self.sweep.eval_tasks if self.sweep.eval_tasks else None
        )
        for w in workers:
            if local_dataset.exists():
                await w.send_dataset(local_dataset, w._ws(self.sweep.base_dataset))  # noqa: SLF001
            else:
                logger.warning(
                    "dataset %s not found locally; skipping distribution to %s "
                    "(real runs need it pre-staged — issue #3 §4.5)",
                    local_dataset,
                    w.name,
                )
            if local_tasks is not None and local_tasks.exists():
                await w.send_dataset(local_tasks, w._ws(self.sweep.eval_tasks))  # noqa: SLF001

    # ---- persistence ----------------------------------------------------- #
    async def _persist_state(self, *, aborted: bool) -> None:
        state = {
            "sweep": self.sweep.name,
            "updated": self.clock(),
            "total": self._total,
            "settled": len(self._settled),
            "aborted": aborted,
            "jobs": {jid: asdict(s) for jid, s in self._job_state.items()},
            "workers": {n: asdict(s) for n, s in self._worker_state.items()},
        }
        path = self.out_dir / "state.json"
        # Unique temp name per write so concurrent writers can never clobber a
        # shared temp (and so this stays correct even if an await is added later).
        tmp = path.with_suffix(f".json.tmp.{os.getpid()}.{next(self._write_seq)}")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)  # atomic rename on the same filesystem


def _read_task_mean(report_path: Path) -> float | None:
    if not report_path.is_file():
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    val = data.get("task_mean")
    return float(val) if isinstance(val, (int, float)) else None


def _is_finite(x: float) -> bool:
    import math

    return math.isfinite(x)
