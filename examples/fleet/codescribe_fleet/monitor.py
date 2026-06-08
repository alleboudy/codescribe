"""Fleet-wide GPU monitoring (codescribe issue #3 §7).

Polls ``nvidia-smi`` on every worker on an interval and appends one JSON record
per worker per round to a JSONL file. Lightweight by design — for >10 workers the
issue suggests Prometheus + ``nvidia_gpu_exporter`` instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from .config import WorkerSpec
from .transport import Transport

logger = logging.getLogger(__name__)

# The columns we ask nvidia-smi for, in order. Keep in sync with parse_smi_csv.
SMI_QUERY_FIELDS = (
    "name",
    "temperature.gpu",
    "power.draw",
    "memory.used",
    "memory.total",
    "utilization.gpu",
)


def build_smi_command() -> str:
    fields = ",".join(SMI_QUERY_FIELDS)
    return f"nvidia-smi --query-gpu={fields} --format=csv,noheader,nounits"


def parse_smi_csv(line: str, fields: tuple[str, ...] = SMI_QUERY_FIELDS) -> dict[str, str]:
    """Parse one ``--format=csv,noheader`` row into ``{field: value}``.

    Values are kept as trimmed strings (nvidia-smi mixes units/text); callers cast
    as needed. Extra/short columns are tolerated rather than raising.
    """
    cells = [c.strip() for c in line.split(",")]
    return {field: (cells[i] if i < len(cells) else "") for i, field in enumerate(fields)}


async def poll_fleet(
    workers: list[WorkerSpec],
    transport_factory: Callable[[WorkerSpec], Transport],
    *,
    out_jsonl: str | Path,
    interval: float = 5.0,
    rounds: int | None = None,
    clock: Callable[[], float] = time.time,
) -> Path:
    """Poll every worker ``rounds`` times (forever if ``rounds is None``).

    One :class:`~codescribe_fleet.transport.Transport` is created per worker up
    front and reused across rounds. Returns the JSONL path written.
    """
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transports: dict[str, Transport] = {w.name: transport_factory(w) for w in workers}
    # The whole body is inside try/finally so every transport that was opened is
    # closed even if a connect() fails mid-setup. A single unreachable worker must
    # NOT abort monitoring of the rest (a monitor must never crash the loop).
    unreachable: set[str] = set()
    try:
        for w in workers:
            connect = getattr(transports[w.name], "connect", None)
            if connect is None:
                continue  # LocalTransport / fakes need no connect
            try:
                await connect()
            except Exception as exc:  # noqa: BLE001
                logger.warning("monitor: %s unreachable at startup: %s", w.name, exc)
                unreachable.add(w.name)

        count = 0
        with out_path.open("a", encoding="utf-8") as fh:
            while rounds is None or count < rounds:
                ts = clock()
                for w in workers:
                    if w.name in unreachable:
                        record: dict[str, object] = {
                            "ts": ts, "worker": w.name, "status": "unreachable"
                        }
                    else:
                        record = await _poll_one(w, transports[w.name], ts)
                    fh.write(json.dumps(record) + "\n")
                fh.flush()
                count += 1
                if rounds is None or count < rounds:
                    await asyncio.sleep(interval)
    finally:
        for t in transports.values():
            try:
                await t.close()
            except Exception:  # noqa: BLE001 - cleanup must not raise
                pass
    return out_path


async def _poll_one(worker: WorkerSpec, transport: Transport, ts: float) -> dict:
    record: dict[str, object] = {"ts": ts, "worker": worker.name}
    try:
        result = await transport.run(build_smi_command(), timeout=30.0)
        if result.returncode == 0 and result.stdout.strip():
            record.update(parse_smi_csv(result.stdout.strip().splitlines()[0]))
            record["status"] = "ok"
        else:
            record["status"] = "smi_failed"
            record["rc"] = result.returncode
    except Exception as exc:  # noqa: BLE001 - a monitor must never crash the loop
        logger.warning("monitor: %s unreachable: %s", worker.name, exc)
        record["status"] = "unreachable"
        record["error"] = str(exc)
    return record
