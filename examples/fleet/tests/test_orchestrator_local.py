"""End-to-end test: the REAL LocalTransport driving the REAL mock trainer.

This exercises the whole Path A pipeline as a user would run the demo — subprocess
launch, config push, train, adapter check, eval, pull, summarise — with no GPU and
no SSH. It is the closest thing to the `python -m codescribe_fleet sweep --mode local`
command in the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

from codescribe_fleet.config import (
    CoordinatorSpec,
    FleetConfig,
    TrainLauncher,
    WorkerSpec,
)
from codescribe_fleet.orchestrator import Orchestrator
from codescribe_fleet.summary import best, summarise
from codescribe_fleet.transport import LocalTransport

from .conftest import make_sweep

EXAMPLE = Path(__file__).resolve().parents[1]


async def test_local_end_to_end(tmp_path: Path) -> None:
    # Two localhost "workers"; workspace is the example dir so `scripts/fake_train.py`
    # resolves after the orchestrator `cd`s into it.
    specs = [
        WorkerSpec(name=f"local-0{i}", ssh_host="localhost", workspace=str(EXAMPLE))
        for i in (1, 2)
    ]
    fleet = FleetConfig(
        coordinator=CoordinatorSpec(),
        workers=specs,
        # use this interpreter explicitly so the test never depends on PATH
        train=TrainLauncher(runner=sys.executable, script="scripts/fake_train.py"),
    )
    sweep = make_sweep({"optim.lr": [1e-4, 2e-4], "lora.r": [8, 16]})
    out = tmp_path / "demo"

    orch = Orchestrator(
        fleet,
        sweep,
        out_dir=out,
        transport_factory=lambda s: LocalTransport(s.name),
        base_config_dir=EXAMPLE,
        empty_poll_interval=0.01,
    )
    result = await orch.run()

    assert len(result.done) == 4, [j.error for j in result.failed]
    assert not result.aborted

    rows = summarise(out)
    # the mock's task_mean peaks at lr=2e-4, r=16 -> that combo must win.
    win = best(rows)
    assert win is not None
    assert win.task_mean is not None
    # grid keys sort to [lora.r, optim.lr]; index 3 == (r=16, lr=2e-4)
    assert win.job_id == "combo-003-r=16-lr=0.0002"

    # every done job produced a real adapter + report that was pulled back
    for j in result.done:
        assert (out / j.id / "adapter_model.safetensors").is_file()
        assert (out / j.id / "eval-report.json").is_file()
