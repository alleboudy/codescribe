# codescribe_fleet — fleet QLoRA orchestrator

Runnable reference for codescribe issue #3: drive N workstation laptops to
fine-tune Qwen 2.5 Coder 7B with QLoRA. Path A (parallel HP sweep, recommended)
and Path B (distributed DDP, rarely worth it).

## Cross-cutting hard rules
- Python 3.12; `uv` for deps (never `pip install`); type hints throughout.
- No cloud calls. The only network egress is SSH to the operator's own workers
  (production transport) — no telemetry, no hosted services.
- Network defaults to loopback. No `0.0.0.0`.
- No `print(...)` in library code — `logger = logging.getLogger(__name__)` only.
  `__main__.py` (CLI) and `scripts/` may print.
- **All remote command strings are built with `shlex.quote` on every interpolated
  path/value.** Subprocess execution is confined to `transport.py` — the single
  audited boundary. `tests/test_command_safety.py` pins both invariants.
- Heavy/production-only deps are extras: `ssh` (asyncssh) for the real fleet;
  `dev` (pytest, pytest-asyncio) for the suite. The test suite + local demo need
  neither asyncssh nor a GPU.

## Architecture
```
sweep.yaml grid ─► grid.expand_grid ─► Job[] ─► asyncio.Queue
                                                   │
fleet.yaml ─► Worker(spec, Transport, TrainLauncher) ◄┘  (one loop per worker)
   per job:  send_config → run_training → score_job → pull_results
   Transport:  LocalTransport (demo) | SSHTransport (prod) | FakeTransport (tests)
   robustness: re-queue (max_requeue) · quarantine (quarantine_after) · finite-loss abort
   state:      state.json (atomic) ─► summary.summarise ─► sorted table
Path B:  ddp.build_torchrun_command + ddp.nccl_env  ◄─  scripts/run_ddp.sh
```

## Where things live
| File | What |
|---|---|
| `codescribe_fleet/config.py` | typed loaders for fleet.yaml / sweep.yaml |
| `codescribe_fleet/grid.py` | cartesian expansion + per-job config render + seed |
| `codescribe_fleet/transport.py` | Local/SSH transports — the only subprocess site |
| `codescribe_fleet/worker.py` | per-worker ops (quoted command building) |
| `codescribe_fleet/orchestrator.py` | Path A scheduler (queue, re-queue, quarantine) |
| `codescribe_fleet/ddp.py` | Path B torchrun/NCCL builders |
| `codescribe_fleet/monitor.py` | fleet-wide nvidia-smi → JSONL |
| `codescribe_fleet/summary.py` | results table sorted by task_mean |
| `codescribe_fleet/__main__.py` | CLI: sweep / status / summarise / ddp-plan / monitor |
| `scripts/fake_train.py` | mock trainer honouring the real `train` CLI contract |
| `scripts/run_ddp.sh` | Path B per-node launcher |

## The train contract this orchestrator drives
The orchestrator invokes a training CLI on each worker. Production points at the
real package from issue #2 (`train.module: codescribe_train.train`); the demo/tests
point at `scripts/fake_train.py`. Either must accept:
```
<prefix> run   --config C --dataset D --out O      # writes O/adapter_model.safetensors
<prefix> eval  --adapter O [--tasks T] --report R  # writes R = {"task_mean": float}
```
`fake_train.py` is NOT a trainer — it sleeps and writes deterministic artefacts so
the whole pipeline runs with no GPU. Swap in the real trainer to train for real.
