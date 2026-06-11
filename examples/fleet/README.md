# codescribe-fleet — parallel & distributed QLoRA across a laptop fleet

A **runnable** reference implementation of [codescribe issue #3](https://github.com/alleboudy/codescribe/issues/3):
put N workstation laptops (Ada-class, 8 GB) to work fine-tuning Qwen 2.5
Coder 7B with QLoRA. It turns the issue's §5.3 pseudocode and §6 recipes into
working code, and ships a mock trainer so you can run the **entire pipeline on one
machine with no GPU** before pointing it at real hardware.

Read the companion walkthrough first: [`docs/14-fleet-training.md`](../../docs/14-fleet-training.md).

> **Path A vs Path B (the one decision that matters).** On 8 GB-VRAM laptops over
> a gigabit LAN, **Path A — a parallel hyperparameter sweep, one config per
> worker — almost always beats Path B — distributed data-parallel of a single
> run.** Path B's gradient-sync overhead eats the speedup below 10 Gbps. This repo
> implements both; default to Path A.

## Install

```bash
uv sync --extra dev          # test/demo env — no asyncssh, no GPU needed
uv sync --extra ssh --extra dev   # add the real SSH transport for a real fleet
```

## Try it with no GPU and no SSH (the demo)

Runs a 9-point `lr × rank` sweep across two localhost "workers" using the mock
trainer (`scripts/fake_train.py`), which writes deterministic adapters + eval
reports so the schedule → train → score → pull → rank flow is real:

```bash
uv run python -m codescribe_fleet sweep \
    --fleet configs/fleet.local.yaml \
    --sweep sweeps/lr-rank.yaml \
    --out  sweeps/demo/ \
    --mode local

uv run python -m codescribe_fleet status    --sweep sweeps/demo/
uv run python -m codescribe_fleet summarise --sweep sweeps/demo/
```

The summary is sorted by `task_mean`; the mock's score peaks at the QLoRA
reference point (`lr=2e-4, r=16`), so that combo wins — exactly the kind of signal
a real sweep gives you.

## Run it on a real fleet (Path A)

1. Build the single-host stack from issue #2 on each laptop (the real `train`
   package). Do the per-laptop setup from issue #3 §4 (OEM thermal profile,
   SSH keys, pre-stage the base model, preflight smoke).
2. Edit `configs/fleet.yaml` (your workers, absolute workspaces, SSH identity) and
   `sweeps/lr-rank.yaml` (your grid + dataset).
3. Run:
   ```bash
   uv run python -m codescribe_fleet sweep \
       --fleet configs/fleet.yaml \
       --sweep sweeps/lr-rank.yaml \
       --out  sweeps/lr-rank-$(date +%F)/ \
       --mode ssh
   ```
4. Watch the fleet: `uv run python -m codescribe_fleet monitor --fleet configs/fleet.yaml --mode ssh`
5. Export the winning adapter to GGUF with the real `train export` (issue #2),
   then serve it (issue #1).

## Path B (distributed DDP — only if you've read issue #3 §3)

Generate the per-node `torchrun` commands + NCCL env, then launch on each node:

```bash
uv run python -m codescribe_fleet ddp-plan \
    --fleet configs/fleet.yaml --sweep sweeps/lr-rank.yaml \
    --rdzv-host trainer-01
# ...or run scripts/run_ddp.sh on each node with NODE_RANK / RDZV_HOST / NCCL_SOCKET_IFNAME set.
```

## Test

```bash
uv run pytest -q
```

36 tests: config/grid/ddp/summary units, an orchestrator unit test for the
re-queue / quarantine / finite-loss-abort rules (in-memory fake transport), a full
end-to-end run through `LocalTransport` + the mock trainer, and static safety
checks (no `shell=True`, subprocess confined to `transport.py`, command quoting,
logger-only, no `0.0.0.0`).

## What this is / isn't

- **Is:** the orchestration layer (scheduling, robustness, monitoring, DDP launch)
  plus a runnable harness to exercise it.
- **Isn't:** the trainer. Real QLoRA training is issue #2's `train` package (needs
  CUDA). Point `train.module` in your fleet config at it. `scripts/fake_train.py`
  is a stand-in that honours the same CLI contract.
