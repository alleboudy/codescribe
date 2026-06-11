# 14 — Fleet training: parallel & distributed QLoRA across N laptops

You built the single-host stack ([issue #2](https://github.com/alleboudy/codescribe/issues/2)):
one laptop can take your codebase and produce a QLoRA fine-tune of Qwen 2.5 Coder
7B. Now you have **several** laptops — a fleet of Dell Precision mobile
workstations — and you want them all working. This doc is the ground-up guide to
that, and it pairs with a **runnable reference implementation** at
[`examples/fleet/`](../examples/fleet/) so you can see the orchestration work
end-to-end on one machine before you touch real hardware.

The authoritative procedure is [issue #3](https://github.com/alleboudy/codescribe/issues/3);
this doc is the readable on-ramp to it.

---

## The one decision that matters

> On **8 GB-VRAM laptops over a gigabit LAN**, a **parallel hyperparameter sweep
> (Path A)** — each laptop trains an independent config — almost always beats
> **distributed data-parallel (Path B)** — all laptops collaborate on one run.

Why: Path B has to synchronise gradients across the network every step. On 1 Gbps
Ethernet that sync overhead eats the per-step speedup — you pay ~3× the compute for
~1.4× wall-clock. Path A has *zero* cross-node traffic during training, so it
scales linearly with worker count. Path B only wins when (a) you have 10 Gbps+
end-to-end **and** (b) you have one specific config you need finished fast rather
than many configs to explore.

```
Are you exploring multiple configs (LR, rank, seq_len, …)?
  └─ yes ──────────────────────────────────► PATH A (parallel sweep). Done.
  └─ no, I have ONE run I want faster
        └─ LAN ≥ 10 Gbps end-to-end?
              └─ yes ──────────────────────► PATH B may be worth it.
              └─ no
                    └─ one run > 10 h on a single laptop?
                          └─ yes ──────────► PATH B gives ~1.5–2× on 1 Gbps.
                          └─ no ───────────► PATH A (run it once; sweep the rest).
```

Budget, 4× RTX 2000 Ada on 1 Gbps, sweep of 16 configs (~3K-sample dataset):

| Approach | Setup | Wall-clock | Compute (node-h) | Verdict |
|---|---|---|---|---|
| 1 laptop, sequential | minimal | ~80 h | 80 | only if you have one |
| **Path A, 4-laptop sweep** | ~2 h | **~20 h** | 80 | **default** |
| Path B, 4-laptop DDP @ 1 Gbps | 1–3 d debug | ~192 h | 768 | don't |
| Path B, 4-laptop DDP @ 10 Gbps | 1–3 d debug | ~54 h | 216 | only with 10 GbE + single-config need |

---

## Reading order

If you're starting cold, read in this order. Each doc is short; come back to the
phase-specific ones as you reach them.

1. [`01-overview.md`](01-overview.md) — what the whole stack does end-to-end.
2. [`02-fine-tuning.md`](02-fine-tuning.md) — **what one QLoRA run is.** This is
   the unit of work the fleet parallelises: LoRA/QLoRA, the config YAML, the
   `smoke`/`run`/`eval`/`export` CLI, the 70/20/10 data mix, the split-by-file rule.
3. [`11-hardware.md`](11-hardware.md) — VRAM math (why `seq_len=1024` on 8 GB), the
   compute-capability map (RTX 2000 Ada = **sm_89**), and laptop thermal/power.
4. [`03-models.md`](03-models.md) — why Qwen 2.5 Coder 7B Instruct is the base.
5. **[issue #3](https://github.com/alleboudy/codescribe/issues/3)** — the fleet
   guide itself (decision tree, per-laptop setup, orchestrator, monitoring,
   anti-patterns, budget).
6. [issue #2](https://github.com/alleboudy/codescribe/issues/2) — the single-host
   stack the fleet **assumes you've already built** (the `train` package).
7. [issue #1](https://github.com/alleboudy/codescribe/issues/1) — serving the
   winning adapter as a GGUF after the sweep picks it.

Then open [`examples/fleet/`](../examples/fleet/) and run the demo (below).

---

## The hardware: RTX 2000 Ada laptops

| Spec | Value | Why it matters |
|---|---|---|
| Architecture | Ada Lovelace (AD107), **sm_89** | Modern enough for Flash-Attention 2; CUDA 12.x is clean. llama.cpp build flag: `-DCMAKE_CUDA_ARCHITECTURES=89` (NOT `120`/Blackwell, NOT `75`/Turing). |
| VRAM | 8 GB GDDR6 | Qwen 7B fits in 4-bit + a small adapter; nothing bigger fits. `seq_len=1024` is the hard ceiling. |
| TGP | 35–75 W, OEM-configurable | Higher = faster. Set to max in BIOS. |
| Mem bandwidth | ~256 GB/s | ~60–70% of an RTX 5070 Laptop → expect ~28–35 s/step (vs ~21 s), ~4.5–5.5 h for a 3-epoch run on a ~3K-sample dataset. |

Measure your own per-step time first (the preflight smoke, below) — those numbers
are extrapolated.

### Dell Precision per-laptop checklist (do once, sanity-check before every long run)

These are the failure modes that silently halve throughput:

1. **BIOS thermal profile → "Ultra Performance"** (F2 at boot → Performance →
   Thermal Management). The default "Optimized" throttles under sustained load.
2. **Dell Power Manager → "Ultra Performance"** power plan on AC.
3. **Run on AC, always.** On battery the GPU drops to ~30 W (vs 75 W) → ~3–5×
   slower. Use the 130 W+ USB-C PD or barrel-jack adapter, not a 65 W charger.
4. **Don't stack laptops.** They pull air through the bottom; stacked → 95 °C →
   throttle. Cooling pads with rear fans buy 5–10 °C if your fleet runs hot.
5. **Verify power actually lands:**
   ```bash
   sudo nvidia-smi -i 0 -pm 1        # persistent mode (less driver init overhead)
   sudo nvidia-smi -i 0 -pl 75       # set power limit to 75 W if BIOS allows
   nvidia-smi --query-gpu=power.draw,temperature.gpu,clocks.current.sm --format=csv -l 2
   ```
   Healthy under load: `power.draw` 65–75 W steady (not 35 W), `temperature.gpu`
   75–88 °C, `clocks.current.sm` ≥ 1500 MHz. Stuck at 35 W with low temps = BIOS
   throttling; re-check the thermal profile.

### Per-laptop one-time setup (≈1–2 h each), summarised from issue #3 §4

1. **Drivers + compute cap:** `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`
   → expect `... RTX 2000 Ada ..., 8.9`. Rebuild llama.cpp with `-DCMAKE_CUDA_ARCHITECTURES=89`
   if the binary came from a different-arch host.
2. **Networking:** static hostnames (or Tailscale for a mesh across networks);
   open ports `29400:29500` + `22` for Path B; verify wired speed with `ethtool`.
3. **SSH key auth from the coordinator to every worker** (`ssh-copy-id`, a
   `~/.ssh/config` entry per worker). The orchestrator drives workers over this.
4. **Pre-stage the base model on each laptop** (`hf download Qwen/Qwen2.5-Coder-7B-Instruct`,
   ~15 GB). Without this every job re-downloads it. **Never share `~/.cache/huggingface`
   over NFS — it corrupts on concurrent writes.** Each laptop owns its own.
5. **Preflight smoke (DO NOT SKIP):** a 10-step `train smoke` on each laptop.
   Record final `train_loss` (finite + decreasing), wall-clock, and peak
   `memory.used` (≤ 7.5 GB). A laptop >1.5× the median is throttling — fix it
   before adding it to the fleet.

---

## Path A — the parallel sweep (recommended)

**Idea:** each worker runs an *independent* training job with a different
hyperparameter combination. No gradient sync, no NCCL, no shared state during
training. The coordinator schedules jobs onto idle workers and pulls back each
`adapter_model.safetensors` + `eval-report.json`.

The reference implementation lives in [`examples/fleet/`](../examples/fleet/). Its
shape:

```
sweep.yaml grid ─► expand_grid ─► Job[] ─► asyncio.Queue
fleet.yaml ─► one Worker per laptop, one loop each, pulling jobs:
                send_config → run_training → score_job → pull_results
robustness:    re-queue failed jobs (max_requeue) · quarantine a flapping worker
               (quarantine_after) · abort on non-finite loss
output:        state.json (live) ─► summarise → table sorted by task_mean
```

### See it run with no GPU (the demo)

The example ships a **mock trainer** (`scripts/fake_train.py`) that honours the
real `train` CLI contract but just writes deterministic artefacts — so the whole
schedule/train/score/pull/rank flow runs on your laptop, no GPU, no SSH:

```bash
cd examples/fleet
uv sync --extra dev
uv run python -m codescribe_fleet sweep \
    --fleet configs/fleet.local.yaml \
    --sweep sweeps/lr-rank.yaml \
    --out  sweeps/demo/ --mode local
uv run python -m codescribe_fleet summarise --sweep sweeps/demo/
```

You'll get a table like (the mock's score peaks at the QLoRA reference point, so
`lr=2e-4, r=16` wins — the same signal a real sweep gives you):

```
job_id                      worker     status   task_mean train_loss tries
combo-004-r=16-lr=0.0002    local-02   done         0.620      0.300     1
combo-007-r=32-lr=0.0002    local-01   done         0.579      0.391     1
...
best: combo-004-r=16-lr=0.0002 (task_mean=0.620) on local-02
```

### Run it for real

1. On each laptop: build the `train` package (issue #2) and do the per-laptop
   setup above.
2. Edit `configs/fleet.yaml` (your workers, **absolute** workspaces, SSH identity;
   `train.module: codescribe_train.train`) and `sweeps/lr-rank.yaml` (your grid +
   dataset + eval tasks).
3. Run the sweep over SSH:
   ```bash
   uv run python -m codescribe_fleet sweep \
       --fleet configs/fleet.yaml --sweep sweeps/lr-rank.yaml \
       --out sweeps/lr-rank-$(date +%F)/ --mode ssh
   ```
4. Watch it: `uv run python -m codescribe_fleet monitor --fleet configs/fleet.yaml --mode ssh`
   (and `status --sweep <dir>` for job/worker state).
5. Take the top `adapter_model.safetensors`, `train export` it to a Q4_K_M GGUF
   (issue #2 / [`02-fine-tuning.md`](02-fine-tuning.md#export-to-gguf-the-historic-landmine)),
   and serve it (issue #1).

### Path A pitfalls (designed-for in the reference impl)

- **A worker drops mid-sweep** → its job is re-queued onto another idle worker
  (`max_requeue`). A worker that fails repeatedly is quarantined (`quarantine_after`)
  so it stops poisoning the queue.
- **One worker consistently slower** → almost always thermal. Re-run the Dell
  checklist.
- **A worker silently OOMs** → check its training log for `CUDA out of memory`;
  peak VRAM at `seq_len=1024` should be ≤ 7.5 GB. If it's at 8 GB you're over
  `seq_len` or the data loader leaks.
- **Same seed everywhere** → the orchestrator sets `seed = sweep_seed + job_index`
  so no two jobs are identical and any run is reproducible.

---

## Path B — distributed data-parallel (advanced; usually skip)

Read the decision tree again first. If you genuinely need it:

`torchrun` launches the same training entry point on every node; one node is the
rendezvous master. The reference impl gives you the exact commands:

```bash
uv run python -m codescribe_fleet ddp-plan \
    --fleet configs/fleet.yaml --sweep sweeps/lr-rank.yaml \
    --rdzv-host dell-precision-01
```

That prints, per node, the NCCL env + the `torchrun` line (varying only
`--node-rank`). Or run `scripts/run_ddp.sh` on each node with `NODE_RANK`,
`RDZV_HOST`, and `NCCL_SOCKET_IFNAME` set.

The NCCL settings that actually work on consumer LAN (no InfiniBand):

```bash
export NCCL_IB_DISABLE=1            # no InfiniBand on a laptop
export NCCL_NET=Socket             # force TCP
export NCCL_SOCKET_IFNAME=eno1     # your WIRED interface (`ip -br link`)
export NCCL_DEBUG=INFO             # leave on for the first runs
export TORCH_DIST_INIT_BARRIER_TIMEOUT=600
```

Three things that bite:

- **`NCCL_SOCKET_IFNAME` wrong** → `NCCL WARN Could not find network address`.
- **WiFi** → NCCL hangs. Always wired.
- **The Unsloth-vs-DDP tension** → Unsloth's fused kernels conflict with DDP
  wrapping on some versions (`Expected to mark a variable ready only once`, NaN on
  step 2). Workarounds in order: check Unsloth release notes → drop Unsloth for
  vanilla HF `Trainer` + `accelerate` (loses the 4× speedup, may end up slower than
  Path A) → try FSDP. Budget 2–3 days. **Try Path A first.**
- **Effective batch size changes** → DDP multiplies it by node count; lower LR by
  ~`sqrt(N)` (the reference impl exposes `scale_lr_for_nodes`).

---

## Monitoring the fleet

```bash
uv run python -m codescribe_fleet monitor --fleet configs/fleet.yaml --mode ssh \
    --interval 5 --out logs/fleet.jsonl
```

Polls `nvidia-smi` (name/temp/power/mem/util) on every worker into JSONL. For a
quick eyeball without the tool, issue #3 §7 has a `watch` one-liner. For 10+
laptops, graduate to Prometheus + `nvidia_gpu_exporter`.

---

## Anti-patterns (issue #3 §8)

- **Don't put the coordinator on a worker laptop** — if that laptop's training
  crashes you lose the orchestrator state too. Use a separate box (a NUC, a Pi,
  any spare machine).
- **Don't share `~/.cache/huggingface` over NFS/SMB** — concurrent-write corruption.
- **Never train on battery.**
- **Don't run two training jobs on one laptop** — 8 GB fits exactly one 7B QLoRA job.
- **Don't run distributed *inference* across laptops** — a 7B Q4_K_M fits on one
  card; pin inference to one worker.
- **Don't mix native-Ubuntu and WSL2 workers in one DDP job** — NCCL is sensitive
  to network-stack differences.

---

## When to outgrow the fleet (issue #3 §9, [`11-hardware.md`](11-hardware.md#when-to-escalate-beyond-laptop-class))

A sweep that takes >1 week even with every worker busy; a model bigger than 7B
(needs ≥12 GB even in 4-bit); >32K context; or being DDP-bound on the LAN. The
sane next step before cloud is **one desktop/tower with an RTX 4090 (24 GB, ~€2K)**
— it replaces a 4-laptop fleet for most workloads, Path B's sync overhead vanishes,
and each Path A job runs ~2.5× faster.

---

## Related

- [`examples/fleet/`](../examples/fleet/) — the runnable orchestrator (this doc's companion).
- [issue #3](https://github.com/alleboudy/codescribe/issues/3) — the authoritative fleet guide.
- [issue #2](https://github.com/alleboudy/codescribe/issues/2) — the single-host stack you build first.
- [issue #1](https://github.com/alleboudy/codescribe/issues/1) — serving the winning GGUF.
- [`02-fine-tuning.md`](02-fine-tuning.md), [`03-models.md`](03-models.md), [`11-hardware.md`](11-hardware.md) — the concepts.
