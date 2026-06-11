# Training paths — single-host, parallel sweep (A), distributed (B)

The QLoRA trainer (`codescribe_train.train`) is the unit of work. There are three
ways to run it, depending on how much hardware you have and what you're optimising
for. The trainer itself is the same in all three — only how you *launch* it changes.

The CLI contract (identical across paths):

```bash
python -m codescribe_train.train smoke  --config configs/train/qwen7b_qlora.yaml --dataset datasets/<repo>/ --out checkpoints/smoke/
python -m codescribe_train.train run    --config configs/train/qwen7b_qlora.yaml --dataset datasets/<repo>/ --out checkpoints/<repo>-lora-v1/
python -m codescribe_train.train eval   --adapter checkpoints/<repo>-lora-v1/ [--dataset datasets/<repo>/] [--tasks evals/sample_tasks.json] --report eval-report.json
python -m codescribe_train.train export --adapter checkpoints/<repo>-lora-v1/ --base-model Qwen/Qwen2.5-Coder-7B-Instruct --merged checkpoints/merged/ --gguf checkpoints/model-q4_k_m.gguf --llama-cpp vendor/llama.cpp
```

`eval`'s `--dataset` is optional: with it you also get held-out perplexity; without
it you get the task-suite score only (this is what the sweep orchestrator uses to
rank adapters by `task_mean`).

---

## Path 0 — single-host (how this stack was originally trained)

One machine, one GPU. The default. Run the four steps in sequence:

```bash
python -m codescribe_train.train smoke  --config configs/train/qwen7b_qlora.yaml --dataset datasets/<repo>/ --out checkpoints/smoke/   # 10-step gate
python -m codescribe_train.train run    --config configs/train/qwen7b_qlora.yaml --dataset datasets/<repo>/ --out checkpoints/<repo>-lora-v1/
python -m codescribe_train.train eval   --adapter checkpoints/<repo>-lora-v1/ --dataset datasets/<repo>/ --report eval-report.json
python -m codescribe_train.train export --adapter checkpoints/<repo>-lora-v1/ --base-model Qwen/Qwen2.5-Coder-7B-Instruct --merged checkpoints/merged/ --gguf checkpoints/model-q4_k_m.gguf --llama-cpp vendor/llama.cpp
```

~3.4 h for a 3-epoch run on a ~3K-sample dataset (a consumer Blackwell-class laptop GPU (example hardware), FA2, seq_len=1024).
**Always run `smoke` first on new hardware** — if it doesn't print a finite,
decreasing `train_loss` in a few minutes, stop and diagnose before the full run.

---

## Path A — parallel hyperparameter sweep (recommended for a fleet)

If you have several machines, the highest-throughput use of them is **not** to make
one run faster — it's to run many *different* configs at once, one per machine, and
pick the best adapter. No gradient sync, no shared state; linear speedup with worker
count. This is the recommended path for a laptop fleet on a gigabit LAN.

The orchestrator for this lives in the companion **fleet example** (`examples/fleet/`,
[issue #3](https://github.com/example-org/codescribe-train/issues/3)). It drives this exact
trainer on each worker — its `train.module` is `codescribe_train.train`:

```bash
# from examples/fleet/  (configs/fleet.yaml: train.module = codescribe_train.train)
python -m codescribe_fleet sweep --fleet configs/fleet.yaml --sweep sweeps/lr-rank.yaml --out sweeps/run/ --mode ssh
python -m codescribe_fleet summarise --sweep sweeps/run/      # ranks adapters by task_mean
```

Each worker runs **Path 0's `run`** on a different point in the grid, then the
orchestrator calls `eval` (task-suite only — hence the optional `--dataset`) and
pulls back the adapter + `eval-report.json`. See the fleet example's README for the
full setup.

---

## Path B — distributed data-parallel (one run, many GPUs)

When you genuinely need a *single* run finished faster and have ≥10 Gbps between
nodes, you can data-parallel it. **`trl`/HF `Trainer` pick up DDP automatically from
`torchrun`'s environment variables**, so Path B is just Path 0's `run` launched under
`torchrun` — no separate entry point. Each rank loads the full dataset; the Trainer's
distributed sampler shards it; only rank 0 writes the adapter.

```bash
# on EVERY node, varying only --node-rank (0 on the rendezvous master):
torchrun \
    --nproc-per-node=1 --nnodes=4 --node-rank=$NODE_RANK \
    --rdzv-id=ft-001 --rdzv-backend=c10d --rdzv-endpoint=<master-host>:29400 \
    -m codescribe_train.train run \
        --config configs/train/qwen7b_qlora.yaml --dataset datasets/<repo>/ --out checkpoints/<repo>-ddp-v1/
```

The fleet example generates these per-node commands for you (`codescribe_fleet
ddp-plan`) and ships a launcher (`scripts/run_ddp.sh`) plus the NCCL settings that
work on consumer LAN (`NCCL_IB_DISABLE=1`, `NCCL_NET=Socket`, `NCCL_SOCKET_IFNAME`).

**Two honest caveats:**

1. **Path A almost always wins on ≤1 Gbps.** DDP's gradient-sync overhead eats the
   per-step speedup; you pay ~3× the compute for ~1.4× wall-clock. Only reach for
   Path B with 10 Gbps+ end-to-end and a single config you must finish fast. The
   numbers + decision tree are in [issue #3](https://github.com/example-org/codescribe-train/issues/3) §3.
2. **Unsloth vs DDP.** The single-GPU loop uses Unsloth's fused kernels + gradient
   checkpointing, which conflict with DDP wrapping on some versions (`Expected to
   mark a variable ready only once`, NaN on step 2). If you hit this, drop Unsloth
   for vanilla HF `Trainer` + `accelerate` for the distributed run (you lose the
   ~4× Unsloth speedup, so re-check that Path B still beats Path A). Track Unsloth's
   release notes — DDP support is occasionally fixed.

---

## Which path?

| You have… | Use |
|---|---|
| one machine | Path 0 |
| several machines + many configs to try | **Path A** (parallel sweep) |
| several machines, one config, ≥10 Gbps LAN, must finish fast | Path B (DDP) |
| several machines, one config, ≤1 Gbps LAN | Path 0 on one; sweep the rest with Path A |
