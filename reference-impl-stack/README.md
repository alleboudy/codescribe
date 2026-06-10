# codescribe-train

A **strictly-local**, agentic coding assistant.

Fine-tunes a small open-weights coder (**Qwen 2.5 Coder 7B**) on a target git repository, then drives a vendored CLI agent harness against the result. Designed for private repositories under contract or NDA where **no code, datasets, weights, or telemetry can leave the machine**.

> `sample` / `example-org` / [`../your-repo`](../your-repo/) are placeholders for **your own** target repository — the codebase you fine-tune on. This stack shares no code with it; it's only the training-data source.

---

## Status — all phases complete

| Phase | Status | What it does |
|---|---|---|
| 0 — scaffold | ✅ | Repo skeleton, tooling, CI-equivalent ruff + pytest |
| 1 — `data/` | ✅ | Any git repo → train/val/test JSONL splits |
| 2 — `train/` | ✅ | QLoRA fine-tuning via Unsloth on 8 GB VRAM |
| FA2 perf bump | ✅ | Cross-built on a second (builder) machine, 4.1× |
| 3 — `harness/` | ✅ | Pluggable Rust agent harness via `ultraworkers/claw-code` |
| 4 — `backends/` | ✅ | Pluggable OpenAI-compat inference layer (default `llama-server`) |

**End-to-end DoD verified** with the egress audit: zero non-loopback `connect()` calls during a full session. ⇒ [`docs/known-issues.md § Egress audit`](docs/known-issues.md#egress-audit--passed-substitute-model-fine-tuned-model).

Continuity playbook for outage recovery / future sessions: [issue #1](https://github.com/example-org/codescribe-train/issues/1).

---

## Documentation index

### Background — what & why

| Doc | What it covers |
|---|---|
| **[`docs/CONCEPTS.md`](docs/CONCEPTS.md)** | **Start here.** Conceptual map of the three pillars (fine-tuning, RAG, MCP) — what each is, pros/cons, how `codescribe-train` applies each to `../your-repo`. Curated entry point to all of the detail docs below. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System overview, four-phase composition, trust posture, what's deliberately out of scope |
| [`docs/FLASH-ATTENTION.md`](docs/FLASH-ATTENTION.md) | What attention is, why O(N²) memory matters, FA1/FA2/FA3, the `cu128` × `torch2.10` × `sm_120` × `cp313` ABI break, the cross-build solution |
| [`docs/QLoRA-AND-UNSLOTH.md`](docs/QLoRA-AND-UNSLOTH.md) | LoRA → QLoRA → Unsloth — what each layer of optimisation does, hyperparameter rationale, the three sample formatters |
| [`docs/HARDWARE-AND-PERFORMANCE.md`](docs/HARDWARE-AND-PERFORMANCE.md) | Hardware split (training machine trains, builder machine builds), constraints, the seq_len 2048→1024 decision with measurements, swap-thrash forensics |
| [`docs/STRICTLY-LOCAL-POSTURE.md`](docs/STRICTLY-LOCAL-POSTURE.md) | Threat model, vendored components & their audit, defence-in-depth layers, the egress audit method, hardening tips |
| [`docs/RAG.md`](docs/RAG.md) | What retrieval-augmented generation is, embedder/vector-store choices, when RAG vs fine-tuning vs tools, local-only stack |
| [`docs/MCP-SERVERS.md`](docs/MCP-SERVERS.md) | Model Context Protocol — JSON-RPC shape, Python SDK, claw-code integration, useful server patterns |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | Quick lookup for project jargon (FIM, GGUF, NF4, sm_120, …) |

### Operational

| Doc | What it covers |
|---|---|
| [`docs/known-issues.md`](docs/known-issues.md) | Diagnoses + rerun commands. Currently: FA2 install record, egress audit result |
| [`vendor/SECURITY-NOTES.md`](vendor/SECURITY-NOTES.md) | Audit findings for vendored claw-code and llama.cpp |
| [`vendor/README-VENDOR.md`](vendor/README-VENDOR.md) | Vendoring policy, SHA pinning, fallback remotes |

### Per-package

| Package | Path | What it does |
|---|---|---|
| `data` | [`codescribe_train/data/README.md`](codescribe_train/data/README.md) | Generic git repo → JSONL pipeline (walker, filters, dedup, splitter, three formatters) |
| `train` | [`codescribe_train/train/README.md`](codescribe_train/train/README.md) | QLoRA SFT loop, eval, GGUF export |
| `harness` | [`codescribe_train/harness/README.md`](codescribe_train/harness/README.md) | Pluggable Harness ABC, ClawCodeHarness, NoOpHarness, Docker sandbox |
| `backends` | [`codescribe_train/backends/README.md`](codescribe_train/backends/README.md) | Pluggable Backend ABC, llama-server / vllm / ollama / remote, hardware probe |

### Design rationale

The **why** behind the design — including alternatives considered and rejected — lives in [`docs/CONCEPTS.md`](docs/CONCEPTS.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), with the detail docs linked from there.

---

## Quick start

### One-time machine setup

**Training machine** (the one that trains and serves day-to-day):

```bash
# 1. CUDA toolkit 12.8 (NVIDIA's WSL repo)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-12-8

# 2. Add CUDA to PATH (idempotent script, edits ~/.bashrc)
bash scripts/setup_wsl.sh && source ~/.bashrc

# 3. Python deps
uv sync --extra data --extra train --extra dev

# 4. Build vendored claw-code (Rust harness)
bash scripts/build_claw_code.sh

# 5. Build vendored llama.cpp with sm_120 / Blackwell support
BUILD_JOBS=2 bash scripts/build_llama_cpp.sh   # ~50 minutes

# 6. (Optional but recommended) install Flash Attention 2
#    Requires a cross-build on a second machine — see docs/FLASH-ATTENTION.md
```

**Builder machine** (only needed for FA2 cross-build, optional inference role): see [`docs/FLASH-ATTENTION.md § 5`](docs/FLASH-ATTENTION.md#5-the-cross-build-solution) for the full setup.

### Phase 1 — build a dataset

```bash
python -m codescribe_train.data build \
    --repo /path/to/your/repo \
    --config configs/sample.yaml \
    --out datasets/sample/
```

Outputs `train.jsonl` / `val.jsonl` / `test.jsonl` plus `manifest.json`. ⇒ [`codescribe_train/data/README.md`](codescribe_train/data/README.md).

### Phase 2 — fine-tune

Verify the loop on your hardware first:

```bash
python -m codescribe_train.train smoke \
    --config configs/train/qwen7b_qlora.yaml \
    --dataset datasets/sample/ \
    --out checkpoints/sample-qwen7b-lora-smoke/
```

10 steps. Should print `train_loss=...` finite and decreasing. With FA2 + seq_len=1024: ~3.5 minutes wall-clock.

Then run the full training:

```bash
python -m codescribe_train.train run \
    --config configs/train/qwen7b_qlora.yaml \
    --dataset datasets/sample/ \
    --out checkpoints/sample-qwen7b-lora-v1/
```

ETA on the reference 8 GB-VRAM laptop GPU with FA2 + seq_len=1024: **~3.4 hours** for 3 epochs.

Then evaluate:

```bash
python -m codescribe_train.train eval \
    --adapter checkpoints/sample-qwen7b-lora-v1/ \
    --dataset datasets/sample/ \
    --tasks evals/sample_tasks.json \
    --report eval-report.json
```

Then export to a `Q4_K_M` GGUF for serving:

```bash
python -m codescribe_train.train export \
    --adapter checkpoints/sample-qwen7b-lora-v1/ \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
    --merged checkpoints/sample-qwen7b-merged-v1/ \
    --gguf checkpoints/sample-qwen7b-q4_k_m.gguf \
    --llama-cpp vendor/llama.cpp
```

⇒ [`codescribe_train/train/README.md`](codescribe_train/train/README.md), [`docs/QLoRA-AND-UNSLOTH.md`](docs/QLoRA-AND-UNSLOTH.md).

### Phase 3 + 4 — launch the agent

```bash
python -m codescribe_train.cli run \
    --repo /path/to/your/repo \
    --adapter checkpoints/sample-qwen7b-q4_k_m.gguf \
    --backend llama-server \
    --harness claw \
    --port 8080 \
    --model openai/sample-qwen7b-local
```

This:
1. Probes hardware, picks defaults.
2. Spawns `llama-server` from `vendor/llama.cpp/build/bin/` against your fine-tuned GGUF on `127.0.0.1:8080`.
3. Polls `/v1/models` until the server is ready.
4. Renders `.claude.json` / `.claw.json` config files into your repo.
5. Spawns `claw` from `vendor/claw-code/rust/target/release/` with `--model openai/sample-qwen7b` and the appropriate env vars.
6. Forwards stdin/stdout to the harness.
7. On exit, kills the server.

For the most paranoid posture, add `--sandbox docker` to wrap the harness in `--network none`. ⇒ [`codescribe_train/harness/README.md`](codescribe_train/harness/README.md), [`docs/STRICTLY-LOCAL-POSTURE.md`](docs/STRICTLY-LOCAL-POSTURE.md).

### Optional — RAG sidecar

To wire the v2 RAG sidecar (the `repo-rag` MCP server) into claw, the harness renders an MCP-server registry into claw's discovered config location (`.claw/settings.json`) so the `repo-rag` and `repo-grep` servers are registered.

Once that registry is in place, `claw mcp list` should show the `repo-rag` and `repo-grep` servers.

Build or refresh the underlying index with `python -m codescribe_train.rag index`.

---

## Hardware

The reference setup assumes two machines; both live on the same local network.

The specs below are one representative setup — treat them as an example class, not a hard requirement.

**Training machine** — primary, runs everything day-to-day:
- a recent consumer GPU with **~8 GB VRAM** (e.g. Blackwell-class, `sm_120`), CUDA toolkit 12.8
- a modern multi-core x86 CPU, DDR5
- WSL2 Ubuntu 24.04, with visible RAM deliberately kept tight (~8 GiB)

**Builder machine** — cross-builds the FA2 wheel; optionally serves 14B-class models over the local network:
- a Pascal-class GPU with ~11 GB VRAM (`sm_61`)
- a multi-core desktop CPU, DDR4
- Ubuntu 24.04 Server, ~32 GiB DDR4

The detailed split, performance numbers, and the seq_len decision: [`docs/HARDWARE-AND-PERFORMANCE.md`](docs/HARDWARE-AND-PERFORMANCE.md).

---

## Why local-only

Intended targets are private repositories under contract or NDA. Code, datasets, weights, and telemetry never leave the machine. The constraint isn't "it'd be preferable not to send data out" — it's "data going out is a *failure*."

Concrete consequences:
- No cloud GPU. Ever. Not for training, not for compilation, not as an "escape hatch."
- No remote inference endpoints in default configs. `RemoteBackend` exists in code but refuses to start without an explicit env-var opt-in.
- No telemetry from any component, including vendored components — the audit is in [`vendor/SECURITY-NOTES.md`](vendor/SECURITY-NOTES.md).
- Loopback by default. Optional Docker `--network none` sandbox for the agent harness.
- Empirical egress audit verifies the policy holds — most recent run: zero non-loopback `connect()` calls.

⇒ [`docs/STRICTLY-LOCAL-POSTURE.md`](docs/STRICTLY-LOCAL-POSTURE.md) for the full threat model and defence layers.

---

## Tests

```bash
.venv/bin/pytest tests/                  # whole suite
.venv/bin/pytest tests/data/             # phase 1 — 45 tests
.venv/bin/pytest tests/train/            # phase 2 — 5 torch-free tests
.venv/bin/pytest tests/harness/          # phase 3 — 34 tests
.venv/bin/pytest tests/backends/         # phase 4 — 40+ tests
```

**Total: 159+ tests, all passing on `main`.**

`tests/test_top_cli.py` and `tests/test_lazy_imports_top.py` exercise the orchestrator end-to-end with mocked `Backend` and `Harness`. The `smoke` subcommand of `codescribe_train.train` is the actual hardware verification — pytest doesn't try to mock `torch` or `unsloth`.

---

## Performance summary

| Configuration | Per-step wall clock | 3-epoch full training ETA |
|---|---|---|
| Xformers fallback @ seq_len=2048 | 85.8 s | ~14 h |
| FA2 @ seq_len=2048 | 34–300 s (variable, swap thrash) | unstable |
| **FA2 @ seq_len=1024 (current default)** | **20.9 s** | **~3.4 h** |

⇒ [`docs/HARDWARE-AND-PERFORMANCE.md § 3`](docs/HARDWARE-AND-PERFORMANCE.md#3-the-seq_len-20481024-decision) for the full measurements + reasoning.

---

## License

Private. Not licensed for redistribution.

Vendored components retain their original licenses (see `vendor/claw-code/LICENSE`, `vendor/llama.cpp/LICENSE`).
