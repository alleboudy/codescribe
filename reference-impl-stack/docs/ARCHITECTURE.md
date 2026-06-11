# Architecture

How `codescribe-train` is composed, why it's composed that way, and what's deliberately out of scope.

> **New to the project?** Start with [`CONCEPTS.md`](CONCEPTS.md) for the 30,000-foot view of *what* fine-tuning + RAG + MCP each contribute and *why* the project stacks them; this doc is the structural answer to "and how do the modules fit together?".
>
> **Single source of truth for *design rationale*** — including alternatives considered and rejected — is the project's design plan. This doc summarises *what is*; the plan documents *why this and not that*.

---

## 1. The four-phase composition

```
                                                                    ┌─────────────────┐
                          repo-specific YAML config ───┐            │ sample/         │
                                                       ▼            │ other-repo/     │ (any git repo)
                       ┌──────────────────────┐  ┌──────────┐       │ ...             │
                       │  data/  (Python)     │──│ datasets │◀──────│  walks repo    │
                       │  collect→clean→split │  │ /JSONL   │       └─────────────────┘
                       └──────────────────────┘  └────┬─────┘
                                                      ▼
                       ┌──────────────────────┐  ┌──────────┐
                       │  train/ (Python)     │──│  LoRA    │
                       │  QLoRA + eval        │  │ adapter  │
                       └──────────────────────┘  └────┬─────┘
                                                      ▼
                                                 (export → GGUF)
                                                      │
                                                      ▼
                       ┌────────────────────────────────────────┐
                       │  backends/  (pluggable; OpenAI-compat) │
                       │  llama-server | vllm | ollama | remote │
                       └─────────────────┬──────────────────────┘
                                         │ HTTP (chat/completions)
                                         ▼
                       ┌────────────────────────────────────────┐
                       │  harness/   (pluggable, low-trust)     │
                       │  default: ultraworkers/claw-code (Rust)│
                       │  iface ready for native/C++20 harness  │
                       └────────────────────────────────────────┘
                                         │
                                         ▼
                                 user CLI session
```

Build order — locked by the original plan: **`data/` → `train/` → `harness/` → `backends/`**. The order is incremental: each phase produces an artefact the next phase needs, but each is independently usable. By the end of `train/` you have a fine-tuned model you can serve with `llama-server` from any OpenAI-compatible client; `harness/` and `backends/` add the integrated CLI experience on top.

The top-level glue is `codescribe_train/cli.py`, exposed as `codescribe-train run`. It probes hardware → starts the chosen backend → prepares the chosen harness against the backend's URL → wires stdin/stdout into a session → cleans up backend on exit.

---

## 2. Each phase in one paragraph

### `codescribe_train/data/` — generic repo → JSONL pipeline

Walks any git repo via `git ls-files -z`, applies a YAML-driven filter (extensions, gitignore-style path globs, size bounds), deduplicates by SHA-256 of file content, splits the survivors deterministically by file path (90/5/5 train/val/test, salt-keyed so the same file always lands in the same bucket), then emits text-only training samples through three orthogonal formatters: **document** (full file with `<|repo_name|>`/`<|file_sep|>` markers), **FIM** (Fill-in-the-Middle — random line-span masked and surrounded by `<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>`), and **diff_instr** (commit subject + body → ChatML user/assistant pair where the assistant message is the diff). The CLI command is `python -m codescribe_train.data build --repo X --config Y --out Z`. The output is `train.jsonl`, `val.jsonl`, `test.jsonl`, plus a `manifest.json` with provenance.

See: [`codescribe_train/data/README.md`](../codescribe_train/data/README.md) and [`docs/QLoRA-AND-UNSLOTH.md`](QLoRA-AND-UNSLOTH.md#why-three-formatters) for why we use three formats.

### `codescribe_train/train/` — QLoRA fine-tuning

Uses [Unsloth](https://github.com/unslothai/unsloth) to load Qwen 2.5 Coder 7B Instruct in 4-bit NF4 quantization (≈ 4.5 GB on GPU), attaches a **LoRA** adapter (rank 16, ~40 M trainable parameters out of 7.6 B), and fine-tunes via `trl.SFTTrainer` with `paged_adamw_8bit` + `bf16`. `report_to="none"` and a battery of telemetry-disabling env vars enforce the strictly-local posture. Three CLI subcommands: `smoke` (10-step verification), `run` (full training), `eval` (perplexity + handwritten task suite scoring), `export` (LoRA → merged fp16 → GGUF Q4_K_M).

See: [`codescribe_train/train/README.md`](../codescribe_train/train/README.md) and [`docs/QLoRA-AND-UNSLOTH.md`](QLoRA-AND-UNSLOTH.md) and [`docs/FLASH-ATTENTION.md`](FLASH-ATTENTION.md).

### `codescribe_train/harness/` — pluggable agent harness

Defines a `Harness` ABC with four methods (`prepare`, `start_session`, `health_check`, `sandbox_args`) and ships two implementations: `ClawCodeHarness` (default; vendored Rust binary from [`ultraworkers/claw-code`](https://github.com/ultraworkers/claw-code), pinned at SHA `357629d`, audited for telemetry in `vendor/SECURITY-NOTES.md`) and `NoOpHarness` (echo loop used as a swap-test that the abstraction holds). The Rust binary is launched as a subprocess with `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` env vars and `--model openai/<id>` on the CLI (the prefix avoids the upstream Qwen-routing trap documented in the audit). A Docker sandbox path (`Dockerfile.claw`) wraps the launch in `docker run --network none` for write-tool sessions.

See: [`codescribe_train/harness/README.md`](../codescribe_train/harness/README.md) and [`vendor/SECURITY-NOTES.md`](../vendor/SECURITY-NOTES.md).

### `codescribe_train/backends/` — pluggable inference backend

Defines a `Backend` ABC (`start` → endpoint URL, `stop`, `health_check`) and ships four implementations: **`LlamaServerBackend`** (default; spawns `llama-server` from vendored llama.cpp built for sm_120 — see [`docs/FLASH-ATTENTION.md`](FLASH-ATTENTION.md) for why we pin the build), **`VllmBackend`** (lazy-imported; not in any default config), **`OllamaBackend`** (assumes `ollama serve` is already running externally), **`RemoteBackend`** (interface stub only — `start()` raises `RemoteBackendDisabled` unless `CODESCRIBE_ENABLE_REMOTE_BACKEND=1`). A `probe.py` reads `nvidia-smi` (or `pynvml` if installed) and recommends a backend + key options based on free VRAM and model size.

See: [`codescribe_train/backends/README.md`](../codescribe_train/backends/README.md).

### Top-level `codescribe_train/cli.py` — `codescribe-train run`

The orchestrator. `run` subcommand: probe → backend.start (poll health for up to `--health-timeout-s`) → harness.prepare → harness.start_session (forwards stdin/stdout) → on exit, `try/finally` calls backend.stop. `probe` subcommand: just prints the hardware profile + recommendation. Designed so `import codescribe_train.cli` doesn't pull in `httpx` / `psutil` / `pynvml` / any concrete harness or backend — every heavy module is loaded lazily when actually needed.

---

## 3. Data flow at runtime

```
$ codescribe-train run --repo ../your-repo --adapter checkpoints/sample-qwen7b-q4_k_m.gguf \
                 --backend llama-server --harness claw

  1. probe_hardware() → HardwareProfile
                        (GPU name, VRAM, RAM, CPU threads)
  2. recommend_backend(profile, model_size_gib=4.6)
                        → BackendChoice
  3. LlamaServerBackend.start(model_id=<gguf path>, adapter=None, port=8080)
                        → spawns llama-server -m <gguf> --host 127.0.0.1
                        → polls /v1/models until 200 OK
                        → returns "http://127.0.0.1:8080"
  4. ClawCodeHarness.prepare(workdir=../your-repo,
                             backend_url="http://127.0.0.1:8080",
                             model="openai/sample-qwen7b-local",
                             harness_config=<yaml>)
                        → renders .claude.json + .claw.json into ../your-repo
                        → composes env (OPENAI_*, ANTHROPIC_BASE_URL pinned to local)
  5. ClawCodeHarness.start_session(stdin, stdout)
                        → subprocess: vendor/claw-code/rust/target/release/claw \
                                       --model openai/sample-qwen7b-local
                        → forwards parent stdin/stdout
                        → returns claw's exit code when the user quits
  6. finally: LlamaServerBackend.stop()
                        → SIGTERM, wait 10s, SIGKILL if still alive
```

Network surface during the session — verified by `strace -f -e connect`:
- httpx (orchestrator) → `127.0.0.1:8080/v1/models` (health check, 6 calls)
- claw → `127.0.0.1:8080/v1/chat/completions` (1 call per session turn)
- nothing else

(See [`docs/known-issues.md`](known-issues.md#egress-audit--passed-substitute-model-fine-tuned-model) for the audit method and rerun checklist.)

---

## 4. Trust posture

The default stack vendors **two third-party C++ / Rust components** — llama.cpp and claw-code — and treats them as low-trust subprocess black boxes. The defence-in-depth layers, top to bottom:

1. **Pinned-SHA submodules.** `vendor/claw-code` and `vendor/llama.cpp` are git submodules at exact commits. Auto-updates are impossible because we never run `git submodule update --remote`. Bumping the SHA is a deliberate, reviewed change.
2. **Audit on each SHA bump.** The audit method and the previous findings live in [`vendor/SECURITY-NOTES.md`](../vendor/SECURITY-NOTES.md), with a re-audit checklist for each component. Audit findings include any `https?://` literal in runtime paths, every `libcurl` / `httplib::Client` consumer, every env-var-reading code path that could open a side channel.
3. **Loopback-only by default.** `configs/backends/local-default.yaml` pins `host: 127.0.0.1`; `RemoteBackend` exists in code but is never wired into any shipped config (and refuses to start unless an explicit env var is set).
4. **Optional Docker sandbox** (`harness/claw.py`'s `--sandbox docker` mode) launches the harness in a container with `--network none` — a hard kernel-enforced "no egress" if the user wants to be paranoid.
5. **Empirical egress audit.** `strace -f -e connect` is run end-to-end and the result is committed (`docs/known-issues.md`). Zero non-loopback `connect()` calls is the expected output; any non-zero is a regression.

Out of scope: supply-chain attacks on PyPI / GitHub / Hugging Face during install. We pull `torch`, `unsloth`, `transformers`, etc. from PyPI without pinning hashes; we pull base model weights from the HF Hub. If you don't trust those origins, you'll need a separate offline mirror of those artefacts.

---

## 5. Why "pluggable" matters

The two ABCs (`Harness` and `Backend`) exist for one practical reason: **the user explicitly does not fully trust** `claw-code`, the vendored Rust harness. A thin abstraction means that if claw-code grows misbehaviour, gets unmaintained, or someone wants to write a lighter alternative, the rest of the project doesn't have to move.

Concretely:

- A future native C++20 harness (the project name's original ambition) would implement `Harness` and slot in. `data/`, `train/`, `backends/` don't change.
- The `NoOpHarness` swap test (`tests/harness/test_swap.py`) is the abstraction's correctness check — proves you can run end-to-end without ever touching claw-code.
- Backends are similarly pluggable; the most likely future addition is a `RemoteBackend` instance pointing at a second machine on your LAN (e.g. a Pascal-class GPU, 32 GB DDR4 / 11 GB VRAM) for serving 14 B-class models that don't fit on the training machine's 8 GB.

The plan treats these abstractions as load-bearing. Any change that adds direct dependencies on `claw` or `llama-server` outside their adapter modules is a regression.

---

## 6. What's not in here (deliberately)

Things that came up during planning and were rejected:

- **Cloud GPU training** (RunPod, Modal, Lambda) — would void the strictly-local posture.
- **A homegrown agent harness** for MVP — too much surface area; vendor `claw-code` instead.
- **Pre-quantized model downloads as the conversion contract** — we use llama.cpp's `convert_hf_to_gguf.py` so the export path is reproducible from a fine-tuned LoRA adapter, not from someone else's GGUF.
- **C++20 implementations of the harness or backends in MVP** — deferred. The `Harness` / `Backend` ABCs are designed so a native impl can replace the Python adapter without disturbing anything else.
- **Multi-host distributed training** (training-machine + builder-machine GPUs together) — mismatched architectures (Blackwell sm_120 + Pascal sm_61) plus cross-host NCCL is a debugging swamp for negligible payoff at this scale.

If you find yourself wanting one of these, read the design plan's section on it before starting — there's a documented reason it was rejected.

---

## 7. Pointers to deeper docs

| You want to understand … | Read |
|---|---|
| Why Flash Attention matters and how we built our own wheel | [`docs/FLASH-ATTENTION.md`](FLASH-ATTENTION.md) |
| What QLoRA is, why we use Unsloth, and what the hyperparameters mean | [`docs/QLoRA-AND-UNSLOTH.md`](QLoRA-AND-UNSLOTH.md) |
| Performance numbers, the seq_len 2048→1024 decision, swap-thrash forensics | [`docs/HARDWARE-AND-PERFORMANCE.md`](HARDWARE-AND-PERFORMANCE.md) |
| The strictly-local threat model + audits | [`docs/STRICTLY-LOCAL-POSTURE.md`](STRICTLY-LOCAL-POSTURE.md) |
| Project jargon (FIM, GGUF, NF4, sm_120, …) | [`docs/GLOSSARY.md`](GLOSSARY.md) |
| Known issues and their rerun commands | [`docs/known-issues.md`](known-issues.md) |
| Per-package conventions | `codescribe_train/<package>/README.md` |
| Decision history | the project's design plan |
