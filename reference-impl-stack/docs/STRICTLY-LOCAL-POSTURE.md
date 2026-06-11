# Strictly-local posture

What "strictly local" means in this project, why it's the central design constraint, what the threat model is, and how it's enforced in code, configuration, and verification.

---

## 1. The headline

This project is built for **fine-tuning and serving LLMs against private code repositories that cannot leave the user's machine.** The constraint is not "we'd prefer not to send data out" — it's "data going out is a *failure*". Every design decision in the repo flows from that.

Concretely:

- **No cloud GPU.** Ever. Not for training, not for compilation, not as an "escape hatch".
- **No remote inference endpoints in default config.** `RemoteBackend` exists in code, but it's an interface stub that refuses to start without an explicit env var. No shipped config wires it in.
- **No telemetry, analytics, error reporting, or auto-update** from any component — including the vendored Rust harness `claw-code` and the vendored `llama.cpp`. Both have been audited; findings are in [`vendor/SECURITY-NOTES.md`](../vendor/SECURITY-NOTES.md).
- **No upload paths** — no `huggingface-cli upload`, no Weights & Biases, no Sentry, no automatic push of any artefact. `WANDB_DISABLED=true`, `HF_HUB_DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1` are set defensively at training start.
- **Loopback by default.** `llama-server` binds to `127.0.0.1`. The optional Docker sandbox path uses `--network none` — kernel-enforced "no egress, period".
- **Empirical verification.** `strace -f -e connect` is run end-to-end and any non-loopback `connect()` is treated as a regression. Most recent audit: zero non-loopback connect()s; see [`docs/known-issues.md`](known-issues.md#egress-audit--passed-substitute-model-fine-tuned-model).

---

## 2. Threat model

The user is fine-tuning agents on **proprietary code** that's confidential under contract or NDA. The threat model isn't "a sophisticated APT compromises the machine" — it's the much more mundane:

| Threat | Mitigation |
|---|---|
| A vendored binary phones home on first run | Vendor at pinned SHA; audit before bumping; firewall in Docker sandbox |
| A Python library has telemetry on by default | Set `DO_NOT_TRACK`, `HF_HUB_DISABLE_TELEMETRY`, `WANDB_DISABLED`, `WANDB_MODE=disabled` |
| A model upload happens "by accident" via misconfigured HF integration | `report_to="none"` in `SFTConfig`; never call `model.push_to_hub`; never call `huggingface-cli upload` |
| A future config change leaks something | The egress audit (`strace -f`) is reproducible and committed; it acts as a regression guard |
| A vendored binary's tools (file read, shell exec) accidentally read sensitive paths | Docker sandbox mode mounts target repo only, with `--network none` so even if it tries to upload, it can't |
| Someone clones the repo and runs default training, expecting privacy by default | The defaults are correct — the "wrong" config has to be done explicitly via env var |

The model we don't try to defend against:

- Compromised PyPI / HF Hub / GitHub during package installation. We pull `torch`, `transformers`, etc. from PyPI without hash-pinning. If you don't trust those origins, you need an offline mirror.
- A determined operator on the user's own machine manually uploading the weights or dataset. We can't prevent intentional exfiltration; we can prevent *accidental* exfiltration.
- Bugs in the kernel that bypass `--network none`. We trust the Linux network namespace primitive.

---

## 3. The vendored components and their audit

Two third-party native binaries run on the training machine during a typical session:

### `vendor/claw-code` (Rust agent harness)

Pinned at SHA `357629dbd9b300cbea7f484b6df263a862e20a84`. Fallback remote: `ultraworkers/claw-code-parity` (the canonical `ultraworkers/claw-code` was temporarily locked for ownership transfer in early 2026; the parity remote is its mirror).

Audit findings ([`vendor/SECURITY-NOTES.md`](../vendor/SECURITY-NOTES.md)):

| Severity | Finding | Mitigation |
|---|---|---|
| low | First-party `telemetry` crate exists — local sinks only (memory + JSONL) | Accept |
| medium | Default `https://api.anthropic.com` for Anthropic provider | `ANTHROPIC_BASE_URL` env var pinned by our adapter to the local backend URL |
| medium | Default OpenAI/xAI/DashScope endpoints | `OPENAI_BASE_URL` env var pinned by our adapter |
| medium | **Routing trap**: bare `qwen-*` model names route to DashScope by prefix match | Our default config uses `model: openai/<id>` so the OpenAI-compatible client wins |
| low | Honors `HTTP_PROXY` / `HTTPS_PROXY` env vars (standard `reqwest` behaviour) | Accept — not a covert channel |
| low | `plugins.autoUpdate` config field exists in the validator | We never enable plugins; field is unused |

No telemetry phone-home. No auto-update at runtime. No covert side channels found.

### `vendor/llama.cpp` (inference server)

Pinned at SHA `58e68df0f91dd16ff56423ee5ef44062ed73bdfc`. The `llama-server` binary is the only artefact launched at runtime; `llama-quantize` is build-time only.

Audit findings:

| Severity | Finding | Mitigation |
|---|---|---|
| **medium (opt-in)** | `--hf-repo` / `--hf-file` / `--model-url` flags fetch from HuggingFace via libcurl | Our wrapper always passes `-m <local-path>`, never these flags |
| low | `common/download.cpp` reads `HF_TOKEN` env var when fetching | Wrapper never sets `HF_TOKEN`; user's shell may, but it's only consumed by the opt-in fetch path |
| none | Server defaults to `--host 127.0.0.1 --port 8080` | Pinned in our default config |
| none | `tools/server/webui/package-lock.json` lists `@opentelemetry/api` as a peer-optional dep | Webui ships as static assets — not loaded at runtime |
| none | Storybook fixture files reference `analytics.example.com` | Test fixtures, not runtime code |
| low | Build-time fetches in `Makefile` / CI scripts | Our build path is `cmake --build` only — no `git pull`, no CI script invocation |

No telemetry phone-home in the runtime path. The HF fetcher is opt-in; we don't opt in.

### Re-audit on SHA bump

Both `vendor/SECURITY-NOTES.md` sections include a **rerun checklist** to apply the same audit when bumping a SHA. The minimum:

```bash
cd vendor/<component>
rg -i 'telemetry|analytics|sentry|posthog|amplitude|mixpanel|phone.?home'
rg 'https?://' --type rust --type cpp --type c | grep -vE '(github|huggingface|wikipedia|jina\.ai|prometheus\.io|arxiv|opensource)'
rg -i 'auto.?update|self.?update|check.?update'
```

Any new finding gets a row in the table with mitigation, before the SHA bump merges.

---

## 4. The Python stack

The Python side has fewer covert channels than C++/Rust binaries — most ML libraries either don't phone home or are easy to disable. We set the kill switches anyway, defence in depth.

`codescribe_train/train/sft.py:_enforce_local_only_env()` runs at training start:

```python
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "0")
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "0")
os.environ["DO_NOT_TRACK"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
```

The `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` envs are deliberately *opt-in* — they're set to `0` by default because we still need to download the base model the first time. After the first run, the user can set them to `1` in their shell to fully bar HF Hub access.

`SFTConfig(report_to="none", ...)` in the same file ensures TRL never tries to log to W&B / TensorBoard / MLflow / etc. even if the libraries are present.

### What we *do* talk to over the network

| Network endpoint | When | Why |
|---|---|---|
| PyPI (`pypi.org` or mirror) | Once at `uv sync` time | Install Python packages |
| `download.pytorch.org/whl/cu128` | Once at `uv sync` time | Install PyTorch CUDA wheels |
| HuggingFace Hub | Once on first training run | Pull base model weights — `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` |
| `developer.download.nvidia.com` | Once during machine setup | Install CUDA toolkit |
| `astral.sh/uv/install.sh` | Once during machine setup | Install uv |
| **(during a training run)** | **never** | — |
| **(during an `codescribe-train run` session)** | **never** | — |

Steady-state, the project is fully offline. The setup-time downloads are bounded, well-known, and one-time.

---

## 5. The Docker sandbox

For the most paranoid posture, `harness/claw.py` supports `--sandbox docker`. This launches the Rust harness inside a container with:

```
docker run --rm -i \
    -v <workdir>:/work -w /work \
    --network none \
    claw-code:vendored
```

`--network none` is the kernel network namespace primitive: the container gets *no network interface at all* (no loopback, no eth0, nothing). Any `connect()` syscall returns `ENETUNREACH`. The harness can still read/write to the mounted `/work` (the target repo) but cannot exfiltrate anything.

The sandbox image is built once from `Dockerfile.claw` (multi-stage, `rust:1.83-slim` builder + `debian:bookworm-slim` runtime, non-root `claw` user). The image contains only the vendored claw-code binary and its runtime deps; no shell, no curl, no Python.

Caveat: if the user wants the agent to actually *fetch* anything (e.g., a docs page), `--network none` makes that impossible, and the agent will fail those operations. For most coding tasks against a known repo, this is fine. For tasks where the agent legitimately needs the internet, run without `--sandbox` and accept the relaxed posture for that session.

---

## 6. Verification — the egress audit

Trust but verify. The strictly-local posture has been measured end-to-end via `strace -f -e trace=connect`:

```bash
strace -f -e trace=connect -o /tmp/codescribe-train-egress.strace \
    .venv/bin/python -m codescribe_train.cli run \
        --repo $HOME/repos/sample \
        --adapter checkpoints/test-base/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
        --backend llama-server \
        --harness claw \
        --port 8082 \
        --model openai/qwen-test
```

The trace covers:
- The Python orchestrator (`codescribe_train.cli`)
- `llama-server` (subprocess of the orchestrator)
- `claw` (subprocess of the orchestrator)
- All children of those

Result on the most recent run (full results in [`docs/known-issues.md`](known-issues.md#egress-audit--passed-substitute-model-fine-tuned-model)):

- 116 total `connect()` syscalls
- **7 AF_INET connect()s, all to `127.0.0.1:8082`**:
  - 6 from the Python orchestrator (httpx health-check probes during backend startup)
  - 1 from the `claw` subprocess (single OpenAI-compat chat completion to the local backend)
- 0 AF_INET6 connect()s
- 109 AF_UNIX (local IPC sockets) and AF_NETLINK (kernel queries) — local-only by definition

**Zero non-loopback connect() calls. Zero DNS lookups against any non-loopback target.**

This audit is reproducible and should be re-run after:
- Bumping `vendor/claw-code` SHA
- Bumping `vendor/llama.cpp` SHA
- Adding any new backend default config
- Changing `LlamaServerBackend._build_argv` or any harness adapter

The rerun method is documented in `docs/known-issues.md`. If a future audit shows a non-loopback `connect()`, that's a regression — investigate before merging.

---

## 7. Practical hardening for the operator

If you want to take the posture from "strictly local in normal operation" to "strictly local even under operator error", you can add:

### Firewall the training machine

`ufw default deny outgoing; ufw allow out to 192.168.1.0/24` — blocks all WAN traffic but leaves LAN reachable. The egress-blocking is kernel-level; even a misbehaving binary couldn't escape it.

### Detach from the internet during training

Disable Wi-Fi or unplug Ethernet for the duration of a training session. The base model is already cached locally; nothing in the training path needs network access. This is the simplest possible mitigation.

### Run the harness in `--sandbox docker --network none`

For sessions where the agent might run shell commands you'd rather it not (read random files, run `curl`, etc.), the Docker sandbox is the kernel-enforced fallback. Documented in section 5 above.

### Periodic egress audits

Schedule the `strace` audit above to run weekly or after dependency updates. A short script could `diff` the destinations against an allowlist (`127.0.0.1` only) and fail loudly on regression.

---

## 8. Things that would break the posture (don't do these)

- **Setting `CODESCRIBE_ENABLE_REMOTE_BACKEND=1`** unless you've thought about which "remote" you're talking to. The flag exists for the topology where a second machine on your LAN serves the model; it's still local in the geopolitical sense, but loopback is no longer the only target.
- **Running `--sandbox none` with `--repo` pointing at a path you don't fully trust the harness to read** (e.g., `~/`). The default sandbox is `none`. Raise the sandbox if you're unsure.
- **Calling `model.push_to_hub` from a notebook**. The infrastructure here can't prevent it; just don't.
- **Checking `unsloth_compiled_cache/` or `wandb/` directories into git**. Both are gitignored already, but `git add -A` after a rebase could re-add them. Verify before committing.
- **Editing `report_to` to anything other than `"none"`**. Code review should reject any such change.
- **Using `huggingface-cli login`** if you don't intend to upload. The login persists a token in `~/.cache/huggingface/token` that some libraries auto-discover.

---

## 9. The escape valves (when you actually need network)

There are legitimate operations that need the internet:

- Downloading a new base model the first time → run from a shell, then run training
- Bumping a vendored SHA → done at submodule update time, not at runtime
- Pulling a CUDA toolkit update → builder-machine and training-machine setup
- Fetching a public dataset for benchmarking → out-of-band, before the training session

The pattern: **do all your downloading first, then take the machine offline, then train**. The training itself never needs to talk to the world.
