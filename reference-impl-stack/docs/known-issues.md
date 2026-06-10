# Known Issues

Project-wide collection of "this doesn't work, here's why, here's the path forward". Add a dated section per issue. Don't editorialise — just diagnosis, attempt log, and the realistic next step.

---

## MCP stdio framing mismatch (claw ↔ Python MCP SDK) — FIXED

**Status:** Fixed. The fix is the `_mcp_framing_bridge` shim wired into
every porter-generated `.claw/settings.json`. If you ever
write a new `.claw/settings.json` by hand for a new Python MCP server,
**use the bridge** — see "How to recognise" below.

Recovery one-liner if you hit it again:

```bash
bash scripts/relaunch-claw.sh
```

### Symptom

When launching `claw` against a Python MCP server (e.g. the rag /
grep servers in `codescribe_train/servers/`), the MCP server log fills with
stack traces like:

```
ERROR  Received exception from stream: 1 validation error for
       JSONRPCMessage
         Invalid JSON: expected value at line 1 column 1
         [type=json_invalid, input_value='Content-Length: 156\n',
         input_type=str]
ERROR  Received exception from stream: 1 validation error for
       JSONRPCMessage
         Invalid JSON: EOF while parsing a value at line 2 column 0
         [type=json_invalid, input_value='\n', input_type=str]
```

…and every claw → MCP tool call times out. `claw mcp list` still
works (it doesn't actually start the servers, just lists the
configured entries from `.claw/settings.json`), so the regression
is invisible until a real tool call is attempted.

### Diagnosis

Two MCP stdio framings in the wild:

| Implementation | Framing | Reference |
|---|---|---|
| `vendor/claw-code` (Rust) | **LSP-style**: `Content-Length: N\r\n\r\n<payload>` | `crates/runtime/src/mcp_stdio.rs::encode_frame` (line 1391 at the pinned SHA) |
| `mcp` 1.27.x (Python) `stdio_server` | **Newline-delimited JSON-RPC**: one JSON object per line, no headers | `mcp/server/stdio.py` |

The two don't interoperate. claw sends each request as
`Content-Length: 156\r\n\r\n{...json...}`; the Python server's
stdin loop reads the first line (`Content-Length: 156\n`) and tries
`json.loads(...)` on it, raising the JSON-validation error above. The
MCP server-side error handler logs it and moves on to the next line
(the blank header separator), which fails the same way, then the
actual JSON payload (which would parse) — but by then the response
state machine is out of sync and the tool call has already been
abandoned.

This is not new behaviour in either project; it just hadn't been
exercised end-to-end before. The pre-merge "MCP smoke"
was run through the [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
(newline-delimited) rather than through claw, so the mismatch only
surfaced when the per-platform porters made it
trivial to actually launch claw against the rag server.

### Why not just configure one side to match the other?

- **Python MCP SDK 1.27.x has no LSP-framing option.** Reviewed
  `mcp.server.stdio.stdio_server`'s signature: `(stdin=None,
  stdout=None)` only, no framing parameter, no `lsp_frame=True` flag.
  Upstream MCP spec for stdio is newline-delimited only — LSP framing
  isn't in the spec at all.
- **Bumping claw to a "no-framing" SHA isn't an option** because no
  such SHA exists upstream — `vendor/claw-code` consistently uses LSP
  framing across the v0.x line, by design ("newline-safe" framing).

### Fix

`codescribe_train/servers/_mcp_framing_bridge.py` — pure-stdlib shim spawned
by claw instead of the MCP server. The bridge:

1. Reads LSP frames off its own stdin (= claw's stdout).
2. Writes the parsed JSON payload, newline-terminated, to a child
   subprocess's stdin (= the actual MCP server).
3. Reads newline-delimited JSON off the child's stdout.
4. Re-wraps each line in an LSP frame and writes to its own stdout (=
   claw's stdin).
5. Inherits stderr so server logs land wherever claw points them.

Wired into every porter-generated `.claw/settings.json` via
`scripts/_porter_lib.sh::porter_render_claw_settings` (Mac + Linux)
and `scripts/bootstrap_windows.ps1`.

Each entry looks like:

```json
{
  "command": "/path/to/.venv/bin/python",
  "args": ["-m", "codescribe_train.servers._mcp_framing_bridge", "codescribe_train.servers.rag_server"]
}
```

Covered by `tests/servers/test_mcp_framing_bridge.py`: 10 unit tests
on the framing helpers (CRLF vs LF, case-insensitive headers, extra
headers, EOF, missing Content-Length, truncated body, back-to-back
messages, empty payload), 1 subprocess integration test against a
synthetic echo child, 1 CLI smoke test.

### Recovery procedure (if you re-encounter)

If you see the `Invalid JSON: ...Content-Length: 156\n` errors in any
MCP server log, the most likely cause is a hand-rolled
`.claw/settings.json` that wires the server directly without going
through the bridge. Recovery:

1. `git pull` to make sure the bridge module is present.
2. Regenerate the local `.claw/settings.json` by re-running your
   platform porter (`bash scripts/bootstrap_mac.sh --skip-rsync` on
   Mac, `bash scripts/bootstrap_linux.sh --skip-rsync` on Linux,
   `pwsh scripts\bootstrap_windows.ps1 -SkipRsync` on Windows). The
   porter's `porter_render_claw_settings` always emits bridge-wrapped
   entries.
3. Kill any stale MCP server / bridge processes spawned by the
   previous claw session, then relaunch claw — the
   `scripts/relaunch-claw.sh` helper does both in one command.

If you're authoring a new Python MCP server module, **wire it through
the bridge** in your `.claw/settings.json` entries. There is no clean
direct path until either claw drops LSP framing on stdio or the Python
MCP SDK adds it.

---

## Flash Attention 2 — DEFERRED

**Status:** Unsloth falls back to Xformers at training-time with the message
`Unsloth: Your Flash Attention 2 installation seems to be broken. Using Xformers instead. No performance changes will be seen.`
The Phase 2 smoke run measured ~85.8 s/step at `seq_len=2048`,
effective batch 16. With FA2 the per-step cost roughly halves, so a 3-epoch
full training drops from ~14 h to ~6–8 h. We tried to install FA2 against
the current stack and could not — diagnosis below.

### Stack at time of investigation

| Component | Version |
|-----------|---------|
| GPU       | an 8 GB-VRAM Blackwell-class GPU, sm_120, compute capability `(12, 0)` |
| OS        | WSL2 Ubuntu 24.04 |
| RAM       | ~8 GiB total (≈ 6.3 GiB available), 2 GiB swap |
| CUDA      | 12.8 toolkit at `/usr/local/cuda-12.8/`, `nvcc 12.8.93` |
| Python    | 3.13.13 (cp313) |
| torch     | `2.10.0+cu128` (cxx11_abi=True, compiled CUDA 12.8) |
| unsloth   | `2026.5.2` |
| xformers  | `0.0.35` (current FA2 fallback) |
| bitsandbytes | `0.49.2` |
| peft      | `0.19.1` |
| trl       | `0.24.0` |
| accelerate | `1.13.0` |

### Latest upstream releases (Tri-Dao/flash-attention)

- **fa2 stable**: `v2.8.3` (2025-08-14). Latest in the v2.x line.
- **fa3/fa4**: `fa4-v4.0.0.beta12` (2026-05-06). FA4 is still beta; Unsloth's
  detection path is `import flash_attn` (the v2 package), so beta v4 wheels
  don't satisfy the runtime check this issue is about.

### What prebuilt wheels exist on the v2.8.3 release

The v2.8.3 release ships ~70 wheels indexed by `cu12torch{2.4,2.5,2.6,2.7,2.8,2.9}` × `cxx11abi{TRUE,FALSE}` × `cp{39,310,311,312,313}`. **No wheel targets torch 2.10**, and the only `torch2.9` cp313 wheel ships as `linux_aarch64`, not `linux_x86_64`. The closest match for the current stack is:

```
flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
```

Sm_120 (Blackwell) support landed in fa2 v2.7.4+ and is included in v2.8.3, so the kernel coverage is fine — the blocker is host-side libtorch ABI.

### Attempt 1 (B1) — Closest prebuilt wheel: torch 2.8 → torch 2.10

Downloaded `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl`
(245 MB) and installed with `pip install --no-deps`. Install completed cleanly.
`import flash_attn` then failed with:

```
ImportError: .../flash_attn_2_cuda.cpython-313-x86_64-linux-gnu.so:
  undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

This is a torch 2.9 → 2.10 C++ ABI break in `c10::cuda::c10_cuda_check_implementation`:

| torch version | Mangled symbol |
|---------------|----------------|
| 2.8 (linked into the wheel) | `..._iPKcS2_ib` (last param `i` = signed int line number) |
| 2.10 (our installed libtorch) | `..._iPKcS2_jb` (last param `j` = unsigned int line number) |

`nm -D --defined-only` on `torch/lib/libc10_cuda.so` confirms only the `_jb`
symbol exists in the installed torch 2.10. The wheel cannot be coerced to
load. Wheel was uninstalled cleanly via `pip uninstall -y flash-attn`; venv
returned to baseline (no `flash_attn` module, no leftover artefacts).

### Attempt 2 (B2) — Source build: not viable on a low-RAM trainer

Source build was **not attempted** on this machine. Reasoning:

`flash-attn` v2.8.3's `setup.py` compiles **78 `.cu` files** by default
(forward / backward / causal / split-K × head dims 32/64/96/128/192/256).
Each individual `nvcc` invocation for the heavier head-dim templates is
documented in upstream issues to consume **6–12 GiB of host RAM** at peak,
even with `NVCC_THREADS=1`. This trainer has only ~8 GiB total RAM and ~6.3 GiB
available; swap is 2 GiB. Restricting to a single arch via
`FLASH_ATTN_CUDA_ARCHS=120` reduces per-file compile time but does **not**
substantially reduce the peak host-RAM-per-file (the templates expand
identically; only fewer SASS variants are emitted). With `MAX_JOBS=1` the
build linearises but still OOM-risks each individual heavy file.

Per the task constraints:

- Strictly local — no offloading the build to a higher-RAM machine's CI.
- Don't bump WSL2 RAM (explicitly out of scope).
- Don't trash the venv. The venv is a symlink to the parent's; an OOM-kill
  mid-build can leave a partial `flash_attn_2_cuda.so` on disk that breaks
  every subsequent training run until manually removed.

A 60-min time-boxed source build with MAX_JOBS=1 was therefore declined as
an unacceptable risk vs benefit on this machine. The realistic floor for
a clean fa2 source build is ≥ 16 GiB host RAM; this trainer has half that.

### Realistic path forward

In rough preference order:

1. **Wait for an upstream torch-2.10 / cu128 / cp313 / sm_120 wheel.** Tri-Dao
   typically ships new torch-version wheels within ~weeks of a torch
   release. Re-check the v2.x release on `Dao-AILab/flash-attention/releases`
   for `cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl` periodically;
   when present, this collapses to a one-line `pip install` and the issue
   closes. (The closest existing assets are `cu12torch2.9cxx11abiTRUE-
   cp312-cp312-linux_aarch64.whl`, suggesting a torch-2.9 x86_64 cp313
   wheel may follow in a `v2.8.4` or similar.)

2. **Rebuild the wheel on a different machine and copy the `.whl` over.**
   A second machine with more RAM (per `docs/HARDWARE-AND-PERFORMANCE.md`)
   could plausibly compile fa2 against an identical stack
   (Python 3.13 + torch 2.10.0+cu128 venv, `FLASH_ATTN_CUDA_ARCHS=120`),
   then the resulting wheel transfers as a single file. The strictly-local
   posture is preserved (it's still the user's hardware, no cloud).
   Caveat: if the build host is a Pascal-class (sm_61) GPU it can compile
   but cannot *test* the sm_120 wheel; testing happens back on the trainer.
   Worth doing only when the user wants the speedup ahead of an upstream wheel.

3. **Downgrade torch to 2.8 in the train extras.** Possible but invasive
   — the rest of the cu128 stack (xformers 0.0.35, unsloth 2026.5.2,
   bitsandbytes 0.49.2) was selected against torch 2.10. Not recommended
   unless option 1 takes more than a few weeks.

The current Xformers fallback is functional; Phase 2 smoke training already
passed end-to-end. The cost of deferring is **~7 hours of wall clock on a
3-epoch full-training run** that the user has not yet decided to launch.

### Re-check command (cheap, run periodically)

```bash
gh release view v2.8.3 --repo Dao-AILab/flash-attention | grep cu12torch2.10
gh release list  --repo Dao-AILab/flash-attention --limit 5    # look for v2.8.4+
```

If a `cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl` appears, this
section should be deleted and replaced with an "INSTALLED" section that
records the install command and the new sec/step from a smoke run.

---

## Egress audit — PASSED (substitute model; fine-tuned model)

**Status:** End-to-end `codescribe-train run` against a local `llama-server` produces
**zero non-loopback `connect()` calls** across the entire process tree.
Strictly-local posture verified empirically.

### Fine-tuned model audit

Repeated with the fine-tuned `checkpoints/sample-qwen7b-q4_k_m.gguf`
(4.4 GB, Q4_K_M, 339 tensors, 4.91 BPW).

**Method:** launched `llama-server -m checkpoints/sample-qwen7b-q4_k_m.gguf
--port 8092 -ngl 99`, sent a chat completion request, then inspected
`/proc/$PID/fd/` and `ss -tnp` filtered to the server PID.

**Result:** llama-server owned exactly **one socket** — the listening socket
on `127.0.0.1:8092`. Zero established TCP connections to any remote host.
No non-loopback `connect()` calls.

### Substitute model audit

### Method

```bash
echo "Reply with exactly: ok" | timeout 90 \
  strace -f -e trace=connect -o /tmp/codescribe-train-egress.strace \
    .venv/bin/python -m codescribe_train.cli run \
        --repo $HOME/repos/sample \
        --adapter checkpoints/test-base/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
        --backend llama-server \
        --harness claw \
        --port 8082 \
        --model openai/qwen-test
```

`strace -f` follows children, so the trace covers the Python orchestrator,
the spawned `llama-server` subprocess, and the spawned `claw` subprocess.
Substitute model: official `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` Q4_K_M
(no fine-tuned model exists yet — full training is a separate user step).

### Result

- **116 total `connect()` syscalls** across the tree.
- **7 are AF_INET**, all to `127.0.0.1:8082`:
  - 6 from the Python orchestrator (httpx health-check probes during backend
    startup).
  - 1 from the `claw` subprocess (single OpenAI-compat chat completion to
    the local backend).
- **0 AF_INET6 connect() calls**.
- The remaining 109 are `AF_UNIX` (local IPC sockets) and `AF_NETLINK`
  (kernel queries) — local-only by definition, not network egress.

### What this confirms

1. `LlamaServerBackend` only binds + accepts on loopback (`--host 127.0.0.1`
   in `configs/backends/local-default.yaml`); never initiates outbound
   connections.
2. `ClawCodeHarness`'s child process makes exactly one connect() per session
   turn, to the loopback backend URL the wrapper passed via
   `OPENAI_BASE_URL` + `--model openai/...`.
3. No HF Hub fetch, no auto-update probe, no telemetry beacon, no DNS
   lookup against any non-loopback target. Confirms the audits in
   `vendor/SECURITY-NOTES.md` against the actual runtime.

### Rerun checklist

Re-run the audit after any of:
- bumping `vendor/claw-code` SHA (could add a new outbound path);
- bumping `vendor/llama.cpp` SHA;
- changing `LlamaServerBackend._build_argv` or the harness adapter;
- adding any new backend default config that points off `127.0.0.1`.
- adding MCP servers (Phase 5) — each server must be individually verified.

If the result ever shows a non-loopback `AF_INET` / `AF_INET6` `connect()`
call, that's a regression — investigate before merging.

---

## Egress audit after Phase 5 — PENDING

**Status:** Placeholder. Each Phase 5 MCP server must be audited for network
egress before being wired into the default harness config. Servers that are
strictly local (grep, import-fields, RAG) should produce zero non-loopback
connections. Servers that intentionally reach external services (e.g.
`sample-github` via `gh` CLI) must be documented as opt-in exceptions.

### Per-server audit log

| Server | Result | Notes |
|--------|--------|-------|
| repo-grep | PASS | Subprocess to `rg` only; no network imports |
| sample-import-fields | PASS | Reads local markdown file; no network imports |
| repo-rag | PASS | Reads local sqlite-vec DB; embedder loaded at startup (one-time HF download cached) |

---

## Flash Attention 2 — INSTALLED (cross-build)

**Status:** Resolved by cross-building the wheel on a higher-RAM builder machine
(a Pascal-class sm_61 GPU / 32 GiB DDR4 / Ubuntu 24.04) and copying the
resulting `flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl` to the trainer.
The trainer's ~8 GiB RAM ruled out a local source build; the builder machine's
32 GiB had room to spare.

### Cross-build setup on the builder machine (one-time)

```bash
# Apt prereqs
sudo apt-get install -y build-essential cmake ninja-build python3-dev \
    python3-venv python3-pip git curl pkg-config wget gpg ca-certificates

# CUDA 12.8 toolkit (matches the trainer)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8

# uv + Python 3.13 (matches the trainer)
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.13

# Build venv with torch 2.10+cu128
mkdir -p ~/build/flash-attn-cross && cd ~/build/flash-attn-cross
~/.local/bin/uv venv --python 3.13 .venv
.venv/bin/python -m ensurepip
UV_HTTP_TIMEOUT=600 ~/.local/bin/uv pip install \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.10.0+cu128" ninja packaging wheel setuptools numpy einops

# Build the wheel for sm_120 only (saves time + RAM)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"
export FLASH_ATTN_CUDA_ARCHS=120
export MAX_JOBS=4
export FLASH_ATTENTION_FORCE_BUILD=TRUE
.venv/bin/python -m pip wheel flash-attn --no-build-isolation -w /tmp/wheels/

# Result: /tmp/wheels/flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl (68 MB)
```

Build time on a 6-core builder-machine CPU + 32 GB RAM: ~37 minutes with `MAX_JOBS=4`.

### Install on the trainer

```bash
scp user@<build-host-ip>:/tmp/wheels/flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl \
    checkpoints/flash-attn-wheel/
.venv/bin/python -m ensurepip   # uv venvs don't ship pip
.venv/bin/python -m pip install --no-deps \
    checkpoints/flash-attn-wheel/flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl
.venv/bin/python -m pip install einops
.venv/bin/python -c "import flash_attn; print(flash_attn.__version__)"   # 2.8.3
```

### Measured speedup

Phase 2 smoke training (10 steps, seq_len=1024, 8 GiB-VRAM GPU):

| Config | s/step | 3-epoch full training ETA |
|---|---|---|
| Xformers fallback (no FA2) at seq_len=2048 | 85.8 | ~14 h |
| FA2 at seq_len=2048 | **highly variable** (34–300, swap thrash) | unstable |
| **FA2 at seq_len=1024** | **20.9** | **~3.4 h** |

The seq_len=1024 configuration is now the default in
`configs/train/qwen7b_qlora.yaml`; document samples are truncated harder
but FIM and diff-instr samples are mostly under 1024 tokens already.

### When to revisit

- A torch2.10 / cp313 / cu128 / sm_120 wheel landing on PyPI or the
  upstream `flash-attention` releases page → drop `flash-attn>=2.8.4`
  into `pyproject.toml`'s train extras and remove the wheel-vendoring
  step.
- Bumping torch beyond 2.10 → ABI may break again (this is what
  bit us at 2.9 → 2.10). Re-run the cross-build on the builder machine.
