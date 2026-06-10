# vendor — vendoring notes

The `vendor/` tree contains **git submodules** pinned to specific
upstream commit SHAs. Each submodule is treated as **opaque, not fully
trusted** code that we link to but do not audit line-by-line; the trust
posture for each is documented in `SECURITY-NOTES.md` next to this file.
These vendor-side docs are kept **outside** the submodules so each
submodule worktree stays clean (matching its upstream tree) and a future
SHA bump is a clean fast-forward, not a merge.

## claw-code

## What's pinned

| | |
|---|---|
| Upstream | `https://github.com/ultraworkers/claw-code` |
| Pinned SHA | recorded in the parent repo's `.gitmodules` + index |
| Build artefact | `rust/target/release/claw` (built via `scripts/build_claw_code.sh`) |

## Fallback remote

The canonical `ultraworkers/claw-code` repository was **temporarily
locked** during ownership transfer. While that lock was in effect, the
authors maintained a parity mirror at
`https://github.com/ultraworkers/claw-code-parity`. If a future submodule
update against the canonical remote fails because the repo is locked or
unreachable, switch the submodule URL via:

```bash
git submodule set-url vendor/claw-code https://github.com/ultraworkers/claw-code-parity
git submodule sync vendor/claw-code
git -C vendor/claw-code fetch origin
git -C vendor/claw-code checkout <sha>
```

The two repositories have shared lineage; SHAs may not align exactly, so
choose a parity SHA that matches the upstream commit content you trusted.

## Update policy

- **Never auto-update.** The parent repo's Python wrapper does not pull,
  fetch, or checkout in the submodule.
- **Bump = code review.** Bumping the pinned SHA must accompany a
  re-audit of `SECURITY-NOTES.md`, with diffs from the previous SHA
  inspected for new network calls, new auto-update logic, or new
  telemetry sinks pointed at remote endpoints.
- **No upstream patches in-place.** If a fix is required, submit it
  upstream or fork — never edit files inside `vendor/claw-code/`. The
  submodule pointer should always reference an upstream-shaped tree.

## How the parent repo uses this

- `codescribe_train/harness/claw.py` builds `cmd = [target/release/claw, ...]`
  and invokes it as a subprocess inside the user's working repo.
- `.claude.json` and `.claw.json` are rendered from
  `configs/harness/*.yaml` and written next to the workdir; the binary
  picks them up via its standard discovery rules. We do not edit any
  files inside this submodule to influence runtime behaviour.
- The optional Docker sandbox path (`Dockerfile.claw`,
  `--sandbox docker`) wraps execution in `--network none`, mounting the
  workdir read-write and *only* the workdir. Image `claw-code:vendored`.

---

## llama.cpp

`vendor/llama.cpp/` pins `ggml-org/llama.cpp` to a specific upstream
commit SHA. The build artefact `vendor/llama.cpp/build/bin/llama-server`
is the inference server `LlamaServerBackend` launches.

### What's pinned

| | |
|---|---|
| Upstream | `https://github.com/ggml-org/llama.cpp` |
| Pinned SHA | `58e68df0f91dd16ff56423ee5ef44062ed73bdfc` (master HEAD as of 2026-05-08) |
| Build artefact | `vendor/llama.cpp/build/bin/llama-server` (built via `scripts/build_llama_cpp.sh`) |
| CUDA target | `sm_120` (Blackwell, a Blackwell-class laptop GPU) — `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` |
| CUDA toolkit | `/usr/local/cuda-12.8/` (set in `scripts/build_llama_cpp.sh`) |

### Recheck command

To verify the pinned SHA in the submodule index matches the upstream
master HEAD or a chosen tag:

```bash
git -C vendor/llama.cpp log -1 --format=%H
git ls-remote https://github.com/ggml-org/llama.cpp.git HEAD
```

### Update policy

- **Never auto-update.** `scripts/build_llama_cpp.sh` is idempotent —
  it never `git pull`s in the submodule. Bumps are deliberate, reviewed
  changes accompanied by an audit refresh in `SECURITY-NOTES.md`.
- **Bump = code review.** Diffs from the previous SHA must be inspected
  for new outbound surfaces (`libcurl` integrations, HF fetcher
  expansions, telemetry sinks). The re-audit checklist is in
  `SECURITY-NOTES.md` under the "llama.cpp" section.
- **No upstream patches in-place.** The submodule pointer should always
  reference an upstream-shaped tree. If a fix is required, submit it
  upstream or fork — never edit files inside `vendor/llama.cpp/`.

### How the parent repo uses this

- `codescribe_train/backends/llama_server.py` builds
  `cmd = [build/bin/llama-server, -m <gguf>, --host 127.0.0.1, --port <p>, ...]`
  and invokes it as a subprocess. The `-m <local-path>` form is the
  only one used — never `--hf-repo` / `--model-url`, so no Hugging Face
  fetch ever occurs in the runtime path.
- `scripts/build_llama_cpp.sh` runs `cmake -DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release` from
  `/usr/local/cuda-12.8/` and builds the `llama-server` and
  `llama-quantize` targets only. The script is rebuild-aware: if the
  binary is newer than the submodule HEAD's commit time, it skips.
