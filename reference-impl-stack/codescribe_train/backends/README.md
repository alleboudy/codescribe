# `codescribe_train/backends/`

Pluggable inference-backend layer. Phase 4 of `codescribe-train`.

Defines a `Backend` ABC and ships four implementations:
- **`LlamaServerBackend`** — default. Spawns `llama-server` from vendored llama.cpp built for the host GPU's compute capability (set via `CMAKE_CUDA_ARCHITECTURES`). All on loopback.
- **`VllmBackend`** — alternative full-GPU path. Lazy-imported; not in any default config.
- **`OllamaBackend`** — assumes user runs `ollama serve` separately. Validates URL only.
- **`RemoteBackend`** — interface stub. Refuses to start unless `CODESCRIBE_ENABLE_REMOTE_BACKEND=1`. Reserved for the optional topology where another machine on your LAN serves the model.

Plus `probe.py` — hardware detection (`nvidia-smi` / `pynvml` + `psutil`) that recommends a backend + key options.

> Background:
> - [`docs/ARCHITECTURE.md § 2`](../../docs/ARCHITECTURE.md#2-each-phase-in-one-paragraph) — phase composition
> - [`docs/STRICTLY-LOCAL-POSTURE.md § 1`](../../docs/STRICTLY-LOCAL-POSTURE.md) — why `RemoteBackend` is stub-only
> - [`vendor/SECURITY-NOTES.md § llama.cpp`](../../vendor/SECURITY-NOTES.md) — audit of the vendored inference engine

---

## Public surface

```bash
python -m codescribe_train.backends probe
python -m codescribe_train.backends health --endpoint http://127.0.0.1:8080
```

Both subcommands are diagnostics-only. The intended invocation path is via the top-level **`codescribe-train run`** command (`codescribe_train/cli.py`), which constructs the chosen `Backend`, calls `start()` to bring up the server, polls `health_check()` until ready, hands the endpoint URL to the harness, and calls `stop()` in a `finally` block on session exit.

The ABC:

```python
class Backend(ABC):
    def start(self, *, model_id: str, adapter: Path | None,
              port: int) -> str:
        """Spawn or attach. Return the OpenAI-compat endpoint URL."""

    def stop(self) -> None:
        """Idempotent. Safe to call on a never-started or already-stopped backend."""

    def health_check(self, endpoint: str, *,
                     timeout_s: float = 5.0) -> bool:
        """Probe /v1/models. Return True if 200 OK with a body."""
```

---

## File layout

```
backends/
├── __init__.py
├── __main__.py            # → python -m codescribe_train.backends <subcommand>
├── cli.py                 # probe + health subcommands
├── base.py                # Backend ABC + small dataclasses
├── llama_server.py        # LlamaServerBackend — default
├── vllm.py                # VllmBackend — lazy-imported, not default
├── ollama.py              # OllamaBackend — external server, URL-only
├── remote.py              # RemoteBackend — stub; refuses to start without env
└── probe.py               # HardwareProfile + recommend_backend
```

`__init__.py` and `cli.py` import nothing heavy — concrete backends are loaded lazily inside the dispatch in `cli.main`. Verified by `tests/backends/test_cli_lazy_imports.py`.

---

## `LlamaServerBackend` — the default

Spawns `vendor/llama.cpp/build/bin/llama-server` as a subprocess with these args (built in `_build_argv`):

```
llama-server
    -m <gguf-path>             # always a local path; never --hf-repo
    --host 127.0.0.1           # loopback only
    --port <port>              # default 8080
    -ngl <gpu_layers>           # -1 = auto-pick (puts everything on GPU it fits)
    --ctx-size <ctx>            # default 8192
    [--no-mmap]                 # default true on tight-VRAM laptops
    [--api-key <key>]           # optional; default unset → server doesn't require auth
```

Captures stdout/stderr to `logs/llama-server.log` (gitignored). Detects readiness by tailing that log for a ready marker (e.g. `"HTTP server listening"`); times out after `startup_timeout_s` (default 60 s).

`stop()` sends SIGTERM, waits 10 s, sends SIGKILL if still alive. Idempotent.

### Why we always pass `-m <local-path>`

llama.cpp supports `--hf-repo` and `--model-url` for fetching models from the network on demand. Our wrapper **never** uses these — `model_id` is always a local filesystem path, typically the GGUF produced by `codescribe_train.train.export`. This is the load-bearing constraint for the strictly-local posture; see [`vendor/SECURITY-NOTES.md § llama.cpp`](../../vendor/SECURITY-NOTES.md) row #2.

---

## `VllmBackend`

Spawns `vllm serve <model_id> --host 127.0.0.1 --port <port>`. Lazy-imports `vllm`; raises a clear error if not installed. **Not in default extras** — vLLM has heavyweight deps and doesn't fit Qwen 7B in 8 GB without aggressive tuning.

If you want it: `uv pip install vllm` and run `--backend vllm`. Use cases:
- High-throughput multi-user serving (vLLM excels there)
- Specific model architectures llama.cpp doesn't support yet
- A100/H100 hosts where the FlashInfer / Triton paths shine

Not the right choice for a consumer single-GPU laptop.

---

## `OllamaBackend`

**Does NOT start ollama itself.** Assumes the user has run `ollama serve` separately and has loaded a model. The wrapper validates the URL (default `http://127.0.0.1:11434`) by probing `/api/tags`, and otherwise just passes the configured model name through to whatever speaks at the URL.

Use case: a user who already has Ollama as their day-to-day model server and wants the harness to talk to it.

---

## `RemoteBackend` — interface stub only

```python
class RemoteBackend(Backend):
    def start(self, ...) -> str:
        if os.environ.get("CODESCRIBE_ENABLE_REMOTE_BACKEND") != "1":
            raise RemoteBackendDisabled(
                "RemoteBackend is disabled by default per the strictly-local "
                "posture. Set CODESCRIBE_ENABLE_REMOTE_BACKEND=1 to enable."
            )
        # Otherwise: validate the URL via health_check and return it as-is.
```

The class exists for **ABC parity** — without it, the `Backend` interface couldn't accommodate "another machine I trust on your LAN" without an architecture change. With it, a future user who wants a second, larger-VRAM machine on their LAN to serve 14B models for a lighter client's harness can flip the env and edit a config; the interface is already shaped for it.

`configs/backends/lan-remote.yaml` exists as a documented **example only** of how that config would look. It is not loaded by any default code path.

---

## `probe.py` — hardware detection

```python
from codescribe_train.backends.probe import probe_hardware, recommend_backend

profile = probe_hardware()
# HardwareProfile(gpu_name="...", gpu_vram_gib=8.0,
#                 system_ram_gib=16.0, cpu_count=16)

choice = recommend_backend(profile, model_size_gib=4.6)
# BackendChoice(backend_name="llama-server",
#               options={"gpu_layers": 99, "ctx_size": 8192, "no_mmap": False},
#               rationale="...")
```

`probe_hardware()` tries `pynvml` first (richer info — clocks, utilisation), falls back to parsing `nvidia-smi --query-gpu=...`. `psutil.virtual_memory()` for system RAM, `os.cpu_count()` for CPU.

`recommend_backend()` is rule-based: e.g. an 8 GB-VRAM GPU + 4.6 GiB Q4_K_M model → `llama-server` with `gpu_layers=99, ctx_size=8192, no_mmap=False`. The rule set is documented in the function's docstring.

CLI: `python -m codescribe_train.backends probe` prints a one-line summary plus the recommendation.

---

## Conventions

- **Loopback by default.** No backend's default config binds to anything other than `127.0.0.1`.
- **`RemoteBackend` is never wired into a default config.** Its existence is for the LAN topology when (if) the user opts in.
- **Vendored llama.cpp pinned at SHA `58e68df`.** Audit on every bump (`vendor/SECURITY-NOTES.md`).
- **Lazy imports for vllm / pynvml / heavy deps.** `import codescribe_train.backends.cli` doesn't pull them.
- **`stop()` is idempotent.** Code calling it must not assume the backend is still alive — if it crashed, `stop()` should still do the right thing without raising.

---

## Tests

`tests/backends/`:
- `test_base.py` — ABC contract
- `test_llama_server.py` — argv construction; log file path; health-check poll loop (subprocess mocked)
- `test_vllm.py` — argv construction; raises clear error if vllm unimportable (mocked)
- `test_ollama.py` — health check probes `/api/tags` (httpx mocked)
- `test_remote.py` — `RemoteBackendDisabled` raised unless env var set
- `test_probe.py` — fake `nvidia-smi` invocation; assert `recommend_backend` returns sensible defaults
- `test_cli_lazy_imports.py` — import discipline

Plus `tests/test_top_cli.py` and `tests/test_lazy_imports_top.py` exercise the orchestrator (probe → backend.start → harness.prepare → harness.start_session → backend.stop) with fully mocked Backend + Harness.

40+ tests, all passing.

---

## Hardware prereqs

For `LlamaServerBackend`:
- `vendor/llama.cpp/build/bin/llama-server` exists. Run `scripts/build_llama_cpp.sh` once. ~50 minutes wall-clock with `BUILD_JOBS=2`. The script is idempotent.
- A GGUF on disk. Either pulled from HF (e.g., `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`) or produced by `codescribe_train.train.export` from a fine-tuned LoRA.

For `VllmBackend`:
- `uv pip install vllm` (out-of-tree; not in default extras).

For `OllamaBackend`:
- `ollama serve` running externally with the model already loaded.

For probe:
- `psutil` is in the `backends` extras; `pynvml` is optional (auto-falls-back to `nvidia-smi` parsing).

---

## Adding a new backend

1. Create `<your_backend>.py` with a class implementing `Backend`.
2. Register it in `cli.py`'s dispatch and the `--backend` choices.
3. Add tests covering: argv construction (or URL validation), health check loop, `stop()` idempotency, `start()` failure modes.
4. If the new backend has heavyweight deps, lazy-import them inside the methods that need them.
5. If it would make sense as a default for some hardware profile, update `probe.recommend_backend`.

---

## See also

- [`codescribe_train/harness/README.md`](../harness/README.md) — what consumes the endpoint URL we expose
- [`codescribe_train/train/README.md`](../train/README.md) — produces the GGUF this layer serves
- [`vendor/SECURITY-NOTES.md`](../../vendor/SECURITY-NOTES.md) — llama.cpp audit
