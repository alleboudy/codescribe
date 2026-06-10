# vendor — security audit notes

This file records the audits for each vendored submodule. Append, never
overwrite — older audits document prior trust decisions and changes
between SHA bumps.

## claw-code

This file records the **outbound network surface, telemetry, and auto-
update behaviour** of the vendored `ultraworkers/claw-code` repository at
the pinned commit SHA. The audit is recorded once, here, so future SHA
bumps are forced to either re-confirm or update each row.

The audit was performed with:

```bash
cd vendor/claw-code
rg -i 'telemetry|analytics|sentry|posthog|amplitude|mixpanel' --type rust
rg 'https?://' --type rust
rg -i 'auto.?update|self.?update|check.?update' --type rust
rg -i 'phone.?home|"track"|"event"' --type rust
rg -l 'reqwest|hyper::Client|ureq' --type rust
```

across `rust/crates/` and `install.sh`.

## Findings

| # | file:line(s) | what was found | severity | mitigation in `codescribe-train` |
|---|---|---|---|---|
| 1 | `rust/crates/telemetry/src/lib.rs:1-525` | A first-party `telemetry` crate exposes `MemoryTelemetrySink` and `JsonlTelemetrySink`. Both write to **local destinations only** (in-memory `Vec`, or a JSONL file at a caller-supplied `Path`). No HTTP transport in the crate; nothing in this crate phones home by itself. | low | Accept. Sink is local. The wrapper does not enable any sink. |
| 2 | `rust/crates/telemetry/src/lib.rs:135-156` | Defines `AnalyticsEvent { namespace, action, properties }`. It's only a data shape — a `TelemetrySink` is what would *send* it. Local-only sinks are the only impls in-tree. | low | Accept. Data shape, not a transport. |
| 3 | `rust/crates/api/src/providers/anthropic.rs:25` | `pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";` | **medium** | Wrapper sets `ANTHROPIC_BASE_URL` to the local backend URL in `_build_env()` so a model-name typo cannot leak to Anthropic. Wrapper never injects an `ANTHROPIC_API_KEY` from a default. |
| 4 | `rust/crates/api/src/providers/openai_compat.rs:19-21` | Default base URLs for OpenAI (`https://api.openai.com/v1`), xAI (`https://api.x.ai/v1`), DashScope (`https://dashscope.aliyuncs.com/compatible-mode/v1`). Each is overridable via `OPENAI_BASE_URL`, `XAI_BASE_URL`, `DASHSCOPE_BASE_URL`. | **medium** | Wrapper sets `OPENAI_BASE_URL` to the local backend. **Caveat:** `qwen-*` and `qwen/*` model names route to DashScope by default — the wrapper avoids those names; use `openai/<id>` or `gpt-<id>` model names for local inference. |
| 5 | `rust/crates/api/src/providers/mod.rs:188-218` | `metadata_for_model()` routes `qwen-*`, `qwen/*`, `kimi-*`, `kimi/*` to DashScope via prefix match (uses `DASHSCOPE_API_KEY` / `DASHSCOPE_BASE_URL`). Without the right env override, a `qwen-` model name will try to contact DashScope. | **medium** | Documented in `codescribe_train/harness/claw.py` module docstring. Recommend `openai/qwen2.5-coder-7b-sample` style model names for local routing. |
| 6 | `rust/crates/api/src/http_client.rs:154-320` | Reads `HTTP_PROXY`, `HTTPS_PROXY`, `http_proxy`, `https_proxy` env vars, builds a `reqwest` client honouring them. Standard proxy support; no covert channel. | low | Accept. The wrapper does not unset proxy vars; if the user has them set, the harness will honour them like any HTTP tool. |
| 7 | `install.sh:1-30` (`curl https://sh.rustup.rs \| sh`) | Upstream's installer fetches rustup. Only relevant if the user runs `install.sh`; the parent repo's `scripts/build_claw_code.sh` does not invoke it. | low | Accept. Not on the parent-repo execution path. |
| 8 | `Containerfile` (`FROM rust:bookworm`, runs `apt-get update`, etc.) | Upstream's container build pulls debian + apt packages from network. Build-time only; not runtime. | low | Accept. Our `Dockerfile.claw` does the equivalent — build-time fetch, runtime is `--network none`. |
| 9 | `rust/crates/runtime/src/config_validate.rs` matches `autoUpdate` | Validates a `plugins.autoUpdate` config field. Validation only — does not implement an updater. No HTTP fetch on the parent-repo execution path. | low | Accept. We never enable plugins / `autoUpdate` in `configs/harness/*.yaml`. |
| 10 | `rust/crates/api/src/providers/anthropic.rs:1058-1424` | `https://console.test/...`, `https://example.test` — RFC 6761 reserved test TLDs used in unit tests, not in runtime paths. | none | Accept. Test-only URLs do not resolve. |

## Posture summary

- **No telemetry phone-home.** The first-party `telemetry` crate is
  local-sink-only. There is no HTTP transport in `crates/telemetry/`.
- **No auto-update at runtime.** The only `auto.?update` hits are
  `plugins.autoUpdate` config validation; we don't enable plugins.
- **HTTP egress is provider-routed, env-controlled.** All outbound HTTP
  goes through `crates/api/`'s provider clients, which read base-URL env
  vars. The wrapper sets those env vars to the local backend.
- **DashScope routing trap.** `qwen-*` model names route to DashScope by
  default. The wrapper documents this and recommends `openai/`-prefixed
  model names; in the docker sandbox, `--network none` is the backstop.

## Re-audit checklist on SHA bump

When bumping `vendor/claw-code` to a newer SHA:

1. Re-run all four `rg` commands at the top of this file.
2. For each new `https?://` literal in non-test runtime paths, decide
   whether it represents a new outbound surface and add a row above.
3. If a new HTTP client is added (`reqwest`, `hyper`, `ureq`, `surf`,
   `isahc`, `awc`, etc.), document the env-var override path used to
   point it at a local backend.
4. Inspect any new `mod telemetry` / `mod analytics` / `mod metrics`
   crates and confirm the sinks are still local-only.
5. Confirm `--network none` Docker sandbox still produces a usable
   session — that's the strictly-local backstop and is the most
   trustworthy evidence that nothing important phones home.

---

## llama.cpp

`vendor/llama.cpp/` is `ggml-org/llama.cpp` pinned at SHA
`58e68df0f91dd16ff56423ee5ef44062ed73bdfc` (master HEAD as of 2026-05-08).
Built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` for Blackwell
(a Blackwell-class laptop GPU sm_120). The `llama-server` binary is the only artefact
launched at runtime by `codescribe-train`; the rest of the tree is build-time
or unused tools.

Audit was performed with:

```bash
cd vendor/llama.cpp
rg -i 'telemetry|analytics|sentry|posthog|amplitude|mixpanel|phone.?home' \
    src/ tools/server/ -t cpp -t c
rg 'https?://' src/ tools/server/ -t cpp -t c | grep -vE 'github|huggingface|wikipedia|jina\.ai|prometheus\.io|arxiv|opensource|openai\.com'
rg -i 'auto.?update|self.?update|check.?update' src/ tools/server/ -t cpp -t c
rg -l 'libcurl|curl_easy_setopt|httplib::Client' src/ common/ tools/server/
```

### Findings

| # | file:line(s) | what was found | severity | mitigation in `codescribe-train` |
|---|---|---|---|---|
| 1 | `src/`, `tools/server/` (all C++/C runtime sources) | **No telemetry, analytics, sentry, posthog, amplitude, mixpanel, or phone-home strings.** All matches in the wider repo are documentation comments, build-config strings, or storybook fixture files (e.g. `tools/server/webui/tests/stories/fixtures/data-analysis.ts` is a sample dataset). | none | Accept. No action needed. |
| 2 | `common/common.cpp:~260` (`model_endpoint = "https://huggingface.co/"`) | `arg.cpp` exposes `--hf-repo <name>` and `--hf-file <name>` flags that fetch GGUF files from Hugging Face via libcurl. The endpoint is overridable via `HF_ENDPOINT`. The flags are **opt-in**: with `-m <local-path.gguf>` (the path `codescribe-train run` always uses), no HF fetch happens. | **medium (opt-in)** | We always pass `-m <gguf-path>` from the wrapper — see `LlamaServerBackend._build_argv`. We never set `--hf-repo` or `--hf-file`. The optional `--offline` flag (`common/arg.cpp`) is the documented backstop; we don't set it because we never trigger a fetch in the first place. |
| 3 | `common/download.cpp` (libcurl-driven downloader for `--hf-repo` / `--model-url`) | Library used only when the user opts into a `hf://` or remote-URL model spec. Honors `HF_TOKEN` env var and `--offline` flag. | low | Accept. Not on the wrapper's execution path. |
| 4 | `common/arg.cpp` (`HF_TOKEN` env var read) | Reads `HF_TOKEN` from env to authenticate HF downloads. Only relevant when `--hf-repo` is used. | low | Accept. Wrapper never sets `HF_TOKEN`; user's shell may, but it's only consumed by the opt-in fetch path. |
| 5 | `tools/server/server.cpp` (`--host`, `--port`) | The server binds to `--host 127.0.0.1 --port 8080` by default for our wrapper (see `configs/backends/local-default.yaml`). No automatic LAN binding, no UPnP, no mDNS. | none | Wrapper config locks `host: 127.0.0.1`. |
| 6 | `tools/server/webui/package-lock.json` (`@opentelemetry/api` peer-optional) | Listed as an **optional peer dependency** of an upstream package. `webui/src/**` does not import `@opentelemetry/api` and the build does not pull it. The webui is shipped as static assets served by the C++ binary; it does not phone home. | none | Accept. Optional peer dep, not exercised by the build or runtime. |
| 7 | `tools/server/webui/tests/stories/fixtures/data-analysis.ts` (storybook fixture mentioning `analytics.example.com`, "Amplitude") | Sample data inside a storybook test fixture for a markdown-rendering UI test. Strings only — no HTTP code. | none | Accept. Test fixture, not runtime. |
| 8 | `Makefile`, `CMakeLists.txt`, `flake.nix`, `ci/`, `scripts/` (build-time fetches: cmake-fetched dependencies, Hugging Face download in CI, etc.) | Build-time only, not runtime. Our `scripts/build_llama_cpp.sh` only invokes `cmake --build` against an already-checked-out submodule — no `git pull`, no `cmake -P` of the upstream CI scripts. | low | Accept. Build path is `git submodule update --init` (one-time, local) + `cmake --build`. |

### Posture summary

- **No telemetry phone-home in the runtime path.** No `telemetry`,
  `analytics`, `sentry`, `posthog`, `amplitude`, `mixpanel` modules in
  the runtime C++/C sources.
- **HF fetcher is opt-in.** `llama-server` only contacts Hugging Face
  when invoked with `--hf-repo` / `--hf-file` / `--model-url`. Our
  wrapper always uses `-m <local-gguf-path>`, so no fetch occurs.
- **Loopback by default.** Wrapper config pins `host: 127.0.0.1`. The
  binary itself supports `--host 0.0.0.0`, but our default never sets it.
- **Webui is static + offline.** No analytics or telemetry deps loaded
  by `tools/server/webui` at runtime.

### Re-audit checklist on SHA bump

When bumping `vendor/llama.cpp` to a newer SHA:

1. Re-run the `rg` commands at the top of this section.
2. For each new `https?://` literal in non-test runtime paths
   (`src/`, `tools/server/*.cpp`), decide whether it represents a new
   outbound surface and add a row above.
3. Inspect any new HTTP client integrations
   (`libcurl`, `httplib::Client`, `cpp-httplib` wrappers) and confirm
   they are gated behind explicit user-facing flags.
4. Confirm `tools/server/webui/package.json` has not added an analytics
   or telemetry dependency in `dependencies` (peer-optional in
   `package-lock.json` is acceptable; runtime imports are not).
5. Re-confirm that `LlamaServerBackend._build_argv` still passes
   `-m <local-path>` and never any HF / URL-based model spec.
