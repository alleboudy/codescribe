# `codescribe_train/harness/`

Pluggable agent-harness layer. Phase 3 of `codescribe-train`.

Defines a `Harness` ABC and ships two implementations:
- **`ClawCodeHarness`** — wraps the vendored Rust binary [`ultraworkers/claw-code`](https://github.com/ultraworkers/claw-code) at pinned SHA `357629d`. Default for end-user sessions. **Not fully trusted** — vendored, audited, sandboxable.
- **`NoOpHarness`** — echoes stdin to stdout. Used as a swap-test that the abstraction holds without depending on claw-code.

> Full audit findings for the vendored binary: [`vendor/SECURITY-NOTES.md § claw-code`](../../vendor/SECURITY-NOTES.md). Trust posture overview: [`docs/STRICTLY-LOCAL-POSTURE.md`](../../docs/STRICTLY-LOCAL-POSTURE.md).

---

## Public surface

CLI:

```bash
python -m codescribe_train.harness run \
    --backend http://127.0.0.1:8080 \
    --model openai/sample-qwen7b-local \
    --workdir /path/to/repo \
    --harness {claw|noop} \
    [--sandbox docker] \
    [--config configs/harness/<repo>.yaml] \
    [--skip-health-check]
```

The intended invocation path is via the **top-level `codescribe-train run`** command (in `codescribe_train/cli.py`), which orchestrates backend startup → harness preparation → session → cleanup. This package's CLI is for harness-only testing.

---

## File layout

```
harness/
├── __init__.py
├── __main__.py             # → python -m codescribe_train.harness <subcommand>
├── cli.py                  # argparse + dispatch; lazy imports of impls
├── base.py                 # Harness ABC
├── claw.py                 # ClawCodeHarness — vendored Rust binary subprocess
└── noop.py                 # NoOpHarness — echo loop, swap-test target
```

The ABC has four methods:

```python
class Harness(ABC):
    def prepare(self, workdir: Path, backend_url: str, model: str,
                *, harness_config: dict) -> None: ...
    def start_session(self, stdin, stdout) -> int: ...
    def health_check(self, backend_url: str) -> bool: ...
    def sandbox_args(self) -> list[str]: ...
```

`prepare()` does all the I/O setup (renders config files, composes env vars) but doesn't spawn anything. `start_session()` actually launches the harness subprocess and forwards stdin/stdout. `sandbox_args()` returns a prefix to prepend to the launch command — `[]` for direct, `["docker", "run", "--network", "none", ...]` for sandboxed mode.

---

## `ClawCodeHarness` — the default

Wraps `vendor/claw-code/rust/target/release/claw` (a Rust binary). The binary speaks OpenAI Chat Completions; we forward the local backend URL via env vars and pass `--model openai/<id>` on the CLI.

Critical implementation details:

- **`--model` is a CLI flag, not just an env var.** Claw's provider routing keys off the prefix of the `--model` flag value (`openai/...` → OpenAI-compat client). Setting `OPENAI_MODEL` env alone doesn't work — the `--model` flag value's prefix is what drives provider routing.
- **`ANTHROPIC_BASE_URL` is pinned to the local backend** as belt-and-braces. Some claw paths read it when the model resolves to the Anthropic provider; pinning prevents an off-by-one routing decision from triggering an outbound request to api.anthropic.com.
- **No `ANTHROPIC_API_KEY` is invented.** We forward the user's if they supply one via the standard env, but never set a placeholder.
- **The binary's working directory is the target repo** (`workdir`). Claw treats the cwd as the project root for tool-use scoping.
- **Config rendering**: `prepare()` writes `.claude.json` and `.claw.json` into `workdir` based on the YAML in `--config`. Allowed tools, slash command preferences, MCP servers (empty by default), prompt overlays.

### Sandbox modes

```python
sandbox_args() == []                      # mode='none' — direct launch
sandbox_args() == ["docker", "run", "--rm", "-i",
                   "--network", "none",
                   "-v", f"{workdir}:/work", "-w", "/work",
                   "claw-code:vendored"]   # mode='docker'
```

`--sandbox docker` requires `Dockerfile.claw` to have been built into the `claw-code:vendored` image. The image is multi-stage (`rust:1.83-slim` builder + `debian:bookworm-slim` runtime) and runs as a non-root user. Build with `docker build -t claw-code:vendored -f Dockerfile.claw .` from the repo root. This is the kernel-enforced "no egress" mode for paranoid sessions.

---

## `NoOpHarness` — the abstraction's correctness check

```python
class NoOpHarness(Harness):
    def prepare(self, ...): self._prepared = True
    def start_session(self, stdin, stdout) -> int:
        for line in stdin:
            stdout.write(f"echo: {line}")
        return 0
    def health_check(self, ...) -> bool: return True  # backend doesn't matter
    def sandbox_args(self) -> list[str]: return []
```

Used in `tests/harness/test_swap.py` to verify that **the rest of the orchestration works without claw-code**. If you ever fork claw or write a native C++20 replacement, the swap test is what proves your replacement satisfies the contract.

The CLI also exposes it: `python -m codescribe_train.harness run --harness noop` launches an echo session against any backend URL (or with `--skip-health-check`, against no backend at all).

---

## Conventions

- **Vendored = pinned-SHA. No exceptions.** Don't `git submodule update --remote` without a code review.
- **Audit on bump.** Every SHA bump of `vendor/claw-code` runs the audit checklist in `vendor/SECURITY-NOTES.md` and updates the findings table if anything changed.
- **Lazy imports.** `import codescribe_train.harness.cli` doesn't pull `httpx`, `yaml`, or either concrete harness. Verified by `tests/harness/test_cli_lazy_imports.py`.
- **No claw-specific code outside `claw.py`.** The CLI dispatch and tests should never import `claw` unless the user picked `--harness claw`. The `Harness` ABC is the only thing the orchestrator knows about.

---

## Configs

`configs/harness/sample.yaml` — repo-specific defaults for the e-commerce target. Allowed tools, MCP server list (empty), prompt overlay path (`../your-repo/CLAUDE.md`).

`configs/harness/generic.yaml` — sane defaults for any target.

The Python adapter consumes these YAMLs and renders them into the `.claude.json` / `.claw.json` files claw-code expects on disk in the workdir at session start.

---

## Tests

`tests/harness/`:
- `test_base.py` — Harness ABC has the right method signatures (introspection)
- `test_noop.py` — NoOpHarness round-trips lines correctly
- `test_claw_config.py` — config rendering produces valid JSON; subprocess launch is mocked
- `test_cli_lazy_imports.py` — `import codescribe_train.harness.cli` doesn't pull httpx / yaml / either concrete harness
- `test_swap.py` — proves abstraction holds: `NoOpHarness` and `ClawCodeHarness` both satisfy the protocol; CLI dispatches to noop without loading claw

34 tests, all passing.

---

## Hardware prereqs

- **Rust toolchain** (`rustup`, `cargo`) for building `vendor/claw-code/rust/`. Run `scripts/build_claw_code.sh` once.
- **Docker** (optional) — only needed for `--sandbox docker` mode. Build the image: `docker build -t claw-code:vendored -f Dockerfile.claw .`

---

## Adding a new harness

If you want to plug in a different harness (your own native binary, a different open-source agent, a fork of claw-code you trust more):

1. Create `harness/<your_harness>.py` with a class implementing `Harness`.
2. Add it to the dispatch in `cli.py` and the `--harness` choices.
3. Vendor the source code at a pinned commit if it's third-party. Audit it. Update `vendor/SECURITY-NOTES.md`.
4. Add tests in `tests/harness/`. The minimum: ABC contract test, sandbox-args test, end-to-end smoke test against `NoOpHarness` for orchestration parity.

---

## See also

- [`vendor/SECURITY-NOTES.md`](../../vendor/SECURITY-NOTES.md) — claw-code audit
- [`vendor/README-VENDOR.md`](../../vendor/README-VENDOR.md) — vendoring policy and SHA tracking
- [`docs/STRICTLY-LOCAL-POSTURE.md`](../../docs/STRICTLY-LOCAL-POSTURE.md) — overall trust model
- [`docs/ARCHITECTURE.md § 4`](../../docs/ARCHITECTURE.md#4-trust-posture) — trust posture in context
