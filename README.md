# codescribe

Recipes and reference code for building a **strictly-local fine-tune-on-codebase stack**: take a private codebase, fine-tune a 7B-class open-weights model on it, serve the result locally via an OpenAI-compatible HTTP endpoint, and (optionally) plug it into any modern coding harness with retrieval-augmented context from your bug tracker and version-control history.

Nothing in this repo runs in the cloud. The training, the inference, the retrieval index, and the harness all live on your hardware. The model and your code never leave the box.

This is a **planning + documentation** repo. It contains:

- The **canonical issues** (#1–#5) describing how to build, scale, retrieve, and reference-implement each layer.
- The **documentation tree under `docs/`** explaining every concept from the ground up — for an intern joining the project cold, or for an AI coding agent that needs background before working on any issue.

---

## When to use this

You want an AI coding assistant that knows *your specific codebase* — its module names, its idioms, its historical bugs and the PRs that fixed them — and your codebase is private (NDA, contract, security clearance, regulated industry).

You have **at minimum**:
- A workstation or laptop with an NVIDIA GPU (≥ 8 GB VRAM, Ampere or newer for training; older OK for inference).
- ~50 GB free disk for model weights + datasets + indices.
- Python 3.12.
- A source codebase with at least a few thousand non-vendored source files — origin doesn't matter: a Git repo, a Perforce depot/workspace, a Subversion checkout, or just a plain directory on disk all work. The data layer walks files, not commits. (Smaller corpora teach the model nothing useful.)
- Optional but useful: a bug tracker (Bugzilla, GitHub Issues, JIRA) and a way to link bugs to fixes.

You do **not** want:
- Cloud-only solutions (Copilot Chat as your primary harness, hosted vector stores, hosted embedding APIs).
- A model that pretends to know your codebase by reading it inline every time.
- A vendor lock-in via a proprietary inference stack.

---

## Repository contents

### Issues (planning + spec)

| # | Title | What it gives you |
|---|---|---|
| [#1](https://github.com/alleboudy/codescribe/issues/1) | Runbook: serve the fine-tuned GGUF and drive it with `claw` | The operator runbook. Spin up `llama-server`, tunnel from a remote box, run a smoke test, drive the model with `claw`. |
| [#2](https://github.com/alleboudy/codescribe/issues/2) | Meta-plan: fine-tune-on-codebase stack via GitHub Copilot subagents | The build-off plan. Five per-phase issue bodies you file as GitHub issues and assign to Copilot (or an intern). Includes a 9-dimensional scoring rubric for comparing implementations. |
| [#3](https://github.com/alleboudy/codescribe/issues/3) | Fleet guide: parallel & distributed QLoRA across N workstation laptops | How to scale beyond one machine. Two paths: parallel HP sweep (recommended) and distributed data-parallel training (rarely worth it on 1 Gbps LAN). the OEM-specific power/thermal config. |
| [#4](https://github.com/alleboudy/codescribe/issues/4) | RAG plan: bugs+fixes from Perforce/Bugzilla as MCP-served context | How to add inference-time retrieval over your bug-tracker and VCS history. Six per-package phases. Deep MCP primer. Strict-by-default bug↔CL pairing. |
| [#5](https://github.com/alleboudy/codescribe/issues/5) | Reference Python skeletons for all RAG-stack tooling | Annotated near-runnable templates for every script: Perforce client, Bugzilla REST client, sqlite-vec store, embedder, hybrid retriever, MCP server, llama-server handle, test fixtures. |
| [#6](https://github.com/alleboudy/codescribe/issues/6) | RAG-only path: skip fine-tuning; plug MCP into Copilot Chat | The shortcut. Skip #2's Phases 1–4 (no training, no `llama-server`). Build only the RAG indexer + MCP server from #4 and wire it into Copilot Chat. Cheap and fast — but **cloud-coupled**, not strictly-local. Read §3 of the issue before committing. |

### Documentation (`docs/`)

| File | Read it before |
|---|---|
| [`docs/01-overview.md`](docs/01-overview.md) | Doing anything else in this repo |
| [`docs/02-fine-tuning.md`](docs/02-fine-tuning.md) | Working on issue #2's Phase 2 (the `train/` package) |
| [`docs/03-models.md`](docs/03-models.md) | Wondering why Qwen 2.5 Coder 7B and not another model |
| [`docs/04-inference.md`](docs/04-inference.md) | Working on issue #2's Phase 3 (the `serve/` package) |
| [`docs/05-harnesses.md`](docs/05-harnesses.md) | Deciding which coding harness to drive the model with |
| [`docs/06-github-copilot.md`](docs/06-github-copilot.md) | Handing issue #2's phases to GitHub Copilot |
| [`docs/07-mcp.md`](docs/07-mcp.md) | Working on issue #4's MCP server (Phase 11) |
| [`docs/08-rag.md`](docs/08-rag.md) | Working on issue #4's retrieval pipeline |
| [`docs/09-perforce.md`](docs/09-perforce.md) | Indexing a Perforce-hosted codebase |
| [`docs/10-bugzilla.md`](docs/10-bugzilla.md) | Indexing Bugzilla-hosted bugs |
| [`docs/11-hardware.md`](docs/11-hardware.md) | When something OOMs or refuses to compile |
| [`docs/12-glossary.md`](docs/12-glossary.md) | Hitting a term you don't recognise |
| [`docs/13-further-reading.md`](docs/13-further-reading.md) | Wanting the original papers + external docs |
| [`docs/14-fleet-training.md`](docs/14-fleet-training.md) | Putting a fleet of laptops to work (issue #3) — pairs with [`examples/fleet/`](examples/fleet/) |
| [`docs/20-onboarding-handbook.md`](docs/20-onboarding-handbook.md) | Producing an offline PDF onboarding handbook for a new owner |
| [`docs/22-memory-system-blueprint.md`](docs/22-memory-system-blueprint.md) | Building the deterministic code-memory graph from scratch (agent-implementable spec) |
| [`docs/23-memory-operations-and-lifecycle.md`](docs/23-memory-operations-and-lifecycle.md) | Operating a deployed memory system: setup, schedules, consolidation nights, backups, restores |

---

## How to read this repo

There are two paths through this material.

### Path A — "I'm building the stack"

Read in this order:

1. [`docs/01-overview.md`](docs/01-overview.md) — what the stack does end-to-end.
2. [`docs/11-hardware.md`](docs/11-hardware.md) — confirm your hardware can do this.
3. [Issue #2](https://github.com/alleboudy/codescribe/issues/2) — the build-off plan; how the phases decompose.
4. The doc for each phase as you reach it (e.g., [`02-fine-tuning.md`](docs/02-fine-tuning.md) before Phase 2, [`04-inference.md`](docs/04-inference.md) before Phase 3, etc.).
5. [Issue #5](https://github.com/alleboudy/codescribe/issues/5) when you need the concrete Python skeleton for the file you're about to write.
6. [Issue #1](https://github.com/alleboudy/codescribe/issues/1) when the GGUF is ready and you want to drive it.
7. [Issue #4](https://github.com/alleboudy/codescribe/issues/4) for RAG once the basic stack works.
8. [Issue #3](https://github.com/alleboudy/codescribe/issues/3) only if you outgrow a single machine.

### Path B — "I'm reviewing or auditing"

Read in this order:

1. [`docs/01-overview.md`](docs/01-overview.md) — the design rationale.
2. The "honest caveats" sections in [issue #2](https://github.com/alleboudy/codescribe/issues/2), [#3](https://github.com/alleboudy/codescribe/issues/3), [#4](https://github.com/alleboudy/codescribe/issues/4) — what we're explicit about *not* doing.
3. The scoring rubrics (§16 of #2; §15 of #4) — how to measure whether an implementation is any good.
4. [`docs/07-mcp.md`](docs/07-mcp.md) and [`docs/08-rag.md`](docs/08-rag.md) for the layer that's most novel.

---

## The non-negotiable constraints

These cut across every phase. They're encoded as static tests in [issue #2 §16](https://github.com/alleboudy/codescribe/issues/2) and [issue #4 §15](https://github.com/alleboudy/codescribe/issues/4)'s scoring rubrics, so violations show up in CI:

- **No cloud anything.** No remote embedding APIs, no hosted vector stores, no cloud GPUs. Outbound HTTP is allowlisted to HuggingFace Hub (downloads), PyPI (`uv sync`), GitHub releases (llama.cpp), and your internal Perforce + Bugzilla servers — nothing else.
- **No telemetry.** No `wandb`, no Sentry, no `huggingface-cli upload`. Logs go to stdout/stderr/files; never to a remote sink.
- **Network defaults to `127.0.0.1`.** Any default of `0.0.0.0` is a build break.
- **`uv` is the dependency manager.** Never `pip install` in docs or scripts.
- **Python 3.12 only.** Type hints throughout.
- **No `print(...)` in library code.** Logger only.

These aren't preferences. They're how we know the stack actually stays local.

---

## Implementer (intern or Copilot) expectations

Per issue [#2](https://github.com/alleboudy/codescribe/issues/2) — this repo doubles as the spec for an **intern vs. GitHub Copilot Coding Agent build-off**. Each phase is filed as its own GitHub issue and assigned to `@copilot`; the intern path runs in parallel. The scoring rubric in [#2 §16](https://github.com/alleboudy/codescribe/issues/2) measures:

- CI green on first push (per phase).
- Anti-pattern intro count (target: zero of each banned pattern).
- AGENTS.md fidelity (byte-diff against this repo's verbatim text).
- Wall-clock per phase (reported, not scored).
- Operator-verified end-to-end `task_mean`.
- Surprise / conflict surfacing in PR descriptions.

The scoring is deliberately CI-testable: the goal is to produce a working stack, not just plausible-looking code.

---

## What's NOT in this repo

- **Concrete deliverables.** This is planning + documentation. No `pyproject.toml`, no Python packages. Those get produced by the implementer.
- **Project-specific names.** Issues use `<project>` and `<repo>` placeholders throughout. If you want a concrete project-specific instance, the issues describe how to file phase issues in your own repo.
- **Cloud-coupled options.** GitHub Copilot Chat / Cursor / hosted vector stores are mentioned in the docs for context, but every recommendation defaults to local-only.

---

## License

Everything here is dedicated to the public domain under [CC0 1.0 Universal](LICENSE) — use it, quote it, translate it, build on it, no permission or attribution required (a credit to Ahmad Alleboudy is appreciated, never expected).
---

## Where to start

If you're reading this for the first time: **[`docs/01-overview.md`](docs/01-overview.md)**.
