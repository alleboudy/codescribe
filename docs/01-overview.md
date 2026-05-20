# 01 — The big picture

## What problem this stack solves

You have a private codebase. It's substantial — tens of thousands of files, years of history, lots of project-specific patterns. You'd like an AI coding assistant that *actually knows your code* — names your modules correctly, follows your project's idioms, knows where the bodies are buried. And you can't send any of it to a cloud LLM because the code is under NDA / customer contract / regulatory constraint / national security / whatever.

There are three ways to give a generic LLM project-specific knowledge:

1. **Prompting** — paste the relevant code into the prompt every time. Works for small ad-hoc questions; doesn't scale; the model's "memory" lives only for the duration of one conversation.
2. **Retrieval-Augmented Generation (RAG)** — at runtime, fetch the most relevant snippets from a local index and inject them into the prompt. Adds knowledge without retraining. See [`08-rag.md`](08-rag.md).
3. **Fine-tuning** — train a copy of the model on your codebase. Bakes knowledge into the weights. See [`02-fine-tuning.md`](02-fine-tuning.md).

This stack does **fine-tuning + RAG together** — fine-tuning encodes the *style* and *broad architecture* of the codebase into a personalised model; RAG fills in *specific historical incidents* (similar bugs, the PRs that fixed them) at query time. The two are complementary: fine-tuning ages with the codebase (it's a snapshot); RAG stays current with whatever you re-index.

## The strictly-local posture

This stack is designed for codebases where **data leaving the host is a failure mode**, not just a preference. Concrete consequences:

- No cloud GPUs. Training runs on your own hardware. Inference too.
- No cloud embedding APIs. Embeddings for RAG run locally on the same GPU.
- No cloud vector stores. SQLite + sqlite-vec is the entire vector layer.
- No telemetry — no Weights & Biases, no Sentry, no HuggingFace upload, no `requests.post` to anywhere that isn't on the explicit allowlist (HuggingFace Hub for *downloads*, your Perforce server, your Bugzilla server).
- Default network bind for any HTTP component: `127.0.0.1`. Binding `0.0.0.0` requires an explicit `unsafe_bind_all=True` flag and triggers static-test warnings.

Issue [#1](https://github.com/alleboudy/llm-finetuner/issues/1) operator runbook documents how to verify this posture via an egress audit (`strace -e trace=connect` reports zero non-loopback `connect()` calls during a session).

## The four phases

The stack is decomposed into four packages that depend on each other in strict order:

```
data → train → serve → harness → (RAG, MCP server)
```

1. **`data/`** — Any source tree (Git working tree, Perforce workspace, or plain directory) → JSONL train/val/test splits. The training-data pipeline walks files; the source's version-control system is incidental, used only when present for things like commit-SHA metadata in the manifest. See [`02-fine-tuning.md § Training data`](02-fine-tuning.md#training-data-completion-vs-fim-vs-instruction).
2. **`train/`** — QLoRA fine-tune via Unsloth, eval, GGUF export. See [`02-fine-tuning.md`](02-fine-tuning.md) end-to-end.
3. **`serve/`** — Vendored `llama.cpp` with an OpenAI-compatible HTTP server. See [`04-inference.md`](04-inference.md).
4. **Harness** — Whatever interactive coding agent the operator picks: `claw`, `aider`, `continue.dev`, etc. The harness is *bring-your-own*; this stack only guarantees an OpenAI-compatible endpoint. See [`05-harnesses.md`](05-harnesses.md).

Layered on top:

5. **RAG** — Local indexer over Perforce + Bugzilla (or any other code+issue stack), retrieval served as MCP tools. See [`07-mcp.md`](07-mcp.md) and [`08-rag.md`](08-rag.md). Issue [#4](https://github.com/alleboudy/llm-finetuner/issues/4) is the canonical plan.

## Why a fine-tune AND not just RAG (or vice versa)

| Approach | Strengths | Weaknesses |
|---|---|---|
| Stock LLM + prompting | Zero setup | No project knowledge |
| Stock LLM + RAG | Fast updates; knowledge stays current | Retrieval is shallow; model still sounds generic |
| Fine-tune + no RAG | Model sounds like your codebase | Knowledge frozen at training time; doesn't know recent bugs |
| Fine-tune + RAG | Stylised model with up-to-date retrieval | Most build work; two systems to maintain |

For private codebases under sustained development, fine-tune-plus-RAG is the strongest configuration. The fine-tune encodes how the team writes code; RAG keeps it current.

## Hardware envelope

The reference setup is a single laptop with **8 GB VRAM** (the canonical configuration documented across these issues). Specifically tested on:

- **RTX 5070 Laptop GPU** (Blackwell, sm_120, 8 GB GDDR7) — the trainer.
- **GTX 1080 Ti** (Pascal, sm_61, 11 GB GDDR5X) — the auxiliary inference / FA2 cross-build desktop.
- **An 8 GB Ada-class laptop GPU** (sm_89, GDDR6) — the laptop fleet target (see issue [#3](https://github.com/alleboudy/llm-finetuner/issues/3)).

Apple Metal works for *inference* (post a `fix(build): support macOS CPU/Metal fallback in build_llama_cpp.sh` change in this stack's history); it does NOT work for *training* because Unsloth (the training framework) needs CUDA.

If you have less than 8 GB VRAM, you cannot run this stack as-is for the canonical 7B-parameter model. Options:
- Use a smaller model (Qwen 2.5 Coder 1.5B or 3B variants).
- Use the CPU for training (orders of magnitude slower; not practical for serious work).
- Rent a GPU (defeats the local-first purpose).

See [`11-hardware.md`](11-hardware.md) for the VRAM math.

## How the documents fit together

| Doc | When to read it |
|---|---|
| [`02-fine-tuning.md`](02-fine-tuning.md) | Before working on `train/` or reading issue #2's Phase 2 |
| [`03-models.md`](03-models.md) | When picking a base model or wondering why Qwen 2.5 Coder 7B specifically |
| [`04-inference.md`](04-inference.md) | Before working on `serve/` or wondering what `llama-server` does |
| [`05-harnesses.md`](05-harnesses.md) | Before deciding which harness to drive the model with |
| [`06-github-copilot.md`](06-github-copilot.md) | If the implementer of issue #2 is GitHub Copilot |
| [`07-mcp.md`](07-mcp.md) | Before working on issue #4's RAG MCP server |
| [`08-rag.md`](08-rag.md) | Before working on issue #4's retrieval pipeline |
| [`09-perforce.md`](09-perforce.md) | If the source codebase lives in Perforce |
| [`10-bugzilla.md`](10-bugzilla.md) | If the bug tracker is Bugzilla |
| [`11-hardware.md`](11-hardware.md) | When something OOMs or doesn't compile for your GPU |
| [`12-glossary.md`](12-glossary.md) | When you hit a term you don't recognise |
| [`13-further-reading.md`](13-further-reading.md) | When the in-tree docs aren't deep enough |
