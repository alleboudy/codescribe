# Copilot global instructions

This repo is `codescribe_rag` — a read-only Perforce+Bugzilla RAG indexer + MCP server
(codescribe issue #6, the RAG-only path: no fine-tune, no llama-server).

## Hard rules
- Python 3.12. `uv` for dependencies — never `pip install`.
- Type hints throughout; `from __future__ import annotations` at the top of every module.
- No `print(...)` in library code — use `logging.getLogger(__name__)`. Only CLI `__main__` entry points write stdout.
- No cloud calls. Outbound HTTP is allowlisted to exactly the configured Bugzilla host; Perforce is reached via the local `p4` CLI. No telemetry, no wandb.
- Network defaults to 127.0.0.1; never `0.0.0.0`.
- The MCP server's stdout is the JSON-RPC channel — nothing else may write there. Logger → file only.
- Read-only everywhere: no `p4 submit/edit/add/delete`, no Bugzilla POST/PUT/PATCH/DELETE, no DB writes from the server.
- Credentials live in files (`~/.p4tickets`, `~/.config/codescribe_rag/bugzilla.key`), never inline, never logged.

## Where to look for context
- `docs/superpowers/specs/` — the design spec.
- `docs/superpowers/plans/` — the implementation plan (task-by-task).
- `codescribe_rag/**/AGENTS.md` — per-package rules.
- `docs/SETUP-COPILOT.md` — how this server wires into VS Code Copilot Chat.

## Commit style
Conventional commits scoped to the package: `feat(rag.sources): ...`, `feat(rag_server): ...`, `test(...): ...`, `docs(...): ...`.
