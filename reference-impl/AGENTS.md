# codescribe_rag — RAG-only stack

Read-only indexer (Perforce + Bugzilla) → sqlite-vec/FTS5 store → MCP server for Copilot Chat.
This is the RAG-only path of codescribe issue #6: no fine-tune, no llama-server.

## Cross-cutting hard rules
- Python 3.12; `uv` for deps (never `pip install`); type hints throughout.
- No cloud calls except the configured Perforce + Bugzilla hosts. No telemetry, no wandb.
- Network defaults to 127.0.0.1. No `0.0.0.0` defaults.
- No `print(...)` in library code — logger only. CLI entry points (`__main__`) may write stdout/stderr.
- Credentials live in files (`~/.p4tickets`, `~/.config/codescribe_rag/bugzilla.key`), never inline, never logged.
- Per-package rules live in each package's `AGENTS.md`. The design spec is in `docs/superpowers/specs/`.
