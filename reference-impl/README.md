# codescribe_rag — RAG-only MCP server (codescribe issue #6)

A reference implementation of the **RAG-only path**: index a Perforce + Bugzilla
codebase into a local sqlite-vec/FTS5 store of bug↔fix pairs, and serve them to
**GitHub Copilot Chat** (or any MCP harness) through a local MCP server exposing
`find_similar_bugs` and `get_fix_diff`. No fine-tune, no `llama-server`.

> **Not strictly-local.** The indexer and MCP server run on your machine, but
> Copilot Chat ships your prompt and the retrieved bug/diff context to GitHub's
> cloud. Disqualified for NDA/regulated codebases — read
> [`docs/SETUP-COPILOT.md`](docs/SETUP-COPILOT.md) §0 first.

## Architecture

```
Perforce (p4 -G)  ┐
Bugzilla (REST)   ┘→ sources → extract/pair (regex + confidence) → embed + chunk
                     → store (one indices/rag.db: sqlite-vec + FTS5)
                     → hybrid retriever (vec + BM25 + RRF)
                     → MCP stdio server  → VS Code Copilot Chat
```

## Layout

| Path | What |
|---|---|
| `codescribe_rag/rag/sources/` | read-only Perforce (subprocess `p4 -G`) + Bugzilla (REST) clients, rate limiter |
| `codescribe_rag/rag/extract/` | CL/bug regex link extraction + confidence-scored pairing |
| `codescribe_rag/rag/store/` | `schema.sql`, sqlite-vec/FTS5 writer, hybrid retriever, types |
| `codescribe_rag/rag/embed/` | bge-large `Embedder` + numpy-only `HashingEmbedder` + diff chunker |
| `codescribe_rag/rag/pipelines/` | streaming bootstrap + incremental indexer |
| `codescribe_rag/rag/__main__.py` | `index` / `query` / `status` CLI |
| `codescribe_rag/servers/rag_server/` | MCP stdio server (`find_similar_bugs`, `get_fix_diff`) |
| `configs/rag.yaml` | indexer + retriever config | 
| `configs/mcp.json` | canonical MCP server block to copy into your harness |
| `docs/SETUP-COPILOT.md` | **step-by-step Copilot Chat wiring** |

## Install

```bash
uv sync --extra rag --extra mcp --extra dev      # test/dev env (lean: no torch)
uv sync --extra rag --extra mcp --extra embed    # add the production bge-large embedder
```

Extras: `rag` = sqlite-vec; `mcp` = MCP SDK; `embed` = sentence-transformers (heavy,
production only); `dev` = pytest. The test suite uses the numpy-only
`HashingEmbedder`, so it needs neither torch nor a downloaded model.

## Test

```bash
uv run pytest -m "not slow"     # fast suite (~90 tests)
uv run pytest -m slow           # 100K-vector p99 retrieval-latency check
uv run pytest                   # everything
```

Static safety checks are part of the suite: read-only sources, no `shell=True`, no
`p4python`, egress allowlist, MCP stdout discipline, no unsafe serialisation.

## Run

```bash
# index (after editing configs/rag.yaml + Day-0 auth in docs/SETUP-COPILOT.md)
uv run python -m codescribe_rag.rag --config configs/rag.yaml index --bootstrap
uv run python -m codescribe_rag.rag --config configs/rag.yaml query --query "NPE on startup" --k 5
uv run python -m codescribe_rag.rag --config configs/rag.yaml status

# MCP server (normally spawned by the harness; inspect it directly with:)
npx @modelcontextprotocol/inspector uv run python -m codescribe_rag.servers.rag_server
```

Then wire into Copilot Chat per [`docs/SETUP-COPILOT.md`](docs/SETUP-COPILOT.md).

## Design & plan

- Spec: `docs/superpowers/specs/2026-05-23-codescribe-rag-only-mcp-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-23-codescribe-rag-only-mcp.md`
