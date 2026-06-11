# `.claw/settings.json` — the per-host MCP registry consumed by `claw`

The vendored `claw` binary reads its MCP-server registry and permission gate from
`.claw/settings.json` in the workdir (claw's config discovery merges
`.claw.json` / `.claw/settings.json` / `settings.local.json` — **not** `.claude.json`).

The platform porter (`scripts/bootstrap_linux.sh` / `bootstrap_mac.sh` /
`bootstrap_windows.ps1`) **generates** this file per-host — it is **gitignored, not
committed** — because it uses **absolute** paths (the venv python, the repo's
`indices/` and `logs/`) so claw resolves them from any cwd. `claw` rejects unknown
keys, so the file is strict JSON (no `_comment` or other annotation fields), which is
why the explanation lives here rather than inline.

## Schema

```jsonc
{
  "permissions": { "defaultMode": "dontAsk" },
  "mcpServers": {
    "<name>": {
      "command": "<binary on PATH>",
      "args": ["..."],
      "env": { "...": "..." }
    }
  }
}
```

`permissions.defaultMode` is claw's permission gate: `"dontAsk"` = full tool access
with no per-tool prompts; `"default"` = require interactive approval; `"acceptEdits"`
= allow edits but gate the rest.

Each server is wrapped through `codescribe_train.servers._mcp_framing_bridge`, which
translates between the MCP SDK's newline-delimited stdio and claw's LSP-framed stdio.

## Generated registry

The porter writes three servers:

| Server | Module | Tools | Env |
|---|---|---|---|
| `repo-rag` | `codescribe_train.servers.rag_server` | `find_similar_issues`, `get_pr_diff`, `search_commits`, `search_docs` | `RAG_DB_PATH`, `RAG_LOG_PATH` |
| `repo-grep` | `codescribe_train.servers.repo_grep` | `grep(pattern, path, language, max_results)` | `TARGET_REPO` |
| `repo-docs` | `codescribe_train.servers.repo_docs` | doc lookup scoped to `TARGET_REPO` | `TARGET_REPO` |

## Refreshing

The file is regenerated, never hand-edited. After pulling new code on a host, re-run
the platform porter (`bash scripts/bootstrap_linux.sh` / `bootstrap_mac.sh`) to
rewrite `.claw/settings.json`.
