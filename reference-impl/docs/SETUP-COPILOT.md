# Connecting `codescribe_rag` to GitHub Copilot Chat (VS Code)

Step-by-step wiring of the local `codescribe_rag` MCP server into VS Code Copilot
Chat, so Copilot can call `find_similar_bugs` and `get_fix_diff` against your
indexed Perforce + Bugzilla history. This is the RAG-only path of codescribe
issue #6.

> Paths below use this checkout's location, `/Users/r0gu3/repos/codescribe/reference-impl`.
> Replace it with your own path everywhere it appears.

---

## ⚠️ 0. Read this first — the privacy trade-off

This path is **not strictly-local**. The indexer and the MCP server run on your
machine, but **Copilot Chat is GitHub's cloud**. When Copilot's planner calls
`find_similar_bugs`, the tool result — bug titles, description excerpts, diff
hunks, CL author names — is injected into Copilot's next prompt and **sent to
GitHub's servers, and onward to whichever model provider Copilot routes to**
(Anthropic / OpenAI / Google, depending on your tier and picker).

- For a **public** codebase, or an internal one where cloud AI tools are already
  approved: fine.
- For an **NDA / contract / regulated** codebase: **do not use this path.** Use the
  full strictly-local stack instead (codescribe issues #2 + #4, harness = `claw`).
- Security-embargoed / customer-data bugs will be uploaded the moment Copilot
  retrieves them. Either exclude them at index time, or don't index them.

There is no per-tool-call privacy gate. Get a written sign-off from whoever owns
codebase confidentiality before adopting this. (Full discussion: issue #6 §3.)

---

## 1. Verify your Copilot tier supports MCP

```bash
code --list-extensions --show-versions | grep -i copilot
```

MCP support shipped in recent VS Code Copilot Chat builds; the **setting key has
drifted** between releases. In VS Code, open the Command Palette
(`Cmd/Ctrl+Shift+P`) → **Preferences: Open User Settings (JSON)** and try whichever
your build exposes (search "mcp" in the Settings UI if unsure):

- `github.copilot.chat.mcpServers`
- `chat.mcp.servers`
- a top-level `mcp.servers` / `"mcp": { "servers": { ... } }` block (newer builds use `.vscode/mcp.json` or a `mcp` settings block)

If your build has no MCP support yet, use a **fallback harness** — the same server
works in all of them (same privacy posture, different vendor):

| Harness | Config file | Key |
|---|---|---|
| Cursor | `.cursor/mcp.json` | `mcpServers` |
| Continue.dev | `~/.continue/config.json` | `mcpServers` |
| Claude Code / Claude Desktop | `claude_desktop_config.json` | `mcpServers` |

All take the same server block shown in §5.

---

## 2. One-time setup (Day 0)

```bash
# a. Install p4 + authenticate (creates ~/.p4tickets)
export P4PORT="ssl:perforce.corp.example.com:1666"
export P4USER="<your-user>"
p4 login
p4 changes -m 3 //depot/main/...        # sanity check

# b. Bugzilla API key (Preferences -> API Keys -> Generate), mode 600
mkdir -p ~/.config/codescribe_rag
printf '%s' "<your-key>" > ~/.config/codescribe_rag/bugzilla.key
chmod 600 ~/.config/codescribe_rag/bugzilla.key

# c. Install deps. 'embed' pulls torch + sentence-transformers (~GBs) and is only
#    needed for real semantic retrieval quality; omit it to start on the fallback
#    hashing embedder (set RAG_EMBED_BACKEND=hashing in §5).
cd /Users/r0gu3/repos/codescribe/reference-impl
uv sync --extra rag --extra mcp --extra embed

# d. Pre-stage the embedding model (~1.3 GB), once
hf download BAAI/bge-large-en-v1.5 --local-dir ~/.hf-models/bge-large-en-v1.5
```

Edit `configs/rag.yaml` to point `perforce.*` and `bugzilla.*` at your real
servers (and set `bugzilla.allowlist_hostname` to that exact host). Verify the
clients reach the real servers before indexing:

```bash
uv run python -m codescribe_rag.rag.sources --config configs/rag.yaml p4-probe --last-cl <recent-cl>
uv run python -m codescribe_rag.rag.sources --config configs/rag.yaml bz-probe --since 2026-01-01
```

---

## 3. Build the index

```bash
cd /Users/r0gu3/repos/codescribe/reference-impl
uv run python -m codescribe_rag.rag --config configs/rag.yaml index --bootstrap
# Hours for a 50K-bug / 200K-CL corpus on CPU embedding; run it overnight.
uv run python -m codescribe_rag.rag --config configs/rag.yaml status   # confirm counts
```

This writes `indices/rag.db` (one file). Back it up with a plain copy.

---

## 4. Verify the server boots (MCP Inspector)

Before wiring into Copilot, confirm the server answers over stdio:

```bash
cd /Users/r0gu3/repos/codescribe/reference-impl
npx @modelcontextprotocol/inspector \
  uv run python -m codescribe_rag.servers.rag_server
```

A browser opens. You should see the `initialize` handshake, two tools under
`tools/list`, and be able to run `find_similar_bugs` with
`{"query": "NPE on startup", "k": 3}` and `get_fix_diff` with `{"cl_number": <n>}`.

(No model installed yet? Add `RAG_EMBED_BACKEND=hashing` to the environment for a
keyword-only smoke: `RAG_EMBED_BACKEND=hashing npx @modelcontextprotocol/inspector uv run python -m codescribe_rag.servers.rag_server`.)

---

## 5. Wire it into VS Code Copilot Chat

Command Palette → **Preferences: Open User Settings (JSON)** → add (using the key
your build exposes, per §1):

```jsonc
{
  "github.copilot.chat.mcpServers": {
    "codescribe-rag": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/Users/r0gu3/repos/codescribe/reference-impl",
        "python", "-m", "codescribe_rag.servers.rag_server"
      ],
      "env": {
        "RAG_DB_PATH": "/Users/r0gu3/repos/codescribe/reference-impl/indices/rag.db",
        "RAG_LOG_PATH": "/Users/r0gu3/repos/codescribe/reference-impl/logs/rag-server.log",
        "RAG_CONFIG": "/Users/r0gu3/repos/codescribe/reference-impl/configs/rag.yaml"
        // add  "RAG_EMBED_BACKEND": "hashing"  if you haven't installed the 'embed' extra + model
      }
    }
  }
}
```

This block is also kept in `configs/mcp.json` (the single source of truth — copy
from there to avoid drift).

**If `uv` isn't on VS Code's PATH**, use the venv's Python directly:

```jsonc
"command": "/Users/r0gu3/repos/codescribe/reference-impl/.venv/bin/python",
"args": ["-m", "codescribe_rag.servers.rag_server"]
```

Then Command Palette → **Developer: Reload Window**. Open Copilot Chat; the tool
picker (or the model-info popover) should now list `find_similar_bugs` and
`get_fix_diff` under `codescribe-rag`.

---

## 6. Keep the index fresh (nightly cron)

```bash
crontab -e
# 0 3 * * * cd /Users/r0gu3/repos/codescribe/reference-impl && \
#   /Users/r0gu3/.local/bin/uv run python -m codescribe_rag.rag --config configs/rag.yaml index \
#   >> logs/cron.log 2>&1
```

`index` without `--bootstrap` reads the stored watermarks and pulls only deltas.

---

## 7. Day-to-day

Ask Copilot Chat a project-specific question, e.g. *"How have we historically
handled NPEs in ConfigLoader?"* Copilot's planner decides whether to invoke
`find_similar_bugs`; the retrieved bug+fix pairs ground its answer. Confirm the
tool actually fired:

```bash
tail -f /Users/r0gu3/repos/codescribe/reference-impl/logs/rag-server.log
```

If Copilot never calls the tool, it may be answering from its own knowledge — try
phrasing that explicitly asks for precedent ("what past bugs/CLs are similar to…").

---

## 8. Measuring whether it helps

There's no built-in lift dashboard (this path has no fine-tune baseline). Pick
10–30 questions whose answers live in your bug/fix corpus, ask each twice — once
with this server removed from settings, once with it enabled — and score whether
the answer names a real historical bug / cites a real fix CL / uses real code
names. If the with-RAG answers don't win clearly, revisit pairing quality
(`configs/rag.yaml` → `pairing.threshold`) or confirm the tool is being called.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Tools don't appear in Copilot | Wrong settings key for your build (§1); window not reloaded; `uv` not on PATH (use the venv-python form in §5). |
| Server "fails to start" | Run the §4 Inspector command in a terminal and read `logs/rag-server.log`. Common: `RAG_DB_PATH` doesn't exist (run §3 first), or `embed` extra/model missing (set `RAG_EMBED_BACKEND=hashing`). |
| Garbage / parse errors | Something wrote to stdout. This server logs to file only; if you added code, keep stdout clean (it's the JSON-RPC channel). |
| `p4`/Bugzilla auth errors during indexing | `p4 login` ticket expired; or Bugzilla key wrong/over-permissioned. Re-auth; the indexer surfaces the error. |
