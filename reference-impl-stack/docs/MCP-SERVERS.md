# MCP servers — Model Context Protocol

What MCP is, what shape it has on the wire, what kinds of servers are useful, and how to wire one into the harness in this project.

> **New to the project?** Start with [`CONCEPTS.md §3`](CONCEPTS.md#3-mcp--letting-the-agent-actually-do-things) for the 30,000-foot view of where MCP fits alongside fine-tuning and RAG, then come back here for the depth.

---

## 1. The headline

**MCP (Model Context Protocol)** is an open JSON-RPC protocol designed by Anthropic in 2024 for letting LLM-based agents call tools and read resources hosted by external **servers**. The agent (the *client*) connects to one or many MCP servers and discovers their capabilities at runtime.

Think of MCP as **the LSP of agent tools**. LSP (Language Server Protocol) standardised "what can my editor ask a language toolchain?" so editors and toolchains could mix freely. MCP standardises "what can an agent ask a tool provider?" so harnesses (claw-code, an agent CLI, etc.) and tool servers (your project-specific helpers) can mix freely.

Three kinds of capabilities a server can expose:

| Capability | What it is | Example |
|---|---|---|
| **Tools** | Side-effecting (or read-only) actions the model can invoke with structured arguments | `run_tests(path: str)`, `query_db(sql: str)` |
| **Resources** | Read-only typed data the agent can list and fetch | `sample://specs/SP-042`, `metric://delivery_reliability` |
| **Prompts** | Pre-canned prompt templates the agent can render | `review_pr(pr_number)`, `explain_metric(metric_name)` |

In practice, **tools** are the workhorse. Resources are nice for "show me this file" patterns. Prompts are mostly for power-user editors that surface them as slash commands.

---

## 2. Wire format — JSON-RPC over stdio (or HTTP)

MCP is JSON-RPC 2.0 with a fixed schema. The default transport is **stdio**: the server is a subprocess, requests come on its stdin, responses go to its stdout, log messages to stderr. There's also an HTTP transport (recently a streaming SSE variant), but stdio is what most servers use because it's the simplest secure design — no port to expose, no authentication to handle.

A minimal session looks like:

```
client (claw)                          server (your subprocess)
  │                                          │
  │  initialize(version, capabilities)       │
  │ ────────────────────────────────────────▶│
  │                                          │
  │  ◀───────────────── result(server info) │
  │                                          │
  │  tools/list                              │
  │ ────────────────────────────────────────▶│
  │                                          │
  │  ◀──────── result(name, schema, …)      │
  │                                          │
  │  tools/call(name="run_tests",            │
  │    arguments={"path": "tests/"})         │
  │ ────────────────────────────────────────▶│
  │                                          │
  │  ◀──── result(content=[{type:"text",     │
  │            text: "PASSED 42/42"}])       │
  │                                          │
```

The client decides which tools to expose to the LLM as available actions. The LLM emits tool-call requests; the client routes them to the right server; results come back as plain text (or images, audio, etc. via typed content blocks) and become part of the conversation history.

---

## 3. The server side — what you build

A server is a process that:

1. **Reads JSON-RPC requests on stdin.**
2. **Writes JSON-RPC responses on stdout.**
3. **Implements at least these four methods:**
   - `initialize` — handshake; return your protocol version + capabilities
   - `tools/list` — return the list of tool names + JSON-Schema input schemas
   - `tools/call` — execute a tool with the given args; return content
   - `resources/list` and `resources/read` — if you expose resources

You can write one in any language; the official SDKs are:

| Language | Repo |
|---|---|
| Python | [`modelcontextprotocol/python-sdk`](https://github.com/modelcontextprotocol/python-sdk) |
| TypeScript | [`modelcontextprotocol/typescript-sdk`](https://github.com/modelcontextprotocol/typescript-sdk) |
| Rust | community-maintained; e.g. `rmcp` |
| Go | community-maintained; e.g. `mcp-go` |

For this project's local-only stance, **Python** is the default — the SDK is small, dependency-light, and integrates cleanly with our existing tooling.

A canonical Python server skeleton (using the official SDK):

```python
# servers/sample_specs/server.py
from mcp.server.fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP("sample-specs")
SPECS_DIR = Path("../your-repo/docs/specs/")

@mcp.tool()
def find_specs(query: str, k: int = 5) -> list[dict]:
    """Search sample's spec registry by free-text query.

    Returns up to `k` matching specs as a list of {id, title, path, snippet}.
    """
    # implementation: BM25 over docs/specs/*.md or simple grep
    ...

@mcp.resource("sample://specs/{spec_id}")
def get_spec(spec_id: str) -> str:
    """Return the full Markdown content of one spec by SP-NNN id."""
    spec_path = SPECS_DIR / f"{spec_id}.md"
    if not spec_path.is_file():
        raise ValueError(f"Unknown spec: {spec_id}")
    return spec_path.read_text(encoding="utf-8")

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

The decorators do all the JSON-Schema generation, request routing, and stderr-logging plumbing. The server is run by the client as a subprocess: `python -m servers.sample_specs.server`.

---

## 4. The client side — claw-code's MCP support

`vendor/claw-code/` (the project's vendored Rust agent harness) supports MCP servers via a config file in the workdir. The platform porter (`scripts/_porter_lib.sh`) generates the per-host `.claw/settings.json` MCP registry from `configs/harness/<repo>.yaml`; the `codescribe-train run` orchestrator's `ClawCodeHarness.prepare()` separately renders the per-session `.claude.json` / `.claw.json`.

The shape of the `.claw/settings.json` registry:

```json
{
  "mcpServers": {
    "sample-specs": {
      "command": "python",
      "args": ["-m", "servers.sample_specs.server"],
      "env": {
        "TARGET_REPO": "$HOME/repos/sample"
      }
    },
    "sample-db-schema": {
      "command": "python",
      "args": ["-m", "servers.sample_db_schema.server"],
      "env": {
        "DATABASE_URL": "postgresql://..."
      }
    }
  }
}
```

When claw-code starts, it spawns each listed server as a subprocess, runs the `initialize` handshake, lists their tools, and exposes them to the model as available actions. The model can then call them by name.

---

## 5. Patterns that work

### 5.1 Read-only query servers

The cheapest, safest pattern. The server has read access to a system (DB, filesystem, API) and exposes well-defined queries. Examples:

- **`sample-db-schema`** — lists tables, columns, types, foreign keys. No writes.
- **`sample-import-fields`** — given an import field name, returns the documented mapping from `docs/IMPORT_FIELDS.md`.
- **`sample-issues`** — given an issue number, returns title + body + labels. Cached at server start.

The agent uses these for **lookups**: "what's the type of `orders.shipping_cost_cents`?" → the model calls the tool, gets the answer, doesn't have to remember.

### 5.2 Computational servers

The server runs a calculation on demand. Examples:

- **`sample-metrics-calc`** — given an `order_id`, runs the actual analytics pipeline and returns the score breakdown. Useful for "explain what your scoring would be on this order".
- **`sample-test-runner`** — given a test path, runs `pytest` on it, returns pass/fail + stdout. The agent can iterate on a fix and re-test.
- **`repo-grep`** — fast `rg` over the repo with stuctured output.

These are powerful but mind the **side-effect surface**. A test-runner can use up cycles; a "scoring server" might be slow on big orders. Cap inputs.

### 5.3 Spec / docs servers

The server provides a search-and-retrieve interface over a static doc corpus. Often a thin wrapper around a RAG index ([`docs/RAG.md`](RAG.md)).

- **`sample-specs`** — search SP-NNN specs by query, retrieve full text by ID.
- **`sample-lessons`** — search L-NNN lessons by query, retrieve full text by ID.
- **`sample-architecture`** — search the ARCHITECTURE.md / DEBUGGING.md / etc.

These are the bridge between RAG and MCP: the *retrieval* is internal; the *interface* is JSON-RPC.

### 5.4 Editing servers (with care)

The server can apply changes — `git apply patch`, `sed` substitution, file write. Examples:

- **`sample-migrations`** — given a column spec, generates a new Alembic migration file (read-write to `backend/alembic/versions/`).
- **`sample-config-edit`** — surgical edit of `core/config.py` `Settings` fields.

Editing servers expand the agent's capabilities considerably but also expand the failure modes. Defaults that help:

- **Always atomic.** Write to a temp file, fsync, atomic rename. Don't half-write a file.
- **Always reversible.** Every edit produces a patch the agent can `git apply -R` if needed.
- **Always inside the workdir.** Reject paths outside `os.path.realpath(workdir)`.

---

## 6. Strictly-local considerations

Everything an MCP server can do, it can do without the network — and that's the right default in this project.

- **Stdio transport, not HTTP.** No port to expose, no auth tokens to leak.
- **Subprocess of claw-code.** Lifetime tied to the harness session; dies cleanly when the session ends.
- **Inside the Docker sandbox.** When running `--sandbox docker --network none`, the MCP servers run inside the same container; they can't reach the LAN either.
- **Audit on add.** Every new server is a new code path; treat new servers like new vendored components — review code, audit for `requests.get` / `urllib.request` / shell calls to `curl`, document in `vendor/SECURITY-NOTES.md` or a sibling MCP-NOTES if it grows large.

The standard "no telemetry" env vars (`DO_NOT_TRACK=1`, `HF_HUB_DISABLE_TELEMETRY=1`) propagate from the harness's environment to MCP server subprocesses by default. If you write a server that downloads anything at first run (e.g., a sentence-transformers embedding model for a RAG-as-MCP server), respect those.

---

## 7. When NOT to write an MCP server

- **A bash one-liner would do.** The agent already has a shell tool (most harnesses provide one); if your "tool" is just `git log -n 5`, don't formalise it.
- **A static doc with retrieval would do.** If the data is read-only and queries are mostly free-text, a RAG index is simpler ([`docs/RAG.md`](RAG.md)).
- **The action is too dangerous to allow without per-call confirmation.** Things like `rm -rf` or "send Slack message" need stronger gating than MCP's tool-call interface provides. Use the harness's per-tool permission system instead, or run the action by hand.
- **You don't know what the tool surface should look like.** Design tools by watching the agent fail without them. Premature MCP servers add maintenance burden.

---

## 8. A worked example — `sample-metrics`

The plan: an MCP server that exposes sample's analytics metrics in two ways — by-name lookup of formulas, and by-order runtime computation.

Tool surface:

```python
@mcp.tool()
def list_metrics() -> list[dict]:
    """List all 9 analytics metric dimensions with one-line summaries."""

@mcp.tool()
def get_metric_formula(metric_name: str) -> str:
    """Return the full Markdown documentation for one metric from docs/METRICS.md."""

@mcp.tool()
def compute_metric_for_order(order_id: int, metric_name: str) -> dict:
    """Run the actual scoring pipeline on an order's data; return {score, weights, components}."""
```

Resources:

```python
@mcp.resource("metric://{metric_name}")
def metric_resource(metric_name: str) -> str:
    """Same content as get_metric_formula but exposed as a resource for editor pickers."""
```

Configuration:

```json
{
  "mcpServers": {
    "sample-metrics": {
      "command": ".venv/bin/python",
      "args": ["-m", "servers.sample_metrics.server"],
      "env": {
        "TARGET_REPO": "$HOME/repos/sample",
        "DATABASE_URL": "postgresql://localhost/sample_dev"
      }
    }
  }
}
```

The agent then knows about three new actions. When asked "how is `delivery_reliability` scored?", it calls `get_metric_formula("delivery_reliability")` and quotes the result. When asked "what would `on_time_delivery_rate` score on order 4321?", it calls `compute_metric_for_order(4321, "on_time_delivery_rate")`.

---

## 9. Building, testing, deploying

### Build

```bash
.venv/bin/python -m pip install "mcp>=1.0"
mkdir -p servers/sample_specs
# write servers/sample_specs/server.py + __main__.py
```

### Test (offline, no agent)

The MCP SDK ships with a CLI that drives a server interactively over stdio:

```bash
.venv/bin/python -m mcp.cli servers/sample_specs/server.py
> /tools                       # list tools
> /call find_specs "rate limiting"
> /resource sample://specs/SP-042
```

For CI: write pytest cases that import the server module and call its tool functions directly (the decorators leave the underlying functions callable).

### Wire into the harness

Edit `configs/harness/sample.yaml`:

```yaml
mcp_servers:
  - name: sample-specs
    command: .venv/bin/python
    args: [-m, servers.sample_specs.server]
    env:
      TARGET_REPO: $HOME/repos/sample
```

The platform porter writes this list into the per-host `.claw/settings.json` MCP registry.

---

## 10. Further reading

- [Model Context Protocol introduction](https://modelcontextprotocol.io/introduction) — the official docs, with the protocol spec and tutorials
- [`modelcontextprotocol/python-sdk`](https://github.com/modelcontextprotocol/python-sdk) — the Python SDK we use
- [JSON-RPC 2.0 spec](https://www.jsonrpc.org/specification) — the underlying wire format
- [LSP](https://microsoft.github.io/language-server-protocol/) — the Language Server Protocol that MCP draws inspiration from. Same problem, different domain.
- [Anthropic blog post announcing MCP](https://www.anthropic.com/news/model-context-protocol) — the original framing
