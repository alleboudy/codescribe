# 07 — MCP: Model Context Protocol

## What MCP is, in plain language

**MCP** (Model Context Protocol) is a standardised wire protocol for connecting LLM applications (the *client*) to external tools, resources, and data sources (the *server*). Anthropic published the open spec in late 2024; it has since been adopted by:

- Claude Desktop, Claude Code (Anthropic)
- Cursor (Cursor team)
- Continue.dev (Continue team)
- VS Code Copilot Chat (Microsoft, in preview builds)
- claw-code (ultraworkers, vendored in this stack)
- A growing ecosystem of community MCP servers

Think of MCP as **the USB of LLM tooling**: any device (server) that speaks USB works with any host (client) that speaks USB. Before USB, every peripheral needed a custom port; before MCP, every harness needed custom tool integrations.

## The problem MCP solves

Before MCP existed (and still, for harnesses that don't support MCP yet), if you wrote a "find similar bugs" tool you had to integrate it separately with:

- VS Code Copilot Chat (their custom participants/extensions API).
- Cursor (their custom command system).
- Continue.dev (their context-provider plugin model).
- claw-code (OpenAI-style function definitions in `.claude.json`).
- Aider (a Python plugin in their plugin dir).

Four+ integrations for the same logical tool. With MCP, you write **one server**, and every MCP-compatible client picks it up via auto-discovery.

## The two roles: client and server

- **MCP Client** — the harness (the thing running the LLM session). The client *discovers* available servers and *dispatches* the model's tool calls to them.
- **MCP Server** — a separate process (Python, Node, Go, Rust, anything) that *exposes* tools/resources/prompts and *responds* to discovery and invocation requests.

A typical configuration has one client and 0..N servers. Each server is its own process; they're spawned by the client when a session starts.

## What an MCP server can expose

Three primitives, all optional. A server can implement any combination.

### 1. Tools — callable functions

The most common primitive. Each tool has:

- A **name** (e.g., `find_similar_bugs`).
- A **description** (free-text; the model sees this to decide whether to call).
- An **inputSchema** (JSON Schema describing the argument shape).
- A **handler** in the server's code that produces a response.

Tool responses come back as one or more content blocks (text, image, embedded resource). For our RAG use case, all tool responses are text (formatted Markdown).

### 2. Resources — read-only data

A way for the client to fetch named data items from the server. Example: a server might expose "all open bugs assigned to me" as a resource at `bugs://assigned/me`. The client fetches it via a `resources/read` call.

Less commonly used than tools. For this stack's RAG MCP server, we use only tools.

### 3. Prompts — reusable templates

A server can expose pre-canned prompt templates the user can invoke. Example: a "summarise PR" prompt that takes a PR number, renders into a full system+user message, and is sent to the model. Useful for client UIs that want to surface "slash commands" mapped to server-provided templates.

Also less common in production. We don't use this in the RAG MCP server.

## Transports

The protocol is layered over a transport. Three are spec'd:

| Transport | When to use |
|---|---|
| **stdio** (most common) | Local server spawned as a child process by the client. Stdin/stdout carry JSON-RPC frames; stderr is the server's log channel. This is what we use. |
| **HTTP + SSE** (Server-Sent Events) | Remote server reachable over a network. Used for hosted MCP servers (less common in production as of 2026). |
| **WebSocket** | Less common; mostly for browser-based MCP clients. |

stdio is dramatically simpler — no port management, no TLS, no auth headers. The client launches `python -m my_mcp_server`, pipes stdin/stdout, and JSON-RPC frames flow back and forth.

## The wire protocol: JSON-RPC 2.0

Every MCP message is a JSON-RPC 2.0 object:

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
```

Requests carry an `id`; responses carry the same `id` so the client can match them. Notifications (one-way messages) omit `id`. Errors come back with an `error: {code, message}` field.

For stdio, each JSON-RPC message is sent as a single **line** of text (no framing prefix, no length header). The newline separates messages. Some implementations support Content-Length headers (LSP-style); the Python SDK does both.

## Lifecycle of an MCP session

```
1. Client launches the server as a child process via stdio.

2. Initialize handshake (JSON-RPC):
   Client → {"id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-03-26", "capabilities": {...},
                        "clientInfo": {"name": "claw", "version": "0.1.0"}}}
   Server → {"id": 1, "result": {"serverInfo": {...}, "capabilities": {...}}}

3. Initialized notification (no response):
   Client → {"method": "notifications/initialized"}

4. Discovery:
   Client → {"id": 2, "method": "tools/list"}
   Server → {"id": 2, "result": {"tools": [
     {"name": "find_similar_bugs", "description": "...", "inputSchema": {...}},
     {"name": "get_fix_diff",      "description": "...", "inputSchema": {...}}
   ]}}

5. The client now knows what tools exist. It surfaces them to the model as
   OpenAI-style function specs in the model's next prompt.

6. Tool call (triggered when the model decides to invoke one):
   Client → {"id": 42, "method": "tools/call",
             "params": {"name": "find_similar_bugs",
                        "arguments": {"query": "NPE on startup", "k": 5}}}
   Server → {"id": 42, "result": {"content": [{"type": "text", "text": "..."}]}}

7. The client injects the result back into the model's next turn as a
   `tool`-role message. Model generates a grounded response.

8. Shutdown:
   Client → {"id": 99, "method": "shutdown"}
   Server → {"id": 99, "result": null}
   Client closes stdin. Server's stdin EOF triggers a clean exit.
```

## How MCP relates to OpenAI function calling

These are different layers:

- **OpenAI function calling** is the *in-chat* convention — how the model expresses "call this function" inside a chat-completions response (`tool_calls: [...]`), and how the result is fed back (`role: "tool"`).
- **MCP** is the *discovery and transport* layer — how the harness finds out which functions exist and ferries calls between the model and external processes.

In a typical flow:

```
User prompt
   ↓
Harness prepares request (with MCP-discovered tool defs in the `tools` array)
   ↓
LLM responds with tool_calls JSON (OpenAI function-calling format)
   ↓
Harness extracts tool_calls, routes via MCP `tools/call` to the right server
   ↓
MCP server executes, returns content
   ↓
Harness injects into next turn as `tool`-role message
   ↓
LLM generates grounded response
```

The model never knows anything about MCP. It only sees OpenAI-style tool definitions. The harness handles the MCP plumbing.

## Why build an MCP server (instead of tools inside the harness)

Three reasons:

1. **Reusability across harnesses.** Write the server once; it works in claw, Cursor, Continue, Claude Code, and any future MCP-compatible harness.
2. **Process isolation.** The server is a separate process. If it crashes, the harness is unaffected. If it has stateful resources (a vector DB connection, an open file), they're cleanly bounded.
3. **Security.** The server runs with whatever permissions you grant its process. The harness can be restrictive about which servers it loads, in what mode.

The trade-off: a tiny latency overhead (each tool call is a JSON-RPC round-trip through stdio). For local stdio transport on a modern laptop, this is ~1 ms per call — negligible compared to model inference time.

## Python SDK quickstart

The official Python MCP SDK lives at `pypi.org/project/mcp` (`pip install mcp` or `uv add mcp`). Bare-minimum server:

```python
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as mt


server = Server("hello-mcp")


@server.list_tools()
async def list_tools() -> list[mt.Tool]:
    return [mt.Tool(
        name="echo",
        description="Echo a string back to the caller.",
        inputSchema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mt.TextContent]:
    if name == "echo":
        return [mt.TextContent(type="text", text=f"You said: {arguments['text']}")]
    raise ValueError(f"unknown tool: {name}")


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.get_init_options())


if __name__ == "__main__":
    asyncio.run(main())
```

Save as `hello_mcp.py`. Run via stdio:

```bash
python hello_mcp.py
```

Then test with the MCP Inspector (see below).

## Critical: stdout is the JSON-RPC channel

The most common bug in MCP servers: a stray `print(...)` somewhere in your code corrupts the stdout stream. The client sees a JSON parse error and the server appears to be silently broken. **Configure logging to write to a file ONLY**, before any other code runs. From issue [#4](https://github.com/alleboudy/llm-finetuner/issues/4) §13.5's `rag_server/AGENTS.md`:

> stdout is the JSON-RPC channel. Nothing else goes there. Ever.
> - `logger` is configured to write ONLY to a file.
> - `sys.stdout` is reserved for the MCP SDK; do not `print(...)` anywhere in the package.
> - Static check `test_no_stdout.py` greps for `print(` outside `__main__.py` guards.

Even worse: some libraries (transformers, certain numpy paths) install a `StreamHandler` on the root logger at import time. If your MCP server imports them after configuring its own logger, the StreamHandler can sneak in and start writing to stdout. The fix is to **explicitly remove all handlers from the root logger and install a FileHandler at the very top of your `__main__.py`**, before any other imports. See [issue #5](https://github.com/alleboudy/llm-finetuner/issues/5) §8 for the exact code.

## Testing MCP servers

### MCP Inspector

Official tool from the spec authors. Browser-based UI that connects to a server over stdio, shows the live JSON-RPC traffic, and lets you hand-call tools:

```bash
npx @modelcontextprotocol/inspector python hello_mcp.py
```

Opens a localhost web UI. You can:
- See the `initialize` handshake.
- Browse `tools/list` schemas.
- Invoke tools with arbitrary arguments.
- See JSON-RPC frames in both directions.

**Use this every time you change a tool schema.** It catches issues before the harness ever sees the server.

### In-process tests

For automated tests, the Python SDK provides an in-process transport that avoids the stdio subprocess:

```python
from mcp.shared.memory import create_connected_server_and_client_session


async def test_call_tool():
    async with create_connected_server_and_client_session(server) as session:
        await session.initialize()
        result = await session.call_tool("echo", {"text": "hi"})
        assert "You said: hi" in result.content[0].text
```

Fast; no subprocess; useful for CI. See [issue #5](https://github.com/alleboudy/llm-finetuner/issues/5) §11 for fixture patterns.

## Wiring an MCP server into a harness

Each harness has its own config file format but they all converge on roughly the same JSON shape under a `mcpServers` key.

### `claw` — `.claw-mcp.json`

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["-m", "my_server"],
      "env": {"DB_PATH": "/abs/path/to/db.sqlite"}
    }
  }
}
```

### VS Code Copilot Chat — settings.json (preview)

```json
{
  "github.copilot.chat.mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["-m", "my_server"]
    }
  }
}
```

### Cursor — `.cursor/mcp.json`

Same shape as claw's `.claw-mcp.json`.

### Continue.dev — `~/.continue/config.json`

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["-m", "my_server"]
    }
  }
}
```

### Claude Desktop / Claude Code — `claude_desktop_config.json`

OS-specific path. Same shape under `mcpServers`.

**Practical tip**: pick ONE canonical file in your project repo (e.g., `configs/mcp.json`) and have your orchestrator copy it to each harness's expected location. Avoids drift.

## Security considerations

MCP servers are arbitrary processes. The protocol itself doesn't validate anything. Things to think about:

- **Tool semantics are honour-based.** A tool named `read_file` could secretly delete files; the protocol won't stop it. Use only servers you trust (or audit yourself).
- **The harness controls invocation, not the server.** Permission modes (read-only, etc.) are enforced at the harness layer. If your harness is too permissive, a malicious or buggy server can cause damage.
- **Servers run in the user's process tree.** Same file permissions as the user. Don't run servers you got from a random GitHub gist without reading them.
- **stdio transport has no auth.** Anything that can write to your server's stdin can invoke any tool. This is fine for local stdio (your harness owns the process); be more careful with HTTP+SSE.

For this stack's RAG MCP server, all tools are **read-only**: `find_similar_bugs` and `get_fix_diff` only read from the local SQLite DB. No mutations. No file writes. No network calls beyond the DB. This is the safest possible tool surface.

## The ecosystem (servers worth knowing)

A few existing MCP servers you can learn from or directly use:

- **`@modelcontextprotocol/server-filesystem`** — Expose a directory as a read-only resource set.
- **`@modelcontextprotocol/server-github`** — GitHub API (issues, PRs, search) as tools. (Useful for issue [#4](https://github.com/alleboudy/llm-finetuner/issues/4)'s concrete-project sibling case if you go that route.)
- **`@modelcontextprotocol/server-postgres`** — Read-only SQL access to Postgres.
- **`@modelcontextprotocol/server-puppeteer`** — Browser automation tools.
- Community: a growing list at https://github.com/modelcontextprotocol/servers

## Further reading

- MCP spec: https://modelcontextprotocol.io/specification/
- Python SDK: https://github.com/modelcontextprotocol/python-sdk
- TypeScript SDK: https://github.com/modelcontextprotocol/typescript-sdk
- MCP Inspector: https://github.com/modelcontextprotocol/inspector
- Community server list: https://github.com/modelcontextprotocol/servers
- Anthropic's launch announcement: https://www.anthropic.com/news/model-context-protocol
- Issue [#4 §13](https://github.com/alleboudy/llm-finetuner/issues/4) of this repo has the deep dive applied to our RAG use case.
- Issue [#5 §8](https://github.com/alleboudy/llm-finetuner/issues/5) has the concrete Python skeleton with the logger-isolation pattern.
