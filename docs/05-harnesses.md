# 05 — Harnesses: what they are, why you need one, which to pick

## Dafaq is a harness

The fine-tuned model on `llama-server` is just a text completion endpoint. You send it a list of messages; it sends back the next message. It can't:

- Read files from your filesystem.
- Run shell commands.
- Apply edits to your code.
- Browse the web.
- Search for "similar past bugs" in your bug tracker.
- Remember anything between conversations.

A **harness** (also called an *agent*, *agentic framework*, *AI pair programmer*, *coding agent*) is the program that bolts those abilities on. It:

1. Wraps the LLM endpoint in a conversation loop.
2. Defines a set of *tools* the model can call (`read_file`, `bash`, `edit_file`, etc.).
3. Surfaces those tools to the model as function-call definitions (in the OpenAI chat-completions sense).
4. Intercepts the model's tool-call outputs, executes them in the host environment, and feeds the results back into the next prompt.
5. Manages permissions (which tools require human approval; which run automatically).
6. Persists session state (so you can resume a multi-turn debugging session).

The LLM is the brain. The harness is the rest of the body — eyes, hands, memory, judgement-of-action.

## Anatomy of a harness, more precisely

```
┌────────────────────────────────────────────────────────────────────────┐
│                              Harness                                    │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │ User input   │ -> │ Prompt       │ -> │ HTTP to OpenAI-compat    │  │
│  │ ($ claw...)  │    │ assembler    │    │ endpoint (llama-server)  │  │
│  └──────────────┘    │ + system     │    └──────────┬───────────────┘  │
│                      │ prompt +     │               │                   │
│                      │ tool defs +  │               │ model output      │
│                      │ history      │               │ (tool calls JSON) │
│                      └──────────────┘               ▼                   │
│                                                ┌──────────────────────┐ │
│                              ┌─────────────────│  Tool dispatcher     │ │
│                              │   tool results  └──────────────────────┘ │
│                              ▼                                          │
│                      ┌──────────────────────────────────────────────┐  │
│                      │  Tools:                                       │  │
│                      │   • read_file(path)                          │  │
│                      │   • write_file(path, content)                │  │
│                      │   • bash(cmd)                                │  │
│                      │   • find_similar_bugs(query)  ← MCP server   │  │
│                      │   • ...                                       │  │
│                      └──────────────────────────────────────────────┘  │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

When the model decides "I need to read `src/parser.py` to answer this question", it doesn't read the file directly — it can't. It emits a tool call:

```json
{"tool_calls": [{
  "function": {"name": "read_file", "arguments": "{\"path\": \"src/parser.py\"}"}
}]}
```

The harness sees this in the model's response, executes `read_file("src/parser.py")` in the host process, gets back the file content, and includes it in the model's next turn as a `tool`-role message. The model continues reasoning, now grounded in the file content.

## Permission models

Tools come in flavours:

| Mode | Tools allowed | Use case |
|---|---|---|
| **read-only** | `read_file`, `grep`, `list_directory` | Pair-reviewing existing code; no risk of mutation |
| **workspace-write** | + `write_file`, `edit_file`, `apply_patch` (scoped to the cwd) | Active development; can edit code but can't run arbitrary commands |
| **danger-full-access** | + `bash` with no allowlist | Fully autonomous; agent can do anything |

Good harnesses make the user pick explicitly. `claw` defaults to `workspace-write`; the read-only smoke test in [issue #1](https://github.com/alleboudy/llm-finetuner/issues/1) §4 uses `--permission-mode read-only`.

## Tool-call protocol: OpenAI function calling

This is the *wire format* — how the model expresses "I want to call this tool" inside an OpenAI chat-completions response. Standardised by OpenAI in mid-2023; now the de-facto convention every harness uses.

A typical exchange:

1. **Harness's request** (in OpenAI chat-completions JSON):
   ```json
   {
     "model": "qwen-coder-7b-q4_k_m.gguf",
     "messages": [
       {"role": "system", "content": "You are a coding agent. Tools available: ..."},
       {"role": "user",   "content": "What does parse() do in src/parser.py?"}
     ],
     "tools": [
       {"type": "function", "function": {"name": "read_file", "description": "...", "parameters": {...}}},
       {"type": "function", "function": {"name": "bash",      "description": "...", "parameters": {...}}}
     ]
   }
   ```

2. **Model's response** (one or more tool calls):
   ```json
   {"choices": [{"message": {
     "role": "assistant",
     "tool_calls": [{
       "id": "call_001",
       "function": {"name": "read_file", "arguments": "{\"path\": \"src/parser.py\"}"}
     }]
   }}]}
   ```

3. **Harness executes the tool** then sends the result back as a `tool`-role message in the next request:
   ```json
   {
     "messages": [
       ...prior...,
       {"role": "tool", "tool_call_id": "call_001", "content": "def parse(s):\n    return json.loads(s)"}
     ]
   }
   ```

4. **Model continues**, now knowing what `parse()` looks like, and emits a final assistant message answering the user.

Fine-tuning a model to be good at tool calls is mostly about: did its post-training include diverse, well-formatted function-call examples? Qwen 2.5 Coder 7B Instruct does — it works as a harness target out of the box.

## MCP and how it relates to tools

[MCP](07-mcp.md) (Model Context Protocol) is the *next layer down*: instead of every harness implementing every tool in its own codebase, an MCP server stands as an external process exposing tools. Any MCP-compatible harness can discover and call those tools. So:

- **Tools defined inside the harness** (e.g., `claw`'s built-in `read_file`, `bash`, `edit_file`): hard-coded; live in the harness binary.
- **Tools served by MCP servers** (e.g., this stack's `find_similar_bugs` from the RAG plan in [issue #4](https://github.com/alleboudy/llm-finetuner/issues/4)): live in separate processes; spawned by the harness as children over stdio; their schema is discovered at runtime via `tools/list`.

The model doesn't know the difference. The harness handles routing.

## The harness zoo

### `claw` (claw-code)

A Rust-based coding agent that this stack vendors as a git submodule (`vendor/claw-code/`). Open-source upstream (ultraworkers/claw-code).

**Why we vendor it**: 
- Strictly-local posture — no telemetry; we audit the pinned SHA.
- OpenAI-compat client out of the box (`OPENAI_BASE_URL` env var).
- MCP support is mature.
- Permission system is granular.
- Sandbox modes (including Docker `--network none`) for paranoid usage.

**How to use it**: see [issue #1](https://github.com/alleboudy/llm-finetuner/issues/1) for the runbook.

```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY=local-no-auth \
ANTHROPIC_BASE_URL=http://127.0.0.1:8080/v1 \
./vendor/claw-code/rust/target/release/claw \
    --model openai/qwen-coder-7b-q4_k_m.gguf \
    --permission-mode workspace-write
```

The `--model openai/...` prefix is load-bearing: `claw` routes by prefix (`openai/...` → OpenAI client; `anthropic/...` → Anthropic client; etc.). Without it, `claw` defaults to Anthropic and demands `ANTHROPIC_API_KEY`. See [issue #1](https://github.com/alleboudy/llm-finetuner/issues/1) §4 pitfall #1.

### `aider`

A Python-based pair-programmer (https://github.com/Aider-AI/aider). Mature; large community; good multi-language support.

**Pros**: Python (easy to extend); OpenAI-compatible BYOK works seamlessly; supports multiple LLMs in one session.

**Cons**: Less granular permission model than `claw`; no built-in MCP support as of mid-2026 (community plugin exists).

```bash
aider \
    --model openai/qwen-coder-7b-q4_k_m.gguf \
    --openai-api-base http://127.0.0.1:8080/v1 \
    --openai-api-key local-no-auth
```

### `continue.dev`

A VS Code extension (https://continue.dev/). Lives in the editor sidebar; tightly integrated with the IDE.

**Pros**: In-IDE UX is genuinely good; supports MCP; the config file (`~/.continue/config.json`) is easy to reason about.

**Cons**: Tied to VS Code; requires the editor running.

`~/.continue/config.json`:
```json
{
  "models": [{
    "title": "Local fine-tune",
    "provider": "openai",
    "model": "qwen-coder-7b-q4_k_m.gguf",
    "apiBase": "http://127.0.0.1:8080/v1",
    "apiKey": "local-no-auth"
  }],
  "mcpServers": {
    "rag": {
      "command": "python",
      "args": ["-m", "<project>.servers.rag_server"]
    }
  }
}
```

### `Cursor`

A VS Code fork (https://cursor.com/) with built-in AI features.

**Pros**: Tight IDE integration; large team behind it.

**Cons**: Closed-source; their custom-model support depends on the plan tier; less suited to strictly-local since the default flows go through Cursor's cloud.

### `Claude Code`

Anthropic's CLI agent. Strong agent loop; recent MCP support is excellent.

**Pros**: Mature subagent system (Explore, Plan, general-purpose agents); native MCP.

**Cons**: Defaults to Anthropic's cloud models; you can BYOK to a custom endpoint but the setup is more involved than `claw`/`aider`.

### `GitHub Copilot Chat`

See [`06-github-copilot.md`](06-github-copilot.md) for the deep dive.

## Choosing a harness

Decision tree:

```
Q1: Do you need MCP support (issue #4 RAG)?
  → YES → claw or continue.dev or Claude Code.
  → MAYBE → aider (works without, plugins available).

Q2: Are you driving from an IDE or a terminal?
  → IDE → continue.dev (VS Code), Cursor (its own fork).
  → Terminal → claw, aider, Claude Code.

Q3: Strictly-local matters?
  → YES → claw (vendored + audited) or aider (offline).
  → SOMEWHAT → continue.dev (extension is local; per-model traffic stays local).
  → NO → Copilot Chat with BYOK if the model picker accepts your endpoint.
```

For this stack's canonical use case (issue [#1](https://github.com/alleboudy/llm-finetuner/issues/1)), the default is `claw`. It's the vendored harness; the runbook is written against it; the MCP wiring patterns ([issue #4 §13.7](https://github.com/alleboudy/llm-finetuner/issues/4)) are tested against it. Use anything else if you have a reason.

## What a "good" harness gives you

A short checklist for evaluating any harness you're considering:

- **OpenAI-compat support**: can you point it at an arbitrary `OPENAI_BASE_URL`?
- **Streaming responses**: token-by-token output, not "wait 30 seconds for the whole reply"?
- **Tool calling**: does it support function-call JSON and dispatch?
- **MCP support**: discovers and dispatches MCP server tools?
- **Permission modes**: read-only / workspace-write / full-access switch?
- **Sandbox mode**: optional Docker / namespace isolation for the agent's subprocess execution?
- **Resumable sessions**: pick up where you left off?
- **Multi-file editing**: can it apply patches across multiple files atomically?
- **Sane default model context**: respects the model's `n_ctx_train`; doesn't dump 100K tokens into a 4K-context model?
- **No telemetry**: doesn't phone home; doesn't log your code to a vendor?

`claw` ticks all these boxes. `aider` ticks most. `continue.dev` is strong on most for VS Code users. Anything missing some boxes is a trade-off you make consciously.

## Further reading

- OpenAI function calling docs: https://platform.openai.com/docs/guides/function-calling
- claw-code (ultraworkers/claw-code on GitHub) — pinned SHA in this stack's `vendor/claw-code/`
- aider GitHub: https://github.com/Aider-AI/aider
- continue.dev docs: https://docs.continue.dev/
- Anthropic's Claude Code: https://docs.claude.com/en/docs/claude-code
- A good "agentic patterns" survey: https://huyenchip.com/2025/01/07/agents.html
- See also [`07-mcp.md`](07-mcp.md) for the tool-protocol deep dive and [`06-github-copilot.md`](06-github-copilot.md) for Copilot specifically.
