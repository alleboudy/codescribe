# rag_server — MCP server rules

Exposes `find_similar_bugs` and `get_fix_diff` as MCP tools over stdio.

## Hard rules

- **stdout is the JSON-RPC channel. Nothing else goes there. Ever.**
  - `logger` is configured to write ONLY to a file (`logs/rag-server-<unix_ts>.log`).
  - `sys.stdout` is reserved for the MCP SDK; do not `print(...)` anywhere in the package.
  - Static check `tests/servers/rag_server/test_no_stdout.py` greps for `print(` outside `__main__.py` guards.
- Server lifecycle: synchronous startup (load store + embedder lazy-init), graceful `shutdown` on SIGTERM.
- Both tools are READ-ONLY. No file writes, no DB writes, no network calls beyond the local SQLite store.
- Tool inputs are validated against the declared JSON-Schema. Reject with a clear `InvalidParams` error on mismatch.
- Tool outputs are deterministic for fixed input + DB state (modulo embedder warmup).
- The server inherits the strictly-local posture: no outbound HTTP from this process. `tests/servers/rag_server/test_no_egress.py` monkey-patches `socket.connect` to assert this.

## Anti-patterns

- Do NOT use `print()` for debug. Log to file.
- Do NOT load the embedder eagerly on import — defer to first tool call.
- Do NOT expose any tool that mutates state. If you need to bump a counter, do it in-process.
- Do NOT call the harness or the model from this server. The server is a leaf; it only answers queries.
- Do NOT swallow exceptions silently. Convert to MCP error responses with the right error code.
