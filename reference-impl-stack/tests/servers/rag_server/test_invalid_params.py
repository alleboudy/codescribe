"""Invalid params produce an MCP error response, not a stack trace.

Spec test from the design spec:

  > test_invalid_params.py — `k=999` (above schema max) → MCP error
  > response (not a stack trace to stdout).

The MCP lowlevel ``Server.call_tool`` decorator validates arguments
against the ``inputSchema`` automatically when ``validate_input=True``
(the default). On validation failure it returns a ``CallToolResult``
with ``isError=True`` and a text body describing the violation — this
test asserts the failure surfaces as a clean MCP error result rather
than the server crashing or writing to stdout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.servers.rag_server.conftest import StubEmbedder


@pytest.mark.asyncio
async def test_k_above_schema_max_returns_mcp_error(
    seeded_db: Path,
    stub_embedder: StubEmbedder,
) -> None:
    """``k=999`` (max=20 in schema) returns ``isError=True``."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from codescribe_train.servers.rag_server.server import build_server

    server = build_server(
        db_path=seeded_db,
        log_path=seeded_db.parent / "log.log",
        embedder=stub_embedder,
    )
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool(
            "find_similar_issues",
            arguments={"query": "cart", "k": 999},
        )

    assert result.isError is True, (
        "expected an MCP error response for k above schema max; "
        f"got isError={result.isError!r} content={result.content!r}"
    )
