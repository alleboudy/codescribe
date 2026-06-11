"""``socket.connect`` is never called during tool invocation.

Spec test from the design spec:

  > test_no_egress.py — monkey-patch socket.connect; never called
  > during tool invocation.

The server is strictly local. Tool calls touch only the embedded
sqlite store; no outbound HTTP, no model server, no telemetry. Any
``socket.socket.connect`` call from inside a tool handler fails the
test loudly.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from tests.servers.rag_server.conftest import StubEmbedder


class _ConnectGuard:
    """Monkey-patch helper: every ``socket.socket.connect`` increments a counter."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._original = socket.socket.connect

    def install(self) -> None:
        guard = self

        def _trapped(self_sock, address, *args, **kwargs):  # noqa: ANN001
            guard.calls.append(address)
            raise AssertionError(
                f"socket.connect({address!r}) called from MCP tool — egress is forbidden"
            )

        socket.socket.connect = _trapped  # type: ignore[method-assign]

    def restore(self) -> None:
        socket.socket.connect = self._original  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_no_socket_connect_during_tool_calls(
    seeded_db: Path,
    stub_embedder: StubEmbedder,
) -> None:
    """All three tools complete without any ``socket.connect`` call."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from codescribe_train.servers.rag_server.server import build_server

    server = build_server(
        db_path=seeded_db,
        log_path=seeded_db.parent / "log.log",
        embedder=stub_embedder,
    )

    # Open the in-memory transport BEFORE installing the guard — anyio's
    # internal pipes do not use real sockets, but the SDK's session
    # setup may touch sockets transitively during the handshake; we
    # only care about tool-handler egress.
    async with create_connected_server_and_client_session(server) as client:
        guard = _ConnectGuard()
        guard.install()
        try:
            for tool, args in [
                ("find_similar_issues", {"query": "cart", "k": 5}),
                ("get_pr_diff", {"pr_number": 42, "max_chars": 8000}),
                ("search_commits", {"query": "cart", "k": 5}),
            ]:
                await client.call_tool(tool, arguments=args)
        finally:
            guard.restore()

    assert guard.calls == [], (
        f"socket.connect called during tool invocation: {guard.calls!r}"
    )
