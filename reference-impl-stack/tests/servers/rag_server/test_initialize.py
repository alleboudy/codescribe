"""``initialize`` returns ``serverInfo.name == "repo-rag"``.

Spec test from the design spec. Exercises the MCP handshake end-to-end over the
in-process memory transport — proves the server registers under the
right name and is reachable through the SDK's standard ``ClientSession``
path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codescribe_train.rag.store.writer import Store


@pytest.mark.asyncio
async def test_initialize_reports_sample_rag(tmp_path: Path) -> None:
    """A connected client sees ``serverInfo.name == "repo-rag"``."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from codescribe_train.servers.rag_server.server import build_server

    db_path = tmp_path / "rag.db"
    # Seed an empty store so the server's open() succeeds.
    with Store.open(db_path):
        pass

    log_path = tmp_path / "rag-server.log"
    server = build_server(db_path=db_path, log_path=log_path)
    async with create_connected_server_and_client_session(server) as client:
        init_result = await client.initialize()
        assert init_result.serverInfo.name == "repo-rag"
