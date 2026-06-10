"""``call_tool('search_commits')`` returns Markdown with ``### Commit `` headers.

Spec test from the design spec:

  > test_call_search_commits.py — fixture; assert returned text
  > contains `### Commit `.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp import types

from tests.servers.rag_server.conftest import StubEmbedder


@pytest.mark.asyncio
async def test_call_returns_commit_header(
    seeded_db: Path,
    stub_embedder: StubEmbedder,
) -> None:
    """The returned text contains the canonical ``### Commit `` header."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from codescribe_train.servers.rag_server.server import build_server

    server = build_server(
        db_path=seeded_db,
        log_path=seeded_db.parent / "log.log",
        embedder=stub_embedder,
    )
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool(
            "search_commits",
            arguments={"query": "cart crash", "k": 5},
        )

    assert result.isError is False
    body = result.content[0]
    assert isinstance(body, types.TextContent)
    assert "### Commit " in body.text, (
        f"missing '### Commit ' header in returned text:\n{body.text!r}"
    )
    # The seeded commit's short SHA appears in the header.
    assert "b" * 12 in body.text
