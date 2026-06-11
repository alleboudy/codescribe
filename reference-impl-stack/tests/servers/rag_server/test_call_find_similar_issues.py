"""``call_tool('find_similar_issues')`` returns Markdown with the right headers.

Spec test from the design spec:

  > test_call_find_similar_issues.py — fixture store; call returns
  > Markdown with `### Bug ` / `### Issue ` headers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp import types

from tests.servers.rag_server.conftest import StubEmbedder


@pytest.mark.asyncio
async def test_call_returns_markdown_with_issue_header(
    seeded_db: Path,
    stub_embedder: StubEmbedder,
) -> None:
    """The returned text contains the canonical ``### Issue `` header."""
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
            arguments={"query": "cart crash", "k": 5},
        )

    assert result.isError is False
    assert len(result.content) == 1
    body = result.content[0]
    assert isinstance(body, types.TextContent)
    # The spec accepts either "### Bug " or "### Issue " as the per-hit
    # header; we render "### Issue " uniformly.
    assert "### Issue " in body.text or "### Bug " in body.text, (
        f"missing canonical issue header in returned text:\n{body.text!r}"
    )
    assert "#7" in body.text  # the seeded issue number
