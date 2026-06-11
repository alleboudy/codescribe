"""``call_tool('get_pr_diff')`` returns Markdown starting with ``## PR #``.

Spec test from the design spec:

  > test_call_get_pr_diff.py — fixture; assert returned text starts
  > with `## PR #` and contains a unified-diff `@@`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp import types


@pytest.mark.asyncio
async def test_call_returns_pr_header_and_diff_chunk(seeded_db: Path) -> None:
    """The returned Markdown starts with ``## PR #`` and contains ``@@``."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from codescribe_train.servers.rag_server.server import build_server

    # get_pr_diff doesn't need an embedder — pass None to avoid the
    # live config load.
    server = build_server(
        db_path=seeded_db,
        log_path=seeded_db.parent / "log.log",
        embedder=_NullEmbedder(),
    )
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool(
            "get_pr_diff",
            arguments={"pr_number": 42, "max_chars": 8000},
        )

    assert result.isError is False
    body = result.content[0]
    assert isinstance(body, types.TextContent)
    assert body.text.startswith("## PR #"), (
        f"expected '## PR #' prefix; got {body.text[:80]!r}"
    )
    assert "@@" in body.text, (
        f"expected '@@' (unified diff header) in body; got {body.text!r}"
    )


class _NullEmbedder:
    """Embedder stub for the get_pr_diff path — never called."""

    def embed(self, texts: list[str]):  # noqa: ANN201  # test-only stub
        raise AssertionError("get_pr_diff must not invoke the embedder")
