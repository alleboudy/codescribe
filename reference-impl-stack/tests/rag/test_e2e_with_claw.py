"""End-to-end smoke against the ``repo-rag`` MCP server.

Spec test #3 from the design spec, plus the operator's "acceptable
simplification" note: rather than spawn the vendored ``claw`` Rust
binary (which is not guaranteed to be built in CI and would require
mocking ``llama-server`` too), we exercise the same MCP wire format
claw would use by driving
an in-process :class:`mcp.ClientSession` against a fully-wired
:class:`repo-rag` server built from the seeded fixture store.

This proves the round-trip a real claw session would take:
``initialize`` → ``tools/list`` → ``tools/call(find_similar_issues, ...)``
→ a structured ``CallToolResult`` containing the canonical Markdown
the LLM would receive verbatim.

The post-merge gates on a CUDA GPU machine (the design spec,
"Operator-verified") exercise the actual claw + llama-server path; this
test is the CI gate for the MCP integration shape that does not depend
on those binaries.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp import types

from tests.servers.rag_server.conftest import StubEmbedder


@pytest.mark.asyncio
async def test_end_to_end_find_similar_issues_returns_seeded_issue(
    seeded_db: Path,
    stub_embedder: StubEmbedder,
) -> None:
    """A connected client calls ``find_similar_issues`` and sees Markdown
    naming the fixture's issue #7 plus the canonical ``### Issue `` header.

    What this proves
    -----------------

    * The MCP handshake (``initialize``) completes against a real
      :func:`~codescribe_train.servers.rag_server.server.build_server`-built
      server — i.e. the server registered both the
      ``list_tools`` and ``call_tool`` handlers correctly.
    * The tool ``find_similar_issues`` is discoverable via the standard
      ``tools/list`` SDK call (the surface claw enumerates at session
      start when it prints ``claw mcp list``).
    * Calling that tool round-trips a non-error ``CallToolResult`` whose
      ``TextContent`` Markdown contains the seeded issue and the
      canonical per-hit ``### Issue `` header the model is fine-tuned
      to expect.

    What this does NOT prove
    ------------------------

    * That the vendored Rust ``claw`` binary actually spawns the server.
      That requires a built binary and a running ``llama-server``;
      verified manually on a CUDA GPU machine per the design spec's
      operator gates.
    """
    from mcp.shared.memory import create_connected_server_and_client_session

    from codescribe_train.servers.rag_server.server import build_server

    server = build_server(
        db_path=seeded_db,
        log_path=seeded_db.parent / "rag-server.log",
        embedder=stub_embedder,
    )
    async with create_connected_server_and_client_session(server) as client:
        # 1) handshake — claw does this first thing at session start.
        init = await client.initialize()
        assert init.serverInfo.name == "repo-rag"

        # 2) tools/list — claw exposes this output via `claw mcp list`.
        tools_result = await client.list_tools()
        tool_names = {t.name for t in tools_result.tools}
        assert tool_names == {
            "find_similar_issues",
            "get_pr_diff",
            "search_commits",
            "search_docs",
        }

        # 3) tools/call — the trigger from a real prompt like
        # "Show me a similar past issue to: <X>".
        call_result = await client.call_tool(
            "find_similar_issues",
            arguments={"query": "cart crash on save", "k": 3},
        )

    # The CallToolResult shape claw would forward to the model.
    assert call_result.isError is False
    assert len(call_result.content) == 1
    body = call_result.content[0]
    assert isinstance(body, types.TextContent)
    # The canonical per-hit header the fine-tuned model is taught to expect.
    assert "### Issue " in body.text, (
        f"expected canonical per-hit header missing in:\n{body.text!r}"
    )
    # The seeded issue from tests/servers/rag_server/conftest.py.
    assert "#7" in body.text, (
        f"expected seeded issue #7 not present in response:\n{body.text!r}"
    )
    # And the linked fix PR per the fixture's IssuePRLink at confidence 1.0.
    assert "#42" in body.text, (
        f"expected linked fix PR #42 not present in response:\n{body.text!r}"
    )


@pytest.mark.asyncio
async def test_end_to_end_tools_list_advertises_four_tools(
    seeded_db: Path,
    stub_embedder: StubEmbedder,
) -> None:
    """``claw mcp list`` calls ``tools/list``; we should advertise four.

    The operator-verified gate is "shows the ``repo-rag`` server with
    its tools" — this is the in-process equivalent that runs in CI.
    The fourth tool, ``search_docs``, was added alongside the
    worktree-docs RAG source.
    """
    from mcp.shared.memory import create_connected_server_and_client_session

    from codescribe_train.servers.rag_server.server import build_server

    server = build_server(
        db_path=seeded_db,
        log_path=seeded_db.parent / "rag-server.log",
        embedder=stub_embedder,
    )
    async with create_connected_server_and_client_session(server) as client:
        await client.initialize()
        listed = await client.list_tools()

    assert len(listed.tools) == 4
    for tool in listed.tools:
        assert tool.inputSchema is not None
        assert tool.inputSchema["type"] == "object"
        assert "properties" in tool.inputSchema
