"""``tools/list`` returns the 4 tools with the spec'd schemas.

Spec test from the design spec (updated for search_docs, Task 8).

Asserts both the tool names and the inputSchema match the canonical
constants in :mod:`codescribe_train.servers.rag_server.tools`. If either
the name set or any schema drifts, the generated ``.claw-mcp.json`` will be
mis-wired — catching this early is cheap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codescribe_train.rag.store.writer import Store


@pytest.mark.asyncio
async def test_tools_list_returns_four_tools_with_canonical_schemas(
    tmp_path: Path,
) -> None:
    """Exactly 4 tools, with the names + schemas declared in tools.py."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from codescribe_train.servers.rag_server.server import build_server
    from codescribe_train.servers.rag_server.tools import (
        FIND_SIMILAR_ISSUES_SCHEMA,
        GET_PR_DIFF_SCHEMA,
        SEARCH_COMMITS_SCHEMA,
        SEARCH_DOCS_SCHEMA,
    )

    db_path = tmp_path / "rag.db"
    with Store.open(db_path):
        pass

    server = build_server(db_path=db_path, log_path=tmp_path / "log.log")
    async with create_connected_server_and_client_session(server) as client:
        result = await client.list_tools()

    tools_by_name = {t.name: t for t in result.tools}
    assert set(tools_by_name) == {
        "find_similar_issues",
        "get_pr_diff",
        "search_commits",
        "search_docs",
    }, f"unexpected tool set: {sorted(tools_by_name)}"
    assert len(result.tools) == 4, (
        f"expected exactly 4 tools, got {len(result.tools)}"
    )

    assert tools_by_name["find_similar_issues"].inputSchema == FIND_SIMILAR_ISSUES_SCHEMA
    assert tools_by_name["get_pr_diff"].inputSchema == GET_PR_DIFF_SCHEMA
    assert tools_by_name["search_commits"].inputSchema == SEARCH_COMMITS_SCHEMA
    assert tools_by_name["search_docs"].inputSchema == SEARCH_DOCS_SCHEMA
