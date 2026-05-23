from __future__ import annotations

import asyncio

from mcp.shared.memory import create_connected_server_and_client_session

import codescribe_rag.servers.rag_server.__main__ as srv
from codescribe_rag.rag.store.retrieve import Retriever
from codescribe_rag.servers.rag_server.tools import RagTools


def _server(store, embedder):
    return srv.build_server(_tools=RagTools(store, Retriever(store._conn, embedder)))


def test_tools_list(populated_store, embedder):
    server = _server(populated_store, embedder)

    async def go():
        async with create_connected_server_and_client_session(server) as s:
            await s.initialize()
            return await s.list_tools()

    res = asyncio.run(go())
    assert {t.name for t in res.tools} == {"find_similar_bugs", "get_fix_diff"}


def test_call_find_similar_bugs(populated_store, embedder):
    server = _server(populated_store, embedder)

    async def go():
        async with create_connected_server_and_client_session(server) as s:
            await s.initialize()
            return await s.call_tool("find_similar_bugs", {"query": "NPE on startup", "k": 3})

    res = asyncio.run(go())
    assert "Similar bugs" in res.content[0].text
