from __future__ import annotations

import socket

from codescribe_rag.rag.store.retrieve import Retriever
from codescribe_rag.servers.rag_server.tools import RagTools


def test_tool_call_makes_no_socket(populated_store, embedder, monkeypatch):
    calls = []
    real = socket.socket.connect

    def spy(self, addr):
        calls.append(addr)
        return real(self, addr)

    monkeypatch.setattr(socket.socket, "connect", spy)
    tools = RagTools(populated_store, Retriever(populated_store._conn, embedder))
    tools.find_similar_bugs("npe", k=3, min_confidence=0.8)
    assert calls == []
