"""Scaffold smoke for the rag_server package."""

from __future__ import annotations


def test_rag_server_imports() -> None:
    import codescribe_train.servers.rag_server
    import codescribe_train.servers.rag_server.tools

    assert codescribe_train.servers.rag_server is not None
    assert codescribe_train.servers.rag_server.tools is not None
