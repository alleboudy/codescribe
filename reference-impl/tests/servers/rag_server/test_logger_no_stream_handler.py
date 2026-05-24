from __future__ import annotations

import importlib
import logging


def test_only_file_handler(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_LOG_PATH", str(tmp_path / "s.log"))
    import codescribe_rag.servers.rag_server.__main__ as m
    importlib.reload(m)
    handlers = logging.getLogger().handlers
    assert handlers and all(isinstance(h, logging.FileHandler) for h in handlers)
