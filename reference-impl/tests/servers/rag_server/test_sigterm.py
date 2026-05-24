from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def test_subprocess_sigterm_clean(tmp_path):
    db = tmp_path / "rag.db"
    from codescribe_rag.rag.store.writer import Store
    Store.open(db).close()                       # valid, empty store

    log = tmp_path / "s.log"
    env = {**os.environ, "RAG_DB_PATH": str(db), "RAG_LOG_PATH": str(log),
           "RAG_EMBED_BACKEND": "hashing"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "codescribe_rag.servers.rag_server"],
        stdin=subprocess.PIPE, env=env, cwd=str(Path(__file__).parents[3]))
    time.sleep(2.0)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 0
    assert log.exists() and log.stat().st_size > 0
