"""Bonus subprocess-level smoke for the ``repo-rag`` MCP server.

Spec from the design spec ("If you can additionally write a separate
``tests/rag/test_e2e_with_claw_smoke.py`` that mocks claw as a
subprocess (just verifying that the harness shell-out works), that's a
bonus — but it's NOT a blocker").

This test confirms that ``python -m codescribe_train.servers.rag_server`` is
launchable via ``subprocess`` against an empty rag.db, drains its
shutdown path via SIGTERM, and exits 0 — i.e. the same shape claw uses
when it spawns the stdio server.

Why this is "smoke" and not the primary e2e:
``test_e2e_with_claw.py`` is the primary e2e — it exercises the actual
MCP wire format. This file proves the OS-process boundary works: the
module can be invoked from a shell, it doesn't crash on startup, it
honours $RAG_DB_PATH + $RAG_LOG_PATH, and it terminates cleanly.
Together the two files cover both "the protocol works" and "the
process spawn works", which is what a claw subprocess would need.

Why not actually exchange MCP frames over stdio here?
The stdio JSON-RPC dance is timing-sensitive and brittle to test from
subprocess; the test suite already covers it via the in-process
memory transport. That suite asserts everything this would assert
modulo the OS-level process spawn, which is what we cover here.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from codescribe_train.rag.store.writer import Store


def test_rag_server_module_starts_and_shuts_down_cleanly(tmp_path: Path) -> None:
    """``python -m codescribe_train.servers.rag_server`` spawns, lives, exits 0."""
    db_path = tmp_path / "rag.db"
    log_path = tmp_path / "rag-server.log"
    # Seed an empty store so the read-only open() succeeds.
    with Store.open(db_path):
        pass

    env = {
        **os.environ,
        "RAG_DB_PATH": str(db_path),
        "RAG_LOG_PATH": str(log_path),
    }

    # Spawn the server; it blocks on stdio waiting for a client.
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-m", "codescribe_train.servers.rag_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Give the SIGTERM handler enough time to install. The handler is
    # registered AFTER cold-import + anyio.run() spins up — ~1.5s matches
    # the grace the in-tree subprocess test uses.
    stdout: bytes = b""
    stderr: bytes = b""
    try:
        time.sleep(1.5)
        assert proc.poll() is None, (
            f"rag_server exited unexpectedly during startup; rc={proc.returncode}"
        )

        proc.send_signal(signal.SIGTERM)
        # Use communicate(), not wait(), so we drain stdout/stderr while
        # the child shuts down — otherwise piped buffers can deadlock
        # on a slow shutdown.
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            pytest.fail(
                "rag_server did not exit within 10s of SIGTERM; "
                f"stdout={stdout!r} stderr={stderr!r}"
            )
        assert proc.returncode == 0, (
            f"rag_server exited with rc={proc.returncode}; "
            f"stdout={stdout!r} stderr={stderr!r}"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    # The log file should exist and not be empty (the startup log line is
    # the smoking-gun that the server actually came up).
    assert log_path.is_file(), "rag-server.log was not written"
    log_body = log_path.read_text(encoding="utf-8")
    assert "repo-rag mcp server starting" in log_body, (
        f"unexpected log body:\n{log_body}"
    )
