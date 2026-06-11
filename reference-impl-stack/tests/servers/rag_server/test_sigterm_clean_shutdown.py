"""SIGTERM mid-session shuts the server down cleanly.

Spec test from the design spec:

  > test_sigterm_clean_shutdown.py — subprocess test; SIGTERM
  > mid-call → exit 0 + non-empty log file.

The test:

1. Spawns ``python -m codescribe_train.servers.rag_server`` as a subprocess
   with a fresh tmp_path DB and a tmp_path log file.
2. Waits a short grace period (~1s) for the process to finish
   importing, set up logging, install the SIGTERM handler, and enter
   the stdio loop.
3. Sends ``SIGTERM`` to the process and waits for it to exit.
4. Asserts the exit code is 0 and the log file is non-empty (proves
   ``_configure_logging`` ran AND logging continued through the
   SIGTERM handler's "shutting down" line).

The subprocess uses ``stdin=subprocess.PIPE`` — a real pipe held
open by the parent. The child's ``stdio_server`` blocks reading the
pipe (no EOF until we close our write end), so the process sits
idle until SIGTERM arrives. With ``stdin=DEVNULL`` the child would
see immediate EOF and exit before we ever got to send SIGTERM —
hence the explicit PIPE.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from codescribe_train.rag.store.writer import Store


def test_sigterm_exits_zero_and_logs(tmp_path: Path) -> None:
    """SIGTERM → exit 0; log file is non-empty."""
    db_path = tmp_path / "rag.db"
    log_path = tmp_path / "rag-server.log"

    # Seed an empty store so the server's read-only open() succeeds.
    with Store.open(db_path):
        pass

    env = os.environ.copy()
    env["RAG_DB_PATH"] = str(db_path)
    env["RAG_LOG_PATH"] = str(log_path)

    proc = subprocess.Popen(
        [sys.executable, "-m", "codescribe_train.servers.rag_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    # Startup grace — give the child time to import the world, open
    # the DB, register the SIGTERM handler, and enter the stdio loop.
    time.sleep(1.5)
    assert proc.poll() is None, (
        "server exited before SIGTERM was sent: "
        f"rc={proc.returncode} stdout={proc.stdout.read()!r} stderr={proc.stderr.read()!r}"
    )

    proc.send_signal(signal.SIGTERM)
    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover — defensive
        proc.kill()
        stdout, stderr = proc.communicate()
        raise AssertionError(
            "server did not exit within 10s of SIGTERM; "
            f"stdout={stdout!r} stderr={stderr!r}"
        ) from None

    assert proc.returncode == 0, (
        f"expected exit 0; got rc={proc.returncode}\n"
        f"stdout={stdout!r}\nstderr={stderr!r}"
    )
    assert log_path.exists(), f"log file missing at {log_path}"
    log_text = log_path.read_text(encoding="utf-8")
    assert log_text.strip(), (
        f"log file at {log_path} is empty after shutdown"
    )
