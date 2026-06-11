"""Operator-verified CLI: ``python -m codescribe_train.rag.store init --db ...``.

Acceptance: the operator-verified post-merge gate from the design spec.
The pipeline only ever invokes :class:`Store` directly; this CLI exists
solely so the operator can pre-create the DB file on a fresh box (or
confirm an existing one opens cleanly) before the indexer runs.
"""

from __future__ import annotations

import subprocess
import sys


def test_cli_init_creates_a_fresh_db(tmp_path) -> None:
    db_path = tmp_path / "rag.db"
    assert not db_path.exists()
    proc = subprocess.run(  # noqa: PLW1510 - explicit check below
        [sys.executable, "-m", "codescribe_train.rag.store", "init", "--db", str(db_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert db_path.exists()
    assert "created" in proc.stdout


def test_cli_init_is_idempotent(tmp_path) -> None:
    db_path = tmp_path / "rag.db"
    cmd = [sys.executable, "-m", "codescribe_train.rag.store", "init", "--db", str(db_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
    proc = subprocess.run(  # noqa: PLW1510 - explicit check below
        cmd, check=False, capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    assert "opened existing" in proc.stdout
