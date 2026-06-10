"""Round-trip a few lines through ``NoOpHarness``.

Covers prepare + start_session over an in-memory pipe. Demonstrates that
the harness honours its IO contract without launching a subprocess.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from codescribe_train.harness.noop import NoOpHarness


def test_prepare_records_session_state(tmp_path: Path) -> None:
    h = NoOpHarness()
    assert h.prepared is False
    h.prepare(tmp_path, "http://localhost:8080", "noop-model", harness_config={"foo": "bar"})
    assert h.prepared is True
    assert h.workdir == tmp_path
    assert h.backend_url == "http://localhost:8080"
    assert h.model == "noop-model"
    assert h.extra == {"foo": "bar"}


def test_start_session_round_trips_text(tmp_path: Path) -> None:
    h = NoOpHarness()
    h.prepare(tmp_path, "http://localhost:8080", "noop-model", harness_config={})

    stdin = io.StringIO("hello\nworld\n\nfinal\n")
    stdout = io.StringIO()
    rc = h.start_session(stdin, stdout)
    assert rc == 0
    output = stdout.getvalue()
    assert "echo: hello" in output
    assert "echo: world" in output
    assert "echo: final" in output
    # Blank lines are skipped, so no double-blank in the output stream.
    assert "echo: \n" not in output


def test_start_session_before_prepare_raises(tmp_path: Path) -> None:
    h = NoOpHarness()
    with pytest.raises(RuntimeError):
        h.start_session(io.StringIO(""), io.StringIO())


def test_health_check_always_true() -> None:
    h = NoOpHarness()
    # NoOp has no backend; this is the defined behaviour.
    assert h.health_check("http://example.invalid") is True


def test_sandbox_args_is_empty() -> None:
    assert NoOpHarness().sandbox_args() == []
