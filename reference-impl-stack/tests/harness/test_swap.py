"""Swap test: the abstraction is honest if you can swap implementations.

Drives both ``NoOpHarness`` and ``ClawCodeHarness`` (with ``subprocess.run``
mocked) through the same call sequence and confirms the wider
``Harness``-shaped surface holds. This is the pytest version of the
documented "stub a NoOpHarness and confirm `--harness noop` works without
touching claw-code at all" smoke test in the Phase 3 plan.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest import mock

import pytest

from codescribe_train.harness.base import Harness
from codescribe_train.harness.claw import ClawCodeHarness
from codescribe_train.harness.noop import NoOpHarness


def _exercise_harness(h: Harness, workdir: Path) -> int:
    """Run the same pin-the-tail sequence on any Harness."""
    h.prepare(workdir, "http://localhost:8080", "openai/m", harness_config={})
    stdin = io.StringIO("hello\n")
    stdout = io.StringIO()
    return h.start_session(stdin, stdout)


def test_noop_satisfies_harness_protocol(tmp_path: Path) -> None:
    h = NoOpHarness()
    assert isinstance(h, Harness)
    rc = _exercise_harness(h, tmp_path)
    assert rc == 0


def test_claw_satisfies_harness_protocol(tmp_path: Path) -> None:
    fake_binary = tmp_path / "fake-claw"
    fake_binary.write_text("#!/bin/sh\nexit 0\n")
    fake_binary.chmod(0o755)
    h = ClawCodeHarness(binary_path=fake_binary)
    assert isinstance(h, Harness)
    with mock.patch("codescribe_train.harness.claw.subprocess.run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=0)
        rc = _exercise_harness(h, tmp_path)
    assert rc == 0
    assert run_mock.called


def test_cli_dispatches_noop_without_loading_claw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Driving the CLI with ``--harness noop`` must never import the claw module."""
    import sys

    # Drop any prior import of the claw module; the test is whether the
    # CLI run path re-imports it.
    sys.modules.pop("codescribe_train.harness.claw", None)

    # Pytest captures real stdin/stdout — redirect to in-memory streams
    # so NoOpHarness.start_session can read/write without OSError.
    fake_stdin = io.StringIO("hello\nworld\n")
    fake_stdout = io.StringIO()
    monkeypatch.setattr("sys.stdin", fake_stdin)
    monkeypatch.setattr("sys.stdout", fake_stdout)

    from codescribe_train.harness.cli import main

    rc = main(
        [
            "run",
            "--backend",
            "http://localhost:8080",
            "--model",
            "openai/m",
            "--workdir",
            str(tmp_path),
            "--harness",
            "noop",
            "--skip-health-check",
        ]
    )
    assert rc == 0
    assert "echo: hello" in fake_stdout.getvalue()
    # Crucial: the noop run must not have triggered an import of the
    # claw module. If it did, the abstraction is leaky.
    assert "codescribe_train.harness.claw" not in sys.modules
