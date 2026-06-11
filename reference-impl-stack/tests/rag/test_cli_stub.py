"""CLI subcommand wiring state.

Now that ``index`` + ``status`` and ``query`` are wired, every
top-level subcommand is real. This file used to assert that the
``query`` placeholder exited 2 with "stub-mode"; now it asserts
the negative — none of the subcommands surface the old stub message.
The end-to-end paths for each subcommand live in:

  * ``tests/rag/pipelines/test_index_cli_dispatch.py`` (``index``)
  * ``tests/rag/pipelines/test_status_cli.py`` (``status``)
  * ``tests/rag/store/test_retrieve_query_cli.py`` (``query``)
"""

from __future__ import annotations

import subprocess
import sys


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "codescribe_train.rag", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_no_subcommand_remains_stubbed() -> None:
    """No subcommand returns the ``stub-mode`` exit message."""
    for sub in (["index", "--help"], ["status", "--help"], ["query", "--help"]):
        proc = _run(sub)
        assert "stub-mode" not in proc.stderr, (
            f"{sub} still emits the stub-mode marker: {proc.stderr!r}"
        )
        assert "stub-mode" not in proc.stdout


def test_query_requires_subaction() -> None:
    """``query`` with no sub-action prints a usage error and exits non-zero."""
    proc = _run(["query"])
    assert proc.returncode != 0
    # argparse phrases the missing-required-subparser error in stderr.
    assert "query" in (proc.stderr + proc.stdout).lower()
