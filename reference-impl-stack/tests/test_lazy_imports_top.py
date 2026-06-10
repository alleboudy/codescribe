"""Importing the top-level ``codescribe_train.cli`` must stay cheap.

This is the import the ``codescribe-train`` console script entry point hits.
We must not pull harness implementations, backend implementations,
``httpx``, ``psutil``, or ``pynvml`` until the user actually dispatches
a subcommand that needs them.

Mirrors ``tests/harness/test_cli_lazy_imports.py`` and
``tests/backends/test_cli_lazy_imports.py``.
"""

from __future__ import annotations

import sys

import pytest


def test_import_top_cli_does_not_pull_optional_deps() -> None:
    pre = set(sys.modules)
    from codescribe_train import cli  # noqa: F401 — import is the test

    new_modules = set(sys.modules) - pre
    forbidden = {"httpx", "psutil", "pynvml", "vllm", "yaml"}
    leaked = forbidden & new_modules
    assert not leaked, f"importing codescribe_train.cli leaked: {sorted(leaked)}"


def test_import_top_cli_does_not_pull_concrete_impls() -> None:
    pre = set(sys.modules)
    from codescribe_train import cli  # noqa: F401

    new_modules = set(sys.modules) - pre
    for mod in (
        "codescribe_train.backends.llama_server",
        "codescribe_train.backends.vllm",
        "codescribe_train.backends.ollama",
        "codescribe_train.backends.remote",
        "codescribe_train.backends.probe",
        "codescribe_train.harness.claw",
        "codescribe_train.harness.noop",
    ):
        assert mod not in new_modules, f"{mod} eagerly imported"


def test_top_help_does_not_eager_load_anything(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from codescribe_train.cli import main

    pre = set(sys.modules)
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "run" in captured.out
    assert "probe" in captured.out
    new_modules = set(sys.modules) - pre
    for forbidden in ("httpx", "psutil", "pynvml", "yaml"):
        assert forbidden not in new_modules


def test_run_help_does_not_pull_implementations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Even ``codescribe-train run --help`` must not trigger heavy imports."""
    from codescribe_train.cli import main

    pre = set(sys.modules)
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    new_modules = set(sys.modules) - pre
    for mod in (
        "codescribe_train.backends.llama_server",
        "codescribe_train.harness.claw",
        "codescribe_train.backends.probe",
    ):
        assert mod not in new_modules
