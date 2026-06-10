"""Importing the backends CLI must not eagerly pull heavy or optional deps.

Mirrors ``tests/harness/test_cli_lazy_imports.py``. Contract: someone
running ``python -m codescribe_train.backends --help`` should not trigger an
``import httpx`` / ``import psutil`` / ``import pynvml`` / ``import vllm``
until they actually reach a subcommand that needs those.
"""

from __future__ import annotations

import sys

import pytest


def test_import_cli_does_not_pull_optional_deps() -> None:
    pre = set(sys.modules)
    from codescribe_train.backends import cli  # noqa: F401 — import is the test

    new_modules = set(sys.modules) - pre
    forbidden = {"httpx", "psutil", "pynvml", "vllm"}
    leaked = forbidden & new_modules
    assert not leaked, f"importing codescribe_train.backends.cli leaked: {sorted(leaked)}"


def test_import_cli_does_not_pull_concrete_backends() -> None:
    pre = set(sys.modules)
    from codescribe_train.backends import cli  # noqa: F401

    new_modules = set(sys.modules) - pre
    for mod in (
        "codescribe_train.backends.llama_server",
        "codescribe_train.backends.vllm",
        "codescribe_train.backends.ollama",
        "codescribe_train.backends.remote",
        "codescribe_train.backends.probe",
    ):
        assert mod not in new_modules, f"{mod} eagerly imported"


def test_help_does_not_eager_load_psutil(capsys: pytest.CaptureFixture[str]) -> None:
    from codescribe_train.backends.cli import main

    pre = set(sys.modules)
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "probe" in captured.out
    assert "health" in captured.out
    new_modules = set(sys.modules) - pre
    assert "psutil" not in new_modules
    assert "pynvml" not in new_modules


def test_run_subparser_requires_args() -> None:
    from codescribe_train.backends.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["health"])
    assert exc_info.value.code == 2


def test_health_rejects_unknown_backend() -> None:
    from codescribe_train.backends.cli import main

    with pytest.raises(SystemExit):
        main(
            [
                "health",
                "--endpoint",
                "http://localhost:8080",
                "--backend",
                "ghost",
            ]
        )
