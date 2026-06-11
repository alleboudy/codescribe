"""Importing the harness CLI must not eagerly pull heavy or optional deps.

Mirrors ``tests/train/test_config_and_cli.py``'s lazy-import test. The
contract: someone running ``python -m codescribe_train.harness --help`` should
not trigger an ``import httpx`` or ``import yaml`` until they actually
reach a subcommand that needs those.
"""

from __future__ import annotations

import sys

import pytest


def test_import_cli_does_not_pull_httpx_or_yaml() -> None:
    pre = set(sys.modules)
    from codescribe_train.harness import cli  # noqa: F401 — import is the test

    new_modules = set(sys.modules) - pre
    forbidden = {"httpx", "yaml", "subprocess"}
    leaked = forbidden & new_modules
    assert not leaked, f"importing codescribe_train.harness.cli leaked: {sorted(leaked)}"


def test_import_cli_does_not_pull_concrete_harnesses() -> None:
    pre = set(sys.modules)
    from codescribe_train.harness import cli  # noqa: F401

    new_modules = set(sys.modules) - pre
    # The concrete harness modules are loaded only when --harness=<name>
    # is dispatched. Importing the CLI shouldn't instantiate either.
    assert "codescribe_train.harness.claw" not in new_modules
    assert "codescribe_train.harness.noop" not in new_modules


def test_help_does_not_eager_load_yaml(capsys: pytest.CaptureFixture[str]) -> None:
    from codescribe_train.harness.cli import main

    pre = set(sys.modules)
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "run" in captured.out
    assert "health" in captured.out
    new_modules = set(sys.modules) - pre
    assert "yaml" not in new_modules


def test_run_subparser_requires_args() -> None:
    from codescribe_train.harness.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["run"])
    # argparse exits 2 on missing required args
    assert exc_info.value.code == 2


def test_run_rejects_unknown_harness() -> None:
    from codescribe_train.harness.cli import main

    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--backend",
                "http://localhost:8080",
                "--model",
                "x",
                "--workdir",
                "/tmp",
                "--harness",
                "ghost",
            ]
        )
