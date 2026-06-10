"""Smoke tests for the train module that don't require torch / unsloth installed.

Anything torch-dependent is tested via the actual smoke training run, not
pytest, because (a) it needs a GPU and a real model, and (b) pytest in CI
shouldn't pull a 7B model. These tests just confirm config parsing, CLI arg
parsing, and module imports work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_qwen7b_qlora_yaml_parses() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs/train/qwen7b_qlora.yaml").read_text())
    assert cfg["base_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert cfg["lora"]["r"] == 16
    assert cfg["training"]["seq_len"] == 1024
    assert cfg["training"]["bf16"] is True
    assert cfg["training"]["fp16"] is False  # never on Blackwell
    assert cfg["smoke"]["max_steps"] == 10
    # Strictly-local hint is not in the config (it lives in env vars), but the
    # output paths must be local.
    assert cfg["output"]["adapter_dir"].startswith("checkpoints/")


def test_sample_tasks_json_parses() -> None:
    payload = json.loads((REPO_ROOT / "evals/sample_tasks.json").read_text())
    tasks = payload["tasks"]
    assert len(tasks) >= 10
    seen_ids = set()
    for t in tasks:
        assert t["id"] not in seen_ids, f"duplicate task id: {t['id']}"
        seen_ids.add(t["id"])
        assert t["category"] in {"qa", "howto", "implementation"}
        assert isinstance(t["prompt"], str) and len(t["prompt"]) > 10
        assert isinstance(t["expected_signals"], list)
        assert isinstance(t["forbidden_signals"], list)


def test_train_cli_help_does_not_import_torch() -> None:
    """Importing the CLI must not pull torch/unsloth (they're loaded lazily)."""
    import sys

    pre = set(sys.modules)
    from codescribe_train.train import cli  # noqa: F401 — import is the test

    new_modules = set(sys.modules) - pre
    assert "torch" not in new_modules, (
        f"importing codescribe_train.train.cli pulled in torch (modules: {sorted(new_modules)})"
    )
    assert "unsloth" not in new_modules
    assert "peft" not in new_modules


def test_train_cli_argparse_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    from codescribe_train.train.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "smoke" in captured.out
    assert "run" in captured.out
    assert "eval" in captured.out
    assert "export" in captured.out


def test_train_cli_smoke_subcommand_requires_args() -> None:
    from codescribe_train.train.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["smoke"])
    # argparse exits 2 on missing required args
    assert exc_info.value.code == 2
