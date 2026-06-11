"""Tests for the LoRA → GGUF export module.

These tests don't require torch/unsloth — they verify adapter detection and
dispatch logic using filesystem fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from codescribe_train.train.export_gguf import is_unsloth_adapter, merge_lora


@pytest.fixture()
def unsloth_adapter(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    cfg = {
        "auto_mapping": {
            "base_model_class": "Qwen2ForCausalLM",
            "unsloth_fixed": True,
        },
        "base_model_name_or_path": "unsloth/qwen2.5-coder-7b-instruct-bnb-4bit",
        "peft_type": "LORA",
        "r": 16,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg))
    return adapter_dir


@pytest.fixture()
def vanilla_adapter(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    cfg = {
        "base_model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "peft_type": "LORA",
        "r": 16,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg))
    return adapter_dir


def test_unsloth_adapter_detected(unsloth_adapter: Path) -> None:
    assert is_unsloth_adapter(unsloth_adapter) is True


def test_vanilla_adapter_not_flagged_as_unsloth(vanilla_adapter: Path) -> None:
    assert is_unsloth_adapter(vanilla_adapter) is False


def test_missing_adapter_config_not_flagged(tmp_path: Path) -> None:
    assert is_unsloth_adapter(tmp_path) is False


def test_unsloth_path_dispatched_for_unsloth_adapter(
    unsloth_adapter: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "merged"
    with mock.patch(
        "codescribe_train.train.export_gguf._unsloth_merge", return_value=out_dir
    ) as m_unsloth, mock.patch(
        "codescribe_train.train.export_gguf._peft_merge"
    ) as m_peft:
        merge_lora(unsloth_adapter, "ignored-base-id", out_dir)
    m_unsloth.assert_called_once_with(unsloth_adapter, out_dir)
    m_peft.assert_not_called()


def test_peft_path_dispatched_for_vanilla_adapter(
    vanilla_adapter: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "merged"
    with mock.patch(
        "codescribe_train.train.export_gguf._peft_merge", return_value=out_dir
    ) as m_peft, mock.patch(
        "codescribe_train.train.export_gguf._unsloth_merge"
    ) as m_unsloth:
        merge_lora(vanilla_adapter, "Qwen/Qwen2.5-Coder-7B-Instruct", out_dir)
    m_peft.assert_called_once_with(
        vanilla_adapter, "Qwen/Qwen2.5-Coder-7B-Instruct", out_dir
    )
    m_unsloth.assert_not_called()
