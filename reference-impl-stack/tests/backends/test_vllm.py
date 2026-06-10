"""Tests for ``VllmBackend`` — argv construction + clear error when missing.

vLLM is heavy and almost certainly not installed in the test env. The
backend is structured so that constructing the dataclass and calling
``_build_argv`` does NOT pull ``import vllm``; only ``start`` does, and
when vLLM is absent we want a clear ``VllmNotInstalledError`` rather than
an ``ImportError`` that callers can't easily catch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from codescribe_train.backends.base import Backend
from codescribe_train.backends.vllm import VllmBackend, VllmNotInstalledError


def test_is_a_backend() -> None:
    assert isinstance(VllmBackend(), Backend)


def test_argv_construction_basic() -> None:
    backend = VllmBackend(
        host="127.0.0.1",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
    )
    argv = backend._build_argv(model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct", adapter=None, port=8080)
    assert "serve" in argv
    assert "Qwen/Qwen2.5-Coder-1.5B-Instruct" in argv
    assert "--host" in argv and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8080"
    assert "--gpu-memory-utilization" in argv
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.85"
    assert "--max-model-len" in argv
    assert argv[argv.index("--max-model-len") + 1] == "8192"
    # No LoRA in this scenario.
    assert "--enable-lora" not in argv


def test_argv_includes_lora_when_adapter_passed(tmp_path: Path) -> None:
    backend = VllmBackend()
    adapter = tmp_path / "lora-sample-v1"
    adapter.mkdir()
    argv = backend._build_argv(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct", adapter=adapter, port=8080
    )
    assert "--enable-lora" in argv
    assert "--lora-modules" in argv
    name_eq = argv[argv.index("--lora-modules") + 1]
    assert name_eq.startswith("lora-sample-v1=")
    assert name_eq.endswith(str(adapter))


def test_argv_includes_api_key_when_set() -> None:
    backend = VllmBackend(api_key="local-no-auth")
    argv = backend._build_argv(model_id="x", adapter=None, port=8080)
    assert "--api-key" in argv
    assert argv[argv.index("--api-key") + 1] == "local-no-auth"


def test_start_raises_clear_error_when_vllm_missing() -> None:
    """If vllm isn't installed, ``start`` raises VllmNotInstalledError."""
    backend = VllmBackend(startup_timeout_s=1.0)
    # Replace ``import vllm`` with an ImportError to simulate vllm not installed.
    with mock.patch.dict(sys.modules, {"vllm": None}), pytest.raises(VllmNotInstalledError):
        backend.start(model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct", adapter=None, port=8080)


def test_stop_is_idempotent() -> None:
    backend = VllmBackend()
    backend.stop()
    backend.stop()
