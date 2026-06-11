"""Tests for the hardware probe + recommendation logic.

The probe shells out to ``nvidia-smi`` (and optionally ``pynvml``) to
read GPU specs. Both are mocked here so the tests run in any
environment, with or without an NVIDIA GPU.
"""

from __future__ import annotations

import sys
from unittest import mock

from codescribe_train.backends.probe import (
    BackendChoice,
    HardwareProfile,
    _probe_gpu,
    probe_hardware,
    recommend_backend,
)

# ---------------------------------------------------------------------- #
# Profile / Choice surface
# ---------------------------------------------------------------------- #


def test_profile_summary_with_gpu() -> None:
    p = HardwareProfile(
        gpu_name="Example GPU", gpu_vram_gib=8.0, system_ram_gib=32.0, cpu_count=16
    )
    s = p.summary()
    assert "Example GPU" in s
    assert "8.0 GiB VRAM" in s
    assert "32.0 GiB" in s
    assert "16 threads" in s


def test_profile_summary_without_gpu() -> None:
    p = HardwareProfile(gpu_name=None, gpu_vram_gib=None, system_ram_gib=16.0, cpu_count=8)
    s = p.summary()
    assert "gpu=none" in s


def test_profile_has_gpu() -> None:
    assert HardwareProfile("X", 8.0, 32.0, 16).has_gpu()
    assert not HardwareProfile(None, None, 32.0, 16).has_gpu()
    assert not HardwareProfile("X", 0.0, 32.0, 16).has_gpu()


# ---------------------------------------------------------------------- #
# probe_hardware integration — pynvml + nvidia-smi mocked
# ---------------------------------------------------------------------- #


def test_probe_falls_back_to_nvidia_smi_when_pynvml_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)  # ImportError on import
    fake_result = mock.Mock(returncode=0)
    fake_result.stdout = "NVIDIA GeForce Example GPU, 8192\n"
    with (
        mock.patch("codescribe_train.backends.probe.shutil.which", return_value="/usr/bin/nvidia-smi"),
        mock.patch("codescribe_train.backends.probe.subprocess.run", return_value=fake_result),
    ):
        name, vram = _probe_gpu()
    assert name == "NVIDIA GeForce Example GPU"
    assert vram is not None
    # 8192 MiB → 8.0 GiB
    assert 7.5 < vram < 8.5


def test_probe_returns_none_when_no_gpu_at_all(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)
    with mock.patch("codescribe_train.backends.probe.shutil.which", return_value=None):
        name, vram = _probe_gpu()
    assert name is None
    assert vram is None


def test_probe_hardware_returns_frozen_profile(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)
    fake_result = mock.Mock(returncode=0)
    fake_result.stdout = "Example GPU, 8192\n"
    with (
        mock.patch("codescribe_train.backends.probe.shutil.which", return_value="/usr/bin/nvidia-smi"),
        mock.patch("codescribe_train.backends.probe.subprocess.run", return_value=fake_result),
    ):
        profile = probe_hardware()
    assert isinstance(profile, HardwareProfile)
    # Frozen dataclass — assignment must fail.
    import dataclasses

    assert dataclasses.is_dataclass(profile)
    try:
        profile.gpu_name = "different"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("HardwareProfile must be frozen")


# ---------------------------------------------------------------------- #
# recommend_backend — backend-selection heuristics
# ---------------------------------------------------------------------- #


def test_recommend_for_8gb_vram_qwen7b_q4() -> None:
    """8 GB-VRAM GPU / Qwen 7B Q4_K_M (~4.6 GiB)."""
    profile = HardwareProfile(
        gpu_name="Example GPU", gpu_vram_gib=8.0, system_ram_gib=32.0, cpu_count=16
    )
    choice = recommend_backend(profile, model_size_gib=4.6)
    assert isinstance(choice, BackendChoice)
    assert choice.backend_name == "llama-server"
    # Headroom 8.0 - 4.6 = 3.4 → comfortable; current heuristic returns
    # llama-server with no-mmap=False because headroom > 2.0.
    assert choice.options["gpu_layers"] == 99
    assert choice.options["ctx_size"] == 8192


def test_recommend_for_tight_vram() -> None:
    """Headroom < 2 GiB → no-mmap on, all layers on GPU."""
    profile = HardwareProfile(
        gpu_name="Example GPU", gpu_vram_gib=8.0, system_ram_gib=32.0, cpu_count=16
    )
    choice = recommend_backend(profile, model_size_gib=7.0)  # headroom 1.0
    assert choice.backend_name == "llama-server"
    assert choice.options["gpu_layers"] == 99
    assert choice.options["no_mmap"] is True
    assert "tight" in choice.rationale


def test_recommend_for_oversized_model() -> None:
    """Model > VRAM → still llama-server, partial offload."""
    profile = HardwareProfile(
        gpu_name="Example GPU", gpu_vram_gib=8.0, system_ram_gib=32.0, cpu_count=16
    )
    choice = recommend_backend(profile, model_size_gib=14.0)  # headroom -6.0
    assert choice.backend_name == "llama-server"
    assert choice.options["no_mmap"] is True
    assert "partial GPU offload" in choice.rationale


def test_recommend_for_cpu_only_box() -> None:
    """No GPU → llama-server CPU-only, smaller ctx, with a slow-warning."""
    profile = HardwareProfile(gpu_name=None, gpu_vram_gib=None, system_ram_gib=16.0, cpu_count=8)
    choice = recommend_backend(profile, model_size_gib=4.6)
    assert choice.backend_name == "llama-server"
    assert choice.options["gpu_layers"] == 0
    assert "no GPU" in choice.rationale.lower() or "slow" in choice.rationale.lower()


def test_recommend_never_picks_vllm_on_8gb_gpu() -> None:
    """vLLM is never the default on a tight-VRAM GPU, regardless of headroom."""
    profile = HardwareProfile(
        gpu_name="Example GPU", gpu_vram_gib=8.0, system_ram_gib=32.0, cpu_count=16
    )
    for size in (1.0, 4.6, 7.0, 14.0):
        choice = recommend_backend(profile, model_size_gib=size)
        assert choice.backend_name != "vllm"
