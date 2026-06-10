"""Tests for ``LlamaServerBackend``.

Subprocess and HTTP are mocked — no actual ``llama-server`` is launched
and no network call is made. The point is to lock down argv construction,
the log-file path, the readiness-watcher logic, and the start/stop
lifecycle.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest import mock

import pytest

from codescribe_train.backends.base import Backend
from codescribe_train.backends.llama_server import (
    DEFAULT_LLAMA_SERVER_BINARY,
    LlamaServerBackend,
    poll_health_until,
)


def _make_backend(tmp_path: Path, **overrides) -> LlamaServerBackend:
    """Build a backend with a fake binary + log dir under tmp_path."""
    fake_binary = tmp_path / "fake-llama-server"
    fake_binary.write_text("#!/bin/sh\nexit 0\n")
    fake_binary.chmod(0o755)
    kwargs: dict = {
        "binary_path": fake_binary,
        "log_dir": tmp_path / "logs",
        "startup_timeout_s": 1.0,  # tests must not hang
    }
    kwargs.update(overrides)
    return LlamaServerBackend(**kwargs)


def test_is_a_backend(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    assert isinstance(backend, Backend)


def test_argv_construction(tmp_path: Path) -> None:
    backend = _make_backend(
        tmp_path,
        host="127.0.0.1",
        ctx_size=8192,
        gpu_layers=99,
        no_mmap=True,
        api_key="secret",
        extra_args=["--threads", "8"],
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")
    argv = backend._build_argv(model_path=model, port=8080)

    # Order matters for some llama-server flags; pin it.
    assert argv[0] == str(backend.binary_path)
    assert argv[1] == "-m"
    assert argv[2] == str(model)
    assert "--host" in argv and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8080"
    assert "-ngl" in argv and argv[argv.index("-ngl") + 1] == "99"
    assert "--ctx-size" in argv and argv[argv.index("--ctx-size") + 1] == "8192"
    assert "--no-mmap" in argv
    assert "--api-key" in argv and argv[argv.index("--api-key") + 1] == "secret"
    # extra_args are appended verbatim.
    assert argv[-2:] == ["--threads", "8"]


def test_argv_omits_no_mmap_when_disabled(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, no_mmap=False)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")
    argv = backend._build_argv(model_path=model, port=8080)
    assert "--no-mmap" not in argv


def test_argv_omits_api_key_when_unset(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, api_key=None)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")
    argv = backend._build_argv(model_path=model, port=8080)
    assert "--api-key" not in argv


def test_start_rejects_separate_adapter(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")
    adapter = tmp_path / "lora-adapter"
    adapter.mkdir()
    with pytest.raises(ValueError, match="does not accept a separate"):
        backend.start(model_id=str(model), adapter=adapter, port=8080)


def test_start_raises_when_model_missing(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    with pytest.raises(FileNotFoundError, match="GGUF model not found"):
        backend.start(model_id=str(tmp_path / "nope.gguf"), adapter=None, port=8080)


def test_start_raises_when_binary_missing(tmp_path: Path) -> None:
    # Construct a backend pointing at a non-existent binary.
    backend = LlamaServerBackend(
        binary_path=tmp_path / "missing-binary",
        log_dir=tmp_path / "logs",
        startup_timeout_s=1.0,
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="llama-server binary not found"):
        backend.start(model_id=str(model), adapter=None, port=8080)


def test_default_binary_path_points_at_vendored_build() -> None:
    """The default points where build_llama_cpp.sh is documented to land."""
    assert str(DEFAULT_LLAMA_SERVER_BINARY).endswith("vendor/llama.cpp/build/bin/llama-server")


def test_start_writes_marker_then_returns_endpoint(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")

    fake_proc = mock.Mock()
    fake_proc.poll.return_value = None  # process is "running"
    fake_proc.returncode = None

    def _popen_factory(cmd, **kwargs):
        # Simulate the binary writing the ready marker to its log.
        log_handle = kwargs["stdout"]
        log_handle.write("HTTP server listening on 127.0.0.1:8080\n")
        log_handle.flush()
        return fake_proc

    with mock.patch(
        "codescribe_train.backends.llama_server.subprocess.Popen",
        side_effect=_popen_factory,
    ):
        endpoint = backend.start(model_id=str(model), adapter=None, port=8123)

    assert endpoint == "http://127.0.0.1:8123"
    # Log file lives under the configured log_dir.
    assert (tmp_path / "logs" / "llama-server.log").exists()


def test_start_times_out_when_marker_never_appears(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")

    fake_proc = mock.Mock()
    fake_proc.poll.return_value = None

    def _popen_factory(cmd, **kwargs):
        # Never write a ready marker; let the timeout trip.
        return fake_proc

    with (
        mock.patch(
            "codescribe_train.backends.llama_server.subprocess.Popen",
            side_effect=_popen_factory,
        ),
        pytest.raises(TimeoutError, match="did not become ready"),
    ):
        backend.start(model_id=str(model), adapter=None, port=8080)


def test_start_raises_when_process_exits_early(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")

    fake_proc = mock.Mock()
    # Process has already exited with non-zero by the time we look.
    fake_proc.poll.return_value = 1
    fake_proc.returncode = 1

    def _popen_factory(cmd, **kwargs):
        return fake_proc

    with (
        mock.patch(
            "codescribe_train.backends.llama_server.subprocess.Popen",
            side_effect=_popen_factory,
        ),
        pytest.raises(RuntimeError, match="exited with code 1"),
    ):
        backend.start(model_id=str(model), adapter=None, port=8080)


def test_stop_is_idempotent_when_never_started(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    backend.stop()
    backend.stop()  # second call must still succeed


def test_stop_terminates_running_process(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    fake_proc = mock.Mock()
    fake_proc.poll.return_value = None  # running
    backend._process = fake_proc
    backend._log_file_handle = io.StringIO()

    backend.stop()
    fake_proc.terminate.assert_called_once()
    fake_proc.wait.assert_called()
    assert backend._process is None


def test_stop_kills_unresponsive_process(tmp_path: Path) -> None:
    import subprocess as sp

    backend = _make_backend(tmp_path)
    fake_proc = mock.Mock()
    fake_proc.poll.return_value = None
    fake_proc.wait.side_effect = [sp.TimeoutExpired(cmd="x", timeout=10), None]
    backend._process = fake_proc

    backend.stop()
    fake_proc.terminate.assert_called_once()
    fake_proc.kill.assert_called_once()
    assert backend._process is None


# ---------------------------------------------------------------------- #
# poll_health_until — the helper the top-level CLI uses to block on readiness.
# ---------------------------------------------------------------------- #


def test_poll_health_until_returns_true_when_backend_healthy(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    with mock.patch.object(backend, "health_check", return_value=True) as hc:
        assert poll_health_until(backend, "http://x", interval_s=0.01, timeout_s=1.0)
    hc.assert_called()


def test_poll_health_until_returns_false_on_timeout(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    with mock.patch.object(backend, "health_check", return_value=False):
        assert not poll_health_until(backend, "http://x", interval_s=0.01, timeout_s=0.05)
