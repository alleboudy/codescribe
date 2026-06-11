"""End-to-end orchestration test for ``codescribe-train run``.

Backend + harness are mocked; the test confirms the wiring:

* probe → backend.start → poll-health → harness.prepare → harness.start_session
  → backend.stop (always, in finally)

The full Phase 4 DoD ("user can run a slash command + tool call against the
local fine-tuned model") cannot be exercised end-to-end without a real
fine-tuned GGUF; this test verifies the orchestration path with mocks.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest import mock

import pytest


def test_help_prints_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    from codescribe_train.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "run" in captured.out
    assert "probe" in captured.out


def test_run_orchestration_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Probe → start → harness → stop, in the right order."""
    from codescribe_train import cli

    # Prepare a fake "repo" workdir, a fake GGUF, and a fake backend config.
    workdir = tmp_path / "repo"
    workdir.mkdir()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    backend_cfg = tmp_path / "backend.yaml"
    backend_cfg.write_text(
        "backend: llama-server\nport: 8080\nhost: 127.0.0.1\n"
        f"gguf_path: {gguf}\nmodel: openai/test\n",
        encoding="utf-8",
    )
    harness_cfg = tmp_path / "harness.yaml"
    harness_cfg.write_text("default_permission_mode: ask\nallowed_tools: []\n", encoding="utf-8")

    # Mock the backend and harness builders to return controlled fakes.
    fake_backend = mock.Mock()
    fake_backend.start.return_value = "http://127.0.0.1:8080"
    fake_backend.health_check.return_value = True

    fake_harness = mock.Mock()
    fake_harness.start_session.return_value = 0

    monkeypatch.setattr(cli, "_build_backend", lambda name, config: fake_backend)
    monkeypatch.setattr(cli, "_build_harness", lambda name, *, sandbox: fake_harness)

    # Stub stdin/stdout so harness.start_session sees real streams.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdout", io.StringIO())

    rc = cli.main(
        [
            "run",
            "--repo",
            str(workdir),
            "--adapter",
            str(gguf),
            "--backend",
            "llama-server",
            "--harness",
            "noop",
            "--port",
            "8080",
            "--health-timeout-s",
            "1",
            "--backend-config",
            str(backend_cfg),
            "--harness-config",
            str(harness_cfg),
        ]
    )
    assert rc == 0

    # Assert the orchestration order using the mocked Mock.method_calls
    # property, which records both backend and harness calls in a flat list.
    fake_backend.start.assert_called_once()
    fake_harness.prepare.assert_called_once()
    fake_harness.start_session.assert_called_once()
    fake_backend.stop.assert_called_once()

    # backend.start MUST come before harness.prepare.
    backend_start_calls = [c for c in fake_backend.method_calls if c[0] == "start"]
    backend_stop_calls = [c for c in fake_backend.method_calls if c[0] == "stop"]
    harness_prepare_calls = [c for c in fake_harness.method_calls if c[0] == "prepare"]
    assert backend_start_calls
    assert harness_prepare_calls
    assert backend_stop_calls

    # backend.start was called with the resolved gguf path + port + adapter=None.
    kwargs = fake_backend.start.call_args.kwargs
    assert kwargs["model_id"] == str(gguf)
    assert kwargs["adapter"] is None
    assert kwargs["port"] == 8080

    # harness.prepare was called with the workdir, the endpoint, and the model.
    args, kwargs = fake_harness.prepare.call_args
    assert args[0].resolve() == workdir.resolve()
    assert args[1] == "http://127.0.0.1:8080"
    assert args[2] == "openai/test"


def test_run_calls_backend_stop_even_when_harness_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the harness explodes, the backend still gets stopped."""
    from codescribe_train import cli

    workdir = tmp_path / "repo"
    workdir.mkdir()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    backend_cfg = tmp_path / "backend.yaml"
    backend_cfg.write_text(
        f"backend: llama-server\nport: 8080\ngguf_path: {gguf}\nmodel: openai/test\n",
        encoding="utf-8",
    )

    fake_backend = mock.Mock()
    fake_backend.start.return_value = "http://127.0.0.1:8080"
    fake_backend.health_check.return_value = True

    fake_harness = mock.Mock()
    fake_harness.start_session.side_effect = RuntimeError("kaboom")

    monkeypatch.setattr(cli, "_build_backend", lambda name, config: fake_backend)
    monkeypatch.setattr(cli, "_build_harness", lambda name, *, sandbox: fake_harness)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdout", io.StringIO())

    with pytest.raises(RuntimeError, match="kaboom"):
        cli.main(
            [
                "run",
                "--repo",
                str(workdir),
                "--adapter",
                str(gguf),
                "--backend",
                "llama-server",
                "--harness",
                "noop",
                "--port",
                "8080",
                "--health-timeout-s",
                "1",
                "--backend-config",
                str(backend_cfg),
            ]
        )
    fake_backend.stop.assert_called_once()


def test_run_aborts_when_health_check_never_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the backend never becomes healthy, raise — and still call stop."""
    from codescribe_train import cli

    workdir = tmp_path / "repo"
    workdir.mkdir()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    backend_cfg = tmp_path / "backend.yaml"
    backend_cfg.write_text(
        f"backend: llama-server\nport: 8080\ngguf_path: {gguf}\nmodel: openai/x\n",
        encoding="utf-8",
    )

    fake_backend = mock.Mock()
    fake_backend.start.return_value = "http://127.0.0.1:8080"
    fake_backend.health_check.return_value = False  # never goes healthy

    fake_harness = mock.Mock()
    monkeypatch.setattr(cli, "_build_backend", lambda name, config: fake_backend)
    monkeypatch.setattr(cli, "_build_harness", lambda name, *, sandbox: fake_harness)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdout", io.StringIO())

    with pytest.raises(RuntimeError, match="did not become healthy"):
        cli.main(
            [
                "run",
                "--repo",
                str(workdir),
                "--adapter",
                str(gguf),
                "--backend",
                "llama-server",
                "--harness",
                "noop",
                "--port",
                "8080",
                "--health-timeout-s",
                "0.1",
                "--backend-config",
                str(backend_cfg),
            ]
        )
    fake_backend.stop.assert_called_once()
    fake_harness.prepare.assert_not_called()


def test_run_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from codescribe_train.cli import main

    workdir = tmp_path / "repo"
    workdir.mkdir()
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--repo",
                str(workdir),
                "--adapter",
                "/tmp/x.gguf",
                "--backend",
                "ghost",
                "--harness",
                "noop",
            ]
        )


def test_run_requires_repo_to_exist(tmp_path: Path) -> None:
    from codescribe_train.cli import main

    nope = tmp_path / "definitely-not-here"
    with pytest.raises(FileNotFoundError, match="--repo does not exist"):
        main(
            [
                "run",
                "--repo",
                str(nope),
                "--adapter",
                "/tmp/x.gguf",
                "--backend",
                "llama-server",
                "--harness",
                "noop",
            ]
        )


def test_probe_subcommand_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from codescribe_train import cli
    from codescribe_train.backends.probe import HardwareProfile

    fake_profile = HardwareProfile(
        gpu_name="Example GPU", gpu_vram_gib=8.0, system_ram_gib=32.0, cpu_count=16
    )
    monkeypatch.setattr("codescribe_train.backends.probe.probe_hardware", lambda: fake_profile)
    rc = cli.main(["probe"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "hardware" in captured.out
    assert "Example GPU" in captured.out
