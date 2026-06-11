"""Render ``.claude.json`` / ``.claw.json`` from harness config dicts.

Exercises the pure config-rendering helpers and ``ClawCodeHarness.prepare``.
``start_session`` is exercised with the subprocess mocked out — we never
actually launch the Rust binary in pytest.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from codescribe_train.harness.claw import (
    ClawCodeHarness,
    render_claude_json,
    render_claw_json,
)


def test_render_claude_json_minimal() -> None:
    payload = render_claude_json({}, backend_url="http://localhost:8080", model="openai/qwen-7b")
    assert payload["model"] == "openai/qwen-7b"
    assert payload["permissions"]["defaultMode"] == "ask"
    assert payload["_metadata"]["backendUrl"] == "http://localhost:8080"
    assert "mcpServers" not in payload  # empty by default


def test_render_claude_json_full_overlay() -> None:
    config: dict[str, Any] = {
        "default_permission_mode": "ask",
        "allowed_tools": ["Read", "Edit"],
        "denied_tools": ["Bash"],
        "prompt_overlay": "../your-repo/CLAUDE.md",
        "slash_commands": ["help", "status"],
        "mcp_servers": {"fs": {"command": "fs-mcp"}},
        "claude_json_extra": {"customField": 42},
    }
    payload = render_claude_json(
        config, backend_url="http://localhost:8080", model="openai/qwen-7b"
    )
    assert payload["permissions"]["allowedTools"] == ["Read", "Edit"]
    assert payload["permissions"]["deniedTools"] == ["Bash"]
    assert payload["promptOverlay"] == "../your-repo/CLAUDE.md"
    assert payload["slashCommands"] == ["help", "status"]
    assert payload["mcpServers"] == {"fs": {"command": "fs-mcp"}}
    assert payload["customField"] == 42


def test_render_claw_json_aliases_and_extras() -> None:
    config: dict[str, Any] = {
        "aliases": {"quick": "haiku"},
        "claw_json_extra": {"foo": "bar"},
    }
    payload = render_claw_json(config)
    assert payload == {"aliases": {"quick": "haiku"}, "foo": "bar"}


def test_render_claw_json_empty() -> None:
    assert render_claw_json({}) == {}


def test_prepare_writes_config_files(tmp_path: Path) -> None:
    h = ClawCodeHarness()
    config: dict[str, Any] = {
        "default_permission_mode": "ask",
        "allowed_tools": ["Read", "Grep"],
        "aliases": {"q": "qwen"},
    }
    h.prepare(
        tmp_path,
        "http://localhost:8080",
        "openai/qwen2.5-coder-7b-sample",
        harness_config=config,
    )
    claude = json.loads((tmp_path / ".claude.json").read_text())
    claw = json.loads((tmp_path / ".claw.json").read_text())
    assert claude["model"] == "openai/qwen2.5-coder-7b-sample"
    assert claude["permissions"]["allowedTools"] == ["Read", "Grep"]
    assert claw["aliases"] == {"q": "qwen"}
    assert h.prepared is True


def test_prepare_rejects_missing_workdir(tmp_path: Path) -> None:
    h = ClawCodeHarness()
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        h.prepare(missing, "http://localhost:8080", "openai/m", harness_config={})


def test_sandbox_args_none_returns_empty() -> None:
    h = ClawCodeHarness(sandbox="none")
    assert h.sandbox_args() == []


def test_sandbox_args_docker_requires_workdir(tmp_path: Path) -> None:
    h = ClawCodeHarness(sandbox="docker")
    with pytest.raises(RuntimeError):
        h.sandbox_args()
    h.prepare(tmp_path, "http://localhost:8080", "openai/m", harness_config={})
    args = h.sandbox_args()
    assert args[:5] == ["docker", "run", "--rm", "-i", "--network"]
    assert "none" in args
    assert str(tmp_path) in " ".join(args)
    assert h.docker_image in args


def test_sandbox_args_unknown_mode_raises() -> None:
    h = ClawCodeHarness(sandbox="bwrap")  # not implemented yet
    with pytest.raises(ValueError):
        h.sandbox_args()


def test_start_session_before_prepare_raises() -> None:
    h = ClawCodeHarness()
    with pytest.raises(RuntimeError):
        h.start_session(io.StringIO(""), io.StringIO())


def test_start_session_uses_subprocess_run(tmp_path: Path) -> None:
    """Mock subprocess.run so we don't actually launch the binary."""
    fake_binary = tmp_path / "fake-claw"
    fake_binary.write_text("#!/bin/sh\necho ok\n")
    fake_binary.chmod(0o755)

    h = ClawCodeHarness(binary_path=fake_binary)
    h.prepare(tmp_path, "http://localhost:8080", "openai/m", harness_config={})

    with mock.patch("codescribe_train.harness.claw.subprocess.run") as run_mock:
        run_mock.return_value = mock.Mock(returncode=0)
        rc = h.start_session(io.StringIO(""), io.StringIO())
    assert rc == 0
    assert run_mock.called
    call_args = run_mock.call_args
    cmd = call_args.args[0] if call_args.args else call_args.kwargs["args"]
    assert cmd[0] == str(fake_binary)
    env = call_args.kwargs.get("env") or {}
    assert env["OPENAI_BASE_URL"] == "http://localhost:8080"
    assert env["OPENAI_MODEL"] == "openai/m"
    # Defence in depth: ANTHROPIC_BASE_URL pinned to the local backend so
    # a model-name typo can't leak to api.anthropic.com.
    assert env["ANTHROPIC_BASE_URL"] == "http://localhost:8080"
    # API key sentinel must not be a real-looking secret.
    assert env["OPENAI_API_KEY"] == "local-no-auth"


def test_start_session_missing_binary_raises(tmp_path: Path) -> None:
    h = ClawCodeHarness(binary_path=tmp_path / "absent")
    h.prepare(tmp_path, "http://localhost:8080", "openai/m", harness_config={})
    with pytest.raises(FileNotFoundError):
        h.start_session(io.StringIO(""), io.StringIO())


def test_render_claude_json_mcp_servers_v2() -> None:
    """v2 MCP servers (repo-grep + rag-server) render into .claude.json."""
    config: dict[str, Any] = {
        "mcp_servers": {
            "repo-grep": {
                "command": "python",
                "args": ["-m", "codescribe_train.servers.repo_grep"],
            },
            "rag-server": {
                "command": "python",
                "args": ["-m", "codescribe_train.servers.rag_server"],
                "env": {"RAG_DB_PATH": "indices/rag.db"},
            },
        },
    }
    payload = render_claude_json(
        config, backend_url="http://localhost:8080", model="openai/qwen-7b"
    )
    assert "mcpServers" in payload
    servers = payload["mcpServers"]
    assert "repo-grep" in servers
    assert servers["repo-grep"]["args"] == ["-m", "codescribe_train.servers.repo_grep"]
    assert "rag-server" in servers
    assert servers["rag-server"]["env"]["RAG_DB_PATH"] == "indices/rag.db"


def test_prepare_writes_mcp_servers_to_claude_json(tmp_path: Path) -> None:
    h = ClawCodeHarness()
    config: dict[str, Any] = {
        "mcp_servers": {
            "repo-grep": {
                "command": "python",
                "args": ["-m", "codescribe_train.servers.repo_grep"],
            },
        },
    }
    h.prepare(tmp_path, "http://localhost:8080", "openai/m", harness_config=config)
    claude = json.loads((tmp_path / ".claude.json").read_text())
    assert "mcpServers" in claude
    assert "repo-grep" in claude["mcpServers"]


def test_health_check_returns_false_on_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When httpx is installed but the backend is unreachable, return False."""
    pytest.importorskip("httpx")
    h = ClawCodeHarness()
    # Use an unroutable address; httpx.get raises HTTPError.
    assert h.health_check("http://127.0.0.1:1") is False


def test_health_check_handles_missing_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    h = ClawCodeHarness()
    if isinstance(__builtins__, dict):
        real_import = __builtins__["__import__"]  # type: ignore[index]
    else:
        real_import = __builtins__.__import__

    def fail_httpx(name: str, *args: Any, **kwargs: Any):
        if name == "httpx":
            raise ImportError("simulated missing httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_httpx)
    assert h.health_check("http://localhost:8080") is False
