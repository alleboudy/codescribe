"""``configs/harness/sample.yaml`` advertises both MCP servers.

Per the design spec: the harness YAML feeds the
``mcp_servers:`` block into ``render_claude_json``, which writes them
into the per-session ``.claude.json`` next to the workdir. For the
``codescribe-train run`` orchestration path (the harness-driven session)
this is how the LLM ends up seeing the ``repo-rag`` tools. An earlier step
removed the v1 ``repo-rag`` entry from this file; a later step adds the v2
entry back alongside the kept ``repo-grep`` entry.

The committed YAML can use repo-relative paths (``../your-repo``) because
``codescribe-train run`` runs from a known workdir. ``.claw-mcp.json``, by
contrast, uses absolute paths because it's discovered by the
``claw`` binary at an arbitrary cwd.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_YAML = REPO_ROOT / "configs" / "harness" / "sample.yaml"


def test_sample_yaml_lists_both_mcp_servers() -> None:
    """Both ``repo-rag`` and ``repo-grep`` are wired in the harness YAML."""
    config = yaml.safe_load(HARNESS_YAML.read_text(encoding="utf-8"))
    mcp_servers = config.get("mcp_servers") or {}
    assert "repo-grep" in mcp_servers, "the spec keeps repo-grep; it must stay"
    assert "repo-rag" in mcp_servers, (
        "the spec adds repo-rag back to the harness YAML so the orchestrated "
        "session also exposes the RAG tools to claw"
    )


def test_sample_rag_entry_targets_v2_module_path() -> None:
    """The ``repo-rag`` entry spawns the v2 module path."""
    config = yaml.safe_load(HARNESS_YAML.read_text(encoding="utf-8"))
    entry = config["mcp_servers"]["repo-rag"]
    assert entry["command"] == "python"
    assert entry["args"] == ["-m", "codescribe_train.servers.rag_server"]
    env = entry.get("env") or {}
    # In the harness YAML we use repo-relative paths — the orchestrator's
    # cwd is the target workdir, not $HOME/... .
    assert "RAG_DB_PATH" in env
    assert "RAG_LOG_PATH" in env


def test_repo_grep_entry_unchanged() -> None:
    """The kept ``repo-grep`` entry still points at the v2 in-package path."""
    config = yaml.safe_load(HARNESS_YAML.read_text(encoding="utf-8"))
    entry = config["mcp_servers"]["repo-grep"]
    assert entry["command"] == "python"
    assert entry["args"] == ["-m", "codescribe_train.servers.repo_grep"]
