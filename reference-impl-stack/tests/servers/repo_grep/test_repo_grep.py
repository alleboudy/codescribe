"""Tests for the repo-grep MCP server.

Subprocess is mocked — no actual ``rg`` binary is required.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest import mock

from codescribe_train.servers.repo_grep.server import grep


def _rg_json_line(path: str, line_number: int, text: str) -> str:
    """Build one ripgrep JSON match line."""
    return json.dumps({
        "type": "match",
        "data": {
            "path": {"text": path},
            "line_number": line_number,
            "lines": {"text": text + "\n"},
        },
    })


def _make_rg_output(*match_tuples: tuple[str, int, str]) -> str:
    """Build multi-line rg --json stdout from (path, lineno, text) tuples."""
    summary = json.dumps({"type": "summary", "data": {}})
    lines = [_rg_json_line(p, n, t) for p, n, t in match_tuples]
    lines.append(summary)
    return "\n".join(lines)


@mock.patch("codescribe_train.servers.repo_grep.server.REPO_DIR", Path("/fake/repo"))
class TestGrep:
    """All tests run with a mocked REPO_DIR to avoid filesystem access."""

    @mock.patch("codescribe_train.servers.repo_grep.server.subprocess.run")
    def test_grep_parses_rg_json_output(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=_make_rg_output(
                ("/fake/repo/src/main.py", 10, "def hello():"),
                ("/fake/repo/src/lib.py", 42, "    hello()"),
            ),
            stderr="",
        )
        results = grep("hello")
        assert len(results) == 2
        assert results[0] == {"src_path": "src/main.py", "line": 10, "match": "def hello():"}
        assert results[1] == {"src_path": "src/lib.py", "line": 42, "match": "    hello()"}

    def test_grep_rejects_invalid_regex(self) -> None:
        results = grep("[invalid")
        assert len(results) == 1
        assert "error" in results[0]
        assert "Invalid regex" in results[0]["error"]

    @mock.patch("codescribe_train.servers.repo_grep.server.subprocess.run")
    def test_grep_respects_max_results(self, mock_run: mock.Mock) -> None:
        matches = [("/fake/repo/f.py", i, f"line {i}") for i in range(100)]
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=_make_rg_output(*matches),
            stderr="",
        )
        results = grep("line", max_results=3)
        assert len(results) == 3

    @mock.patch("codescribe_train.servers.repo_grep.server.subprocess.run")
    def test_grep_scopes_to_path(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        grep("pattern", path="src/subdir")
        args = mock_run.call_args[0][0]
        assert str(Path("/fake/repo/src/subdir")) in args

    def test_grep_rejects_path_traversal(self) -> None:
        results = grep("pattern", path="../../etc/passwd")
        assert len(results) == 1
        assert "error" in results[0]
        assert "escapes" in results[0]["error"]

    @mock.patch(
        "codescribe_train.servers.repo_grep.server.subprocess.run",
        side_effect=FileNotFoundError("rg"),
    )
    def test_grep_handles_rg_not_found(self, mock_run: mock.Mock) -> None:
        results = grep("pattern")
        assert len(results) == 1
        assert "error" in results[0]
        assert "not installed" in results[0]["error"]
