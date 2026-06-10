"""MCP server exposing ripgrep-based lexical search over a local repo."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO_DIR = Path(os.environ.get("TARGET_REPO", "../your-repo")).resolve()

mcp = FastMCP("repo-grep")


@mcp.tool()
def grep(  # noqa: PLR0912  # tool dispatch + result parser; flat is clearer than nested
    pattern: str,
    path: str | None = None,
    language: str | None = None,
    max_results: int = 50,
) -> list[dict]:
    """Fast lexical search over the repo using ripgrep."""
    # Reject patterns that aren't valid regex.
    try:
        re.compile(pattern)
    except re.error as exc:
        return [{"error": f"Invalid regex pattern: {exc}"}]

    search_dir = REPO_DIR
    if path is not None:
        scoped = (REPO_DIR / path).resolve()
        if not str(scoped).startswith(str(REPO_DIR)):
            return [{"error": "Path escapes the repo directory"}]
        search_dir = scoped

    cmd: list[str] = ["rg", "--json", pattern]
    if language is not None:
        cmd.extend(["--type", language])
    cmd.append(str(search_dir))

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
    except FileNotFoundError:
        return [{"error": "ripgrep (rg) is not installed or not on PATH"}]
    except subprocess.TimeoutExpired:
        return [{"error": "ripgrep timed out after 30 seconds"}]

    if proc.returncode not in (0, 1):
        return [{"error": f"rg failed (exit {proc.returncode}): {proc.stderr.strip()}"}]

    results: list[dict] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "match":
            continue
        data = obj["data"]
        abs_path = Path(data["path"]["text"])
        try:
            rel = abs_path.relative_to(REPO_DIR)
        except ValueError:
            rel = abs_path
        results.append({
            "src_path": str(rel),
            "line": data["line_number"],
            "match": data["lines"]["text"].rstrip("\n"),
        })
        if len(results) >= max_results:
            break

    return results
