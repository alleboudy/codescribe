"""No ``print(`` calls anywhere in the ``rag_server`` package.

Spec test from the design spec:

  > test_no_stdout.py — capture stdout during a session; only
  > JSON-RPC frames present.

Two checks:

1. **Static grep**: no ``print(`` occurrence in any source file. Even
   inside ``if __name__ == "__main__"`` guards — copy-paste mistakes
   from there have leaked into helpers in past projects.
2. **Live capture**: spin up the in-process server, run a tool call,
   and confirm the process did not write anything to stdout outside
   the SDK's JSON-RPC channel (the in-memory transport bypasses
   stdio entirely, so capturing stdout during the test catches any
   accidental ``print`` or ``sys.stdout.write`` in our code).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from tests.servers.rag_server.conftest import StubEmbedder

_PKG_ROOT = Path(__file__).resolve().parents[3] / "codescribe_train" / "servers" / "rag_server"


def test_static_no_print_in_package_sources() -> None:
    """Grep every ``*.py`` in the package — no ``print(`` may appear."""
    offenders: list[tuple[Path, int, str]] = []
    for source in _PKG_ROOT.rglob("*.py"):
        for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if "print(" in line:
                offenders.append((source, lineno, line.rstrip()))
    assert not offenders, (
        "print( occurrences found in rag_server sources — stdio is sacred:\n"
        + "\n".join(f"  {p}:{ln}: {text}" for p, ln, text in offenders)
    )


@pytest.mark.asyncio
async def test_live_session_writes_nothing_to_stdout(
    seeded_db: Path,
    stub_embedder: StubEmbedder,
) -> None:
    """An in-process session + tool call writes zero bytes to ``sys.stdout``."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from codescribe_train.servers.rag_server.server import build_server

    server = build_server(
        db_path=seeded_db,
        log_path=seeded_db.parent / "log.log",
        embedder=stub_embedder,
    )

    captured = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = captured
    try:
        async with create_connected_server_and_client_session(server) as client:
            for tool, args in [
                ("find_similar_issues", {"query": "cart", "k": 5}),
                ("get_pr_diff", {"pr_number": 42, "max_chars": 8000}),
                ("search_commits", {"query": "cart", "k": 5}),
            ]:
                await client.call_tool(tool, arguments=args)
    finally:
        sys.stdout = real_stdout

    assert captured.getvalue() == "", (
        f"unexpected stdout output during MCP session:\n{captured.getvalue()!r}"
    )
