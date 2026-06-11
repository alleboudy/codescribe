"""Tool handlers never write to the rag store.

Spec test from the design spec:

  > test_no_db_writes.py — wrap SQLite connection to assert no
  > INSERT/UPDATE/DELETE during tool calls.

Uses sqlite3's ``set_authorizer`` callback, which fires once per
SQL action and lets the test reject any write attempt by returning
``SQLITE_DENY``. A denied write surfaces as ``OperationalError`` —
the test asserts no such error fires and that the authorizer never
saw a write action code.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.servers.rag_server.conftest import StubEmbedder

# sqlite3 action codes for the write paths we care about. The full list
# lives in https://www.sqlite.org/c3ref/c_alter_table.html — these are
# the codes that surface as INSERT/UPDATE/DELETE/CREATE/DROP.
_WRITE_ACTION_CODES = frozenset({
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_CREATE_VTABLE,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_DROP_VTABLE,
})


@pytest.mark.asyncio
async def test_tool_calls_attempt_no_writes(
    seeded_db: Path,
    stub_embedder: StubEmbedder,
) -> None:
    """All three tool calls complete without firing a write action code."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from codescribe_train.servers.rag_server.server import build_server

    server = build_server(
        db_path=seeded_db,
        log_path=seeded_db.parent / "log.log",
        embedder=stub_embedder,
    )
    # The server stashes the connection on the server instance so
    # __main__.py can close it at shutdown — we reuse it here to install
    # the authorizer on the same handle the tool calls go through.
    conn: sqlite3.Connection = server._sample_rag_conn  # type: ignore[attr-defined]

    write_attempts: list[tuple[int, str | None, str | None]] = []

    def _authorizer(action_code, arg1, arg2, _db_name, _trigger_name):  # noqa: ANN001
        if action_code in _WRITE_ACTION_CODES:
            write_attempts.append((action_code, arg1, arg2))
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(_authorizer)
    try:
        async with create_connected_server_and_client_session(server) as client:
            for tool, args in [
                ("find_similar_issues", {"query": "cart", "k": 5}),
                ("get_pr_diff", {"pr_number": 42, "max_chars": 8000}),
                ("search_commits", {"query": "cart", "k": 5}),
            ]:
                result = await client.call_tool(tool, arguments=args)
                # If a write was attempted the handler would have raised
                # OperationalError, which the lowlevel server converts
                # to isError=True. Surface that immediately so the test
                # message is useful, then keep checking write_attempts.
                assert result.isError is False, (
                    f"{tool} returned an error response (likely a denied "
                    f"write): {result.content!r}"
                )
    finally:
        conn.set_authorizer(None)

    assert write_attempts == [], (
        f"tool handlers attempted writes: {write_attempts!r}"
    )
