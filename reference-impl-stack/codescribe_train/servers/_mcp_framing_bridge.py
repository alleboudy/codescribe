"""LSP↔newline-delimited stdio framing bridge for MCP servers.

claw-code's stdio MCP transport uses **LSP-style Content-Length framing**
(``Content-Length: N\\r\\n\\r\\n<payload>``); the Python MCP SDK
(``mcp`` package, 1.27.x) uses **newline-delimited JSON-RPC** on stdio.
This bridge sits between them: spawned by claw, it speaks LSP framing
upstream and newline-delimited downstream, where the actual MCP server
(``codescribe_train.servers.rag_server``, ``codescribe_train.servers.repo_grep``)
runs as a child.

Usage:

    python -m codescribe_train.servers._mcp_framing_bridge <module> [args...]

The bridge re-executes the same interpreter (``sys.executable``) to
spawn ``python -m <module> [args...]`` as the downstream MCP server, so
the venv that hosts the bridge also hosts the server. stderr is
inherited so server logs land wherever claw points them.

This file is pure-stdlib (no MCP imports) so it doesn't load the rag
extra into the bridge process — only the spawned child needs the
``rag`` extra.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
from typing import BinaryIO


def _read_lsp_frame(reader: BinaryIO) -> bytes | None:  # noqa: PLR0911 — guard-clause parser; flat early-returns read clearer than nesting
    """Read one LSP-framed message off *reader*; return JSON payload bytes
    (no trailing newline), or ``None`` on EOF / malformed framing.

    Header lines are case-insensitive ASCII ``Name: value`` pairs ending
    in CRLF or LF, terminated by an empty line. We require a
    ``Content-Length`` header; other headers are tolerated and ignored.
    """
    headers: dict[str, str] = {}
    while True:
        line = reader.readline()
        if not line:
            return None  # EOF before the header block ended
        if line in (b"\r\n", b"\n"):
            break  # blank line terminates header block
        try:
            decoded = line.decode("ascii").rstrip("\r\n")
        except UnicodeDecodeError:
            return None
        name, _, value = decoded.partition(":")
        if name:
            headers[name.strip().lower()] = value.strip()
    raw_length = headers.get("content-length")
    if raw_length is None:
        return None
    try:
        length = int(raw_length)
    except ValueError:
        return None
    if length < 0:
        return None
    payload = reader.read(length)
    if len(payload) != length:
        return None  # truncated
    return payload


def _write_lsp_frame(writer: BinaryIO, payload: bytes) -> None:
    """Wrap *payload* in an LSP frame and write it to *writer* unbuffered."""
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
    writer.write(header)
    writer.write(payload)
    writer.flush()


def _parent_to_child(parent_reader: BinaryIO, child_writer: BinaryIO) -> None:
    """Pump claw → server: parse LSP frames, write newline-delimited."""
    try:
        while True:
            payload = _read_lsp_frame(parent_reader)
            if payload is None:
                break
            child_writer.write(payload)
            if not payload.endswith(b"\n"):
                child_writer.write(b"\n")
            child_writer.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        with contextlib.suppress(OSError):
            child_writer.close()


def _child_to_parent(child_reader: BinaryIO, parent_writer: BinaryIO) -> None:
    """Pump server → claw: read JSON lines, wrap in LSP frames."""
    try:
        for line in child_reader:
            payload = line.rstrip(b"\r\n")
            if not payload:
                continue  # ignore blank lines from the child
            _write_lsp_frame(parent_writer, payload)
    except (BrokenPipeError, OSError):
        pass


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "usage: python -m codescribe_train.servers._mcp_framing_bridge "
            "<module> [args...]",
            file=sys.stderr,
        )
        return 2

    module, *module_args = args
    cmd = [sys.executable, "-m", module, *module_args]

    child = subprocess.Popen(  # noqa: S603 — args constructed locally
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        # inherit stderr so the child's logs land where claw expects them
        env=os.environ.copy(),
    )

    if child.stdin is None or child.stdout is None:  # pragma: no cover
        return 3

    pump_in = threading.Thread(
        target=_parent_to_child,
        args=(sys.stdin.buffer, child.stdin),
        daemon=True,
        name="bridge:parent_to_child",
    )
    pump_out = threading.Thread(
        target=_child_to_parent,
        args=(child.stdout, sys.stdout.buffer),
        daemon=True,
        name="bridge:child_to_parent",
    )
    pump_in.start()
    pump_out.start()

    rc = child.wait()
    # Drain remaining output so the last server response makes it through.
    pump_out.join(timeout=2.0)
    return int(rc)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
