"""Unit + integration tests for the LSP↔newline MCP framing bridge.

The bridge is the only thing that lets claw (LSP-framed stdio MCP)
talk to ``codescribe_train.servers.rag_server`` / ``repo_grep`` (Python
MCP SDK's newline-delimited stdio_server). A regression here breaks
claw's MCP tools silently — claw will spawn the server, the server
will fail to parse the framing, and tool calls will time out.
"""

from __future__ import annotations

import io
import subprocess
import sys
import textwrap

from codescribe_train.servers import _mcp_framing_bridge as bridge

# ---------------------------------------------------------------------- #
# Unit tests — framing helpers
# ---------------------------------------------------------------------- #


def test_read_lsp_frame_parses_well_formed_message() -> None:
    """A standard LSP frame returns its JSON payload, headers stripped."""
    body = b'{"jsonrpc":"2.0","id":1}'
    frame = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    assert bridge._read_lsp_frame(io.BytesIO(frame)) == body


def test_read_lsp_frame_handles_lf_only_line_endings() -> None:
    """Some senders use LF instead of CRLF. We must accept both."""
    body = b'{"x":1}'
    frame = b"Content-Length: " + str(len(body)).encode() + b"\n\n" + body
    assert bridge._read_lsp_frame(io.BytesIO(frame)) == body


def test_read_lsp_frame_is_case_insensitive_on_header_name() -> None:
    """``content-length`` lowercase must parse the same as the canonical form."""
    body = b'{"x":1}'
    frame = b"content-length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    assert bridge._read_lsp_frame(io.BytesIO(frame)) == body


def test_read_lsp_frame_ignores_extra_headers() -> None:
    """Other headers (Content-Type, etc.) must not break parsing."""
    body = b'{"x":1}'
    frame = (
        b"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"\r\n" + body
    )
    assert bridge._read_lsp_frame(io.BytesIO(frame)) == body


def test_read_lsp_frame_returns_none_on_eof_before_headers() -> None:
    """Empty stream → ``None`` (callers loop on this)."""
    assert bridge._read_lsp_frame(io.BytesIO(b"")) is None


def test_read_lsp_frame_returns_none_on_missing_length() -> None:
    """A header block without Content-Length is malformed — bail."""
    assert bridge._read_lsp_frame(io.BytesIO(b"Content-Type: x\r\n\r\n")) is None


def test_read_lsp_frame_returns_none_on_truncated_body() -> None:
    """If the payload is shorter than Content-Length claims, bail rather
    than return garbage. Defends against a misbehaving sender."""
    frame = b"Content-Length: 100\r\n\r\nshort"
    assert bridge._read_lsp_frame(io.BytesIO(frame)) is None


def test_read_lsp_frame_handles_back_to_back_messages() -> None:
    """Two LSP frames concatenated must parse as two separate payloads —
    this is the normal case for a request followed by another request."""
    a = b'{"id":1}'
    b = b'{"id":2}'
    stream = io.BytesIO(
        b"Content-Length: " + str(len(a)).encode() + b"\r\n\r\n" + a
        + b"Content-Length: " + str(len(b)).encode() + b"\r\n\r\n" + b
    )
    assert bridge._read_lsp_frame(stream) == a
    assert bridge._read_lsp_frame(stream) == b
    assert bridge._read_lsp_frame(stream) is None  # EOF


def test_write_lsp_frame_emits_canonical_headers() -> None:
    """The writer side uses canonical ``Content-Length`` + CRLF separators."""
    buf = io.BytesIO()
    bridge._write_lsp_frame(buf, b'{"x":1}')
    assert buf.getvalue() == b'Content-Length: 7\r\n\r\n{"x":1}'


def test_write_lsp_frame_handles_empty_payload() -> None:
    """Edge case: a zero-byte payload still produces a valid header."""
    buf = io.BytesIO()
    bridge._write_lsp_frame(buf, b"")
    assert buf.getvalue() == b"Content-Length: 0\r\n\r\n"


# ---------------------------------------------------------------------- #
# Integration test — spawn the bridge with a synthetic child
# ---------------------------------------------------------------------- #


def test_bridge_translates_lsp_to_newline_and_back(tmp_path) -> None:
    """End-to-end: an LSP frame in → newline-delimited JSON to child →
    child's newline reply gets re-wrapped → LSP frame out.

    The synthetic child reads one line of JSON from stdin and echoes a
    modified version back. No MCP SDK dependency, so this test runs
    even on a host without the rag extra installed."""

    # Inline child: read a line of JSON, parse the id, respond with a
    # newline-delimited JSON reply that's clearly attributable.
    child_script = tmp_path / "echo_child.py"
    child_script.write_text(
        textwrap.dedent(
            """
            import json, sys
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                msg = json.loads(line)
                reply = {"jsonrpc": "2.0", "id": msg.get("id"), "result": {"echo": msg}}
                sys.stdout.write(json.dumps(reply) + "\\n")
                sys.stdout.flush()
            """
        )
    )

    # Compose two LSP-framed requests back-to-back; the bridge should
    # forward both as newline-delimited JSON to the child, and the
    # child's two replies should come back LSP-framed.
    req1 = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
    req2 = b'{"jsonrpc":"2.0","id":2,"method":"pong"}'
    framed = (
        b"Content-Length: " + str(len(req1)).encode() + b"\r\n\r\n" + req1
        + b"Content-Length: " + str(len(req2)).encode() + b"\r\n\r\n" + req2
    )

    # Bridge command: spawn this interpreter, run the bridge module,
    # pass the path to the inline child script as the module argument.
    # The bridge will then `python -m <child_script>` — but -m expects
    # a module name, not a file path. So instead, run the child script
    # by absolute path via a tiny shim module. To keep this test simple,
    # we shell out to the bridge's internals directly by patching argv.
    #
    # Approach: invoke the bridge with a one-liner module that imports
    # the script via runpy.
    runner_mod = tmp_path / "echo_module.py"
    runner_mod.write_text(
        f"import runpy; runpy.run_path({str(child_script)!r}, run_name='__main__')"
    )

    # The bridge spawns a fresh `python -m <module>` that doesn't share
    # this process's sys.path, so we propagate the tmp dir via PYTHONPATH
    # so the bridge's child Python can find echo_module.
    import os
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    proc = subprocess.run(  # noqa: S603 — args constructed locally
        [sys.executable, "-m", "codescribe_train.servers._mcp_framing_bridge", "echo_module"],
        input=framed,
        capture_output=True,
        env=env,
        timeout=15,
        check=False,
    )
    out = proc.stdout
    err = proc.stderr.decode("utf-8", errors="replace")
    assert proc.returncode == 0, f"bridge exited {proc.returncode}; stderr=\n{err}"

    # Parse the two LSP frames the bridge wrote back.
    stream = io.BytesIO(out)
    reply1 = bridge._read_lsp_frame(stream)
    reply2 = bridge._read_lsp_frame(stream)
    assert reply1 is not None and reply2 is not None, (
        f"missing LSP frames in bridge output: {out!r}, stderr={err!r}"
    )

    import json
    decoded1 = json.loads(reply1)
    decoded2 = json.loads(reply2)
    assert decoded1["id"] == 1 and decoded1["result"]["echo"]["method"] == "ping"
    assert decoded2["id"] == 2 and decoded2["result"]["echo"]["method"] == "pong"


# ---------------------------------------------------------------------- #
# CLI smoke
# ---------------------------------------------------------------------- #


def test_bridge_cli_rejects_missing_module_arg() -> None:
    """Running the bridge with no arguments must exit 2 with a usage line."""
    result = subprocess.run(  # noqa: S603 — args constructed locally
        [sys.executable, "-m", "codescribe_train.servers._mcp_framing_bridge"],
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 2
    assert b"usage:" in result.stderr.lower() or b"usage:" in result.stdout.lower()
