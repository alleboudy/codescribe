"""MCP server entrypoint: ``python -m codescribe_train.servers.rag_server``.

Wires the :func:`build_server` factory from ``server.py`` into the
stdio JSON-RPC transport. Critical rules (see ``AGENTS.md``):

* **stdout is the JSON-RPC channel.** The root logger is reset to a
  single :class:`logging.FileHandler` before anything else runs so no
  log line can leak to stdout/stderr and corrupt the transport.
* **Read-only DB.** The server opens the rag store in URI mode
  ``mode=ro``.
* **No network egress.** The handlers only ever touch the local
  store; the embedder loads from a local snapshot directory.
* **Clean SIGTERM.** A handler swaps the default SIGTERM behaviour
  for one that raises :class:`KeyboardInterrupt` — anyio's stdio task
  group treats that as a graceful shutdown signal, the
  ``stdio_server`` context manager exits cleanly, the ``finally`` block
  closes the sqlite connection and flushes the log, and the process
  exits 0.

Env vars (read at startup):

* ``RAG_DB_PATH`` — rag store path; default ``./indices/rag.db``
  (relative to CWD at spawn time).
* ``RAG_LOG_PATH`` — log file path; default
  ``./logs/rag-server-<unix_ts>.log``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import sys
import time
from pathlib import Path


def _configure_logging(log_path: Path) -> None:
    """Reset the root logger to a single :class:`FileHandler` at ``log_path``.

    This is the verbatim snippet from the spec. Called before any
    other module-level imports that may log on import (Store, Embedder).
    Strips every pre-existing handler so a parent process (e.g. the
    Rust harness) that already wired a StreamHandler cannot
    leak log output to stdout/stderr — which would corrupt the
    JSON-RPC channel.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(fh)
    root.setLevel(logging.INFO)


def _default_log_path() -> Path:
    """Compute the default log path used when ``$RAG_LOG_PATH`` is unset."""
    return Path.cwd() / "logs" / f"rag-server-{int(time.time())}.log"


def _default_db_path() -> Path:
    """Compute the default DB path used when ``$RAG_DB_PATH`` is unset."""
    return Path.cwd() / "indices" / "rag.db"


def _install_sigterm_handler() -> None:
    """Forward SIGTERM to SIGINT so anyio's built-in cancel path runs.

    anyio installs its own SIGINT handler when ``anyio.run`` starts (see
    ``anyio/_backends/_asyncio.py::_on_sigint``); that handler cancels
    the main task cleanly, the ``stdio_server`` async context manager
    exits, and the ``finally`` block in :func:`main` runs the cleanup
    (close sqlite, flush log). Raising ``KeyboardInterrupt`` directly
    from a SIGTERM handler doesn't reliably unblock anyio's selector
    on every platform; routing through SIGINT delegates to a path that
    anyio has already engineered for graceful shutdown.
    """

    def _on_sigterm(_signum: int, _frame: object) -> None:
        logging.getLogger(__name__).info("SIGTERM received; shutting down")
        os.kill(os.getpid(), signal.SIGINT)

    signal.signal(signal.SIGTERM, _on_sigterm)


async def _serve_stdio(server) -> None:  # noqa: ANN001  # server is mcp.Server, defer import
    """Run ``server`` against the stdio transport until the client disconnects."""
    from mcp.server.stdio import stdio_server

    from codescribe_train.servers.rag_server.server import initialization_options

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, initialization_options(server))


def main() -> int:
    """Entrypoint for ``python -m codescribe_train.servers.rag_server``."""
    log_path = Path(os.environ.get("RAG_LOG_PATH", str(_default_log_path())))
    _configure_logging(log_path)

    logger = logging.getLogger(__name__)
    db_path = Path(os.environ.get("RAG_DB_PATH", str(_default_db_path())))
    logger.info("repo-rag mcp server starting db_path=%s log_path=%s", db_path, log_path)

    # Lazy import — _configure_logging must run before any module that
    # might log on import.
    import anyio

    from codescribe_train.servers.rag_server.server import build_server

    server = build_server(db_path=db_path, log_path=log_path)
    _install_sigterm_handler()
    try:
        anyio.run(_serve_stdio, server)
    except KeyboardInterrupt:
        logger.info("stdio loop interrupted; clean shutdown")
    finally:
        try:
            conn = server._sample_rag_conn  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover — defensive
            conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover — defensive close
                logger.exception("error closing sqlite connection")
        for h in list(logging.getLogger().handlers):
            with contextlib.suppress(Exception):  # pragma: no cover — defensive flush
                h.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
