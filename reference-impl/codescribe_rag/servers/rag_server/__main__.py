from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from pathlib import Path


def _configure_logging() -> None:
    """File-only logging. Runs FIRST so no imported library can install a
    StreamHandler on the root logger that would corrupt the stdout JSON-RPC channel."""
    log_path = Path(os.environ.get("RAG_LOG_PATH", f"./logs/rag-server-{int(time.time())}.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    root.setLevel(logging.INFO)


_configure_logging()
logger = logging.getLogger(__name__)

import mcp.types as mcp_types                        # noqa: E402
from mcp.server import NotificationOptions, Server   # noqa: E402
from mcp.server.models import InitializationOptions  # noqa: E402
from mcp.server.stdio import stdio_server            # noqa: E402

from .tools import FIND_SIMILAR_BUGS_SCHEMA, GET_FIX_DIFF_SCHEMA, RagTools  # noqa: E402

_TOOLS: "RagTools | None" = None


def build_server(_tools: "RagTools | None" = None) -> Server:
    """Build the MCP server. Production passes nothing (loads from env); tests
    inject a fixture-backed RagTools."""
    tools = _tools if _tools is not None else RagTools.load(
        Path(os.environ.get("RAG_DB_PATH", "./indices/rag.db")))
    server = Server("codescribe-rag")

    @server.list_tools()
    async def handle_list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="find_similar_bugs",
                description="Return up to k bugs most similar to a query, with fix CL excerpts.",
                inputSchema=FIND_SIMILAR_BUGS_SCHEMA),
            mcp_types.Tool(
                name="get_fix_diff",
                description="Return the full unified diff of a changelist.",
                inputSchema=GET_FIX_DIFF_SCHEMA),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        try:
            if name == "find_similar_bugs":
                text = tools.find_similar_bugs(
                    query=arguments["query"], k=arguments.get("k", 5),
                    min_confidence=arguments.get("min_confidence", 0.8))
            elif name == "get_fix_diff":
                text = tools.get_fix_diff(
                    cl_number=arguments["cl_number"], max_chars=arguments.get("max_chars", 8000))
            else:
                raise ValueError(f"unknown tool: {name}")
            return [mcp_types.TextContent(type="text", text=text)]
        except Exception:
            logger.exception("tool %s failed", name)
            raise

    return server


def _shutdown(signum, _frame) -> None:
    # Signal-safe shutdown. The stdio reader may be blocked in a worker thread,
    # so flush + hard-exit rather than unwinding the loop around a blocked read.
    # Both tools are read-only, so dropping the SQLite connection is safe.
    logger.info("rag-server received signal %d; shutting down cleanly", signum)
    try:
        if _TOOLS is not None:
            _TOOLS.close()
    except Exception:
        logger.exception("error closing store during shutdown")
    logging.shutdown()
    os._exit(0)


async def main() -> None:
    global _TOOLS
    _TOOLS = RagTools.load(Path(os.environ.get("RAG_DB_PATH", "./indices/rag.db")))
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server = build_server(_tools=_TOOLS)
    async with stdio_server() as (read, write):
        # Returns when the client closes stdin (EOF); SIGTERM/SIGINT exit via _shutdown.
        await server.run(read, write, InitializationOptions(
            server_name="codescribe-rag", server_version="0.1.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(), experimental_capabilities={})))
    _TOOLS.close()
    logger.info("rag-server shutting down cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
