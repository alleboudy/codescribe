"""MCP server factory for the ``repo-rag`` stdio server.

Lives in its own module (separate from ``__main__.py``) so the test
suite can build a fully-wired :class:`mcp.server.Server` instance
without running the stdio transport. ``__main__.py`` consumes this
factory and bolts the stdio + signal-handling code on top.

Key design rules (see ``AGENTS.md`` next door):

* The server is **read-only**. The wrapped sqlite connection is opened
  in read-only URI mode (``file:...?mode=ro``) so any tool that ever
  tried to write would fail at the SQL layer.
* The embedder is **lazy**: it's constructed at server build time but
  the model weights are not loaded until the first ``find_similar_issues``
  or ``search_commits`` call (the :class:`Embedder` contract).
* The server name is **exactly** ``"repo-rag"`` — the tools-list test
  asserts on this string, and the Rust harness's ``.claw-mcp.json`` will
  reference it. Don't rename it without coordinating with the harness.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlite_vec
from mcp import types
from mcp.server import NotificationOptions, Server

from codescribe_train.servers.rag_server.tools import (
    FIND_SIMILAR_ISSUES_SCHEMA,
    GET_PR_DIFF_SCHEMA,
    SEARCH_COMMITS_SCHEMA,
    SEARCH_DOCS_SCHEMA,
    RagTools,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Server identifier. The Rust harness (.claw-mcp.json) will
# reference this string verbatim; don't rename without coordinating.
SERVER_NAME = "repo-rag"
SERVER_VERSION = "0.1.0"


class _ReadOnlyStoreHandle:
    """Thin wrapper exposing only the ``conn`` attribute the retriever needs.

    The :class:`~codescribe_train.rag.store.retrieve.Retriever` only ever
    touches ``store.conn``, never the higher-level upsert methods. So
    for the read-only server path we hand it a small object that holds
    a sqlite connection opened in URI-mode read-only — this way any
    attempt to write fails at the SQL layer, not at the Python layer.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn


def _open_readonly_connection(db_path: Path) -> sqlite3.Connection:
    """Open ``db_path`` in URI-mode read-only and load the sqlite-vec ext.

    sqlite-vec is loaded explicitly so the kNN ``MATCH`` queries the
    retriever issues against ``*_vectors`` virtual tables work. The
    file-not-found case raises a clear ``FileNotFoundError`` instead
    of sqlite's opaque ``unable to open database file``.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"rag store not found at {db_path}")
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    # Disable further extension loading so accidental loads can't happen
    # under a long-lived MCP session.
    conn.enable_load_extension(False)
    # Defensive — even in ?mode=ro this should already be on, but pin
    # query_only so the retriever's behaviour stays loud if anything
    # ever attempts a write.
    conn.execute("PRAGMA query_only = ON")
    return conn


def _build_tools_list() -> list[types.Tool]:
    """The four tools as MCP ``Tool`` objects.

    Schemas are imported from ``tools.py`` so the constants stay in
    one place; the tools-list test compares both names and schemas.
    """
    return [
        types.Tool(
            name="find_similar_issues",
            description=(
                "Search the rag store for issues semantically similar to "
                "a natural-language query. Returns up to k issues ordered "
                "by RRF score, each enriched with its highest-confidence "
                "linked fix PR (filtered by min_confidence)."
            ),
            inputSchema=FIND_SIMILAR_ISSUES_SCHEMA,
        ),
        types.Tool(
            name="get_pr_diff",
            description=(
                "Return the unified diff for a PR by number, optionally "
                "truncated to max_chars with a deterministic marker."
            ),
            inputSchema=GET_PR_DIFF_SCHEMA,
        ),
        types.Tool(
            name="search_commits",
            description=(
                "Search the rag store for commits semantically similar to "
                "a natural-language query. Returns up to k commits ordered "
                "by RRF score; commits without a linked PR are first-class "
                "results."
            ),
            inputSchema=SEARCH_COMMITS_SCHEMA,
        ),
        types.Tool(
            name="search_docs",
            description=(
                "Search the target repo's working-tree documentation "
                "(README, Makefile, docs/**/*.md, docker-compose, "
                "Dockerfiles) for chunks semantically similar to a "
                "natural-language query. Returns up to k chunks ordered "
                "by RRF score, each with its source path + nearest heading. "
                "Use this to find operational/run/build instructions; "
                "follow up with the repo-docs read_doc tool for the full file."
            ),
            inputSchema=SEARCH_DOCS_SCHEMA,
        ),
    ]


def build_server(
    db_path: Path,
    log_path: Path,  # noqa: ARG001  # consumed by __main__.py; kept here for symmetry
    embedder: Any | None = None,
) -> Server:
    """Build a fully-wired ``repo-rag`` MCP server.

    ``db_path`` must point at an existing rag store; the function opens
    it in URI-mode read-only. ``log_path`` is accepted for the symmetric
    signature with ``__main__.py`` but isn't used here — logging setup
    is the caller's responsibility (so tests can drop a tmp_path
    FileHandler before the server is built).

    ``embedder`` defaults to a freshly-constructed :class:`Embedder` built
    from the YAML config; tests may pass a stub. The embedder is
    lazy-loaded inside the retriever — building this server does not
    cost a model load.
    """
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    conn = _open_readonly_connection(db_path)
    store = _ReadOnlyStoreHandle(conn)

    if embedder is None:
        # Lazy import — keeps the test that exercises only get_pr_diff
        # cheap; the embedder builds without loading the model weights.
        # Reuse the same config-loading rules as the rag CLI: honour
        # $RAG_CONFIG, then fall back to configs/rag.yaml under CWD,
        # then all-default RagConfig.
        import os

        from codescribe_train.rag.config import load
        from codescribe_train.rag.embed.embedder import Embedder

        cfg_path = os.environ.get("RAG_CONFIG")
        default = Path.cwd() / "configs" / "rag.yaml"
        if cfg_path:
            config = load(Path(cfg_path))
        elif default.exists():
            config = load(default)
        else:
            from codescribe_train.rag.config import RagConfig

            config = RagConfig()
        embedder = Embedder(
            model_path=config.embed.model_path,
            device=config.embed.device,
            batch_size=config.embed.batch_size,
            max_length=config.embed.max_length,
        )

    retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
    rag_tools = RagTools(store=store, embedder=embedder, retriever=retriever)

    server: Server = Server(name=SERVER_NAME, version=SERVER_VERSION)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return _build_tools_list()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> Iterable[types.TextContent]:
        if name == "find_similar_issues":
            text = rag_tools.find_similar_issues(
                query=arguments["query"],
                k=arguments.get("k", 5),
                min_confidence=arguments.get("min_confidence", 0.8),
            )
        elif name == "get_pr_diff":
            text = rag_tools.get_pr_diff(
                pr_number=arguments["pr_number"],
                max_chars=arguments.get("max_chars", 8000),
            )
        elif name == "search_commits":
            text = rag_tools.search_commits(
                query=arguments["query"],
                k=arguments.get("k", 5),
            )
        elif name == "search_docs":
            text = rag_tools.search_docs(
                query=arguments["query"],
                k=arguments.get("k", 5),
            )
        else:
            raise ValueError(f"unknown tool: {name!r}")
        return [types.TextContent(type="text", text=text)]

    # Stash the connection on the server so __main__.py can close it
    # at shutdown without having to plumb a second reference around.
    server._sample_rag_conn = conn  # type: ignore[attr-defined]

    return server


def initialization_options(server: Server) -> Any:
    """Build the ``InitializationOptions`` for the server.

    Wrapper so ``__main__.py`` can stay tidy. Uses the SDK helper
    ``server.create_initialization_options`` so the capabilities are
    derived from the registered handlers (``list_tools`` +
    ``call_tool``).
    """
    return server.create_initialization_options(
        notification_options=NotificationOptions(),
        experimental_capabilities={},
    )
