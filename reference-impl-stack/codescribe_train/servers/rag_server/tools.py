"""MCP tool implementations for the ``repo-rag`` server.

Each method on :class:`RagTools` returns a **Markdown string** (or, for
structured data like doc chunks, a JSON string). The caller (``server.py``)
wraps that string in a single :class:`mcp.types.TextContent` and lets the
SDK build the ``CallToolResult``.

JSON-Schemas are spec'd verbatim from the design spec and are exported
as module constants so ``server.py`` and the tools-list test share
the same source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codescribe_train.rag.embed.embedder import Embedder
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import (
        PRDiff,
        RetrievedCommit,
        RetrievedDocChunk,
        RetrievedIssue,
    )


# ---------------------------------------------------------------------------
# JSON Schemas — verbatim from the design spec
# ---------------------------------------------------------------------------

FIND_SIMILAR_ISSUES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        "min_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.8,
        },
    },
    "required": ["query"],
}

GET_PR_DIFF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pr_number": {"type": "integer", "minimum": 1},
        "max_chars": {"type": "integer", "minimum": 100, "default": 8000},
    },
    "required": ["pr_number"],
}

SEARCH_COMMITS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
    },
    "required": ["query"],
}

SEARCH_DOCS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
    },
    "required": ["query"],
}


# Sha display length in commit headers. Twelve chars is the git default
# `git log --abbrev-commit` width and matches the rag CLI output.
_SHA_SHORT_LEN = 12


class RagTools:
    """The four MCP tools, each returning a Markdown string.

    Holds a borrowed store + embedder + retriever. The store and
    embedder are kept as attributes (rather than reaching into the
    retriever) so a future tool that bypasses the retriever — e.g. a
    raw "list tables" diagnostic — can ride on the same instance
    without re-plumbing dependencies.
    """

    def __init__(
        self,
        store: Any,  # _ReadOnlyStoreHandle, but typed Any to avoid an import cycle
        embedder: Embedder | None,
        retriever: Retriever,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.retriever = retriever

    # ----- find_similar_issues --------------------------------------------

    def find_similar_issues(
        self,
        query: str,
        k: int = 5,
        min_confidence: float = 0.8,
    ) -> str:
        """Return a Markdown report of the top-k similar issues."""
        hits = self.retriever.find_similar_issues(
            query=query, k=k, min_confidence=min_confidence,
        )
        return _render_issues(query, hits)

    # ----- get_pr_diff ----------------------------------------------------

    def get_pr_diff(self, pr_number: int, max_chars: int = 8000) -> str:
        """Return a Markdown rendering of one PR's diff + metadata."""
        out = self.retriever.get_pr_diff(pr_number=pr_number, max_chars=max_chars)
        return _render_pr_diff(out)

    # ----- search_commits -------------------------------------------------

    def search_commits(self, query: str, k: int = 5) -> str:
        """Return a Markdown report of the top-k commits."""
        hits = self.retriever.search_commits(query=query, k=k)
        return _render_commits(query, hits)

    # ----- search_docs ----------------------------------------------------

    def search_docs(self, query: str, k: int = 5) -> str:
        """Return a Markdown report of the top-k worktree-doc chunks matching ``query``."""
        hits = self.retriever.search_docs(query=query, k=k)
        return _render_docs(query, hits)


# ---------------------------------------------------------------------------
# Markdown rendering — pure functions, easy to test in isolation
# ---------------------------------------------------------------------------


def _render_issues(query: str, hits: list[RetrievedIssue]) -> str:
    """Markdown render for ``find_similar_issues`` results.

    Header shape per the spec ("### Issue #N — title (score X.YY)").
    The "### Bug " variant is also accepted by the spec test;
    we use "### Issue " uniformly because the store talks about issues,
    not bugs.
    """
    if not hits:
        return f"# Similar issues for: {query}\n\n_No issues matched._\n"
    parts: list[str] = [f"# Similar issues for: {query}", ""]
    for h in hits:
        parts.append(f"### Issue #{h.issue_number} — {h.title}  (score {h.score:.2f})")
        parts.append(f"**Status:** {h.state}")
        if h.fix_pr_number is not None:
            parts.append(
                f"**Fix PR:** #{h.fix_pr_number} (confidence {h.confidence:.2f})"
            )
        if h.body_excerpt:
            parts.append("")
            parts.append("> " + h.body_excerpt.replace("\n", "\n> "))
        if h.fix_excerpt:
            parts.append("")
            parts.append("```diff")
            parts.append(h.fix_excerpt)
            parts.append("```")
            parts.append("(call `get_pr_diff` for the full diff)")
        parts.append("")
    return "\n".join(parts)


def _render_pr_diff(out: PRDiff) -> str:
    """Markdown render for ``get_pr_diff``.

    Starts with ``## PR #<n>`` per spec; the spec test asserts the text
    contains a unified-diff ``@@`` header — the underlying retriever
    keeps the original diff text verbatim, so any non-empty diff will
    already contain ``@@``.
    """
    header = (
        f"## PR #{out.pr_number} — {out.title}\n"
        f"**Author:** {out.author or '(unknown)'}  "
        f"**Merged:** {out.merged_at or '(unmerged)'}  "
        f"**Files:** {out.files}\n"
    )
    if out.truncated:
        header += "_(diff truncated)_\n"
    return f"{header}\n```diff\n{out.diff_text}\n```\n"


def _render_commits(query: str, hits: list[RetrievedCommit]) -> str:
    """Markdown render for ``search_commits``.

    Header shape: ``### Commit <sha-short> — <author>, <date>``. The
    spec test asserts the body contains ``### Commit `` (with a
    trailing space) — keep that prefix stable.
    """
    if not hits:
        return f"# Commit search for: {query}\n\n_No commits matched._\n"
    parts: list[str] = [f"# Commit search for: {query}", ""]
    for h in hits:
        short = h.sha[:_SHA_SHORT_LEN]
        author = h.author or "(unknown)"
        date = h.authored_at or "(unknown)"
        # The header substring "### Commit " is asserted by the spec test
        # — keep it byte-stable.
        parts.append(f"### Commit {short} — {author}, {date}")
        first_line = (h.message or "").splitlines()[0] if h.message else ""
        parts.append(f"**Message:** {first_line}")
        parts.append(f"_score: {h.score:.4f}; files: {h.files}_")
        if h.diff_excerpt:
            parts.append("")
            parts.append("```diff")
            parts.append(h.diff_excerpt)
            parts.append("```")
        parts.append("")
    return "\n".join(parts)


def _render_docs(query: str, hits: list[RetrievedDocChunk]) -> str:
    """Markdown render for ``search_docs``.

    Header shape: ``### <doc_path> § <heading>``. Mirrors the structure
    of :func:`_render_commits` — a top-level heading referencing the
    query, then one block per hit.
    """
    if not hits:
        return f"# Doc search for: {query}\n\n_No docs matched._\n"
    parts: list[str] = [f"# Doc search for: {query}", ""]
    for h in hits:
        parts.append(f"### {h.doc_path} § {h.heading}")
        parts.append(f"_score: {h.score:.4f}_")
        if h.text_excerpt:
            parts.append("")
            parts.append(h.text_excerpt)
        parts.append("")
    return "\n".join(parts)
