"""MCP server exposing documentation files from the target repo.

Two tools:

- ``list_docs(subdir=None, pattern=None)`` — enumerate documentation
  files under ``TARGET_REPO`` (relative paths), filtered by
  conventional doc/build/run-script names. Optional ``subdir`` scopes
  the listing to a subdirectory of the repo; optional ``pattern`` is
  a substring match against the path for narrowing large lists.
- ``read_doc(path, max_chars=50000)`` — read one file by relative
  path, with path-traversal sandboxing identical to ``repo-grep``'s
  pattern (resolve then check the resolved path starts with
  ``TARGET_REPO``).

Conventional doc paths included by ``list_docs``:

- ``README*.md`` at the repo root
- ``Makefile``, ``makefile`` (case-insensitive equivalents are filesystem-
  dependent so we list both)
- ``docs/**/*.md`` (recursive)
- ``docker-compose*.yml`` / ``docker-compose*.yaml`` at the repo root
- ``Dockerfile`` / ``Dockerfile.*`` anywhere under the repo
- ``*.sh`` at the repo root (top-level build / setup shell scripts)

Excluded path components (anywhere in the path): ``.git``,
``node_modules``, ``.venv``, ``venv``, ``__pycache__``, ``dist``,
``build`` — these can otherwise produce false positives via vendored /
generated files.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Path sandboxing root — same env-var convention as ``repo-grep`` so
# operators set TARGET_REPO once and both servers honour it.
REPO_DIR = Path(os.environ.get("TARGET_REPO", "../your-repo")).resolve()

# Hard cap on read_doc output. The MCP client side may impose its own
# limit; this is a defensive ceiling so a pathological multi-megabyte
# file doesn't blow up the protocol layer.
MAX_READ_CHARS = 200_000

EXCLUDED_DIR_PARTS = frozenset({
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
})

# Glob patterns evaluated against ``REPO_DIR`` for a full-repo listing.
# Calibrated for the typical project layout: top-level README + Makefile +
# compose, ``docs/**/*.md`` for documentation, ``**/Dockerfile`` for
# service-specific containers, top-level shell scripts. Each match's
# relative path is added to the result set, deduplicated.
ROOT_GLOB_PATTERNS = (
    "README*.md",
    "Makefile",
    "makefile",
    "docs/**/*.md",
    "docker-compose*.yml",
    "docker-compose*.yaml",
    "Dockerfile",
    "Dockerfile.*",
    "**/Dockerfile",
    "**/Dockerfile.*",
    "*.sh",
)

# When the caller scopes ``list_docs`` to a subdirectory, the
# ``docs/**`` and ``*.sh`` patterns above no longer line up — we're
# already *inside* the subtree. Use recursive ``**/`` patterns that
# match anywhere under the scope, then rely on EXCLUDED_DIR_PARTS to
# filter out vendored / generated noise.
SUBDIR_GLOB_PATTERNS = (
    "**/*.md",
    "**/Makefile",
    "**/makefile",
    "**/docker-compose*.yml",
    "**/docker-compose*.yaml",
    "**/Dockerfile",
    "**/Dockerfile.*",
    "**/*.sh",
)

mcp = FastMCP("repo-docs")


def _list_doc_paths(scope: Path) -> list[str]:
    """Return relative-to-REPO_DIR string paths for all docs under *scope*.

    Pure function — exposed for unit testing without going through the
    MCP layer. Pattern set switches based on whether *scope* is the
    repo root (use ROOT_GLOB_PATTERNS) or a subdirectory (use the
    recursive SUBDIR_GLOB_PATTERNS, since the root patterns assume
    ``docs/`` / top-level ``*.sh`` shapes that don't apply inside a
    sub-tree).
    """
    patterns = ROOT_GLOB_PATTERNS if scope == REPO_DIR else SUBDIR_GLOB_PATTERNS
    found: set[str] = set()
    for pattern in patterns:
        for path in scope.glob(pattern):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(REPO_DIR)
            except ValueError:
                # `scope` was outside REPO_DIR — should have been rejected
                # already by the caller; skip defensively.
                continue
            if any(part in EXCLUDED_DIR_PARTS for part in rel.parts):
                continue
            found.add(str(rel))
    return sorted(found)


@mcp.tool()
def list_docs(
    subdir: str | None = None,
    pattern: str | None = None,
) -> dict:
    """List documentation files in the target repo.

    Args:
        subdir: Optional subdirectory of the repo to scope to (e.g.
            ``"docs"``). Relative to the repo root; absolute paths and
            ``..`` traversal are rejected.
        pattern: Optional case-insensitive substring filter applied to
            the relative path. Useful to narrow large lists, e.g.
            ``pattern="docker"`` to find all docker-related files.

    Returns:
        ``{"repo": "<abs-path>", "docs": ["README.md", "docs/...", ...]}``
        on success, or ``{"error": "..."}`` on rejection.
    """
    scope = REPO_DIR
    if subdir is not None:
        candidate = Path(subdir)
        if candidate.is_absolute():
            return {"error": f"absolute paths not allowed: {subdir}"}
        resolved = (REPO_DIR / candidate).resolve()
        if not str(resolved).startswith(str(REPO_DIR)):
            return {"error": f"subdir escapes the repo directory: {subdir}"}
        if not resolved.is_dir():
            return {"error": f"subdir does not exist or is not a directory: {subdir}"}
        scope = resolved

    paths = _list_doc_paths(scope)
    if pattern is not None:
        needle = pattern.lower()
        paths = [p for p in paths if needle in p.lower()]

    return {"repo": str(REPO_DIR), "docs": paths}


@mcp.tool()
def read_doc(path: str, max_chars: int = 50_000) -> dict:
    """Read one documentation file by relative path.

    Args:
        path: Relative path from the repo root (e.g. ``"README.md"``,
            ``"docs/DEVELOPMENT.md"``, ``"docker-compose.yml"``).
            Absolute paths and ``..`` traversal are rejected.
        max_chars: Maximum characters to return (defaults to 50 000;
            hard-capped at ``MAX_READ_CHARS`` = 200 000). If the file
            exceeds the limit, the content is truncated and a trailing
            marker is appended.

    Returns:
        ``{"path": "<rel-path>", "content": "...", "truncated": bool,
        "size_bytes": int}`` on success, or ``{"error": "..."}`` on
        rejection / read failure.
    """
    if not path:
        return {"error": "path is required"}

    requested = Path(path)
    if requested.is_absolute():
        return {"error": f"absolute paths not allowed: {path}"}

    resolved = (REPO_DIR / requested).resolve()
    if not str(resolved).startswith(str(REPO_DIR)):
        return {"error": f"path escapes the repo directory: {path}"}
    if not resolved.is_file():
        return {"error": f"not a file: {path}"}

    try:
        size_bytes = resolved.stat().st_size
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"error": f"read failed: {e}"}

    effective_cap = min(max_chars, MAX_READ_CHARS)
    truncated = len(content) > effective_cap
    if truncated:
        content = (
            content[:effective_cap]
            + f"\n\n[TRUNCATED: original was {len(content)} chars; "
            f"max_chars={effective_cap} (hard cap {MAX_READ_CHARS})]"
        )

    try:
        rel = resolved.relative_to(REPO_DIR)
    except ValueError:  # pragma: no cover — guarded above
        rel = resolved

    return {
        "path": str(rel),
        "content": content,
        "truncated": truncated,
        "size_bytes": size_bytes,
    }
