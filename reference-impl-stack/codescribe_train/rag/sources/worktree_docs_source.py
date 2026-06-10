"""Read-only source: enumerate + chunk working-tree documentation files.

No subprocess, no network — just globs + reads under ``repo_path``.
Mirrors the ``repo-docs`` MCP server's file-selection rules so the
"search docs" (this source → rag store) and "read doc" (repo-docs
MCP) surfaces agree on what counts as documentation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from codescribe_train.rag.embed.doc_chunker import DocChunk, chunk_document

logger = logging.getLogger(__name__)

_GLOB_PATTERNS = (
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
_EXCLUDED_DIR_PARTS = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
})


@dataclass(frozen=True)
class Doc:
    """One documentation file + its heading-delimited chunks."""

    path: str  # relative to repo root
    chunks: list[DocChunk]


class WorktreeDocsSource:
    """Read-only enumerator over a checkout's documentation files."""

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path).resolve()

    def _candidate_paths(self) -> list[Path]:
        found: set[Path] = set()
        for pattern in _GLOB_PATTERNS:
            for path in self.repo_path.glob(pattern):
                if not path.is_file():
                    continue
                rel = path.relative_to(self.repo_path)
                if any(part in _EXCLUDED_DIR_PARTS for part in rel.parts):
                    continue
                found.add(path)
        return sorted(found)

    def iter_docs(self) -> Iterator[Doc]:
        """Yield one :class:`Doc` per documentation file (with >=1 chunk)."""
        for path in self._candidate_paths():
            rel = str(path.relative_to(self.repo_path))
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.warning("skipping unreadable doc %s: %s", rel, e)
                continue
            chunks = chunk_document(rel, content)
            if chunks:
                yield Doc(path=rel, chunks=chunks)
