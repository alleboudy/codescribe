"""Split a working-tree doc into heading-delimited chunks for embedding.

Markdown (``*.md``) splits on ATX headings (``#``..``######``). Each
chunk carries the nearest preceding heading text + the body down to the
next heading. Non-markdown docs (Makefile, Dockerfile, compose,
``*.sh``) are returned as a single chunk (they're small + structured;
heading-splitting them is meaningless). Any chunk exceeding ``max_chars``
is hard-split on character count so no chunk overflows the embedder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_DEFAULT_MAX_CHARS = 2000


@dataclass(frozen=True)
class DocChunk:
    """One chunk of a document: its nearest heading + the body text."""

    heading: str
    text: str
    chunk_index: int


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into <= max_chars pieces on character boundaries."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def chunk_document(
    path: str,
    content: str,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> list[DocChunk]:
    """Return heading-delimited chunks for ``content``.

    ``path`` only decides markdown-vs-not (by the ``.md`` suffix);
    nothing is read from disk.
    """
    if not content or not content.strip():
        return []

    is_markdown = path.lower().endswith(".md")

    # Build (heading, body) sections.
    sections: list[tuple[str, str]] = []
    if not is_markdown:
        sections.append(("", content.strip()))
    else:
        current_heading = ""
        current_lines: list[str] = []
        for line in content.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                # Flush the previous section.
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_heading, body))
                current_heading = m.group(2).strip()
                current_lines = [line]  # keep the heading line in the body too
            else:
                current_lines.append(line)
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))

    # Expand oversized sections + assign global chunk_index.
    chunks: list[DocChunk] = []
    idx = 0
    for heading, body in sections:
        pieces = _hard_split(body, max_chars) if len(body) > max_chars else [body]
        for piece in pieces:
            if not piece.strip():
                continue
            chunks.append(DocChunk(heading=heading, text=piece, chunk_index=idx))
            idx += 1
    return chunks
