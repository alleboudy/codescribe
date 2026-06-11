"""Diff-aware PR chunker.

Splits a PR's unified diff into per-file, per-hunk chunks; never breaks
inside a ``@@ -... @@`` hunk header. The output :class:`PRChunk` rows
are written by ``Store.upsert_pull`` as the ``pr_chunks`` +
``pr_chunk_vectors`` fan-out (one chunk → one chunk vector). The
PR-level summary string (used for ``pr_vectors``) is produced by
:func:`summarise_pr_for_embedding`.

Pure functional; no I/O, no subprocess. Consumed by the pipeline.

NOTE on signature: ``PullRequest`` (from
``codescribe_train/rag/sources/github_source.py``) intentionally does NOT
carry the diff blob — the diff is fetched lazily via
``GitHubSource.get_pr_diff(pr.number)`` so a re-index pass can skip the
extra API call when the PR's ``head_sha`` is unchanged. The spec
signature ``chunk_pr_diff(pr, max_chars=4000)`` is therefore extended
to ``chunk_pr_diff(pr, diff_text, max_chars=4000)`` so the chunker has
something to chunk. The same goes for
``summarise_pr_for_embedding(pr, diff_text)``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from codescribe_train.rag.sources.github_source import PullRequest

logger = logging.getLogger(__name__)


# Match a unified-diff file header: "diff --git a/foo b/bar".
_FILE_HEADER_RE = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+?)$")
# Match a unified-diff hunk header: "@@ -x,y +z,w @@ ..."
_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")

# Per-PR cap on body chars used in the summary. From the spec wording
# in the design spec: "concatenate (title \n body[:2000] \n diff_summary)".
_BODY_SUMMARY_MAX_CHARS = 2000


@dataclass(frozen=True)
class PRChunk:
    """One chunk of a PR diff — column-shaped for the ``pr_chunks`` table.

    Fields mirror the ``pr_chunks`` schema exactly (minus the
    auto-assigned ``chunk_id`` primary key, which the store generates).
    """

    pr_number: int
    file_path: str
    hunk_index: int
    chunk_text: str


def _iter_file_blocks(diff_text: str) -> list[tuple[str, list[str]]]:
    """Split a unified diff into (file_path, lines) blocks.

    Returns the list of ``(file_path, lines)`` pairs in the diff. Lines
    include the file's metadata headers (``diff --git``, ``index``,
    ``--- a/...``, ``+++ b/...``) BEFORE the first ``@@`` hunk, plus
    every line up to (but not including) the next ``diff --git`` block.
    A diff with zero ``diff --git`` headers returns an empty list.
    """
    blocks: list[tuple[str, list[str]]] = []
    current_path: str | None = None
    current_lines: list[str] = []

    for line in diff_text.splitlines():
        match = _FILE_HEADER_RE.match(line)
        if match:
            if current_path is not None:
                blocks.append((current_path, current_lines))
            # Use the "b/" name (post-image); for renames git emits both.
            current_path = match.group("b")
            current_lines = [line]
            continue
        if current_path is not None:
            current_lines.append(line)

    if current_path is not None:
        blocks.append((current_path, current_lines))
    return blocks


def _split_into_hunks(file_lines: list[str]) -> list[str]:
    """Split a single file's diff lines into hunks.

    The pre-hunk preamble (the ``diff --git`` + ``index`` + ``---`` /
    ``+++`` lines) is dropped — it is not useful as embedding context
    and inflates the chunk count. If a file has no ``@@`` hunks (e.g.
    a pure rename or mode change) the list is empty.
    """
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for line in file_lines:
        if _HUNK_HEADER_RE.match(line):
            if current is not None:
                hunks.append(current)
            current = [line]
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        hunks.append(current)
    return ["\n".join(h) for h in hunks]


def chunk_pr_diff(
    pr: PullRequest,
    diff_text: str,
    max_chars: int = 4000,
) -> list[PRChunk]:
    """Split ``diff_text`` into per-file, per-hunk :class:`PRChunk` rows.

    The chunker emits ONE chunk per hunk; a single hunk is never split
    across PRChunks even if it exceeds ``max_chars`` (splitting inside
    ``@@ ... @@`` would corrupt the patch as a context for the
    embedder). ``max_chars`` is therefore a soft hint, not a hard cap;
    in practice sample PRs have hunks well under 4 KB, and the rare
    oversized hunk is preserved intact so the embedder sees the full
    surrounding context.

    A diff with zero recognisable file blocks (or an empty string)
    returns an empty list — callers can still call
    :meth:`Store.upsert_pull` with ``chunks=[]``.
    """
    if not diff_text:
        return []

    out: list[PRChunk] = []
    for file_path, file_lines in _iter_file_blocks(diff_text):
        hunks = _split_into_hunks(file_lines)
        for hunk_index, hunk_text in enumerate(hunks):
            if len(hunk_text) > max_chars:
                logger.debug(
                    "PR #%s file %s hunk %s exceeds max_chars=%d (%d chars); "
                    "preserving the whole hunk to keep the header intact.",
                    pr.number,
                    file_path,
                    hunk_index,
                    max_chars,
                    len(hunk_text),
                )
            out.append(
                PRChunk(
                    pr_number=pr.number,
                    file_path=file_path,
                    hunk_index=hunk_index,
                    chunk_text=hunk_text,
                )
            )
    return out


def _diff_summary(diff_text: str) -> str:
    """Build a one-paragraph summary of ``diff_text`` for the PR vector.

    The summary lists the touched files and the number of ``+`` /
    ``-`` lines per file. Cheaper than embedding the whole diff and
    sufficient for the PR-level vector (chunks carry the per-hunk
    detail).
    """
    if not diff_text:
        return ""
    blocks = _iter_file_blocks(diff_text)
    if not blocks:
        return ""

    parts: list[str] = []
    for file_path, lines in blocks:
        added = sum(
            1 for ln in lines if ln.startswith("+") and not ln.startswith("+++")
        )
        deleted = sum(
            1 for ln in lines if ln.startswith("-") and not ln.startswith("---")
        )
        parts.append(f"{file_path} (+{added}/-{deleted})")
    return "Files: " + ", ".join(parts)


def summarise_pr_for_embedding(pr: PullRequest, diff_text: str = "") -> str:
    """Build the string fed into the PR-level summary embedding.

    The shape — ``title \\n body[:2000] \\n diff_summary`` — comes
    verbatim from the design spec. The ``body[:2000]`` cap keeps the
    summary in the embedder's 512-token window even for issue-templates
    that paste large stack traces.
    """
    body = (pr.body or "")[:_BODY_SUMMARY_MAX_CHARS]
    pieces: list[str] = [pr.title, body]
    summary = _diff_summary(diff_text)
    if summary:
        pieces.append(summary)
    return "\n".join(pieces)
