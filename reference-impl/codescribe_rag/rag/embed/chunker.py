from __future__ import annotations

import re
from dataclasses import dataclass

HUNK_HEADER_RE = re.compile(r"^@@ [^@]+ @@", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class CLChunk:
    file_path: str
    hunk_index: int
    text: str


def chunk_diff(cl, max_chars: int = 4000) -> list[CLChunk]:
    """Split each file's diff into 1+ chunks of <= max_chars (UTF-8 bytes)."""
    chunks: list[CLChunk] = []
    for file_path, block in _split_by_file(cl.diff_text):
        for hunk_idx, hunk in enumerate(_split_by_hunk(block)):
            for sub in _safe_split(hunk, max_chars):
                chunks.append(CLChunk(file_path, hunk_idx, sub))
    return chunks


def _split_by_file(diff: str) -> list[tuple[str, str]]:
    """Split a unified diff into per-file blocks based on `+++ b/path` markers."""
    blocks: list[tuple[str, str]] = []
    current_path: str | None = None
    current_lines: list[str] = []
    for line in diff.splitlines(keepends=True):
        if line.startswith("+++ b/"):
            if current_path is not None:
                blocks.append((current_path, "".join(current_lines)))
                current_lines = []
            current_path = line[len("+++ b/"):].rstrip()
        current_lines.append(line)
    if current_path is not None and current_lines:
        blocks.append((current_path, "".join(current_lines)))
    return blocks


def _split_by_hunk(block: str) -> list[str]:
    """Split a file's diff block at `@@ ... @@` hunk headers."""
    splits = HUNK_HEADER_RE.split(block)
    headers = HUNK_HEADER_RE.findall(block)
    if not headers:
        return [block]
    result = [splits[0]]  # file header (often `--- a/x\n+++ b/x\n`)
    for header, body in zip(headers, splits[1:], strict=True):
        result.append(header + body)
    return result


def _safe_split(text: str, max_chars: int) -> list[str]:
    """Split text into <= max_chars chunks. Never splits a multi-byte UTF-8 char."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_chars:
        return [text]
    out: list[str] = []
    i = 0
    while i < len(encoded):
        end = min(i + max_chars, len(encoded))
        # Walk back to a UTF-8 boundary (continuation bytes are 10xxxxxx).
        while end < len(encoded) and (encoded[end] & 0xC0) == 0x80:
            end -= 1
        out.append(encoded[i:end].decode("utf-8", errors="replace"))
        i = end
    return out
