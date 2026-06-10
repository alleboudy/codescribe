"""Diff + commit-message instruction-pair formatter.

For each non-merge commit in the repo's history whose subject is reasonably
descriptive and whose diff is small enough to fit in context, emit a
ChatML-formatted sample where the user asks the commit subject (plus body) and
the assistant responds with the diff.

Many commits don't make for good training data (merges, formatter sweeps,
mass-rename refactors). This formatter applies coarse filters but is
intentionally noisy — the per-repo config gates whether to enable it at all.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path

from codescribe_train.data.formatters.base import Sample

logger = logging.getLogger(__name__)

# Sentinels chosen to be vanishingly unlikely in a real commit message or diff.
_COMMIT_SEP = "===CODESCRIBE:COMMIT==="
_HEADER_END = "===CODESCRIBE:HEADEREND==="


def format_diff_instructions(
    repo: Path,
    *,
    max_diff_lines: int = 400,
    min_subject_chars: int = 12,
    max_subject_chars: int = 200,
    skip_merge_commits: bool = True,
    max_commits: int | None = None,
) -> Iterator[Sample]:
    """Yield ChatML samples from the repo's git history.

    Parameters
    ----------
    repo:
        Absolute path to the git repo root.
    max_diff_lines:
        Drop commits whose patch is longer than this. Keeps samples tractable.
    min_subject_chars / max_subject_chars:
        Subject length bounds. Drops single-word ``WIP`` commits and absurdly
        long subjects.
    skip_merge_commits:
        If true, ``--no-merges`` is passed to ``git log``.
    max_commits:
        If set, only the last N commits are inspected.
    """
    repo = repo.resolve()
    pretty = f"{_COMMIT_SEP}%n%H%n%s%n%b%n{_HEADER_END}"
    args = [
        "git",
        "-C",
        str(repo),
        "log",
        "--pretty=format:" + pretty,
        "--patch",
        "-U0",  # zero context lines so diffs stay compact
    ]
    if skip_merge_commits:
        args.insert(args.index("log") + 1, "--no-merges")
    if max_commits:
        args.append(f"-n{max_commits}")

    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("git log failed in %s: %s", repo, e.stderr.strip())
        return

    sections = result.stdout.split(_COMMIT_SEP + "\n")
    for section in sections:
        if not section.strip():
            continue
        end_idx = section.find(_HEADER_END + "\n")
        if end_idx == -1:
            continue
        header = section[:end_idx].rstrip("\n")
        diff = section[end_idx + len(_HEADER_END) + 1 :]
        header_lines = header.split("\n", 2)
        if len(header_lines) < 2:
            continue
        sha = header_lines[0].strip()
        subject = header_lines[1].strip() if len(header_lines) > 1 else ""
        body = header_lines[2].strip() if len(header_lines) > 2 else ""

        if not (min_subject_chars <= len(subject) <= max_subject_chars):
            continue
        diff_line_count = diff.count("\n")
        if not (2 <= diff_line_count <= max_diff_lines):
            continue

        instruction = (subject + ("\n\n" + body if body else "")).strip()
        text_sample = (
            "<|im_start|>user\n"
            f"{instruction}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{diff.strip()}\n"
            "<|im_end|>"
        )
        yield Sample(
            text=text_sample,
            metadata={
                "format": "diff_instr",
                "sha": sha,
                "subject": subject,
                "diff_lines": diff_line_count,
            },
        )
