"""Read-only Git source: enumerate + describe commits via the ``git`` CLI.

Subprocess only — no clone, no fetch, no write commands. The repo is
assumed to be a working checkout. See ``codescribe_train/rag/sources/AGENTS.md``
for the rules this module obeys; the static checks in
``tests/rag/sources/test_no_writes.py`` and ``test_no_shell_true.py``
enforce them.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Timeout for every git subprocess (seconds). 60 s is generous: the only
# command that can be slow is `git log` over the whole history, and even on
# a multi-thousand-commit repo this finishes in well under a second.
_GIT_TIMEOUT = 60

# Subjects produced by GitHub's "Merge pull request" UI start with this exact
# prefix; the merge-commit-style PR-number extractor relies on it.
_MERGE_PR_RE = re.compile(r"\bMerge pull request #(\d+)\b")
# Squash-merge style: subject ends with " (#42)". Anchored at end-of-string
# so an inline reference like "fixes #42 in foo (#43)" still resolves to 43.
_PAREN_PR_RE = re.compile(r"\(#(\d+)\)\s*$")
# Explicit spec text from the design spec: "Merges PR #N".
_MERGES_PR_RE = re.compile(r"\bMerges PR #(\d+)\b")


@dataclass(frozen=True)
class FileChange:
    """One file touched by a commit, with line counts from ``git --numstat``."""

    path: str
    change_type: str  # "A" | "M" | "D" | "R" — currently always "M" from numstat
    added: int
    deleted: int


@dataclass(frozen=True)
class Commit:
    """One commit with the metadata the RAG pipeline cares about.

    ``diff_text`` is the raw ``git show`` output (subject + body + patch);
    the store layer is responsible for gzipping it before persisting.
    """

    sha: str
    author: str
    author_email: str
    authored_at: str  # ISO-8601 with timezone offset (from `%aI`)
    message: str  # subject line only
    files: list[FileChange] = field(default_factory=list)
    diff_text: str = ""
    pr_number: int | None = None


def _extract_pr_number(subject: str) -> int | None:
    """Return the PR number referenced by a merge-commit subject, or None."""
    for pattern in (_MERGE_PR_RE, _MERGES_PR_RE, _PAREN_PR_RE):
        match = pattern.search(subject)
        if match:
            return int(match.group(1))
    return None


def _run_git(repo_path: Path, *args: str) -> str:
    """Run ``git -C <repo_path> <args>`` and return stdout.

    Always invokes the binary directly (no shell interpolation). Raises
    ``subprocess.CalledProcessError`` on non-zero exit; callers can choose
    to suppress or re-raise.
    """
    cmd = ["git", "-C", str(repo_path), *args]
    proc = subprocess.run(  # noqa: PLW1510 - check=True is set explicitly
        cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
    )
    return proc.stdout


def _parse_log_line(line: str) -> tuple[str, str, str, str, str] | None:
    """Parse one `%H|%aI|%aE|%aN|%s` line. Returns None on malformed input."""
    # Split on the first 4 separators only — the subject may itself contain
    # '|' characters.
    parts = line.split("|", 4)
    if len(parts) != 5:
        return None
    sha, authored_at, author_email, author, subject = parts
    # Sanity: SHA shape (full 40-char hex). Reject anything else.
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        return None
    return sha, authored_at, author_email, author, subject


class GitSource:
    """Read-only enumerator over a local git checkout."""

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path)

    # --- enumeration ---------------------------------------------------

    def iter_commits_since(self, last_sha: str | None) -> Iterator[Commit]:
        """Yield every commit reachable from any ref since ``last_sha``.

        Commits are yielded in ASCENDING commit-date order so a pipeline can
        checkpoint after each one and resume safely. ``last_sha`` is
        exclusive: the commit named by it is not yielded again. Pass
        ``None`` for a full-history walk.

        The yielded ``Commit`` objects carry only the metadata visible in
        ``git log``; callers that need the patch text or file list should
        call :meth:`describe` for the SHAs they care about.
        """
        log_args = [
            "log",
            "--all",
            "--pretty=format:%H|%aI|%aE|%aN|%s",
        ]
        if last_sha:
            # `<last_sha>..HEAD` excludes last_sha and everything reachable
            # from it. Combined with `--all` we get every new commit on any
            # branch.
            log_args.append(f"{last_sha}..HEAD")

        try:
            stdout = _run_git(self.repo_path, *log_args)
        except subprocess.CalledProcessError as exc:
            logger.warning("git log failed: %s", exc.stderr.strip() if exc.stderr else exc)
            return

        commits: list[Commit] = []
        for raw_line in stdout.splitlines():
            line = raw_line.rstrip()
            if not line:
                continue
            parsed = _parse_log_line(line)
            if parsed is None:
                logger.debug("skipping malformed git log line: %r", line)
                continue
            sha, authored_at, author_email, author, subject = parsed
            commits.append(
                Commit(
                    sha=sha,
                    author=author,
                    author_email=author_email,
                    authored_at=authored_at,
                    message=subject,
                    files=[],
                    diff_text="",
                    pr_number=_extract_pr_number(subject),
                )
            )

        # `git log` is newest-first by default. Re-sort ASC by `authored_at`
        # so resumable checkpoints advance monotonically.
        commits.sort(key=lambda c: c.authored_at)
        yield from commits

    # --- single-commit describe ----------------------------------------

    def describe(self, sha: str) -> Commit:
        """Return a fully populated :class:`Commit` for ``sha``.

        Issues three ``git`` calls:

        1. ``git log -1`` for the metadata header.
        2. ``git show --numstat`` for the per-file added/deleted counts.
        3. ``git show --patch -m`` for the full diff text (incl. merge sides).
        """
        if not re.fullmatch(r"[0-9a-f]{4,40}", sha):
            raise ValueError(f"invalid sha: {sha!r}")

        header = _run_git(
            self.repo_path,
            "log",
            "-1",
            "--pretty=format:%H|%aI|%aE|%aN|%s",
            sha,
        )
        first_line = header.splitlines()[0] if header else ""
        parsed = _parse_log_line(first_line)
        if parsed is None:
            raise RuntimeError(f"git log returned unparseable header for {sha!r}: {first_line!r}")
        full_sha, authored_at, author_email, author, subject = parsed

        numstat = _run_git(
            self.repo_path,
            "show",
            "--numstat",
            "--pretty=format:",
            "-m",
            full_sha,
        )
        files: list[FileChange] = []
        for raw_line in numstat.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            added_s, deleted_s, path = cols[0], cols[1], cols[2]
            # `git --numstat` reports "-\t-" for binary files; coerce to 0.
            added = int(added_s) if added_s.isdigit() else 0
            deleted = int(deleted_s) if deleted_s.isdigit() else 0
            files.append(
                FileChange(path=path, change_type="M", added=added, deleted=deleted)
            )

        diff_text = _run_git(
            self.repo_path,
            "show",
            "--pretty=format:%B",
            "--patch",
            "-m",
            "-U3",
            full_sha,
        )

        return Commit(
            sha=full_sha,
            author=author,
            author_email=author_email,
            authored_at=authored_at,
            message=subject,
            files=files,
            diff_text=diff_text,
            pr_number=_extract_pr_number(subject),
        )
