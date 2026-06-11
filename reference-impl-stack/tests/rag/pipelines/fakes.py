"""In-process fakes for the pipeline tests.

The real :class:`GitSource` and :class:`GitHubSource` spawn ``git`` /
``gh`` subprocesses; the pipeline tests don't want either. These fakes
mirror the iterator + ``get_pr_diff`` + ``iter_closing_references``
surface the pipeline depends on, and keep all records in-memory.

The :class:`FakeEmbedder` returns a deterministic ``(N, 1024)``
L2-normalised matrix derived from a string hash — cheap, no torch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from codescribe_train.rag.sources.git_source import Commit
from codescribe_train.rag.sources.github_source import Issue, PullRequest


def _deterministic_vector(text: str) -> np.ndarray:
    """Return a 1024-d L2-normalised float32 vector derived from ``text``."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    v = rng.standard_normal(1024).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


@dataclass
class FakeGitSource:
    """In-memory stand-in for :class:`codescribe_train.rag.sources.git_source.GitSource`."""

    commits: list[Commit] = field(default_factory=list)

    def iter_commits_since(self, last_sha: str | None) -> Iterator[Commit]:
        # Commits ordered ASC by ``authored_at`` (the real GitSource enforces this).
        ordered = sorted(self.commits, key=lambda c: c.authored_at)
        if last_sha is None:
            yield from ordered
            return
        # `last_sha` is exclusive — drop the first commit whose SHA matches and
        # everything before it.
        seen_marker = False
        for commit in ordered:
            if not seen_marker:
                if commit.sha == last_sha:
                    seen_marker = True
                continue
            yield commit

    def describe(self, sha: str) -> Commit:
        for commit in self.commits:
            if commit.sha == sha:
                return commit
        raise KeyError(sha)


@dataclass
class FakeGitHubSource:
    """In-memory stand-in for :class:`GitHubSource`."""

    issues: list[Issue] = field(default_factory=list)
    pulls: list[PullRequest] = field(default_factory=list)
    pr_diffs: dict[int, str] = field(default_factory=dict)
    closing_refs: list[tuple[int, int]] = field(default_factory=list)

    def iter_issues_changed_since(self, since: datetime) -> Iterator[Issue]:
        cutoff_iso = since.isoformat().replace("+00:00", "Z")
        ordered = sorted(self.issues, key=lambda i: i.updated_at)
        for issue in ordered:
            if issue.updated_at >= cutoff_iso:
                yield issue

    def iter_pulls_changed_since(self, since: datetime) -> Iterator[PullRequest]:
        cutoff_iso = since.isoformat().replace("+00:00", "Z")
        ordered = sorted(self.pulls, key=lambda p: p.updated_at)
        for pr in ordered:
            if pr.updated_at >= cutoff_iso:
                yield pr

    def get_pr_diff(self, pr_number: int) -> str:
        return self.pr_diffs.get(pr_number, "")

    def iter_closing_references(self) -> Iterator[tuple[int, int]]:
        yield from self.closing_refs


class FakeEmbedder:
    """Deterministic in-memory embedder; no torch, no model load."""

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[int] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.call_count += 1
        self.calls.append(len(texts))
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        rows = [_deterministic_vector(t) for t in texts]
        return np.stack(rows, axis=0).astype(np.float32)


def make_issue(
    number: int,
    *,
    title: str = "issue",
    body: str = "body",
    state: str = "open",
    closed_at: str | None = None,
    updated_at: str = "2026-05-01T12:00:00Z",
    assignees: list[str] | None = None,
    author: str | None = "example-org",
) -> Issue:
    return Issue(
        number=number,
        title=title,
        body=body,
        state=state,
        state_reason=None,
        labels=[],
        assignees=assignees or [],
        author=author,
        created_at=updated_at,
        updated_at=updated_at,
        closed_at=closed_at,
        raw_json=f'{{"number": {number}}}',
    )


def make_pull(
    number: int,
    *,
    title: str = "pr",
    body: str | None = "body",
    state: str = "merged",
    draft: bool = False,
    updated_at: str = "2026-05-02T12:00:00Z",
    merged_at: str | None = "2026-05-02T12:00:00Z",
    author: str | None = "example-org",
) -> PullRequest:
    return PullRequest(
        number=number,
        title=title,
        body=body,
        state=state,
        head_sha=f"{number:040d}",
        base_branch="main",
        author=author,
        draft=draft,
        created_at=updated_at,
        updated_at=updated_at,
        merged_at=merged_at,
        closed_at=merged_at,
        raw_json=f'{{"number": {number}}}',
    )


def make_commit(
    sha: str,
    *,
    message: str = "fix: something",
    authored_at: str = "2026-05-01T12:00:00Z",
    author: str = "example-org",
    diff_text: str = "diff --git a/x b/x\n@@ -1 +1 @@\n-old\n+new\n",
    pr_number: int | None = None,
) -> Commit:
    return Commit(
        sha=sha,
        author=author,
        author_email=f"{author}@example.com",
        authored_at=authored_at,
        message=message,
        files=[],
        diff_text=diff_text,
        pr_number=pr_number,
    )
