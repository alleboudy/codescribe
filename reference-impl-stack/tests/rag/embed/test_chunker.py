"""Diff-aware chunker: chunk_pr_diff + summarise_pr_for_embedding.

Deliverable 4 in the spec. The bonus "never split inside a hunk header"
property test is also here (called out in the spec deviation notes; not
on the required-test list).

NOTE on signature: ``PullRequest`` carries no ``diff_text`` field, so
:func:`chunk_pr_diff` takes the unified diff string as a second
positional arg. The pipeline fetches the diff via
``GitHubSource.get_pr_diff(pr.number)`` and threads it through.
"""

from __future__ import annotations

from codescribe_train.rag.embed.chunker import (
    PRChunk,
    chunk_pr_diff,
    summarise_pr_for_embedding,
)
from codescribe_train.rag.sources.github_source import PullRequest


def _pr(number: int = 42, title: str = "fix: thing", body: str = "Closes #7.") -> PullRequest:
    return PullRequest(
        number=number,
        title=title,
        body=body,
        state="merged",
        head_sha="a" * 40,
        base_branch="main",
        author="example-org",
        draft=False,
        created_at="2026-05-01T10:00:00Z",
        updated_at="2026-05-02T11:00:00Z",
        merged_at="2026-05-02T11:00:00Z",
        closed_at="2026-05-02T11:00:00Z",
        raw_json='{"number": 42}',
    )


# -- chunk_pr_diff -----------------------------------------------------


def test_chunk_pr_diff_returns_one_chunk_per_hunk(tmp_path) -> None:
    diff = (
        "diff --git a/src/cart.py b/src/cart.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/src/cart.py\n"
        "+++ b/src/cart.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a\n"
        "-old\n"
        "+new\n"
        "@@ -10,2 +10,2 @@\n"
        "-x\n"
        "+y\n"
    )
    chunks = chunk_pr_diff(_pr(7), diff)

    assert all(isinstance(c, PRChunk) for c in chunks)
    assert [c.pr_number for c in chunks] == [7, 7]
    assert [c.file_path for c in chunks] == ["src/cart.py", "src/cart.py"]
    assert [c.hunk_index for c in chunks] == [0, 1]
    assert chunks[0].chunk_text.startswith("@@ -1,3 +1,3 @@")
    assert chunks[1].chunk_text.startswith("@@ -10,2 +10,2 @@")


def test_chunk_pr_diff_handles_multiple_files() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n"
        "index aaaa..bbbb 100644\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+aa\n"
        "diff --git a/b.py b/b.py\n"
        "index cccc..dddd 100644\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -1 +1 @@\n"
        "-b\n"
        "+bb\n"
    )
    chunks = chunk_pr_diff(_pr(11), diff)

    assert [c.file_path for c in chunks] == ["a.py", "b.py"]
    # Hunk index resets per file.
    assert [c.hunk_index for c in chunks] == [0, 0]


def test_chunk_pr_diff_never_splits_inside_a_hunk_header() -> None:
    """Bonus property (spec test #9 wording): chunker never breaks ``@@ ... @@``.

    Build a file whose single hunk EXCEEDS max_chars on its own. The
    chunker must still emit one PRChunk per hunk (we do not sub-split
    hunks; that would silently corrupt unified diff syntax).
    """
    huge_lines = "\n".join(f" line {i}" for i in range(2000))
    diff = (
        "diff --git a/big.py b/big.py\n"
        "index aaaa..bbbb 100644\n"
        "--- a/big.py\n"
        "+++ b/big.py\n"
        "@@ -1,2000 +1,2000 @@\n" + huge_lines + "\n"
    )
    chunks = chunk_pr_diff(_pr(12), diff, max_chars=100)

    assert len(chunks) == 1
    # The whole hunk (including the header) is in one chunk.
    assert chunks[0].chunk_text.startswith("@@ -1,2000 +1,2000 @@")
    assert "line 1999" in chunks[0].chunk_text


def test_chunk_pr_diff_empty_string_returns_no_chunks() -> None:
    assert chunk_pr_diff(_pr(1), "") == []


def test_chunk_pr_diff_non_diff_string_returns_no_chunks() -> None:
    """A PR with no diff (or a malformed body) yields nothing — don't crash."""
    assert chunk_pr_diff(_pr(1), "not a diff at all") == []


# -- summarise_pr_for_embedding ----------------------------------------


def test_summary_concatenates_title_body_and_diff_summary() -> None:
    pr = _pr(99, title="fix: item loader", body="Closes #7. Patches LRU eviction.")
    diff = (
        "diff --git a/a.py b/a.py\n"
        "index aaaa..bbbb 100644\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    summary = summarise_pr_for_embedding(pr, diff)

    assert "fix: item loader" in summary
    assert "Closes #7" in summary
    # The diff summary mentions the touched file.
    assert "a.py" in summary


def test_summary_truncates_long_body() -> None:
    """``body[:2000]`` per the spec — long bodies are capped, not dropped."""
    long_body = "x" * 5000
    pr = _pr(99, title="t", body=long_body)
    summary = summarise_pr_for_embedding(pr, "")

    # The summary must contain at most 2000 'x's from the body.
    body_chunk = summary.split("\n")
    # Look for the body line(s).
    body_xs = sum(line.count("x") for line in body_chunk)
    assert body_xs == 2000


def test_summary_handles_none_body() -> None:
    pr = _pr(99, title="title-only", body=None)
    summary = summarise_pr_for_embedding(pr, "")
    assert "title-only" in summary
