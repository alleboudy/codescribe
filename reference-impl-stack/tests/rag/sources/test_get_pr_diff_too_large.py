"""Regression: GitHub returning HTTP 406 / "diff too_large" must NOT abort.

GitHub returns HTTP 406 / `diff too_large` for very large PR diffs — `gh pr
diff <n>` exits 1 with stderr containing:

    could not find pull request diff: HTTP 406: Sorry, the diff exceeded
    the maximum number of lines (20000) (...)
    PullRequest.diff too_large

Without the soft-fail path, the indexer's run_bootstrap propagated the
CalledProcessError and crashed the whole bootstrap. We treat this as a
known limitation: the PR's metadata still indexes; only its chunks are
skipped. Operators can still call `get_pr_diff` interactively on a
truncated path if they need the diff.
"""

from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from codescribe_train.rag.sources.github_source import GitHubSource


def _too_large_stderr() -> str:
    return (
        "could not find pull request diff: HTTP 406: Sorry, the diff exceeded "
        "the maximum number of lines (20000) "
        "(https://api.github.com/repos/example-org/sample/pulls/315)\n"
        "PullRequest.diff too_large\n"
    )


def test_get_pr_diff_returns_empty_on_too_large() -> None:
    src = GitHubSource("example-org", "sample", rate_limit_rps=1000.0)
    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=1, cmd=["gh", "pr", "diff", "315"], output="", stderr=_too_large_stderr()
        ),
    ):
        assert src.get_pr_diff(315) == ""


def test_get_pr_diff_returns_empty_on_too_large_capitalised() -> None:
    """The string-match is case-insensitive on the `too_large` marker."""
    src = GitHubSource("example-org", "sample", rate_limit_rps=1000.0)
    stderr = "PullRequest.diff TOO_LARGE\n"
    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=1, cmd=["gh", "pr", "diff", "1"], output="", stderr=stderr
        ),
    ):
        assert src.get_pr_diff(1) == ""


def test_get_pr_diff_soft_fails_on_generic_gh_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-too-large gh failure (transient flake, 404, unavailable head ref)
    must NOT abort the index — it returns "" and logs a WARNING.

    Regression: a flaky
    ``gh pr diff`` exited 1 (the diff was fetchable seconds later) and the
    error propagated, which also skipped the docs-indexing stream that runs
    after the pulls stream. A single PR's diff fetch must degrade
    gracefully — the PR metadata still indexes; the diff is backfillable
    with ``index --refresh-pr <N>``.
    """
    src = GitHubSource("example-org", "sample", rate_limit_rps=1000.0)
    stderr = "could not find pull request diff: HTTP 404: Not Found"
    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=1, cmd=["gh", "pr", "diff", "49"], output="", stderr=stderr
        ),
    ), caplog.at_level("WARNING"):
        result = src.get_pr_diff(49)
    assert result == ""
    # Not silent: the skip is logged at WARNING with the PR number so a
    # systematic problem (e.g. broken auth hitting every PR) is visible.
    assert any("49" in rec.message for rec in caplog.records), (
        f"expected a WARNING mentioning PR #49; got {[r.message for r in caplog.records]}"
    )


def test_get_pr_diff_returns_diff_text_on_success() -> None:
    src = GitHubSource("example-org", "sample", rate_limit_rps=1000.0)
    expected = "diff --git a/foo b/foo\n@@ -1 +1 @@\n-x\n+y\n"
    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["gh"], returncode=0, stdout=expected, stderr=""
        ),
    ):
        assert src.get_pr_diff(42) == expected
