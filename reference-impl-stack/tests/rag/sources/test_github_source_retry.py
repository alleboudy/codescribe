"""Tenacity retry behavior: retry on rate-limit / 5xx, fail fast on 401/404."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from unittest import mock

import pytest


def test_retries_on_rate_limit_stderr() -> None:
    """gh stderr containing 'rate limit' triggers retry; success on 2nd attempt."""
    from codescribe_train.rag.sources.github_source import GitHubSource

    rate_limit_err = subprocess.CalledProcessError(
        returncode=1,
        cmd=["gh", "api", "repos/x/y/issues"],
        output="",
        stderr="API rate limit exceeded for user ID 12345",
    )
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")

    # Patch tenacity's sleep so we don't actually wait 1-20 seconds.
    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        side_effect=[rate_limit_err, ok],
    ) as run_mock, mock.patch("tenacity.nap.time.sleep"):
        src = GitHubSource(owner="example-org", repo="sample")
        issues = list(
            src.iter_issues_changed_since(datetime(2026, 1, 1, tzinfo=UTC))
        )

    assert issues == []
    assert run_mock.call_count == 2


def test_retries_on_5xx_stderr() -> None:
    """gh stderr containing 'HTTP 502' triggers retry."""
    from codescribe_train.rag.sources.github_source import GitHubSource

    five_xx_err = subprocess.CalledProcessError(
        returncode=1,
        cmd=["gh", "api", "repos/x/y/issues"],
        output="",
        stderr="HTTP 502: Bad Gateway",
    )
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        side_effect=[five_xx_err, ok],
    ) as run_mock, mock.patch("tenacity.nap.time.sleep"):
        src = GitHubSource(owner="example-org", repo="sample")
        list(
            src.iter_issues_changed_since(
                datetime(2026, 1, 1, tzinfo=UTC)
            )
        )

    assert run_mock.call_count == 2


def test_does_not_retry_on_404() -> None:
    """gh stderr signaling 404 must fail fast (no retry)."""
    from codescribe_train.rag.sources.github_source import GitHubSource

    not_found = subprocess.CalledProcessError(
        returncode=1,
        cmd=["gh", "api", "repos/x/y/issues"],
        output="",
        stderr="gh: HTTP 404: Not Found (/repos/x/y/issues)",
    )

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        side_effect=not_found,
    ) as run_mock, mock.patch("tenacity.nap.time.sleep"):
        src = GitHubSource(owner="example-org", repo="sample")
        with pytest.raises(subprocess.CalledProcessError):
            list(
                src.iter_issues_changed_since(
                    datetime(2026, 1, 1, tzinfo=UTC)
                )
            )

    assert run_mock.call_count == 1, "404 must NOT be retried"


def test_does_not_retry_on_401() -> None:
    """gh stderr signaling 401 must fail fast (no retry)."""
    from codescribe_train.rag.sources.github_source import GitHubSource

    unauthorized = subprocess.CalledProcessError(
        returncode=1,
        cmd=["gh", "api", "repos/x/y/issues"],
        output="",
        stderr="gh: HTTP 401: Bad credentials",
    )

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        side_effect=unauthorized,
    ) as run_mock, mock.patch("tenacity.nap.time.sleep"):
        src = GitHubSource(owner="example-org", repo="sample")
        with pytest.raises(subprocess.CalledProcessError):
            list(
                src.iter_issues_changed_since(
                    datetime(2026, 1, 1, tzinfo=UTC)
                )
            )

    assert run_mock.call_count == 1, "401 must NOT be retried"


def test_retry_gives_up_after_3_attempts() -> None:
    """A persistent rate-limit error must surface after exactly 3 attempts."""
    from codescribe_train.rag.sources.github_source import GitHubSource

    rate_limit_err = subprocess.CalledProcessError(
        returncode=1,
        cmd=["gh", "api", "repos/x/y/issues"],
        output="",
        stderr="API rate limit exceeded",
    )

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        side_effect=rate_limit_err,
    ) as run_mock, mock.patch("tenacity.nap.time.sleep"):
        src = GitHubSource(owner="example-org", repo="sample")
        with pytest.raises(subprocess.CalledProcessError):
            list(
                src.iter_issues_changed_since(
                    datetime(2026, 1, 1, tzinfo=UTC)
                )
            )

    assert run_mock.call_count == 3, "must stop after 3 attempts per spec"
