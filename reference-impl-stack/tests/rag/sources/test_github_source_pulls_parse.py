"""Fixture-driven GitHub pulls parser tests."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

FIXTURES = Path(__file__).parent / "fixtures"


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_iter_pulls_yields_pullrequest_dataclasses() -> None:
    from codescribe_train.rag.sources.github_source import GitHubSource, PullRequest

    payload = (FIXTURES / "pulls_page1.json").read_text()
    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(stdout=payload),
    ):
        src = GitHubSource(owner="example-org", repo="sample")
        pulls = list(
            src.iter_pulls_changed_since(datetime(2026, 1, 1, tzinfo=UTC))
        )

    # Ascending updated_at — PR #11 first, then #13, then #14.
    numbers = [p.number for p in pulls]
    assert numbers == [11, 13, 14]

    merged = pulls[0]
    assert isinstance(merged, PullRequest)
    assert merged.title == "feat: closes #1"
    assert merged.state == "merged"  # GitHub returns "closed" + merged_at set
    assert merged.merged_at == "2026-02-01T09:00:00Z"
    assert merged.head_sha == "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    assert merged.base_branch == "main"
    assert merged.author == "example-org"
    assert merged.draft is False

    draft = pulls[1]
    assert draft.draft is True  # The pairing pipeline will use this.
    assert draft.merged_at is None
    assert draft.state == "open"

    closed_not_merged = pulls[2]
    assert closed_not_merged.merged_at is None
    assert closed_not_merged.state == "closed"
    assert closed_not_merged.body is None


def test_iter_pulls_respects_since_watermark() -> None:
    """PRs older than `since` are filtered out client-side."""
    from codescribe_train.rag.sources.github_source import GitHubSource

    payload = (FIXTURES / "pulls_page1.json").read_text()
    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(stdout=payload),
    ):
        src = GitHubSource(owner="example-org", repo="sample")
        pulls = list(
            src.iter_pulls_changed_since(
                datetime(2026, 4, 1, tzinfo=UTC)
            )
        )

    # Only the two PRs with updated_at >= 2026-04-01 survive.
    assert [p.number for p in pulls] == [13, 14]


def test_iter_pulls_uses_pulls_endpoint() -> None:
    from codescribe_train.rag.sources.github_source import GitHubSource

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(stdout="[]"),
    ) as run_mock:
        src = GitHubSource(owner="example-org", repo="sample")
        list(src.iter_pulls_changed_since(datetime(2026, 1, 1, tzinfo=UTC)))

    args = run_mock.call_args.args[0]
    assert args[0] == "gh"
    assert "api" in args
    assert any("repos/example-org/sample/pulls" in a for a in args)
    assert any("sort=updated" in a for a in args)
    assert any("direction=desc" in a for a in args)


def test_get_pr_diff_invokes_gh_pr_diff() -> None:
    from codescribe_train.rag.sources.github_source import GitHubSource

    diff = (
        "diff --git a/foo.py b/foo.py\n"
        "index 1..2 100644\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(stdout=diff),
    ) as run_mock:
        src = GitHubSource(owner="example-org", repo="sample")
        out = src.get_pr_diff(11)

    assert out == diff
    args = run_mock.call_args.args[0]
    assert args[:3] == ["gh", "pr", "diff"]
    assert "11" in args
    # The --repo flag scopes the call to the right repo (operator may be in
    # an unrelated working directory).
    assert "--repo" in args
    assert "example-org/sample" in args


def test_get_pr_diff_rejects_non_positive_numbers() -> None:
    import pytest

    from codescribe_train.rag.sources.github_source import GitHubSource

    src = GitHubSource(owner="example-org", repo="sample")
    with pytest.raises(ValueError):
        src.get_pr_diff(0)
    with pytest.raises(ValueError):
        src.get_pr_diff(-5)
