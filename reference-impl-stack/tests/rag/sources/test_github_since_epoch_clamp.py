"""Regression: GitHub's Issues endpoint silently returns [] for pre-2008 `since`.

The first-time bootstrap defaults `since_issue_updated_at` to the
Unix epoch (1970-01-01). GitHub returns an empty array when `since`
predates the platform's founding year, so a 1970 watermark silently
yields zero issues even when the repo has hundreds.

We clamp the wire-level `since` value to `_GH_EPOCH` (2008-01-01) so
the API behaves the way operators expect. The pipeline still uses the
real watermark internally — only the GitHub request is clamped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest import mock

from codescribe_train.rag.sources.github_source import GitHubSource


def test_iter_issues_clamps_since_to_2008() -> None:
    """A 1970 watermark must be promoted to 2008 in the gh URL."""
    src = GitHubSource("example-org", "sample", rate_limit_rps=1000.0)

    captured_paths: list[str] = []

    def _capture(args: list[str]) -> str:
        # args = ["api", "--paginate", "<path>"]
        captured_paths.append(args[-1])
        return "[]"

    with mock.patch.object(src, "_run_gh", side_effect=_capture):
        list(src.iter_issues_changed_since(datetime(1970, 1, 1, tzinfo=timezone.utc)))

    assert len(captured_paths) == 1
    path = captured_paths[0]
    assert "since=2008-01-01T00:00:00Z" in path
    assert "since=1970" not in path


def test_iter_issues_preserves_modern_since() -> None:
    """A post-2008 watermark passes through unchanged."""
    src = GitHubSource("example-org", "sample", rate_limit_rps=1000.0)
    captured_paths: list[str] = []

    def _capture(args: list[str]) -> str:
        captured_paths.append(args[-1])
        return "[]"

    with mock.patch.object(src, "_run_gh", side_effect=_capture):
        list(
            src.iter_issues_changed_since(
                datetime(2026, 5, 21, 3, 25, 11, tzinfo=timezone.utc)
            )
        )

    assert "since=2026-05-21T03:25:11Z" in captured_paths[0]
