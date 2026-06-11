"""Fixture-driven GitHub issues parser tests.

All ``subprocess.run`` calls are monkey-patched; no real ``gh`` binary is
invoked.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

FIXTURES = Path(__file__).parent / "fixtures"


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_iter_issues_yields_issue_dataclasses() -> None:
    """`iter_issues_changed_since` parses `gh api` JSON into Issue rows."""
    from codescribe_train.rag.sources.github_source import GitHubSource, Issue

    payload = (FIXTURES / "issues_page1.json").read_text()
    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(stdout=payload),
    ):
        src = GitHubSource(owner="example-org", repo="sample")
        issues = list(
            src.iter_issues_changed_since(datetime(2026, 1, 1, tzinfo=UTC))
        )

    # PR-as-issue (#9) must be filtered out.
    numbers = [i.number for i in issues]
    assert 9 not in numbers
    assert numbers == [1, 7]  # also asserts ascending updated_at order.

    first = issues[0]
    assert isinstance(first, Issue)
    assert first.title == "First sample issue"
    assert first.state == "closed"
    assert first.state_reason == "completed"
    assert first.labels == ["bug", "infra"]
    assert first.assignees == ["example-org"]
    assert first.author == "example-org"
    assert first.created_at == "2026-01-02T08:00:00Z"
    assert first.updated_at == "2026-01-15T12:34:56Z"
    assert first.closed_at == "2026-01-15T12:34:56Z"
    # Raw payload is preserved for the store layer.
    assert "First sample issue" in first.raw_json

    second = issues[1]
    assert second.body is None  # null bodies round-trip as None, not "null"
    assert second.labels == []
    assert second.closed_at is None


def test_iter_issues_passes_since_to_gh_api() -> None:
    """The `since=<iso>` query parameter is built from the datetime arg."""
    from codescribe_train.rag.sources.github_source import GitHubSource

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(stdout="[]"),
    ) as run_mock:
        src = GitHubSource(owner="example-org", repo="sample")
        list(
            src.iter_issues_changed_since(
                datetime(2026, 5, 1, 12, 30, 45, tzinfo=UTC)
            )
        )

    args = run_mock.call_args.args[0]
    assert args[0] == "gh"
    assert "api" in args
    # The endpoint must reference the configured repo.
    assert any("repos/example-org/sample/issues" in a for a in args)
    # `since` carries ISO-8601 with Z suffix.
    assert any("since=2026-05-01T12:30:45Z" in a for a in args)
    # Resilience: don't crash if `--paginate` isn't there, but it should be
    # per the AGENTS.md spec.
    assert "--paginate" in args


def test_iter_issues_handles_empty_response() -> None:
    """Empty `[]` returns from `gh api` yield no issues, no exception."""
    from codescribe_train.rag.sources.github_source import GitHubSource

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(stdout="[]"),
    ):
        src = GitHubSource(owner="example-org", repo="sample")
        issues = list(
            src.iter_issues_changed_since(datetime(2026, 1, 1, tzinfo=UTC))
        )
    assert issues == []


def test_iter_issues_does_not_use_shell(monkeypatch: Any) -> None:
    """Every `gh` invocation must pass `shell=False` (or omit `shell`)."""
    from codescribe_train.rag.sources.github_source import GitHubSource

    captured: list[dict[str, Any]] = []

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        captured.append({"args": args, "kwargs": kwargs})
        return _completed(stdout="[]")

    monkeypatch.setattr(
        "codescribe_train.rag.sources.github_source.subprocess.run", fake_run
    )
    src = GitHubSource(owner="example-org", repo="sample")
    list(src.iter_issues_changed_since(datetime(2026, 1, 1, tzinfo=UTC)))
    assert captured
    for call in captured:
        assert call["kwargs"].get("shell", False) is False
