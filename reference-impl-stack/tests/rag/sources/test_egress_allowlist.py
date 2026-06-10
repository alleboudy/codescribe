"""Egress allowlist gate — every gh API call must stay on api.github.com.

`gh` defaults to ``api.github.com`` when no explicit hostname is passed,
which is what we want. If the implementation ever started passing a
``--hostname`` flag or an absolute URL that points anywhere else, this
test would fail.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import Any
from unittest import mock
from urllib.parse import urlparse


def _completed(stdout: str = "[]") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _hosts_for_call(args: list[str]) -> list[str]:
    """Return every explicit hostname mentioned in a `gh` invocation.

    Returns an empty list when `gh` is invoked without any explicit host —
    that is fine and means `gh` will default to ``api.github.com``.
    """
    hosts: list[str] = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == "--hostname":
            if i + 1 < len(args):
                hosts.append(args[i + 1])
            skip_next = True
            continue
        if a.startswith("--hostname="):
            hosts.append(a.split("=", 1)[1])
            continue
        # `gh api https://example.com/...` overrides the default host.
        if a.startswith(("http://", "https://")):
            host = urlparse(a).hostname
            if host:
                hosts.append(host)
    return hosts


def _assert_calls_allowed(calls: list[mock._Call]) -> None:
    for call in calls:
        args = call.args[0]
        # Must be a `gh` invocation; anything else in the sources package
        # would be a different egress story we have not opted into.
        assert args[0] == "gh", f"non-gh subprocess in sources: {args!r}"
        for host in _hosts_for_call(args):
            assert host == "api.github.com", (
                f"egress to disallowed host {host!r} in {args!r}"
            )


def test_iter_issues_egress_stays_on_api_github_com() -> None:
    from codescribe_train.rag.sources.github_source import GitHubSource

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(),
    ) as run_mock:
        src = GitHubSource(owner="example-org", repo="sample")
        list(src.iter_issues_changed_since(datetime(2026, 1, 1, tzinfo=UTC)))

    _assert_calls_allowed(run_mock.call_args_list)


def test_iter_pulls_egress_stays_on_api_github_com() -> None:
    from codescribe_train.rag.sources.github_source import GitHubSource

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(),
    ) as run_mock:
        src = GitHubSource(owner="example-org", repo="sample")
        list(src.iter_pulls_changed_since(datetime(2026, 1, 1, tzinfo=UTC)))

    _assert_calls_allowed(run_mock.call_args_list)


def test_get_pr_diff_egress_stays_on_api_github_com() -> None:
    from codescribe_train.rag.sources.github_source import GitHubSource

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(stdout="diff --git a/x b/x\n"),
    ) as run_mock:
        src = GitHubSource(owner="example-org", repo="sample")
        src.get_pr_diff(11)

    _assert_calls_allowed(run_mock.call_args_list)


def test_iter_closing_references_egress_stays_on_api_github_com() -> None:
    from codescribe_train.rag.sources.github_source import GitHubSource

    empty_page = (
        '{"data":{"repository":{"pullRequests":{'
        '"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[]}}}}'
    )
    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(stdout=empty_page),
    ) as run_mock:
        src = GitHubSource(owner="example-org", repo="sample")
        list(src.iter_closing_references())

    _assert_calls_allowed(run_mock.call_args_list)


def test_no_explicit_hostname_means_gh_default() -> None:
    """Documented invariant: omitting `--hostname` lets `gh` default to api.github.com.

    This test asserts that the implementation does NOT pass `--hostname`
    anywhere — relying on gh's default is the strictly-local posture.
    """
    from codescribe_train.rag.sources.github_source import GitHubSource

    captured: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        captured.append(args)
        return _completed()

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        side_effect=fake_run,
    ):
        src = GitHubSource(owner="example-org", repo="sample")
        list(src.iter_issues_changed_since(datetime(2026, 1, 1, tzinfo=UTC)))
        list(src.iter_pulls_changed_since(datetime(2026, 1, 1, tzinfo=UTC)))

    for call_args in captured:
        assert "--hostname" not in call_args
        for token in call_args:
            assert not token.startswith("--hostname=")
