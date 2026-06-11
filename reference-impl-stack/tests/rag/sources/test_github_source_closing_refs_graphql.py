"""Fixture-driven GraphQL closing-references parser tests."""

from __future__ import annotations

import subprocess
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


def test_iter_closing_references_yields_pr_issue_tuples() -> None:
    """A single GraphQL page → list of (pr_number, issue_number) pairs."""
    from codescribe_train.rag.sources.github_source import GitHubSource

    payload = (FIXTURES / "closing_refs_graphql.json").read_text()
    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(stdout=payload),
    ):
        src = GitHubSource(owner="example-org", repo="sample")
        pairs = list(src.iter_closing_references())

    # PR 11 → issue 1; PR 14 → issues 7 and 8; PR 15 has no refs.
    assert pairs == [(11, 1), (14, 7), (14, 8)]


def test_iter_closing_references_paginates() -> None:
    """When hasNextPage=True the iterator issues a follow-up GraphQL call."""
    from codescribe_train.rag.sources.github_source import GitHubSource

    page1 = (FIXTURES / "closing_refs_graphql_page1.json").read_text()
    page2 = (FIXTURES / "closing_refs_graphql_page2.json").read_text()
    responses = [_completed(stdout=page1), _completed(stdout=page2)]

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        side_effect=responses,
    ) as run_mock:
        src = GitHubSource(owner="example-org", repo="sample")
        pairs = list(src.iter_closing_references())

    # Page1 contributes 3 pairs, page2 contributes 1 pair.
    assert pairs == [(11, 1), (14, 7), (14, 8), (20, 9)]
    assert run_mock.call_count == 2
    # The second call must carry the page-1 endCursor in the GraphQL query.
    second_args = run_mock.call_args_list[1].args[0]
    cursor_present = any("Y3Vyc29yOnYyOpHOAbCdEf" in a for a in second_args)
    assert cursor_present, f"page-2 call must include endCursor; got {second_args!r}"


def test_iter_closing_references_uses_graphql_endpoint() -> None:
    from codescribe_train.rag.sources.github_source import GitHubSource

    with mock.patch(
        "codescribe_train.rag.sources.github_source.subprocess.run",
        return_value=_completed(
            stdout='{"data":{"repository":{"pullRequests":{'
            '"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[]}}}}'
        ),
    ) as run_mock:
        src = GitHubSource(owner="example-org", repo="sample")
        list(src.iter_closing_references())

    args: list[str] = run_mock.call_args.args[0]
    assert args[0] == "gh"
    assert "api" in args
    assert "graphql" in args
    # The query body must reference the owner/repo + the closingIssuesReferences field.
    query_body = " ".join(args)
    assert "example-org" in query_body
    assert "sample" in query_body
    assert "closingIssuesReferences" in query_body
    # And `shell=True` is forbidden everywhere in this package.
    kwargs: dict[str, Any] = run_mock.call_args.kwargs
    assert kwargs.get("shell", False) is False
