"""Fixture-driven Git source parser tests.

All ``subprocess.run`` calls are monkey-patched; no real git binary is
invoked.
"""

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


def test_commit_dataclass_has_required_fields() -> None:
    from codescribe_train.rag.sources.git_source import Commit, FileChange

    c = Commit(
        sha="a" * 40,
        author="Alice",
        author_email="alice@example.com",
        authored_at="2026-05-19T10:00:00+00:00",
        message="msg",
        files=[FileChange(path="src/foo.py", change_type="M", added=1, deleted=0)],
        diff_text="diff --git ...",
        pr_number=None,
    )
    assert c.sha == "a" * 40
    assert c.files[0].path == "src/foo.py"
    assert c.pr_number is None


def test_iter_commits_since_parses_pipe_format(tmp_path: Path) -> None:
    """`iter_commits_since` runs `git log` and parses %H|%aI|%aE|%aN|%s lines."""
    from codescribe_train.rag.sources.git_source import GitSource

    log_text = (FIXTURES / "git_log.txt").read_text()
    # git log emits newest-first; reverse to get ASC order for resumability.
    # The fixture is already roughly time-ordered; the parser should not care
    # about input order — it should sort ascending by `authored_at` itself.
    with mock.patch(
        "codescribe_train.rag.sources.git_source.subprocess.run",
        return_value=_completed(stdout=log_text),
    ):
        src = GitSource(repo_path=tmp_path)
        commits = list(src.iter_commits_since(last_sha=None))

    assert len(commits) == 5
    # Ascending by authored_at — first commit is the 10:00 one.
    assert commits[0].sha == "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
    assert commits[0].author == "Alice Author"
    assert commits[0].author_email == "alice@example.com"
    assert commits[0].authored_at == "2026-05-19T10:00:00+00:00"
    assert commits[0].message == "feat(core): add diff parser"
    # Subjects with the pipe character must round-trip (the parser splits
    # only on the first four separators).
    assert commits[-1].message == "docs: clarify usage | also note edge cases"


def test_iter_commits_since_uses_git_log_all_branches(tmp_path: Path) -> None:
    """The git invocation must include `--all` so multi-branch repos resume."""
    from codescribe_train.rag.sources.git_source import GitSource

    with mock.patch(
        "codescribe_train.rag.sources.git_source.subprocess.run",
        return_value=_completed(stdout=""),
    ) as run_mock:
        src = GitSource(repo_path=tmp_path)
        list(src.iter_commits_since(last_sha=None))

    args = run_mock.call_args.args[0]
    assert args[0] == "git"
    assert "-C" in args
    assert str(tmp_path) in args
    assert "log" in args
    assert "--all" in args
    # No `shell=True` — `shell` must be either omitted or False.
    kwargs: dict[str, Any] = run_mock.call_args.kwargs
    assert kwargs.get("shell", False) is False
    assert kwargs.get("check") is True
    assert kwargs.get("capture_output") is True
    assert kwargs.get("text") is True
    assert kwargs.get("timeout") == 60


def test_iter_commits_since_filters_by_last_sha(tmp_path: Path) -> None:
    """Passing `last_sha` excludes that SHA and everything before it."""
    from codescribe_train.rag.sources.git_source import GitSource

    # AGENTS.md spec: enumerate via `git log ... <last_sha>..HEAD --all`
    # We assert the rev-range gets composed correctly.
    with mock.patch(
        "codescribe_train.rag.sources.git_source.subprocess.run",
        return_value=_completed(stdout=""),
    ) as run_mock:
        src = GitSource(repo_path=tmp_path)
        list(src.iter_commits_since(last_sha="deadbeef" * 5))

    args = run_mock.call_args.args[0]
    assert any("deadbeef" * 5 in a for a in args), (
        f"last_sha must appear in git invocation; got {args!r}"
    )


def test_describe_parses_show_output_and_extracts_pr_number(tmp_path: Path) -> None:
    """`describe(sha)` returns Commit with files + diff and merge PR number."""
    from codescribe_train.rag.sources.git_source import GitSource

    log_line = (
        "c3d4e5f60718293a4b5c6d7e8f9012345678901b|"
        "2026-05-19T12:45:00+00:00|alice@example.com|Alice Author|"
        "Merge pull request #43 from feature/branch\n"
    )
    show_text = (FIXTURES / "git_show_merge.txt").read_text()

    # The implementation issues two subprocess calls inside describe:
    #   1. `git log -1 --pretty=format:%H|%aI|%aE|%aN|%s <sha>`
    #   2. `git show <sha> --pretty=format:%B --patch -m -U3`
    # plus a third for `--numstat` to populate FileChange counts.
    numstat_text = "1\t0\tsrc/foo.py\n1\t1\tsrc/bar.py\n"

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        if "log" in args:
            return _completed(stdout=log_line)
        if "show" in args and "--numstat" in args:
            return _completed(stdout=numstat_text)
        if "show" in args:
            return _completed(stdout=show_text)
        raise AssertionError(f"unexpected git call: {args!r}")

    with mock.patch(
        "codescribe_train.rag.sources.git_source.subprocess.run", side_effect=fake_run
    ):
        src = GitSource(repo_path=tmp_path)
        commit = src.describe("c3d4e5f60718293a4b5c6d7e8f9012345678901b")

    assert commit.sha == "c3d4e5f60718293a4b5c6d7e8f9012345678901b"
    assert commit.pr_number == 43  # "Merge pull request #43" → 43
    assert commit.diff_text.startswith("Merge pull request #43")
    paths = {f.path for f in commit.files}
    assert paths == {"src/foo.py", "src/bar.py"}
    foo = next(f for f in commit.files if f.path == "src/foo.py")
    assert foo.added == 1
    assert foo.deleted == 0


def test_describe_extracts_pr_number_from_paren_suffix(tmp_path: Path) -> None:
    """Subjects ending with `(#N)` (squash-merge style) populate pr_number."""
    from codescribe_train.rag.sources.git_source import GitSource

    log_line = (
        "b2c3d4e5f60718293a4b5c6d7e8f90123456789a|"
        "2026-05-19T11:30:00+00:00|bob@example.com|Bob Builder|"
        "fix(parser): handle empty diffs (#42)\n"
    )

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        if "log" in args:
            return _completed(stdout=log_line)
        if "show" in args and "--numstat" in args:
            return _completed(stdout="")
        if "show" in args:
            return _completed(stdout="fix(parser): handle empty diffs (#42)\n")
        raise AssertionError(f"unexpected git call: {args!r}")

    with mock.patch(
        "codescribe_train.rag.sources.git_source.subprocess.run", side_effect=fake_run
    ):
        src = GitSource(repo_path=tmp_path)
        commit = src.describe("b2c3d4e5f60718293a4b5c6d7e8f90123456789a")

    assert commit.pr_number == 42


def test_describe_extracts_pr_number_from_merges_pr_keyword(tmp_path: Path) -> None:
    """Subjects with `Merges PR #N` (per spec text) populate pr_number."""
    from codescribe_train.rag.sources.git_source import GitSource

    log_line = (
        "d4e5f60718293a4b5c6d7e8f90123456789012cd|"
        "2026-05-19T13:00:00+00:00|carol@example.com|Carol Coder|"
        "chore: bump version. Merges PR #44\n"
    )

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        if "log" in args:
            return _completed(stdout=log_line)
        if "show" in args and "--numstat" in args:
            return _completed(stdout="")
        if "show" in args:
            return _completed(stdout="chore: bump version. Merges PR #44\n")
        raise AssertionError(f"unexpected git call: {args!r}")

    with mock.patch(
        "codescribe_train.rag.sources.git_source.subprocess.run", side_effect=fake_run
    ):
        src = GitSource(repo_path=tmp_path)
        commit = src.describe("d4e5f60718293a4b5c6d7e8f90123456789012cd")

    assert commit.pr_number == 44


def test_describe_returns_none_pr_number_for_non_merge(tmp_path: Path) -> None:
    """Plain feature commits with no PR marker have pr_number == None."""
    from codescribe_train.rag.sources.git_source import GitSource

    log_line = (
        "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678|"
        "2026-05-19T10:00:00+00:00|alice@example.com|Alice Author|"
        "feat(core): add diff parser\n"
    )

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        if "log" in args:
            return _completed(stdout=log_line)
        if "show" in args and "--numstat" in args:
            return _completed(stdout="")
        if "show" in args:
            return _completed(stdout="feat(core): add diff parser\n")
        raise AssertionError(f"unexpected git call: {args!r}")

    with mock.patch(
        "codescribe_train.rag.sources.git_source.subprocess.run", side_effect=fake_run
    ):
        src = GitSource(repo_path=tmp_path)
        commit = src.describe("a1b2c3d4e5f60718293a4b5c6d7e8f9012345678")

    assert commit.pr_number is None
