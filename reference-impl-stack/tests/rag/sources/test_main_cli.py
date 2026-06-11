"""`python -m codescribe_train.rag.sources` argparse smoke + dispatch tests."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest


def test_cli_no_args_exits_with_usage(capsys: pytest.CaptureFixture[str]) -> None:
    from codescribe_train.rag.sources.__main__ import main

    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "git-probe" in err or "usage" in err.lower()


def test_cli_lists_three_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    """The three operator-verified probes must be registered."""
    from codescribe_train.rag.sources.__main__ import main

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out + capsys.readouterr().err
    # argparse usually prints --help to stdout; tolerate either stream.
    for sub in ("git-probe", "gh-probe", "gh-closing-refs"):
        assert sub in out, f"subcommand {sub!r} missing from CLI help: {out!r}"


def test_git_probe_dispatches_to_git_source(tmp_path: Path) -> None:
    """`git-probe --repo <p> --last-sha <s>` calls GitSource.iter_commits_since."""
    from codescribe_train.rag.sources.__main__ import main

    fake_commit = mock.MagicMock(
        sha="a" * 40,
        author="Alice",
        authored_at="2026-05-19T10:00:00+00:00",
        message="feat: hi",
        pr_number=None,
    )
    with mock.patch(
        "codescribe_train.rag.sources.__main__.GitSource"
    ) as GS:
        GS.return_value.iter_commits_since.return_value = iter([fake_commit])
        rc = main(["git-probe", "--repo", str(tmp_path), "--last-sha", "HEAD~50"])

    assert rc == 0
    GS.assert_called_once_with(repo_path=tmp_path)
    GS.return_value.iter_commits_since.assert_called_once_with(last_sha="HEAD~50")


def test_gh_probe_dispatches_to_github_source() -> None:
    """`gh-probe --owner X --repo Y --since 2026-01-01` calls both iter methods."""
    from codescribe_train.rag.sources.__main__ import main

    fake_issue = mock.MagicMock(number=1, updated_at="2026-04-01T00:00:00Z", title="x")
    fake_pr = mock.MagicMock(number=11, updated_at="2026-04-15T00:00:00Z", title="y")
    with mock.patch(
        "codescribe_train.rag.sources.__main__.GitHubSource"
    ) as GH:
        GH.return_value.iter_issues_changed_since.return_value = iter([fake_issue])
        GH.return_value.iter_pulls_changed_since.return_value = iter([fake_pr])
        rc = main(
            [
                "gh-probe",
                "--owner",
                "example-org",
                "--repo",
                "sample",
                "--since",
                "2026-01-01",
            ]
        )

    assert rc == 0
    GH.assert_called_once_with(owner="example-org", repo="sample")
    assert GH.return_value.iter_issues_changed_since.called
    assert GH.return_value.iter_pulls_changed_since.called


def test_gh_closing_refs_dispatches_to_github_source() -> None:
    """`gh-closing-refs --owner X --repo Y` calls iter_closing_references."""
    from codescribe_train.rag.sources.__main__ import main

    with mock.patch(
        "codescribe_train.rag.sources.__main__.GitHubSource"
    ) as GH:
        GH.return_value.iter_closing_references.return_value = iter(
            [(11, 1), (14, 7), (14, 8)]
        )
        rc = main(
            ["gh-closing-refs", "--owner", "example-org", "--repo", "sample"]
        )

    assert rc == 0
    GH.assert_called_once_with(owner="example-org", repo="sample")
    GH.return_value.iter_closing_references.assert_called_once_with()


def test_module_invocation_returns_0(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke-test `python -m codescribe_train.rag.sources git-probe` resolves to main."""
    from codescribe_train.rag.sources import __main__ as m

    # Patch out the actual GitSource to avoid touching a real repo.
    fake_commit = mock.MagicMock(
        sha="a" * 40,
        author="Alice",
        authored_at="2026-05-19T10:00:00+00:00",
        message="feat: hi",
        pr_number=None,
    )
    monkeypatch.setattr(m, "GitSource", mock.MagicMock(
        return_value=mock.MagicMock(
            iter_commits_since=mock.MagicMock(return_value=iter([fake_commit]))
        )
    ))
    # Calling main([...]) should return 0 cleanly.
    rc = m.main(["git-probe", "--repo", "/tmp/fake-repo"])
    assert rc == 0
