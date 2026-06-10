"""Tests for :mod:`codescribe_train.data.walker`."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from codescribe_train.data.walker import NotAGitRepo, walk_repo


def _make_repo(root: Path, files: dict[str, str]) -> Path:
    """Create a git repo at ``root`` with ``{relpath: contents}`` files committed."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    for relpath, contents in files.items():
        p = root / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contents)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)
    return root


def test_walks_tracked_files(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path / "r",
        {
            "a.py": "x = 1\n",
            "src/b.ts": "export const y = 2;\n",
            "README.md": "# hi\n",
        },
    )
    records = sorted(walk_repo(repo), key=lambda r: r.relpath)
    assert [r.relpath for r in records] == ["README.md", "a.py", "src/b.ts"]
    assert [r.ext for r in records] == ["md", "py", "ts"]
    assert all(r.size_bytes > 0 for r in records)
    assert all(r.abspath.is_absolute() for r in records)


def test_skips_untracked(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "r", {"a.py": "x = 1\n"})
    (repo / "untracked.txt").write_text("not staged\n")
    relpaths = [r.relpath for r in walk_repo(repo)]
    assert relpaths == ["a.py"]


def test_skips_symlinks(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "r", {"target.py": "x = 1\n"})
    (repo / "link.py").symlink_to(repo / "target.py")
    subprocess.run(["git", "add", "link.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add link"], cwd=repo, check=True)
    relpaths = sorted(r.relpath for r in walk_repo(repo))
    # The symlink is tracked but skipped because it's not a regular file.
    assert relpaths == ["target.py"]


def test_extensionless_files(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path / "r",
        {
            "Makefile": "all:\n\techo hi\n",
            "a.py": "x = 1\n",
        },
    )
    by_relpath = {r.relpath: r for r in walk_repo(repo)}
    assert by_relpath["Makefile"].ext == ""
    assert by_relpath["a.py"].ext == "py"


def test_rejects_non_git_dir(tmp_path: Path) -> None:
    d = tmp_path / "plain"
    d.mkdir()
    with pytest.raises(NotAGitRepo):
        list(walk_repo(d))


def test_rejects_subdir_of_repo(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "r", {"sub/a.py": "x = 1\n"})
    with pytest.raises(NotAGitRepo):
        list(walk_repo(repo / "sub"))


def test_handles_unicode_filenames(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "r", {"حروف.py": "x = 1\n", "正常.txt": "ok\n"})
    relpaths = sorted(r.relpath for r in walk_repo(repo))
    assert relpaths == ["حروف.py", "正常.txt"]


def test_empty_repo_no_commits(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    assert list(walk_repo(repo)) == []
