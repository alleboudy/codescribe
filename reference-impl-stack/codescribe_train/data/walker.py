"""Git-aware repository walker.

Stage 1 of the data pipeline. Enumerates tracked files in a git repository via
``git ls-files``. Produces file metadata only — content loading happens later,
so big files we'll filter out aren't read into memory.
"""

from __future__ import annotations

import stat as stat_mod
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One tracked, regular file in a git repository.

    ``relpath`` is POSIX-style and relative to the repo root; ``abspath`` is the
    absolute resolved path on disk. ``ext`` is the lowercase suffix without the
    leading dot (e.g. ``"py"``, ``"tsx"``); empty string when there is no
    extension.
    """

    relpath: str
    abspath: Path
    size_bytes: int
    ext: str


class NotAGitRepo(ValueError):
    """Raised when the requested path is not the root of a git repository."""


def assert_git_root(repo: Path) -> Path:
    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise NotAGitRepo(f"not a directory: {repo}")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise NotAGitRepo(f"`git rev-parse` failed in {repo}: {e.stderr.strip()}") from e
    toplevel = Path(result.stdout.strip()).resolve()
    if toplevel != repo:
        raise NotAGitRepo(
            f"{repo} is not a git repo root (toplevel is {toplevel}); pass the root explicitly"
        )
    return repo


def walk_repo(repo: Path) -> Iterator[FileRecord]:
    """Yield a :class:`FileRecord` for every tracked regular file in ``repo``.

    Uses ``git ls-files -z`` (null-separated) so the output is robust to weird
    filenames. Symlinks, gitlinks (submodules), and any non-regular entries are
    skipped. Files tracked by git but currently missing on disk are also
    skipped — rare, but can happen during interactive rebases.
    """
    repo = assert_git_root(repo)
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    if not result.stdout:
        return
    for relpath_bytes in result.stdout.split(b"\x00"):
        if not relpath_bytes:
            continue
        relpath = relpath_bytes.decode("utf-8", errors="surrogateescape")
        abspath = repo / relpath
        try:
            st = abspath.lstat()
        except FileNotFoundError:
            continue
        if not stat_mod.S_ISREG(st.st_mode):
            continue
        ext = abspath.suffix[1:].lower() if abspath.suffix else ""
        yield FileRecord(
            relpath=relpath,
            abspath=abspath,
            size_bytes=st.st_size,
            ext=ext,
        )
