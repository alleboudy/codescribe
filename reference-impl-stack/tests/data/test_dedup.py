"""Tests for :mod:`codescribe_train.data.dedup`."""

from __future__ import annotations

from pathlib import Path

from codescribe_train.data.dedup import dedup_records
from codescribe_train.data.walker import FileRecord


def _rec(abspath: Path) -> FileRecord:
    st = abspath.stat()
    return FileRecord(
        relpath=abspath.name,
        abspath=abspath,
        size_bytes=st.st_size,
        ext=abspath.suffix[1:] if abspath.suffix else "",
    )


def test_keeps_unique_files(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    out = [r.relpath for r in dedup_records([_rec(a), _rec(b)])]
    assert sorted(out) == ["a.py", "b.py"]


def test_drops_exact_duplicates(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"  # different name, identical content
    a.write_text("def hello():\n    return 'world'\n")
    b.write_text("def hello():\n    return 'world'\n")
    out = [r.relpath for r in dedup_records([_rec(a), _rec(b)])]
    assert out == ["a.py"]  # first one wins


def test_keeps_first_of_duplicates_in_order(tmp_path: Path) -> None:
    paths = []
    for name in ["c.py", "a.py", "b.py"]:
        p = tmp_path / name
        p.write_text("same\n")
        paths.append(p)
    records = [_rec(p) for p in paths]
    out = [r.relpath for r in dedup_records(records)]
    assert out == ["c.py"]


def test_empty_files_dedupe_correctly(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("")
    b.write_text("")
    out = [r.relpath for r in dedup_records([_rec(a), _rec(b)])]
    assert out == ["a.py"]


def test_skips_files_that_disappear(tmp_path: Path, caplog) -> None:
    a = tmp_path / "a.py"
    a.write_text("real\n")
    rec_a = _rec(a)
    rec_missing = FileRecord(
        relpath="missing.py",
        abspath=tmp_path / "missing.py",
        size_bytes=0,
        ext="py",
    )
    out = [r.relpath for r in dedup_records([rec_a, rec_missing])]
    assert out == ["a.py"]
