"""Tests for :mod:`codescribe_train.data.filters`."""

from __future__ import annotations

from pathlib import Path

from codescribe_train.data.filters import FilterConfig, filter_records
from codescribe_train.data.walker import FileRecord


def _rec(relpath: str, ext: str = "py", size: int = 100) -> FileRecord:
    return FileRecord(relpath=relpath, abspath=Path(f"/tmp/{relpath}"), size_bytes=size, ext=ext)


def test_empty_config_passes_all_through() -> None:
    records = [_rec("a.py"), _rec("b.ts", ext="ts"), _rec("c.md", ext="md")]
    assert list(filter_records(records, FilterConfig())) == records


def test_include_extensions_keeps_only_matching() -> None:
    config = FilterConfig(include_extensions=frozenset({"py", "ts"}))
    records = [_rec("a.py"), _rec("b.ts", ext="ts"), _rec("c.md", ext="md")]
    out = [r.relpath for r in filter_records(records, config)]
    assert out == ["a.py", "b.ts"]


def test_exclude_extensions_wins_over_include() -> None:
    config = FilterConfig(
        include_extensions=frozenset({"py", "lock"}),
        exclude_extensions=frozenset({"lock"}),
    )
    records = [_rec("a.py"), _rec("uv.lock", ext="lock")]
    out = [r.relpath for r in filter_records(records, config)]
    assert out == ["a.py"]


def test_exclude_paths_gitignore_semantics() -> None:
    config = FilterConfig(
        exclude_paths=(
            "**/__pycache__/**",
            "**/migrations/**",
            "**/test_*.py",
            "**/*.lock",
        )
    )
    records = [
        _rec("src/a.py"),
        _rec("src/__pycache__/b.pyc", ext="pyc"),
        _rec("backend/migrations/0001_init.py"),
        _rec("tests/test_a.py"),
        _rec("uv.lock", ext="lock"),
    ]
    out = [r.relpath for r in filter_records(records, config)]
    assert out == ["src/a.py"]


def test_include_paths_acts_as_allowlist() -> None:
    config = FilterConfig(include_paths=("src/**",))
    records = [_rec("src/a.py"), _rec("docs/b.py")]
    out = [r.relpath for r in filter_records(records, config)]
    assert out == ["src/a.py"]


def test_size_bounds() -> None:
    config = FilterConfig(min_bytes=10, max_bytes=100)
    records = [
        _rec("tiny.py", size=5),
        _rec("ok.py", size=50),
        _rec("huge.py", size=1000),
    ]
    out = [r.relpath for r in filter_records(records, config)]
    assert out == ["ok.py"]


def test_from_dict_normalizes_extensions() -> None:
    config = FilterConfig.from_dict(
        {
            "include": {"extensions": [".PY", "Ts", "tsx"]},
            "exclude": {"extensions": ["LOCK"]},
        }
    )
    assert config.include_extensions == frozenset({"py", "ts", "tsx"})
    assert config.exclude_extensions == frozenset({"lock"})


def test_from_dict_handles_empty_input() -> None:
    config = FilterConfig.from_dict(None)
    assert config == FilterConfig()
    config2 = FilterConfig.from_dict({})
    assert config2 == FilterConfig()


def test_from_dict_passes_paths_and_size() -> None:
    config = FilterConfig.from_dict(
        {
            "include": {"paths": ["src/**"]},
            "exclude": {"paths": ["**/test_*.py"]},
            "size": {"min_bytes": 16, "max_bytes": 512_000},
        }
    )
    assert config.include_paths == ("src/**",)
    assert config.exclude_paths == ("**/test_*.py",)
    assert config.min_bytes == 16
    assert config.max_bytes == 512_000
