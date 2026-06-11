"""Config-driven file filtering.

Stage 2 of the data pipeline. Takes the raw stream from :mod:`walker` and
prunes it according to a per-repo YAML config: extension allowlists/denylists,
gitignore-style path patterns, and size bounds.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import pathspec

from codescribe_train.data.walker import FileRecord

_DEFAULT_MAX_BYTES = 1 << 30  # 1 GiB — effectively no upper bound; configs should set this.


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Filtering rules.

    Defaults are intentionally permissive — most filtering should be expressed
    in the per-repo YAML config, not hard-coded here.
    """

    include_extensions: frozenset[str] = frozenset()
    """Lowercase extensions to allow (no leading dot). Empty = allow all."""

    exclude_extensions: frozenset[str] = frozenset()
    """Lowercase extensions to reject. Wins over ``include_extensions``."""

    include_paths: tuple[str, ...] = ()
    """Gitignore-style globs; record path must match at least one. Empty = allow all paths."""

    exclude_paths: tuple[str, ...] = ()
    """Gitignore-style globs; record path matching any of these is rejected. Wins over include."""

    min_bytes: int = 0
    max_bytes: int = _DEFAULT_MAX_BYTES

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> FilterConfig:
        d = d or {}
        include = d.get("include") or {}
        exclude = d.get("exclude") or {}
        size = d.get("size") or {}
        return FilterConfig(
            include_extensions=frozenset(
                e.lower().lstrip(".") for e in include.get("extensions", [])
            ),
            exclude_extensions=frozenset(
                e.lower().lstrip(".") for e in exclude.get("extensions", [])
            ),
            include_paths=tuple(include.get("paths", [])),
            exclude_paths=tuple(exclude.get("paths", [])),
            min_bytes=int(size.get("min_bytes", 0)),
            max_bytes=int(size.get("max_bytes", _DEFAULT_MAX_BYTES)),
        )


def filter_records(records: Iterable[FileRecord], config: FilterConfig) -> Iterator[FileRecord]:
    """Yield records that pass every rule in ``config``."""
    include_path_spec = (
        pathspec.PathSpec.from_lines("gitignore", config.include_paths)
        if config.include_paths
        else None
    )
    exclude_path_spec = (
        pathspec.PathSpec.from_lines("gitignore", config.exclude_paths)
        if config.exclude_paths
        else None
    )
    for record in records:
        if record.size_bytes < config.min_bytes or record.size_bytes > config.max_bytes:
            continue
        if config.include_extensions and record.ext not in config.include_extensions:
            continue
        if record.ext in config.exclude_extensions:
            continue
        if include_path_spec is not None and not include_path_spec.match_file(record.relpath):
            continue
        if exclude_path_spec is not None and exclude_path_spec.match_file(record.relpath):
            continue
        yield record
