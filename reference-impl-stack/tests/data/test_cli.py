"""Integration test for ``python -m codescribe_train.data build``."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from codescribe_train.data.cli import build


def _make_repo(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    for relpath, contents in files.items():
        p = root / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contents)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", "initial scaffold commit"],
        check=True,
    )
    return root


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    files = {f"src/file_{i:03d}.py": f"# file {i}\n" + "x = 1\n" * 30 for i in range(40)}
    files["tests/test_things.py"] = "def test_foo(): pass\n"
    files["package.json"] = '{"name": "x"}\n'
    files["uv.lock"] = "(lockfile)\n"
    files["README.md"] = "# Hello\n\nThis is the repo.\n"
    return _make_repo(tmp_path / "synth", files)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    cfg = {
        "repo_name": "synth",
        "seed": 7,
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        "filter": {
            "include": {"extensions": ["py", "md"]},
            "exclude": {
                "paths": [
                    "**/tests/**",
                    "**/test_*.py",
                    "**/*.lock",
                ]
            },
            "size": {"min_bytes": 4, "max_bytes": 100_000},
        },
        "formatters": {
            "document": {"enabled": True},
            "fim": {"enabled": True, "samples_per_file": 1},
            "diff_instr": {"enabled": False},
        },
    }
    p = tmp_path / "synth.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_build_writes_jsonl_and_manifest(
    synthetic_repo: Path, config_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    args = argparse.Namespace(
        cmd="build", repo=str(synthetic_repo), config=str(config_path), out=str(out)
    )
    rc = build(args)
    assert rc == 0

    for split in ("train", "val", "test"):
        f = out / f"{split}.jsonl"
        assert f.exists(), f"missing {split}.jsonl"
        # Every line is valid JSON with `text` and `meta` keys.
        for line in f.read_text().splitlines():
            row = json.loads(line)
            assert "text" in row
            assert "meta" in row
            assert isinstance(row["text"], str)
            assert "format" in row["meta"]

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["repo_name"] == "synth"
    assert "totals" in manifest
    assert manifest["totals"]["train"] > 0


def test_build_excludes_filtered_paths(
    synthetic_repo: Path, config_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    args = argparse.Namespace(
        cmd="build", repo=str(synthetic_repo), config=str(config_path), out=str(out)
    )
    build(args)
    # No sample should reference files we excluded.
    for split in ("train", "val", "test"):
        for line in (out / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            src = row["meta"].get("src_path", "")
            assert "tests/" not in src
            assert not src.endswith(".lock")


def test_build_split_files_disjoint(
    synthetic_repo: Path, config_path: Path, tmp_path: Path
) -> None:
    """Same source file must never appear in two different splits."""
    out = tmp_path / "out"
    args = argparse.Namespace(
        cmd="build", repo=str(synthetic_repo), config=str(config_path), out=str(out)
    )
    build(args)
    paths_per_split = {}
    for split in ("train", "val", "test"):
        paths = set()
        for line in (out / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            src = row["meta"].get("src_path")
            if src:
                paths.add(src)
        paths_per_split[split] = paths
    assert paths_per_split["train"].isdisjoint(paths_per_split["val"])
    assert paths_per_split["train"].isdisjoint(paths_per_split["test"])
    assert paths_per_split["val"].isdisjoint(paths_per_split["test"])


def test_module_runner_works(synthetic_repo: Path, config_path: Path, tmp_path: Path) -> None:
    """`python -m codescribe_train.data build ...` invocation works end-to-end."""
    out = tmp_path / "out_mod"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "codescribe_train.data",
            "build",
            "--repo",
            str(synthetic_repo),
            "--config",
            str(config_path),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "manifest.json").exists()
