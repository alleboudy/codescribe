"""Tests for :mod:`codescribe_train.data.formatters`."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

from codescribe_train.data.formatters import (
    Sample,
    format_diff_instructions,
    format_document,
    format_fim,
)
from codescribe_train.data.walker import FileRecord, walk_repo


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
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)
    return root


def _rec(abspath: Path) -> FileRecord:
    st = abspath.stat()
    return FileRecord(
        relpath=abspath.name,
        abspath=abspath,
        size_bytes=st.st_size,
        ext=abspath.suffix[1:] if abspath.suffix else "",
    )


# ---- format_document ----


def test_document_emits_one_sample_per_file(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    a.write_text("x = 1\n")
    b = tmp_path / "b.py"
    b.write_text("y = 2\n")
    samples = list(format_document([_rec(a), _rec(b)], repo_name="testrepo"))
    assert len(samples) == 2
    assert all(isinstance(s, Sample) for s in samples)


def test_document_includes_repo_and_file_markers(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    a.write_text("x = 1\n")
    [s] = list(format_document([_rec(a)], repo_name="myrepo"))
    assert "<|repo_name|>myrepo" in s.text
    assert "<|file_sep|>a.py" in s.text
    assert "x = 1" in s.text
    assert s.metadata["src_path"] == "a.py"
    assert s.metadata["format"] == "document"


# ---- format_fim ----


def test_fim_emits_samples_with_all_three_markers(tmp_path: Path) -> None:
    f = tmp_path / "f.py"
    f.write_text("\n".join(f"line_{i} = {i}" for i in range(20)) + "\n")
    rng = random.Random(0)
    samples = list(format_fim([_rec(f)], rng=rng, samples_per_file=3))
    assert len(samples) == 3
    for s in samples:
        assert "<|fim_prefix|>" in s.text
        assert "<|fim_suffix|>" in s.text
        assert "<|fim_middle|>" in s.text
        assert s.metadata["format"] == "fim"


def test_fim_reconstructs_original_when_concatenated(tmp_path: Path) -> None:
    f = tmp_path / "f.py"
    original = "\n".join(f"line_{i} = {i}" for i in range(20)) + "\n"
    f.write_text(original)
    rng = random.Random(42)
    [s] = list(format_fim([_rec(f)], rng=rng, samples_per_file=1))
    # Extract prefix / suffix / middle and verify they reassemble to the source.
    body = s.text
    pref_start = body.index("<|fim_prefix|>") + len("<|fim_prefix|>")
    pref_end = body.index("<|fim_suffix|>")
    suf_end = body.index("<|fim_middle|>")
    prefix = body[pref_start:pref_end]
    suffix = body[pref_end + len("<|fim_suffix|>") : suf_end]
    middle = body[suf_end + len("<|fim_middle|>") :]
    assert prefix + middle + suffix == original


def test_fim_skips_too_short_files(tmp_path: Path) -> None:
    short = tmp_path / "tiny.py"
    short.write_text("x = 1\n")
    rng = random.Random(0)
    samples = list(format_fim([_rec(short)], rng=rng, samples_per_file=3))
    assert samples == []


def test_fim_deterministic_with_fixed_seed(tmp_path: Path) -> None:
    f = tmp_path / "f.py"
    f.write_text("\n".join(f"line_{i} = {i}" for i in range(50)) + "\n")
    samples_a = list(format_fim([_rec(f)], rng=random.Random(7), samples_per_file=4))
    samples_b = list(format_fim([_rec(f)], rng=random.Random(7), samples_per_file=4))
    assert [s.text for s in samples_a] == [s.text for s in samples_b]


# ---- format_diff_instructions ----


def test_diff_instr_yields_samples_for_substantive_commits(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "r", {"a.py": "x = 1\n"})
    # Add a substantive commit on top.
    (repo / "a.py").write_text("x = 1\ny = 2\nz = 3\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "Add y and z constants"],
        check=True,
    )
    samples = list(format_diff_instructions(repo, max_commits=10))
    assert any("Add y and z constants" in s.metadata["subject"] for s in samples)
    chat_sample = next(s for s in samples if "Add y and z constants" in s.metadata["subject"])
    assert "<|im_start|>user" in chat_sample.text
    assert "<|im_start|>assistant" in chat_sample.text
    assert chat_sample.metadata["format"] == "diff_instr"


def test_diff_instr_filters_short_subjects(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "r", {"a.py": "x = 1\n"})
    (repo / "a.py").write_text("x = 99\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "wip"], check=True)
    samples = list(format_diff_instructions(repo))
    assert all(s.metadata["subject"].lower() != "wip" for s in samples)


def test_diff_instr_filters_oversized_diffs(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "r", {"a.py": "x = 1\n"})
    big = "\n".join(f"v{i} = {i}" for i in range(2000)) + "\n"
    (repo / "a.py").write_text(big)
    subprocess.run(["git", "-C", str(repo), "add", "a.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "Add many constants"],
        check=True,
    )
    samples = list(format_diff_instructions(repo, max_diff_lines=50))
    assert all("Add many constants" not in s.metadata["subject"] for s in samples)


def test_diff_instr_uses_walker_paths_only_indirectly(tmp_path: Path) -> None:
    """Sanity: format_diff_instructions doesn't depend on the walker."""
    repo = _make_repo(tmp_path / "r", {"a.py": "x = 1\n", "b.py": "y = 2\n"})
    # No FileRecord input — formatter takes the repo path directly.
    samples = list(format_diff_instructions(repo, max_commits=5))
    assert isinstance(samples, list)
    # And walker still works on the same repo (smoke check that the test
    # repo isn't corrupted by the formatter).
    assert {r.relpath for r in walk_repo(repo)} == {"a.py", "b.py"}
