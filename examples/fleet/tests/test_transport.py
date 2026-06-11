"""Transport contract tests — the push/pull directory-layout invariant.

The key invariant (regression-guarded here): pushing/pulling a *directory* lands
its children DIRECTLY under the destination — no extra nesting level — and this is
identical across LocalTransport and the scp-based SSHTransport. The SSH path nested
one level too deep before the fix because `send_dataset` pre-created the dest dir;
`_scp_push_dest` + send_dataset's parent-only mkdir are what keep them consistent.
"""

from __future__ import annotations

from pathlib import Path

from codescribe_fleet.config import TrainLauncher, WorkerSpec
from codescribe_fleet.transport import LocalTransport, _scp_push_dest
from codescribe_fleet.worker import Worker

from .conftest import FakeTransport


def test_scp_push_dest_dir_goes_to_parent(tmp_path: Path) -> None:
    d = tmp_path / "example-repo"
    d.mkdir()
    # a directory is scp'd into the destination's PARENT (so basename becomes dest)
    assert _scp_push_dest(d, "/home/u/proj/datasets/example-repo") == "/home/u/proj/datasets"


def test_scp_push_dest_file_keeps_full_path(tmp_path: Path) -> None:
    f = tmp_path / "tasks.json"
    f.write_text("[]", encoding="utf-8")
    assert _scp_push_dest(f, "/home/u/proj/evals/tasks.json") == "/home/u/proj/evals/tasks.json"


async def test_local_push_dir_merges_children(tmp_path: Path) -> None:
    src = tmp_path / "src" / "example-repo"
    src.mkdir(parents=True)
    (src / "train.jsonl").write_text("x", encoding="utf-8")
    dst = tmp_path / "dst" / "example-repo"  # does not pre-exist

    await LocalTransport().push(src, dst)
    assert (dst / "train.jsonl").is_file()
    assert not (dst / "example-repo").exists()  # no extra nesting


async def test_local_pull_dir_merges_children(tmp_path: Path) -> None:
    remote = tmp_path / "remote" / "combo-001"
    remote.mkdir(parents=True)
    (remote / "eval-report.json").write_text("{}", encoding="utf-8")
    local = tmp_path / "out" / "combo-001"

    await LocalTransport().pull(remote, local)
    assert (local / "eval-report.json").is_file()
    assert not (local / "combo-001").exists()


async def test_send_dataset_lands_children_under_remote(tmp_path: Path) -> None:
    # local dataset dir with a file
    local = tmp_path / "src" / "example-repo"
    local.mkdir(parents=True)
    (local / "train.jsonl").write_text("data", encoding="utf-8")

    ws = tmp_path / "ws"
    ws.mkdir()
    remote = str(ws / "datasets" / "example-repo")

    spec = WorkerSpec(name="fleet-01", ssh_host="localhost", workspace=str(ws))
    t = FakeTransport(spec.name)
    worker = Worker(spec, t, TrainLauncher(runner="python", script="scripts/fake_train.py"))

    await worker.send_dataset(local, remote)

    # children land directly under remote — not at remote/example-repo/...
    assert (Path(remote) / "train.jsonl").is_file()
    assert not (Path(remote) / "example-repo").exists()
    # the parent was created, not the leaf itself
    mkdirs = [c for c in t.calls if c.startswith("mkdir")]
    assert any(str(ws / "datasets") in c for c in mkdirs)
    assert not any(c.rstrip("'\" ").endswith("example-repo") for c in mkdirs)
