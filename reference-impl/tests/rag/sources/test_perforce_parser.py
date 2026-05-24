from __future__ import annotations

import marshal
import subprocess
import sys
from pathlib import Path

import pytest

from codescribe_rag.rag.sources._ratelimit import TokenBucket
from codescribe_rag.rag.sources.perforce import P4AuthExpired, P4Change, P4Source

STUB = Path(__file__).parent / "fixtures" / "p4_stub.py"


def make_source() -> P4Source:
    # Drive the stub interpreter as the "p4 binary".
    return P4Source(
        p4port="ssl:stub:1666", p4user="tester", ticket_path=Path("/tmp/none"),
        rate_limiter=TokenBucket(1000.0),
        binary=sys.executable, extra_argv=[str(STUB)],
    )


def test_describe_parses_change():
    cl = make_source().describe(1001)
    assert isinstance(cl, P4Change)
    assert cl.cl == 1001
    assert cl.author == "alice"
    assert cl.files and cl.files[0].depot_path.endswith("Foo.java")
    assert "@@ -10,3 +10,5 @@" in cl.diff_text


def test_iter_changes_ascending():
    cls = list(make_source().iter_changes_since(0))
    nums = [c.cl for c in cls]
    assert nums == [1001, 1002] == sorted(nums)


def test_marshal_loop_parses_multiple_records():
    buf = marshal.dumps({b"change": b"1", b"user": b"a"}) + marshal.dumps({b"change": b"2", b"user": b"b"})
    recs = make_source()._decode_marshal(buf)
    assert [r["change"] for r in recs] == ["1", "2"]


def test_auth_expired_raises(monkeypatch):
    src = make_source()

    def boom(*a, **k):
        raise subprocess.CalledProcessError(
            1, "p4", stderr=b"Your session has expired, please login again.")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(P4AuthExpired):
        src.describe(1001)


def test_diff_truncation_detected():
    src = make_source()
    assert src._is_truncated("x" * 1_000_001) is True
    assert src._is_truncated("... truncated") is True
    assert src._is_truncated("small") is False
