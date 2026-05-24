from __future__ import annotations

from datetime import datetime, timezone

from codescribe_rag.rag.config import BugzillaConfig, PerforceConfig, RagConfig
from codescribe_rag.rag.embed.embedder import HashingEmbedder
from codescribe_rag.rag.pipelines.bootstrap import run_bootstrap
from codescribe_rag.rag.sources.bugzilla import Bug, BugComment
from codescribe_rag.rag.sources.perforce import FileChange, P4Change
from codescribe_rag.rag.store.writer import Store

UTC = timezone.utc


def mk_cfg():
    return RagConfig(perforce=PerforceConfig(p4port="x", p4user="y"),
                     bugzilla=BugzillaConfig(base_url="https://bz.x", allowlist_hostname="bz.x"))


def mk_bug(i, comment):
    return (Bug(i, "NPE", "", "core", "major", "RESOLVED", "FIXED",
                datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 10, tzinfo=UTC), "alice", {"id": i}),
            [BugComment(1, "alice", comment, datetime(2026, 1, 10, tzinfo=UTC))])


def mk_cl(n, desc):
    return P4Change(n, "alice", datetime(2026, 1, 12, tzinfo=UTC), desc,
                    (FileChange("//depot/Foo.java", "edit", 5),), "@@ -1 +1 @@\n-a\n+b\n", False)


class FakeSources:
    def __init__(self, bugs_with_comments, cls):
        self._bc = bugs_with_comments
        self._cls = cls

    def iter_bugs(self):
        for b, _ in self._bc:
            yield b

    def comments_for(self, bug_id):
        return next(c for b, c in self._bc if b.id == bug_id)

    def iter_cls(self):
        yield from self._cls


def test_bootstrap_idempotent(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    src = FakeSources([mk_bug(1001, "Fixed in CL 12345")], [mk_cl(12345, "fixes bug 1001")])
    s1 = run_bootstrap(mk_cfg(), src, store, HashingEmbedder())
    s2 = run_bootstrap(mk_cfg(), src, store, HashingEmbedder())
    assert s1.bugs_added == 1 and s1.cls_added == 1 and s1.links_added == 1
    assert s2.bugs_added == 0 and s2.cls_added == 0
    store.close()


def test_late_arriving_cl_pairs(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    src = FakeSources([mk_bug(1001, "no ref here")], [mk_cl(12345, "fixes bug 1001")])
    stats = run_bootstrap(mk_cfg(), src, store, HashingEmbedder())
    assert stats.links_added == 1
    row = store._conn.execute("SELECT cl_number FROM fix_links WHERE bug_id=1001").fetchone()
    assert row[0] == 12345
    store.close()


def test_batch_size_capped(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    bugs = [mk_bug(1000 + i, "x") for i in range(1, 600)]
    src = FakeSources(bugs, [])
    stats = run_bootstrap(mk_cfg(), src, store, HashingEmbedder(), batch_size=256)
    assert stats.bugs_added == 599
    assert max(stats.batch_sizes) <= 256
    store.close()
