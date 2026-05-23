from __future__ import annotations

from codescribe_rag.rag.embed.embedder import HashingEmbedder
from codescribe_rag.rag.pipelines.bootstrap import run_bootstrap
from codescribe_rag.rag.pipelines.incremental import read_watermarks
from codescribe_rag.rag.store.writer import Store

from tests.rag.pipelines.test_bootstrap import FakeSources, mk_bug, mk_cfg, mk_cl


def test_watermarks_advance(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    src = FakeSources([mk_bug(1001, "CL 12345")], [mk_cl(12345, "fixes bug 1001")])
    run_bootstrap(mk_cfg(), src, store, HashingEmbedder())
    last_cl, since = read_watermarks(store)
    assert last_cl == 12345
    assert since.year == 2026
    store.close()
