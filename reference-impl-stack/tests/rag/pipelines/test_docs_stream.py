"""The inner loop drains a docs source + upserts doc chunks."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from codescribe_train.rag.embed.doc_chunker import DocChunk
from codescribe_train.rag.pipelines._common import IndexerDeps, epoch_datetime, run_inner_loop
from codescribe_train.rag.sources.worktree_docs_source import Doc
from codescribe_train.rag.store.writer import Store


class _FakeGit:
    def iter_commits_since(self, last_sha): return []
    def describe(self, sha): raise AssertionError("not called")


class _FakeGitHub:
    def iter_issues_changed_since(self, since): return []
    def iter_pulls_changed_since(self, since): return []
    def get_pr_diff(self, n): return ""
    def iter_closing_references(self): return []


class _FakeDocs:
    def iter_docs(self):
        yield Doc("README.md", [DocChunk("Run", "make up", 0)])
        yield Doc("docs/DEV.md", [DocChunk("Dev", "pytest", 0)])


class _StubEmbedder:
    def embed(self, texts):
        out = np.zeros((len(texts), 1024), dtype=np.float32)
        for i, _ in enumerate(texts):
            out[i, i % 1024] = 1.0
        return out


def test_inner_loop_indexes_docs(tmp_path: Path) -> None:
    db = tmp_path / "rag.db"
    # Touch the store so the schema exists before the loop opens it per-batch.
    with Store.open(db):
        pass
    deps = IndexerDeps(
        git=_FakeGit(),
        github=_FakeGitHub(),
        docs=_FakeDocs(),
        embedder=_StubEmbedder(),
        store_path=db,
    )
    result = asyncio.run(
        run_inner_loop(
            deps,
            since_sha=None,
            since_issue_updated_at=epoch_datetime(),
            since_pull_updated_at=epoch_datetime(),
        )
    )
    assert result.docs == 2
    with Store.open(db) as store:
        n = store.conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
    assert n == 2
