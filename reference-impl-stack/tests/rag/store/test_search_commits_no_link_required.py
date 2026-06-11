"""``search_commits`` returns commits even without a linked PR.

Spec test from the design spec:

  > test_search_commits_no_link_required.py — commit not associated
  > with any PR is retrievable.

The commit pipeline allows ``commits.pr_number`` to be ``None``; the
retriever's commit search must not silently filter those out.
"""

from __future__ import annotations

import numpy as np

from codescribe_train.rag.sources.git_source import Commit, FileChange
from codescribe_train.rag.store.writer import Store


class _StubEmbedder:
    """In-test embedder: returns a fixed query vector regardless of input."""

    def __init__(self, query_vector: np.ndarray) -> None:
        self._q = query_vector

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._q for _ in texts], axis=0).astype(np.float32)


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def _commit(
    sha: str,
    *,
    message: str = "fix something",
    pr_number: int | None = None,
) -> Commit:
    return Commit(
        sha=sha,
        author="Alice",
        author_email="alice@example.com",
        authored_at="2026-05-01T12:00:00Z",
        message=message,
        files=[FileChange(path="src/x.py", change_type="M", added=1, deleted=1)],
        diff_text="diff --git a/x b/x\n@@ -1 +1 @@\n-old\n+new\n",
        pr_number=pr_number,
    )


def test_search_commits_returns_commit_with_null_pr(tmp_path) -> None:
    """A commit with ``pr_number=None`` is retrievable by ``search_commits``."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    # Seed: one commit, no PR linkage.
    target_vec = _unit_vec(seed=42)
    sha = "a" * 40
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_commit(_commit(sha, message="fix cart eviction"), target_vec)

    # Query whose embedded form aligns with the commit's vector — RRF
    # picks it up via either signal.
    embedder = _StubEmbedder(target_vec)
    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
        hits = retriever.search_commits(query="cart", k=5)

    assert len(hits) == 1
    assert hits[0].sha == sha
    assert hits[0].message == "fix cart eviction"
    assert hits[0].author == "Alice"
    assert hits[0].files == 1
    assert "old" in hits[0].diff_excerpt or "new" in hits[0].diff_excerpt
    assert hits[0].score > 0.0


def test_search_commits_mix_linked_and_unlinked(tmp_path) -> None:
    """Both linked and unlinked commits show up in the same query."""
    from codescribe_train.rag.embed.chunker import PRChunk
    from codescribe_train.rag.sources.github_source import PullRequest
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    target_vec = _unit_vec(seed=1)
    pr = PullRequest(
        number=42,
        title="parent",
        body=None,
        state="merged",
        head_sha="0" * 40,
        base_branch="main",
        author="example-org",
        draft=False,
        created_at="2026-05-01T10:00:00Z",
        updated_at="2026-05-01T10:00:00Z",
        merged_at="2026-05-01T10:00:00Z",
        closed_at="2026-05-01T10:00:00Z",
        raw_json='{"number": 42}',
    )

    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_pull(
            pr=pr,
            chunks=[PRChunk(pr_number=42, file_path="x", hunk_index=0, chunk_text="x")],
            chunk_embeddings=[_unit_vec(99)],
            pr_summary_embedding=_unit_vec(100),
        )
        store.upsert_commit(
            _commit("a" * 40, message="cart crash", pr_number=42),
            target_vec,
        )
        store.upsert_commit(
            _commit("b" * 40, message="cart eviction", pr_number=None),
            target_vec,
        )

    embedder = _StubEmbedder(target_vec)
    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
        hits = retriever.search_commits(query="cart", k=5)

    shas = {h.sha for h in hits}
    assert "a" * 40 in shas, "linked commit should appear"
    assert "b" * 40 in shas, "unlinked commit (pr_number IS NULL) must appear"
