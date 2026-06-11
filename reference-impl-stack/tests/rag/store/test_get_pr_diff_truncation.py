"""``get_pr_diff(max_chars=N)`` truncates the decoded diff with a marker.

Spec test from the design spec:

  > test_get_pr_diff_truncation.py — `max_chars=1000` produces text
  > ending in the truncation marker.

The retriever decodes the gzip-compressed ``pulls.diff_text`` blob
written by the store, optionally clips to ``max_chars``, and
appends a deterministic marker so callers can detect truncation
without re-reading the row.
"""

from __future__ import annotations

import numpy as np

from codescribe_train.rag.embed.chunker import PRChunk
from codescribe_train.rag.sources.github_source import PullRequest
from codescribe_train.rag.store.writer import Store


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_pull(number: int) -> PullRequest:
    return PullRequest(
        number=number,
        title="fix: long diff",
        body="Closes #1.",
        state="merged",
        head_sha="a" * 40,
        base_branch="main",
        author="example-org",
        draft=False,
        created_at="2026-05-01T10:00:00Z",
        updated_at="2026-05-02T11:00:00Z",
        merged_at="2026-05-02T11:00:00Z",
        closed_at="2026-05-02T11:00:00Z",
        raw_json=f'{{"number": {number}}}',
    )


def _seed_pr_with_diff(tmp_path, pr_number: int, diff_text: str) -> None:
    """Write one PR (no chunks, summary embedding only) into a fresh store."""
    pr = _make_pull(pr_number)
    with Store.open(tmp_path / "rag.db") as store:
        store.upsert_pull(
            pr=pr,
            chunks=[PRChunk(pr_number=pr_number, file_path="x", hunk_index=0, chunk_text="x")],
            chunk_embeddings=[_emb(0)],
            pr_summary_embedding=_emb(1),
            diff_text=diff_text,
        )


def test_get_pr_diff_truncates_to_max_chars_with_marker(tmp_path) -> None:
    """A 5000-char diff returned with ``max_chars=1000`` ends in the marker."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    big_diff = "diff --git a/x b/x\n" + ("+a line\n" * 800)  # well over 1000 chars
    assert len(big_diff) > 5000
    _seed_pr_with_diff(tmp_path, pr_number=42, diff_text=big_diff)

    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=None, config=RetrieveConfig())
        out = retriever.get_pr_diff(pr_number=42, max_chars=1000)

    assert out.pr_number == 42
    assert out.truncated is True
    # The deterministic marker substring lives in retrieve.py — assert
    # it shows up *at the end* of the returned text.
    assert out.diff_text.endswith("chars]")
    assert "truncated" in out.diff_text
    # The leading body is still the original diff (no re-encoding shenanigans).
    assert out.diff_text.startswith("diff --git a/x b/x")


def test_get_pr_diff_no_max_returns_full_text(tmp_path) -> None:
    """``max_chars=None`` returns the full decoded diff and ``truncated=False``."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    diff = "diff --git a/y b/y\n@@ -1 +1 @@\n-old\n+new\n"
    _seed_pr_with_diff(tmp_path, pr_number=7, diff_text=diff)

    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=None, config=RetrieveConfig())
        out = retriever.get_pr_diff(pr_number=7, max_chars=None)

    assert out.pr_number == 7
    assert out.truncated is False
    assert out.diff_text == diff


def test_get_pr_diff_short_diff_under_max_chars_not_truncated(tmp_path) -> None:
    """A diff shorter than ``max_chars`` is returned verbatim."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig

    diff = "tiny diff\n"
    _seed_pr_with_diff(tmp_path, pr_number=9, diff_text=diff)

    with Store.open(tmp_path / "rag.db") as store:
        retriever = Retriever(store=store, embedder=None, config=RetrieveConfig())
        out = retriever.get_pr_diff(pr_number=9, max_chars=1000)

    assert out.truncated is False
    assert out.diff_text == diff
