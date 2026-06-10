"""Fixture builders for the ``repo-rag`` MCP server tests.

Re-uses the store writer + stub-embedder pattern to seed a small
rag.db that exercises all three tools without booting a real model. Each
fixture returns a path to a fresh tmp_path-scoped DB so tests stay
isolated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codescribe_train.rag.embed.chunker import PRChunk
from codescribe_train.rag.extract.pairing import IssuePRLink
from codescribe_train.rag.sources.git_source import Commit, FileChange
from codescribe_train.rag.sources.github_source import Issue, PullRequest
from codescribe_train.rag.store.writer import Store


class StubEmbedder:
    """In-test embedder: returns a fixed query vector regardless of input.

    Public (not underscored) so test modules can import it from the
    conftest without tripping ruff's "private import" rule.
    """

    def __init__(self, query_vector: np.ndarray) -> None:
        self._q = query_vector

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._q for _ in texts], axis=0).astype(np.float32)


def _basis_vec(index: int) -> np.ndarray:
    """A 1024-d L2-normalised vector with all weight on dimension ``index``."""
    v = np.zeros(1024, dtype=np.float32)
    v[index] = 1.0
    return v


@pytest.fixture
def query_vec() -> np.ndarray:
    """The query direction used by every fixture below (basis dim 0)."""
    return _basis_vec(0)


@pytest.fixture
def stub_embedder(query_vec: np.ndarray) -> StubEmbedder:
    """A :class:`StubEmbedder` that always returns the canonical query vec."""
    return StubEmbedder(query_vec)


@pytest.fixture
def seeded_db(tmp_path: Path, query_vec: np.ndarray) -> Path:
    """A small rag.db with one issue, one PR (linked at conf 1.0), one commit.

    The PR's diff contains a unified-diff header so the
    ``get_pr_diff`` Markdown check (``@@`` substring) passes without
    relying on the chunker producing one — this fixture writes the
    diff text directly.
    """
    db_path = tmp_path / "rag.db"

    issue = Issue(
        number=7,
        title="cart crash on save",
        body="Saving the cart editor crashes when the layer is empty.",
        state="closed",
        state_reason="completed",
        labels=["bug"],
        assignees=["example-org"],
        author="example-org",
        created_at="2026-05-01T12:00:00Z",
        updated_at="2026-05-02T15:00:00Z",
        closed_at="2026-05-02T16:00:00Z",
        raw_json='{"number": 7}',
    )

    pr = PullRequest(
        number=42,
        title="fix: cart save crash on empty layer",
        body="Closes #7. Guards the empty-layer branch before save.",
        state="merged",
        head_sha="a" * 40,
        base_branch="main",
        author="example-org",
        draft=False,
        created_at="2026-05-01T10:00:00Z",
        updated_at="2026-05-02T11:00:00Z",
        merged_at="2026-05-02T11:00:00Z",
        closed_at="2026-05-02T11:00:00Z",
        raw_json='{"number": 42}',
    )
    chunks = [
        PRChunk(
            pr_number=42,
            file_path="src/cart.py",
            hunk_index=0,
            chunk_text="@@ -1 +1 @@\n-old\n+new\n",
        ),
        PRChunk(
            pr_number=42,
            file_path="tests/test_cart.py",
            hunk_index=0,
            chunk_text="@@ -10 +10 @@\n-assert old\n+assert new\n",
        ),
    ]
    diff_text = (
        "diff --git a/src/cart.py b/src/cart.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/src/cart.py\n"
        "+++ b/src/cart.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    commit = Commit(
        sha="b" * 40,
        author="Alice",
        author_email="alice@example.com",
        authored_at="2026-05-02T10:30:00Z",
        message="fix: cart crash on empty layer save",
        files=[FileChange(path="src/cart.py", change_type="M", added=1, deleted=1)],
        diff_text=(
            "diff --git a/src/cart.py b/src/cart.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        ),
        pr_number=42,
    )

    with Store.open(db_path) as store:
        store.upsert_issue(issue, query_vec)
        store.upsert_pull(
            pr=pr,
            chunks=chunks,
            chunk_embeddings=[_basis_vec(100), _basis_vec(101)],
            pr_summary_embedding=query_vec,
            diff_text=diff_text,
        )
        store.upsert_commit(commit, query_vec)
        store.upsert_issue_pr_link(
            IssuePRLink(
                issue_number=7,
                pr_number=42,
                source="closingIssuesReferences",
                confidence=1.0,
            )
        )

    return db_path
