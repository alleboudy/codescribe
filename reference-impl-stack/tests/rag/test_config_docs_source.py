# tests/rag/test_config_docs_source.py
from __future__ import annotations

from pathlib import Path

from codescribe_train.rag.config import RagConfig, WorktreeDocsSourceConfig


def test_default_worktree_docs_config() -> None:
    cfg = RagConfig()
    assert isinstance(cfg.sources.worktree_docs, WorktreeDocsSourceConfig)
    assert cfg.sources.worktree_docs.repo_path == Path("../your-repo")


def test_worktree_docs_repo_path_expands_tilde() -> None:
    cfg = WorktreeDocsSourceConfig(repo_path="~/repos/sample")
    assert "~" not in str(cfg.repo_path)
