"""Regression: YAML-loaded `~` paths must be expanded.

Pydantic v2 does not run `os.path.expanduser` when coercing strings to
`Path`. Without the field validator in `codescribe_train.rag.config`, a YAML
config like `model_path: ~/.hf-models/...` would land as the literal
`~/...` Path, and `sentence-transformers` would fail with
`FileNotFoundError: Path ~/... not found` at indexer bootstrap time.
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

from codescribe_train.rag.config import (
    EmbedConfig,
    GitSourceConfig,
    StoreConfig,
    load,
)


def test_embed_model_path_expands_tilde_from_yaml(tmp_path: Path) -> None:
    yaml_text = dedent(
        """
        embed:
          model_path: ~/.hf-models/bge-large-en-v1.5
        """
    ).strip()
    yaml_path = tmp_path / "rag.yaml"
    yaml_path.write_text(yaml_text)

    cfg = load(yaml_path)
    expected = Path(os.path.expanduser("~/.hf-models/bge-large-en-v1.5"))
    assert cfg.embed.model_path == expected
    assert "~" not in str(cfg.embed.model_path)


def test_store_db_path_expands_tilde_from_yaml(tmp_path: Path) -> None:
    yaml_text = "store:\n  db_path: ~/var/sample/rag.db\n"
    yaml_path = tmp_path / "rag.yaml"
    yaml_path.write_text(yaml_text)

    cfg = load(yaml_path)
    assert cfg.store.db_path == Path(os.path.expanduser("~/var/sample/rag.db"))


def test_git_repo_path_expands_tilde_from_yaml(tmp_path: Path) -> None:
    yaml_text = "sources:\n  git:\n    repo_path: ~/repos/sample\n"
    yaml_path = tmp_path / "rag.yaml"
    yaml_path.write_text(yaml_text)

    cfg = load(yaml_path)
    assert cfg.sources.git.repo_path == Path(os.path.expanduser("~/repos/sample"))


def test_relative_paths_pass_through_unchanged() -> None:
    # Relative paths and absolute paths without `~` must not be perturbed.
    e = EmbedConfig(model_path="/abs/path/to/model")
    assert e.model_path == Path("/abs/path/to/model")

    s = StoreConfig(db_path="indices/rag.db")
    assert s.db_path == Path("indices/rag.db")

    g = GitSourceConfig(repo_path="../your-repo")
    assert g.repo_path == Path("../your-repo")
