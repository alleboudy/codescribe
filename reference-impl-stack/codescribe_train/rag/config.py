"""Typed configuration loaded from configs/rag.yaml.

This module provides the model classes + a YAML loader; downstream
phases consume these. Defaults match `configs/rag.yaml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


def _expanduser(value: Any) -> Any:
    """Expand a leading `~` on Path-like inputs loaded from YAML.

    Pydantic v2 coerces YAML strings straight to `Path` without expanding
    the tilde — so `model_path: ~/.hf-models/bge-large-en-v1.5` would land
    as a literal `~/...` Path that nothing on disk matches. Run this as a
    `mode="before"` field validator on every user-overridable filesystem
    path so the YAML key behaves the way operators expect.
    """
    if value is None:
        return value
    return Path(value).expanduser()


class GitSourceConfig(BaseModel):
    repo_path: Path = Path("../your-repo")
    since_sha: str | None = None

    _expand_repo_path = field_validator("repo_path", mode="before")(_expanduser)


class GitHubSourceConfig(BaseModel):
    owner: str = "example-org"
    repo: str = "sample"
    allowlist_hostname: str = "api.github.com"


class WorktreeDocsSourceConfig(BaseModel):
    repo_path: Path = Path("../your-repo")

    _expand_repo_path = field_validator("repo_path", mode="before")(_expanduser)


class SourcesConfig(BaseModel):
    git: GitSourceConfig = Field(default_factory=GitSourceConfig)
    github: GitHubSourceConfig = Field(default_factory=GitHubSourceConfig)
    worktree_docs: WorktreeDocsSourceConfig = Field(default_factory=WorktreeDocsSourceConfig)
    rate_limit_rps: float = 5.0


class StoreConfig(BaseModel):
    db_path: Path = Path("indices/rag.db")

    _expand_db_path = field_validator("db_path", mode="before")(_expanduser)


class EmbedConfig(BaseModel):
    model_path: Path = Path("~/.hf-models/bge-large-en-v1.5").expanduser()
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512

    _expand_model_path = field_validator("model_path", mode="before")(_expanduser)


class PairingWeights(BaseModel):
    closing_issues_reference: float = 1.0
    # 0.85 per the design spec (canonical implementation weight; reconciled
    # against an earlier draft which used 0.7).
    body_closes_keyword: float = 0.85
    temporal_author_match: float = 0.3


class PairingConfig(BaseModel):
    strict_threshold: float = 0.8
    weights: PairingWeights = Field(default_factory=PairingWeights)


class RagConfig(BaseModel):
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    pairing: PairingConfig = Field(default_factory=PairingConfig)


def load(path: Path | str) -> RagConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return RagConfig.model_validate(data)
