from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _expand(p: str | Path) -> Path:
    return Path(p).expanduser()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PerforceConfig(_Strict):
    p4port: str
    p4user: str
    ticket_path: Path = Path("~/.p4tickets")
    depot_path: str = "//depot/main/..."
    binary: str = "p4"
    subprocess_timeout_s: float = 60.0

    @field_validator("ticket_path", mode="before")
    @classmethod
    def _exp(cls, v):
        return _expand(v)


class BugzillaConfig(_Strict):
    base_url: str
    allowlist_hostname: str
    api_key_path: Path = Path("~/.config/codescribe_rag/bugzilla.key")
    page_size: int = 100
    request_timeout_s: float = 30.0
    connect_timeout_s: float = 5.0

    @field_validator("api_key_path", mode="before")
    @classmethod
    def _exp(cls, v):
        return _expand(v)


class StoreConfig(_Strict):
    db_path: Path = Path("indices/rag.db")


class EmbedConfig(_Strict):
    backend: str = "sentence-transformers"   # | "hashing"
    model_path: Path = Path("~/.hf-models/bge-large-en-v1.5")
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    dim: int = 1024

    @field_validator("model_path", mode="before")
    @classmethod
    def _exp(cls, v):
        return _expand(v)


class PairingWeights(_Strict):
    bug_comment_cites_cl: float = 0.5
    cl_desc_cites_bug: float = 0.5
    temporal_proximity: float = 0.3
    assignee_author_match: float = 0.2
    bug_status_fixed: float = 0.1


class PairingConfig(_Strict):
    weights: PairingWeights = Field(default_factory=PairingWeights)
    threshold: float = 0.8
    temporal_signal_days: int = 7
    candidate_window_days: int = 60
    min_bug_id: int = 1000
    min_cl_id: int = 1000


class RetrieveConfig(_Strict):
    k: int = 5
    k_vec: int = 20
    k_bm25: int = 20
    confidence_threshold: float = 0.8
    rrf_constant: int = 60


class RagConfig(_Strict):
    perforce: PerforceConfig
    bugzilla: BugzillaConfig
    store: StoreConfig = Field(default_factory=StoreConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    pairing: PairingConfig = Field(default_factory=PairingConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    rate_limit_rps: float = 5.0

    @classmethod
    def from_yaml(cls, path: Path) -> "RagConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
