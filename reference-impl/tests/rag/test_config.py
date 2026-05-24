from __future__ import annotations

from pathlib import Path

import pytest

from codescribe_rag.rag.config import RagConfig

CONFIG = Path(__file__).parents[2] / "configs" / "rag.yaml"


def test_loads_example_yaml():
    cfg = RagConfig.from_yaml(CONFIG)
    assert cfg.perforce.p4port.startswith("ssl:")
    assert cfg.bugzilla.allowlist_hostname == "bugzilla.corp.example.com"
    assert cfg.embed.dim == 1024
    assert cfg.pairing.threshold == 0.8
    assert cfg.retrieve.rrf_constant == 60
    assert cfg.rate_limit_rps == 5.0


def test_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("perforce:\n  p4port: x\n  nonsense_key: 1\n")
    with pytest.raises(Exception):
        RagConfig.from_yaml(bad)


def test_expands_user_paths():
    cfg = RagConfig.from_yaml(CONFIG)
    assert not str(cfg.bugzilla.api_key_path).startswith("~")
    assert not str(cfg.perforce.ticket_path).startswith("~")
