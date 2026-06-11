"""Weight reconciliation between two spec sections, part of the work.

An earlier step left `configs/rag.yaml` with `body_closes_keyword: 0.7`
(one section's value) and flagged that the canonical value is `0.85`. A
later step made the canonical-spec value real: 0.85 in both the YAML and
the pydantic default. This test pins both so a future drift is caught at
import time.
"""

from __future__ import annotations

from pathlib import Path

from codescribe_train.rag.config import PairingWeights, load


def test_config_yaml_body_closes_keyword_is_0_85() -> None:
    yaml_path = Path(__file__).resolve().parents[3] / "configs" / "rag.yaml"
    cfg = load(yaml_path)
    assert cfg.pairing.weights.body_closes_keyword == 0.85


def test_pairing_weights_pydantic_default_is_0_85() -> None:
    """Pydantic default mirrors the YAML to avoid drift on omitted keys."""
    assert PairingWeights().body_closes_keyword == 0.85
