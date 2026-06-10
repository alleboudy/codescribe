"""``RetrieveConfig`` ships with the spec's default knobs.

This is a small bonus test — the spec doesn't require it, but it
locks the RRF defaults (``k_vec=20``, ``k_bm25=20``, ``rrf_k=60``)
so a careless change there breaks the suite loudly.
"""

from __future__ import annotations


def test_retrieve_config_default_knobs() -> None:
    from codescribe_train.rag.store.types import RetrieveConfig

    cfg = RetrieveConfig()
    assert cfg.k_vec == 20
    assert cfg.k_bm25 == 20
    assert cfg.rrf_k == 60


def test_retrieve_config_overrides_take_effect() -> None:
    from codescribe_train.rag.store.types import RetrieveConfig

    cfg = RetrieveConfig(k_vec=5, k_bm25=10, rrf_k=42)
    assert (cfg.k_vec, cfg.k_bm25, cfg.rrf_k) == (5, 10, 42)
