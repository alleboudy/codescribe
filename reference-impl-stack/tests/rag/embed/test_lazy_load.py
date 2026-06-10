"""Embedder is lazy: sentence_transformers is NOT imported at module import.

Deliverable 3 in the spec requires "lazy load on first call." A
non-lazy import would pull torch/transformers into every other rag
module's import graph and inflate cold-start by 5+ seconds.
"""

from __future__ import annotations

import importlib
import sys


def test_importing_embedder_module_does_not_pull_sentence_transformers() -> None:
    """The first ``import codescribe_train.rag.embed.embedder`` must not import
    ``sentence_transformers``.

    The test forcibly evicts both modules first so the assertion is
    meaningful even when a prior test (test_dim_1024 etc.) already
    transitively imported sentence_transformers via its mock.
    """
    for mod in [
        m
        for m in list(sys.modules)
        if m in {"codescribe_train.rag.embed.embedder", "sentence_transformers"}
        or m.startswith("sentence_transformers.")
    ]:
        del sys.modules[mod]

    importlib.import_module("codescribe_train.rag.embed.embedder")

    pulled = [m for m in sys.modules if m == "sentence_transformers"]
    assert pulled == [], (
        "module-level import of embedder pulled sentence_transformers; "
        "must be lazy. Imported chain may include: "
        + ", ".join(m for m in sys.modules if m.startswith("sentence_transformers"))
    )


def test_constructing_embedder_does_not_load_model(tmp_path) -> None:
    """Constructor must NOT call ``_load_sentence_transformer``."""
    from unittest import mock

    from codescribe_train.rag.embed.embedder import Embedder

    with mock.patch(
        "codescribe_train.rag.embed.embedder._load_sentence_transformer"
    ) as m_load:
        emb = Embedder(model_path=tmp_path / "model")
        m_load.assert_not_called()
        assert emb._model is None  # noqa: SLF001
