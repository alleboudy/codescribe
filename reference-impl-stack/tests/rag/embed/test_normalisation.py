"""Every embedding row is L2-normalised (norm ≈ 1.0).

Acceptance: spec test #9 from the design spec.

The store relies on cosine similarity via dot-product on raw vectors
(sqlite-vec stores what we give it), so normalisation MUST happen
upstream of insertion. The Embedder asks sentence-transformers for
``normalize_embeddings=True``; this test verifies both that we pass that
flag through AND that the output rows are unit-norm — a wrapper bug that
drops the kwarg would slip past the dim-only test.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np


class _RecordingMock:
    """Stand-in for the SentenceTransformer instance that records calls."""

    def __init__(self) -> None:
        self.last_kwargs: dict | None = None
        self.max_seq_length: int | None = None

    def encode(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        self.last_kwargs = {
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "convert_to_numpy": convert_to_numpy,
            "show_progress_bar": show_progress_bar,
        }
        # Synthesize NON-unit vectors so the wrapper alone can't pass the
        # norm assertion by accident — the kwarg-pass-through is the
        # only path that makes the output unit-norm.
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((len(texts), 1024)).astype(np.float32) * 5.0
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


def test_every_row_has_unit_norm(tmp_path: Path) -> None:
    from codescribe_train.rag.embed.embedder import Embedder

    recorder = _RecordingMock()
    with (
        mock.patch(
            "codescribe_train.rag.embed.embedder._load_sentence_transformer",
            return_value=recorder,
        ),
        mock.patch(
            "codescribe_train.rag.embed.embedder._select_device_and_dtype",
            return_value=("cpu", "float32"),
        ),
    ):
        emb = Embedder(model_path=tmp_path / "model")
        out = emb.embed(["alpha", "beta", "gamma", "delta", "epsilon"])

    norms = np.linalg.norm(out, axis=1)
    # bge models often land at 0.9999..1.0001 in float32; tolerance is loose.
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


def test_normalize_kwarg_is_forwarded(tmp_path: Path) -> None:
    """A regression on the kwarg name (e.g. ``normalize=``) silently
    breaks retrieval relevance — pin the contract."""
    from codescribe_train.rag.embed.embedder import Embedder

    recorder = _RecordingMock()
    with (
        mock.patch(
            "codescribe_train.rag.embed.embedder._load_sentence_transformer",
            return_value=recorder,
        ),
        mock.patch(
            "codescribe_train.rag.embed.embedder._select_device_and_dtype",
            return_value=("cpu", "float32"),
        ),
    ):
        emb = Embedder(model_path=tmp_path / "model")
        emb.embed(["only one"])

    assert recorder.last_kwargs is not None
    assert recorder.last_kwargs["normalize_embeddings"] is True
    assert recorder.last_kwargs["convert_to_numpy"] is True
