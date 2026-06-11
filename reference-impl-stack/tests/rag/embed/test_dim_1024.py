"""Embedder output has shape (N, 1024).

Acceptance: spec test #8 from the design spec.

CI runs on CPU-only Mac, so we mock ``SentenceTransformer.encode`` (and
the lazy device picker) — the test verifies the *contract* of the
Embedder wrapper: shape, dtype, batching pass-through. Live model
inference is operator-verified post-merge.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np


def _fake_sentence_transformer_class() -> mock.MagicMock:
    """Build a MagicMock that imitates SentenceTransformer's surface area."""
    instance = mock.MagicMock()

    def fake_encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        # bge-large-en-v1.5 outputs 1024 dims; mirror that shape.
        n = len(texts)
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((n, 1024)).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / norms
        return arr

    instance.encode.side_effect = fake_encode
    klass = mock.MagicMock(return_value=instance)
    return klass


def test_embed_returns_n_by_1024(tmp_path: Path) -> None:
    from codescribe_train.rag.embed.embedder import Embedder

    with (
        mock.patch(
            "codescribe_train.rag.embed.embedder._load_sentence_transformer",
            _fake_sentence_transformer_class(),
        ),
        mock.patch(
            "codescribe_train.rag.embed.embedder._select_device_and_dtype",
            return_value=("cpu", "float32"),
        ),
    ):
        emb = Embedder(model_path=tmp_path / "model")
        out = emb.embed(["text one", "text two", "text three"])

    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 1024)
    assert out.dtype == np.float32


def test_embed_single_string_input_still_2d(tmp_path: Path) -> None:
    from codescribe_train.rag.embed.embedder import Embedder

    with (
        mock.patch(
            "codescribe_train.rag.embed.embedder._load_sentence_transformer",
            _fake_sentence_transformer_class(),
        ),
        mock.patch(
            "codescribe_train.rag.embed.embedder._select_device_and_dtype",
            return_value=("cpu", "float32"),
        ),
    ):
        emb = Embedder(model_path=tmp_path / "model")
        out = emb.embed(["only one"])

    assert out.shape == (1, 1024)


def test_embed_empty_list_returns_empty_array(tmp_path: Path) -> None:
    from codescribe_train.rag.embed.embedder import Embedder

    with (
        mock.patch(
            "codescribe_train.rag.embed.embedder._load_sentence_transformer",
            _fake_sentence_transformer_class(),
        ),
        mock.patch(
            "codescribe_train.rag.embed.embedder._select_device_and_dtype",
            return_value=("cpu", "float32"),
        ),
    ):
        emb = Embedder(model_path=tmp_path / "model")
        out = emb.embed([])

    assert out.shape == (0, 1024)
