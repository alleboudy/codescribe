from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from codescribe_rag.rag.embed.embedder import Embedder, HashingEmbedder

MODEL = Path("~/.hf-models/bge-large-en-v1.5").expanduser()
# The semantic test needs the model dir AND the heavy 'embed' extra (torch +
# sentence-transformers), which the lean test env intentionally omits.
_REAL_EMBED_AVAILABLE = MODEL.exists() and importlib.util.find_spec("sentence_transformers") is not None


def test_hashing_dim_is_1024():
    assert HashingEmbedder().embed_one("hello world").shape == (1024,)


def test_hashing_l2_normalised():
    v = HashingEmbedder().embed_one("some text here")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_hashing_deterministic():
    a = HashingEmbedder().embed_one("repeatable")
    b = HashingEmbedder().embed_one("repeatable")
    assert np.allclose(a, b)


def test_hashing_empty_input():
    assert HashingEmbedder().embed([]).shape == (0, 1024)


def test_embedder_lazy_load():
    e = Embedder(MODEL)
    assert e._model is None  # constructor must not load


@pytest.mark.skipif(not _REAL_EMBED_AVAILABLE, reason="bge-large model + sentence-transformers not both available")
def test_real_normalisation_semantics():
    e = Embedder(MODEL)
    a = e.embed_one("NullPointerException during startup")
    b = e.embed_one("NPE thrown when the app starts")
    assert float(a @ b) > 0.7
