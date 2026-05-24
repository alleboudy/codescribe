from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)
DIM = 1024


@runtime_checkable
class SupportsEmbedding(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_one(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic, dependency-light embedder for tests / offline wiring.

    Hashes token unigrams+bigrams into a fixed 1024-dim vector, then L2-normalises.
    Not semantic -- same-token overlap drives similarity. Never use in production;
    the real `Embedder` (bge-large) provides semantic retrieval quality.
    """

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self._vec(text)

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        toks = (text or "").lower().split()
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
            v[h % self._dim] += 1.0
        n = float(np.linalg.norm(v))
        if n == 0.0:
            v[0] = 1.0
            n = 1.0
        return (v / n).astype(np.float32)


class Embedder:
    """Local sentence-transformers wrapper (bge-large). Lazy-loads; L2-normalises.

    torch + sentence-transformers are imported only on first use, so importing this
    module (and running the test suite) costs nothing when the model isn't installed.
    """

    def __init__(self, model_path: Path, device: str = "auto",
                 batch_size: int = 32, max_length: int = 512) -> None:
        self._model_path = model_path
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = self._resolve_device(torch)
            logger.info("loading embedder %s on %s", self._model_path, device)
            self._model = SentenceTransformer(str(self._model_path), device=device)
            self._model.max_seq_length = self._max_length
            self._model.encode(["warmup"], batch_size=1, normalize_embeddings=True)
        return self._model

    def _resolve_device(self, torch) -> str:
        if self._device != "auto":
            return self._device
        if not torch.cuda.is_available():
            return "cpu"
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) < (8, 0):  # pre-Ampere: no bf16
            logger.info("GPU capability %d.%d < 8.0; will use fp16", major, minor)
        return "cuda"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        model = self._ensure_loaded()
        emb = model.encode(texts, batch_size=self._batch_size, show_progress_bar=False,
                           normalize_embeddings=True, convert_to_numpy=True)
        return emb.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def make_embedder(cfg) -> SupportsEmbedding:
    """Build the embedder selected by ``EmbedConfig.backend``."""
    if cfg.backend == "hashing":
        return HashingEmbedder(cfg.dim)
    return Embedder(cfg.model_path, cfg.device, cfg.batch_size, cfg.max_length)
