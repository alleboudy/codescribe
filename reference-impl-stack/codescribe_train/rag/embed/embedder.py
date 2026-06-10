"""Local 1024-dim text embedder for the rag sidecar.

Wraps a single SentenceTransformer instance (default model:
``BAAI/bge-large-en-v1.5``, loaded from a local snapshot directory). The
wrapper exists to enforce three contracts the rest of the rag package
relies on:

* shape is always ``(N, 1024)`` — the store schema's vec0 virtual tables
  hard-code FLOAT[1024];
* outputs are L2-normalised so cosine similarity reduces to a dot
  product (sqlite-vec stores raw vectors; normalisation must happen
  upstream);
* device + dtype are selected manually rather than via
  ``device_map='auto'`` so that GPUs older than sm_80 (no bf16 support;
  only fp16) get the fp16 path instead of an unsupported bf16 default.

The sentence-transformers import is lazy — the first call to
:meth:`embed` triggers the load. CI runs on CPU-only hosts where the
model is not available; tests patch ``_load_sentence_transformer`` and
``_select_device_and_dtype`` to keep import-time cost at zero.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# bge-large-en-v1.5 is 1024-dim; the store schema hard-codes this.
_EMBED_DIM = 1024


def _load_sentence_transformer(
    model_path: Path,
    device: str,
) -> Any:
    """Lazy SentenceTransformer constructor.

    Patched out in unit tests; never imported until the first
    :meth:`Embedder.embed` call.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path), device=device)


def _select_device_and_dtype(requested: str) -> tuple[str, str]:
    """Pick (device, dtype) via manual compute-capability detection.

    Returns one of:

      * ``('cpu', 'float32')`` — CPU-only host or ``device='cpu'``;
      * ``('cuda', 'float16')`` — GPUs older than sm_80;
      * ``('cuda', 'bfloat16')`` — Ampere+ (sm_80+);
      * ``('cuda', 'float16')`` — any sm_70..sm_75 (Volta, Turing) GPU.

    The bf16/fp16 split matters: on GPUs below sm_80 the bf16 ops emulate
    via scalar code and lose perf catastrophically, so we never pick bf16
    below sm_80.
    """
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(
            f"unsupported device {requested!r}; expected 'auto' | 'cpu' | 'cuda'"
        )
    if requested == "cpu":
        return "cpu", "float32"

    import torch  # local import — keeps the rag-store-only path import-cheap

    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
        return "cpu", "float32"

    cap = torch.cuda.get_device_capability()
    if cap >= (8, 0):
        return "cuda", "bfloat16"
    # sm_61 and Volta/Turing sm_7x → fp16. Pre-sm_61 (sm_60 and
    # below) lack a fully-baked fp16 path either, but bge-large + sm_<6
    # is not a supported config; we still return fp16 and log a warning.
    if cap < (6, 1):
        logger.warning(
            "GPU compute capability %s is below sm_61; fp16 path is untested.", cap
        )
    return "cuda", "float16"


class Embedder:
    """Local text → 1024-dim embedding wrapper.

    Lazy-loaded; the model is materialised on first :meth:`embed` call.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        self.device_request = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None
        self._device: str | None = None
        self._dtype: str | None = None

    # --- lifecycle -----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        device, dtype = _select_device_and_dtype(self.device_request)
        logger.info(
            "loading embedder model=%s device=%s dtype=%s",
            self.model_path,
            device,
            dtype,
        )
        self._model = _load_sentence_transformer(self.model_path, device=device)
        self._device = device
        self._dtype = dtype
        # max_seq_length lives on the SentenceTransformer instance; the
        # wrapper applies it once so encode() truncates per-batch.
        # Mocked instances in unit tests may not expose this attribute.
        with contextlib.suppress(AttributeError):
            self._model.max_seq_length = self.max_length

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def dtype(self) -> str | None:
        return self._dtype

    # --- inference -----------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(N, 1024)`` float32 L2-normalised matrix.

        Empty input short-circuits to an empty ``(0, 1024)`` array so
        callers can pass through whatever the chunker emits without
        special-casing.
        """
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)

        self._ensure_loaded()
        out = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != _EMBED_DIM:
            raise RuntimeError(
                f"embedder returned dim={arr.shape[1]}, expected {_EMBED_DIM}"
            )
        return arr
