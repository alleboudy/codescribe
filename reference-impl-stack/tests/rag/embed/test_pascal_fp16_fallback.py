"""Pascal (sm_61) auto-select picks fp16, not bf16.

Acceptance: spec test from the design spec — "mock
torch.cuda.get_device_capability to return (6,1); assert dtype=float16
chosen".

A Pascal-class GPU is sm_61. Its bf16 path is emulated and useless;
the rag indexer MUST pick fp16. The wrapper
must also pick fp16 on Volta/Turing (sm_70..sm_75) and bf16 only on
sm_80+ (Ampere and newer).
"""

from __future__ import annotations

from unittest import mock


def test_pascal_sm_61_selects_fp16() -> None:
    """The exact spec assertion: (6, 1) → cuda + float16."""
    from codescribe_train.rag.embed import embedder

    with (
        mock.patch.object(embedder, "_load_sentence_transformer"),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.get_device_capability", return_value=(6, 1)),
    ):
        device, dtype = embedder._select_device_and_dtype("auto")

    assert device == "cuda"
    assert dtype == "float16"


def test_ampere_sm_80_selects_bf16() -> None:
    from codescribe_train.rag.embed import embedder

    with (
        mock.patch.object(embedder, "_load_sentence_transformer"),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.get_device_capability", return_value=(8, 0)),
    ):
        device, dtype = embedder._select_device_and_dtype("auto")

    assert device == "cuda"
    assert dtype == "bfloat16"


def test_blackwell_sm_120_selects_bf16() -> None:
    """A Blackwell-class GPU (sm_120) supports bf16."""
    from codescribe_train.rag.embed import embedder

    with (
        mock.patch.object(embedder, "_load_sentence_transformer"),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.get_device_capability", return_value=(12, 0)),
    ):
        device, dtype = embedder._select_device_and_dtype("auto")

    assert (device, dtype) == ("cuda", "bfloat16")


def test_turing_sm_75_selects_fp16() -> None:
    """Volta/Turing have proper fp16 hardware but no bf16."""
    from codescribe_train.rag.embed import embedder

    with (
        mock.patch.object(embedder, "_load_sentence_transformer"),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
    ):
        device, dtype = embedder._select_device_and_dtype("auto")

    assert (device, dtype) == ("cuda", "float16")


def test_no_cuda_falls_back_to_cpu_fp32() -> None:
    from codescribe_train.rag.embed import embedder

    with (
        mock.patch.object(embedder, "_load_sentence_transformer"),
        mock.patch("torch.cuda.is_available", return_value=False),
    ):
        device, dtype = embedder._select_device_and_dtype("auto")

    assert (device, dtype) == ("cpu", "float32")


def test_pascal_fallback_when_embed_called(tmp_path) -> None:
    """The fp16 dtype must be visible on the Embedder after first embed().

    Validates the END-to-END contract: mock CUDA introspection AND
    SentenceTransformer; embed something; read back ``embedder.dtype``.
    This is the form the spec calls out: "mock torch.cuda... AND
    SentenceTransformer itself".
    """
    import numpy as np

    from codescribe_train.rag.embed.embedder import Embedder

    fake_st_instance = mock.MagicMock()
    fake_st_instance.encode.return_value = np.zeros((1, 1024), dtype=np.float32)
    with (
        mock.patch(
            "codescribe_train.rag.embed.embedder._load_sentence_transformer",
            return_value=fake_st_instance,
        ),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.get_device_capability", return_value=(6, 1)),
    ):
        emb = Embedder(model_path=tmp_path / "model")
        emb.embed(["x"])

    assert emb.device == "cuda"
    assert emb.dtype == "float16"


def test_explicit_cpu_request_skips_cuda_introspection() -> None:
    """device='cpu' must not call torch.cuda APIs at all.

    Some CI runners have torch but no CUDA driver; calling
    is_available() is fine but ``get_device_capability()`` raises. A
    user that explicitly asks for CPU should never trigger either.
    """
    from codescribe_train.rag.embed import embedder

    with (
        mock.patch.object(embedder, "_load_sentence_transformer"),
        mock.patch("torch.cuda.is_available") as m_avail,
        mock.patch("torch.cuda.get_device_capability") as m_cap,
    ):
        device, dtype = embedder._select_device_and_dtype("cpu")

    assert (device, dtype) == ("cpu", "float32")
    m_avail.assert_not_called()
    m_cap.assert_not_called()
