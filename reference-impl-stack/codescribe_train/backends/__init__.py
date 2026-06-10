"""``codescribe_train.backends`` — pluggable inference-server layer.

The backend layer brings up an OpenAI-compatible HTTP server on the user's
own machine, with a fine-tuned model + LoRA, so the harness from
:mod:`codescribe_train.harness` can connect to it locally. The default
implementation, :mod:`codescribe_train.backends.llama_server`, wraps the
**vendored** ``ggml-org/llama.cpp`` build at a pinned commit SHA.

The :class:`~codescribe_train.backends.base.Backend` ABC exists so the harness
never knows or cares whether the model is being served by ``llama-server``,
``vllm``, or a running ``ollama serve`` — they all expose the same
``/v1/chat/completions`` surface and look identical to the harness.

Trust posture: the backend layer is **strictly local**. Concretely:

* The :class:`~codescribe_train.backends.remote.RemoteBackend` exists in code
  only — it is never wired into a shipped config and refuses to start
  unless an explicit ``CODESCRIBE_ENABLE_REMOTE_BACKEND=1`` env var is set.
* No telemetry, analytics, error reporting, or auto-update from any
  shipped component, including the vendored ``llama.cpp``.
* The default :class:`~codescribe_train.backends.llama_server.LlamaServerBackend`
  binds to ``127.0.0.1`` only.

See :class:`~codescribe_train.backends.base.Backend` for the interface contract.
"""

from __future__ import annotations
