"""``Backend`` ABC — the pluggable inference-server interface.

A backend owns the lifecycle of an OpenAI-compatible HTTP server: spawn
the process, wait for it to listen, return its endpoint URL, tear it down
on exit. Concrete implementations (e.g.
:class:`~codescribe_train.backends.llama_server.LlamaServerBackend`,
:class:`~codescribe_train.backends.vllm.VllmBackend`,
:class:`~codescribe_train.backends.ollama.OllamaBackend`) wrap a specific server
binary or pre-running daemon behind this small surface.

Design notes
------------

* **Strictly local by contract.** Implementations must bind to
  ``127.0.0.1`` (or whatever loopback-equivalent their config dictates) and
  never reach the public network. ``health_check`` is the only method
  permitted to make a network call, and only against the backend's own URL.
* **Idempotent ``stop``.** ``stop()`` is safe to call multiple times, on a
  never-started backend, or after the underlying process has already
  exited. Callers wrap ``start`` / ``stop`` in ``try/finally``.
* **Lazy imports.** Optional / heavy dependencies (``httpx``, ``psutil``,
  ``pynvml``, vendor-specific clients) are imported inside the methods
  that need them so ``import codescribe_train.backends.<X>`` stays cheap and the
  CLI ``--help`` doesn't pull a full dependency stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Backend(ABC):
    """Abstract base class for inference-server adapters.

    Concrete subclasses must implement all three methods. The interface is
    deliberately small so swapping ``llama-server`` for ``vllm`` (or a
    remote LAN backend, opt-in only) is a one-line config change.
    """

    @abstractmethod
    def start(self, *, model_id: str, adapter: Path | None, port: int) -> str:
        """Bring up the server. Return the OpenAI-compat endpoint URL it's listening on.

        ``model_id`` is either a path to a model file (e.g. a Q4_K_M
        ``.gguf`` for ``llama-server``) or a server-specific identifier
        (e.g. an HF model id for ``vllm``, an Ollama tag for ``ollama``).
        ``adapter`` is an optional path to a LoRA-adapter directory; the
        ``llama-server`` default expects a fully-merged GGUF and so passes
        ``adapter=None``. ``port`` is the loopback port to bind.

        Returns the full ``http://127.0.0.1:<port>`` URL the harness can
        point at; callers immediately follow with ``health_check`` to
        confirm the listener is up.
        """

    @abstractmethod
    def stop(self) -> None:
        """Gracefully terminate. Idempotent.

        Safe to call multiple times, on a never-started backend, or after
        the underlying process has already exited. Implementations escalate
        from a soft signal (``terminate``) to a hard one (``kill``) on a
        bounded timeout.
        """

    @abstractmethod
    def health_check(self, endpoint: str, *, timeout_s: float = 5.0) -> bool:
        """Probe ``endpoint`` for OpenAI-compat readiness. Return ``True`` if it answers.

        Implementations probe ``GET {endpoint}/v1/models`` (or a backend-
        specific equivalent like ``GET /api/tags`` for Ollama). This is the
        only method allowed to touch the network.
        """
