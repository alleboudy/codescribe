"""``OllamaBackend`` — talk to a pre-running ``ollama serve`` daemon.

Unlike :class:`~codescribe_train.backends.llama_server.LlamaServerBackend`, this
backend **does not start the server itself**. It assumes the user is
already running ``ollama serve`` (Ollama is daemon-style by design); the
backend's ``start`` is a near-no-op that just validates the configured
URL and returns it.

This keeps the backend strictly local: we don't fetch / pull any models,
we don't talk to ollama.com — we only check that ``GET /api/tags`` on the
configured loopback URL responds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from codescribe_train.backends.base import Backend

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass(slots=True)
class OllamaBackend(Backend):
    """Adapter for an externally-managed ``ollama serve`` daemon.

    Construct with the URL Ollama is bound to. ``start`` validates that
    URL and returns it; ``stop`` is a no-op (the user manages the daemon
    lifecycle themselves).
    """

    url: str = DEFAULT_OLLAMA_URL
    require_health_on_start: bool = True
    health_timeout_s: float = 2.0

    # Runtime state — not part of the construction contract.
    _started: bool = field(default=False, init=False)

    # ------------------------------------------------------------------ #
    # Backend interface
    # ------------------------------------------------------------------ #

    def start(self, *, model_id: str, adapter: Path | None, port: int) -> str:
        # ``port`` is unused — Ollama's port is part of ``self.url``. The
        # parameter is kept for ABC parity and so the top-level CLI can
        # use a uniform `start(...)` call across backends.
        del port  # explicit "intentionally unused"
        if adapter is not None:
            logger.warning(
                "OllamaBackend ignores `adapter` (%s); LoRA must be baked into "
                "the Ollama model file via Modelfile + `ollama create`",
                adapter,
            )
        if self.require_health_on_start and not self.health_check(
            self.url, timeout_s=self.health_timeout_s
        ):
            raise RuntimeError(
                f"OllamaBackend: ollama daemon not reachable at {self.url}; "
                "start it with `ollama serve` first"
            )
        logger.info("OllamaBackend ready at %s (model=%s)", self.url, model_id)
        self._started = True
        return self.url

    def stop(self) -> None:
        # Daemon is not owned by us. Mark internal state and return.
        self._started = False

    def health_check(self, endpoint: str, *, timeout_s: float = 5.0) -> bool:
        """Probe ``GET {endpoint}/api/tags`` for the Ollama daemon."""
        try:
            import httpx  # noqa: PLC0415 — lazy import keeps module light
        except ImportError:
            logger.warning("httpx not installed; health_check returning False")
            return False

        url = endpoint.rstrip("/") + "/api/tags"
        try:
            response = httpx.get(url, timeout=timeout_s)
        except httpx.HTTPError as e:
            logger.info("health_check(%s) failed: %s", url, e)
            return False
        return response.status_code < 500
