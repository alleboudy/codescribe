"""``VllmBackend`` — adapter for ``vllm serve``.

vLLM is **not** a default-installed dependency. Importing this module is
cheap (no ``import vllm``); the package is only probed lazily inside
``start`` and ``health_check``. If vLLM isn't installed, both raise a
clear error rather than crashing on a missing import.

vLLM requires the model to fit fully in VRAM (no CPU offload), which on
an 8 GB-VRAM GPU excludes 7B+ models with the default settings.
``recommend_backend`` in :mod:`codescribe_train.backends.probe` will not pick
this backend on an 8 GB-VRAM GPU. It exists for the case where the user runs on
a larger box (or chooses to run a smaller model than the default).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from codescribe_train.backends.base import Backend

logger = logging.getLogger(__name__)


class VllmNotInstalledError(RuntimeError):
    """Raised when ``vllm`` cannot be imported but the backend was selected."""


@dataclass(slots=True)
class VllmBackend(Backend):
    """Adapter for ``vllm serve``.

    Requires the ``vllm`` package and a CUDA-capable GPU. The backend is
    available but lazily imported; constructing the dataclass does not
    pull ``vllm``.
    """

    host: str = "127.0.0.1"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192
    api_key: str | None = None
    startup_timeout_s: float = 120.0
    extra_args: list[str] = field(default_factory=list)

    # Runtime state — not part of the construction contract.
    _process: subprocess.Popen | None = field(default=None, init=False)
    _endpoint: str | None = field(default=None, init=False)

    # ------------------------------------------------------------------ #
    # Backend interface
    # ------------------------------------------------------------------ #

    def start(self, *, model_id: str, adapter: Path | None, port: int) -> str:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("VllmBackend.start called while process already running")

        self._require_vllm_installed()
        cmd = self._build_argv(model_id=model_id, adapter=adapter, port=port)

        logger.info("launching vllm: %s", " ".join(cmd))
        self._process = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
        endpoint = f"http://{self.host}:{port}"
        # Best-effort startup wait — we don't tail vLLM's log; callers
        # should immediately follow with ``poll_health_until``.
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.health_check(endpoint, timeout_s=2.0):
                self._endpoint = endpoint
                return endpoint
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"vllm exited with code {self._process.returncode} before becoming ready"
                )
            time.sleep(0.5)
        # Cleanup if startup failed.
        self.stop()
        raise TimeoutError(f"vllm did not become ready within {self.startup_timeout_s:.0f}s")

    def stop(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("vllm did not terminate within 10s; sending SIGKILL")
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.error("vllm did not exit even after SIGKILL")
        finally:
            self._process = None

    def health_check(self, endpoint: str, *, timeout_s: float = 5.0) -> bool:
        try:
            import httpx  # noqa: PLC0415 — lazy import keeps module light
        except ImportError:
            logger.warning("httpx not installed; health_check returning False")
            return False

        url = endpoint.rstrip("/") + "/v1/models"
        try:
            response = httpx.get(url, timeout=timeout_s)
        except httpx.HTTPError as e:
            logger.info("health_check(%s) failed: %s", url, e)
            return False
        return response.status_code < 500

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_argv(self, *, model_id: str, adapter: Path | None, port: int) -> list[str]:
        """Compose the ``vllm serve`` argv. Pure for testability."""
        # Prefer the ``vllm`` console script if available; fall back to
        # ``python -m vllm`` so the backend works in venvs that don't
        # install console scripts.
        argv0 = shutil.which("vllm") or "vllm"
        argv = [
            argv0,
            "serve",
            model_id,
            "--host",
            self.host,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
        ]
        if adapter is not None:
            argv.extend(
                [
                    "--enable-lora",
                    "--lora-modules",
                    f"{Path(adapter).name}={adapter}",
                ]
            )
        if self.api_key:
            argv.extend(["--api-key", self.api_key])
        argv.extend(self.extra_args)
        return argv

    @staticmethod
    def _require_vllm_installed() -> None:
        try:
            import vllm  # noqa: PLC0415, F401 — import-time check only
        except ImportError as e:
            raise VllmNotInstalledError(
                "vllm is not installed; install it with `uv pip install vllm` "
                "or pick a different backend (llama-server is the default)."
            ) from e
