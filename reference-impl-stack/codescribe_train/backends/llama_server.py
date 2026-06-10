"""``LlamaServerBackend`` — adapter for the vendored ``ggml-org/llama.cpp``.

Wraps the ``llama-server`` binary at
``vendor/llama.cpp/build/bin/llama-server`` as an opaque, low-trust
subprocess. Spawns it bound to loopback only, captures stdout/stderr to a
log file under ``logs/``, watches for the "HTTP server listening" line,
and returns the endpoint URL.

Trust posture:

* The submodule SHA is pinned (see ``.gitmodules`` and
  ``vendor/SECURITY-NOTES.md``); the wrapper never auto-pulls.
* The binary binds to ``127.0.0.1`` only; no network egress beyond the
  user's loopback.
* No LoRA-adapter merging happens here — this backend consumes a pre-
  baked Q4_K_M ``.gguf`` produced at train-time (``codescribe_train.train.export_gguf``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from codescribe_train.backends.base import Backend

logger = logging.getLogger(__name__)

# Repo-relative location of the vendored binary. The repo root is resolved
# from this file's location: codescribe_train/backends/llama_server.py → ../../.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLAMA_SERVER_BINARY = _REPO_ROOT / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server"
DEFAULT_LOG_DIR = _REPO_ROOT / "logs"

# llama-server prints this line on stdout/stderr (varies between versions)
# once it's bound and ready to accept connections.
_READY_MARKERS: tuple[str, ...] = (
    "HTTP server listening",
    "server is listening",
    "starting the main loop",
    "all slots are idle",
)

# Wall-clock cap on how long we wait for the binary to start serving HTTP.
_DEFAULT_STARTUP_TIMEOUT_S: float = 60.0


@dataclass(slots=True)
class LlamaServerBackend(Backend):
    """Adapter for the vendored ``llama-server`` binary.

    Construct with optional overrides; defaults point at the vendored
    binary. ``start`` spawns the subprocess and blocks until it logs a
    ready marker; ``stop`` terminates it idempotently.
    """

    binary_path: Path = field(default_factory=lambda: DEFAULT_LLAMA_SERVER_BINARY)
    host: str = "127.0.0.1"
    ctx_size: int = 8192
    gpu_layers: int = -1  # -1 → "all on GPU", let llama.cpp pick
    no_mmap: bool = True  # default friendly to tight-VRAM (~8 GB) GPUs
    api_key: str | None = None
    log_dir: Path = field(default_factory=lambda: DEFAULT_LOG_DIR)
    startup_timeout_s: float = _DEFAULT_STARTUP_TIMEOUT_S
    extra_args: list[str] = field(default_factory=list)

    # Runtime state — not part of the construction contract.
    _process: subprocess.Popen | None = field(default=None, init=False)
    _log_file_handle: object | None = field(default=None, init=False)
    _endpoint: str | None = field(default=None, init=False)
    _model_path: Path | None = field(default=None, init=False)

    # ------------------------------------------------------------------ #
    # Backend interface
    # ------------------------------------------------------------------ #

    def start(self, *, model_id: str, adapter: Path | None, port: int) -> str:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("LlamaServerBackend.start called while process already running")

        if adapter is not None:
            raise ValueError(
                "LlamaServerBackend does not accept a separate `adapter`. "
                "Pass a fully-merged Q4_K_M GGUF as `model_id`; do LoRA "
                "merging in codescribe_train.train.export_gguf."
            )

        model_path = Path(model_id)
        if not model_path.is_absolute():
            model_path = (_REPO_ROOT / model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found at {model_path}; "
                "export it via codescribe_train.train.export_gguf"
            )
        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"llama-server binary not found at {self.binary_path}; "
                "run scripts/build_llama_cpp.sh first"
            )

        self._model_path = model_path
        cmd = self._build_argv(model_path=model_path, port=port)
        log_path = self._open_log_file()

        logger.info("launching llama-server: %s", " ".join(cmd))
        # noqa: S603 — args are constructed locally, not from user shell
        self._process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=self._log_file_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
        endpoint = f"http://{self.host}:{port}"
        try:
            self._wait_for_ready(log_path)
        except Exception:
            # Best-effort cleanup so a half-started server doesn't leak.
            self.stop()
            raise
        self._endpoint = endpoint
        logger.info("llama-server ready on %s (log: %s)", endpoint, log_path)
        return endpoint

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
                    logger.warning("llama-server did not terminate within 10s; sending SIGKILL")
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.error("llama-server did not exit even after SIGKILL")
        finally:
            handle = self._log_file_handle
            if handle is not None:
                with contextlib.suppress(OSError, ValueError):
                    handle.close()  # type: ignore[union-attr]
            self._log_file_handle = None
            self._process = None

    def health_check(self, endpoint: str, *, timeout_s: float = 5.0) -> bool:
        """Probe ``GET {endpoint}/v1/models`` for OpenAI-compat readiness."""
        try:
            import httpx  # noqa: PLC0415 — lazy import keeps base CLI import light
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

    def _build_argv(self, *, model_path: Path, port: int) -> list[str]:
        """Compose the ``llama-server`` argv. Pure for testability."""
        argv = [
            str(self.binary_path),
            "-m",
            str(model_path),
            "--host",
            self.host,
            "--port",
            str(port),
            "-ngl",
            str(self.gpu_layers),
            "--ctx-size",
            str(self.ctx_size),
        ]
        if self.no_mmap:
            argv.append("--no-mmap")
        if self.api_key:
            argv.extend(["--api-key", self.api_key])
        argv.extend(self.extra_args)
        return argv

    def _open_log_file(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "llama-server.log"
        # Append, line-buffered so the watcher sees marker lines promptly.
        self._log_file_handle = log_path.open("a", buffering=1, encoding="utf-8")
        return log_path

    def _wait_for_ready(self, log_path: Path) -> None:
        """Tail the log until a ready-marker appears or we time out / crash."""
        deadline = time.monotonic() + self.startup_timeout_s
        # Read from byte 0 the first time so we don't miss markers printed
        # before we attached.
        with log_path.open(encoding="utf-8", errors="replace") as f:
            while True:
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited with code {self._process.returncode} "
                        f"before becoming ready (see {log_path})"
                    )
                line = f.readline()
                if line:
                    if any(marker in line for marker in _READY_MARKERS):
                        return
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"llama-server did not become ready within "
                        f"{self.startup_timeout_s:.0f}s (see {log_path})"
                    )
                time.sleep(0.1)


def poll_health_until(
    backend: Backend,
    endpoint: str,
    *,
    interval_s: float = 0.5,
    timeout_s: float = 60.0,
) -> bool:
    """Poll ``backend.health_check(endpoint)`` until it returns ``True`` or we time out.

    Lives here rather than on the ABC so backends remain side-effect-free
    in pure tests; this is the orchestration helper used by the top-level
    ``codescribe-train run`` glue.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        if backend.health_check(endpoint, timeout_s=min(2.0, interval_s * 4)):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval_s)


def _maybe_log_dir_override() -> Path | None:
    override = os.environ.get("CODESCRIBE_LOG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return None
