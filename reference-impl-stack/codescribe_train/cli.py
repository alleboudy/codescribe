"""Top-level CLI: ``codescribe-train <subcommand> ...``.

Subcommands:

* ``run``   — bring up a backend + harness session. The payoff command.
* ``probe`` — print the hardware profile + recommended backend.

The ``run`` flow:

1. Probe hardware, log a one-line summary.
2. Construct the chosen ``Backend`` from the loaded backend config.
3. Call ``backend.start(...)``, then poll ``health_check`` every 0.5 s
   until it returns ``True`` (timeout 60 s).
4. Construct the chosen ``Harness`` from the loaded harness config.
5. Call ``harness.prepare(workdir, backend_url, model, ...)``.
6. Call ``harness.start_session(stdin, stdout)`` and run until the
   harness exits (clean or signal).
7. ``backend.stop()`` in a ``finally`` block — idempotent, always runs.

Lazy imports keep ``import codescribe_train.cli`` cheap. ``codescribe-train --help``
must not pull ``httpx``, ``psutil``, ``pynvml``, the harness, or any
concrete backend.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Repo-relative root for resolving config files. ``codescribe_train/cli.py`` →
# project root is one level up from the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BACKEND_CONFIG = _REPO_ROOT / "configs" / "backends" / "local-default.yaml"
DEFAULT_HARNESS_CONFIG = _REPO_ROOT / "configs" / "harness" / "sample.yaml"

KNOWN_BACKENDS: tuple[str, ...] = ("llama-server", "vllm", "ollama", "remote")
KNOWN_HARNESSES: tuple[str, ...] = ("claw", "noop")
KNOWN_SANDBOXES: tuple[str, ...] = ("none", "docker")


# ---------------------------------------------------------------------- #
# Config loading + construction. All lazy on first use.
# ---------------------------------------------------------------------- #


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # noqa: PLC0415 — lazy import, see module docstring

    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config at {path} did not parse as a mapping")
    return loaded


def _build_backend(name: str, config: dict[str, Any]):
    """Construct a Backend from a name + the loaded config dict. Lazy imports."""
    if name == "llama-server":
        from codescribe_train.backends.llama_server import LlamaServerBackend  # noqa: PLC0415

        return LlamaServerBackend(
            host=str(config.get("host", "127.0.0.1")),
            ctx_size=int(config.get("ctx_size", 8192)),
            gpu_layers=int(config.get("gpu_layers", -1)),
            no_mmap=bool(config.get("no_mmap", True)),
            api_key=(config.get("api_key") or None),
            startup_timeout_s=float(config.get("startup_timeout_s", 60.0)),
        )
    if name == "vllm":
        from codescribe_train.backends.vllm import VllmBackend  # noqa: PLC0415

        return VllmBackend(
            host=str(config.get("host", "127.0.0.1")),
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.85)),
            max_model_len=int(config.get("max_model_len", 8192)),
            api_key=(config.get("api_key") or None),
            startup_timeout_s=float(config.get("startup_timeout_s", 120.0)),
        )
    if name == "ollama":
        from codescribe_train.backends.ollama import OllamaBackend  # noqa: PLC0415

        return OllamaBackend(
            url=str(config.get("url", "http://127.0.0.1:11434")),
            require_health_on_start=bool(config.get("require_health_on_start", True)),
            health_timeout_s=float(config.get("health_timeout_s", 2.0)),
        )
    if name == "remote":
        from codescribe_train.backends.remote import RemoteBackend  # noqa: PLC0415

        return RemoteBackend(
            url=str(config.get("url", "")),
            require_health_on_start=bool(config.get("require_health_on_start", True)),
            health_timeout_s=float(config.get("health_timeout_s", 2.0)),
        )
    raise ValueError(f"unknown backend: {name!r} (known: {KNOWN_BACKENDS})")


def _build_harness(name: str, *, sandbox: str):
    if name == "claw":
        from codescribe_train.harness.claw import ClawCodeHarness  # noqa: PLC0415

        return ClawCodeHarness(sandbox=sandbox)
    if name == "noop":
        from codescribe_train.harness.noop import NoOpHarness  # noqa: PLC0415

        return NoOpHarness()
    raise ValueError(f"unknown harness: {name!r} (known: {KNOWN_HARNESSES})")


# ---------------------------------------------------------------------- #
# Subcommand handlers
# ---------------------------------------------------------------------- #


def _start_backend_and_wait(backend, *, model_id: str, port: int, health_timeout_s: float) -> str:
    """Start the backend, poll its health, raise on timeout. Lazy import."""
    from codescribe_train.backends.llama_server import poll_health_until  # noqa: PLC0415

    endpoint = backend.start(model_id=model_id, adapter=None, port=port)
    if not poll_health_until(backend, endpoint, interval_s=0.5, timeout_s=health_timeout_s):
        raise RuntimeError(
            f"backend at {endpoint} did not become healthy within {health_timeout_s:.0f}s"
        )
    return endpoint


def _cmd_run(args: argparse.Namespace) -> int:
    workdir = Path(args.repo).resolve()
    if not workdir.is_dir():
        raise FileNotFoundError(f"--repo does not exist: {workdir}")

    # 1. Probe and log.
    from codescribe_train.backends.probe import (  # noqa: PLC0415 — lazy import
        probe_hardware,
        recommend_backend,
    )

    profile = probe_hardware()
    logger.info("hardware: %s", profile.summary())

    # 2. Load configs.
    backend_config_path = (
        Path(args.backend_config) if args.backend_config else DEFAULT_BACKEND_CONFIG
    )
    harness_config_path = (
        Path(args.harness_config) if args.harness_config else DEFAULT_HARNESS_CONFIG
    )
    backend_config = _load_yaml(backend_config_path) if backend_config_path.exists() else {}
    harness_config = _load_yaml(harness_config_path) if harness_config_path.exists() else {}

    backend_name = args.backend or str(backend_config.get("backend") or "llama-server")
    if backend_name not in KNOWN_BACKENDS:
        raise ValueError(f"unknown backend: {backend_name!r} (known: {KNOWN_BACKENDS})")

    # Recommend defaults so the operator sees what we'd pick if they
    # hadn't pinned anything; we don't override their config.
    choice = recommend_backend(profile, model_size_gib=float(args.model_size_gib))
    logger.info(
        "probe recommendation: backend=%s rationale=%s",
        choice.backend_name,
        choice.rationale,
    )

    # 3. Resolve the model artefact path. ``--adapter`` is the user-facing
    # flag (kept for symmetry with phase 3 docs); the value is the path
    # to the merged Q4_K_M GGUF, since LoRA-merging is done at train time.
    model_id_raw = args.adapter or backend_config.get("gguf_path") or ""
    if not model_id_raw:
        raise ValueError(
            "no GGUF path: pass --adapter <path-to.gguf> or set gguf_path in the backend config"
        )
    model_id = str(Path(model_id_raw))
    model_name = str(args.model or backend_config.get("model") or "openai/sample-qwen7b-local")
    port = int(args.port or backend_config.get("port") or 8080)

    # 4. Construct backend + harness.
    backend = _build_backend(backend_name, backend_config)
    harness = _build_harness(args.harness, sandbox=args.sandbox)

    # 5. Start backend, run harness, always stop backend on exit.
    rc = 0
    try:
        endpoint = _start_backend_and_wait(
            backend,
            model_id=model_id,
            port=port,
            health_timeout_s=float(args.health_timeout_s),
        )
        logger.info("backend up at %s", endpoint)

        harness.prepare(workdir, endpoint, model_name, harness_config=harness_config)
        rc = harness.start_session(sys.stdin, sys.stdout)
    finally:
        try:
            backend.stop()
        except Exception as e:  # noqa: BLE001 — stop is best-effort cleanup
            logger.warning("backend.stop() raised: %s", e)
    return int(rc)


def _cmd_probe(args: argparse.Namespace) -> int:
    from codescribe_train.backends.probe import (  # noqa: PLC0415 — lazy import
        probe_hardware,
        recommend_backend,
    )

    profile = probe_hardware()
    choice = recommend_backend(profile, model_size_gib=float(args.model_size_gib))
    print(f"hardware: {profile.summary()}")
    print(f"recommend: {choice.backend_name}")
    print(f"options:   {choice.options}")
    print(f"rationale: {choice.rationale}")
    return 0


# ---------------------------------------------------------------------- #
# argparse wiring
# ---------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codescribe-train",
        description=(
            "Bring up a strictly-local OpenAI-compat backend and launch the "
            "agent harness against it. Default stack: vendored llama-server "
            "+ vendored claw-code, both pinned-SHA, both audited. "
            "See docs/STRICTLY-LOCAL-POSTURE.md."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="launch backend + harness session")
    p_run.add_argument("--repo", required=True, help="target git repo / workdir")
    p_run.add_argument(
        "--adapter",
        default=None,
        help="path to the merged Q4_K_M GGUF (overrides backend config gguf_path)",
    )
    p_run.add_argument(
        "--backend",
        default=None,
        choices=KNOWN_BACKENDS,
        help="backend short name; defaults to backend config's `backend:` field",
    )
    p_run.add_argument(
        "--harness", default="claw", choices=KNOWN_HARNESSES, help="harness to launch"
    )
    p_run.add_argument(
        "--sandbox", default="none", choices=KNOWN_SANDBOXES, help="harness sandbox mode"
    )
    p_run.add_argument("--port", default=None, help="loopback port (overrides config)")
    p_run.add_argument(
        "--model",
        default=None,
        help="model id sent to the backend (overrides config); use openai/<id> prefix",
    )
    p_run.add_argument(
        "--health-timeout-s",
        default=60.0,
        help="seconds to wait for the backend to answer /v1/models",
    )
    p_run.add_argument(
        "--model-size-gib",
        default=4.6,
        help="rough GGUF size for the probe recommendation (Qwen7B Q4_K_M ≈ 4.6)",
    )
    p_run.add_argument(
        "--harness-config",
        default=None,
        help="path to harness YAML (default: configs/harness/sample.yaml)",
    )
    p_run.add_argument(
        "--backend-config",
        default=None,
        help="path to backend YAML (default: configs/backends/local-default.yaml)",
    )

    p_probe = sub.add_parser("probe", help="print the hardware profile + recommendation")
    p_probe.add_argument(
        "--model-size-gib",
        default=4.6,
        help="rough GGUF size of the model you plan to serve",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "probe": _cmd_probe,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
