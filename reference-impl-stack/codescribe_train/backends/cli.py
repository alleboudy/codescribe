"""CLI entry point: ``python -m codescribe_train.backends <subcommand> ...``.

Subcommands:

* ``probe``  — print the hardware profile + recommended backend.
* ``health`` — probe a backend URL's OpenAI-compat endpoint.

Lazy imports keep ``import codescribe_train.backends.cli`` cheap: the concrete
backend implementations, ``httpx``, ``psutil`` and ``pynvml`` are not
loaded until the relevant subcommand actually fires. This mirrors the
pattern used by :mod:`codescribe_train.harness.cli`.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)

KNOWN_BACKENDS: tuple[str, ...] = ("llama-server", "vllm", "ollama", "remote")


def _build_backend(name: str):
    """Construct an empty Backend instance from its short name. Lazy imports."""
    if name == "llama-server":
        from codescribe_train.backends.llama_server import LlamaServerBackend  # noqa: PLC0415

        return LlamaServerBackend()
    if name == "vllm":
        from codescribe_train.backends.vllm import VllmBackend  # noqa: PLC0415

        return VllmBackend()
    if name == "ollama":
        from codescribe_train.backends.ollama import OllamaBackend  # noqa: PLC0415

        return OllamaBackend()
    if name == "remote":
        from codescribe_train.backends.remote import RemoteBackend  # noqa: PLC0415

        return RemoteBackend()
    raise ValueError(f"unknown backend: {name!r} (known: {KNOWN_BACKENDS})")


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


def _cmd_health(args: argparse.Namespace) -> int:
    backend = _build_backend(args.backend)
    ok = backend.health_check(args.endpoint, timeout_s=float(args.timeout))
    print("ok" if ok else "fail")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="codescribe_train.backends",
        description=(
            "Backend layer utilities — hardware probe + endpoint health check. Strictly local."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="print hardware profile + recommended backend")
    p_probe.add_argument(
        "--model-size-gib",
        default=4.6,  # default Qwen 2.5 Coder 7B Q4_K_M
        help="rough on-disk size of the GGUF the user plans to serve",
    )

    p_health = sub.add_parser("health", help="probe a backend endpoint")
    p_health.add_argument("--endpoint", required=True, help="backend URL")
    p_health.add_argument(
        "--backend",
        default="llama-server",
        choices=KNOWN_BACKENDS,
        help="which backend's health-check shape to use",
    )
    p_health.add_argument("--timeout", default=5.0, help="HTTP timeout in seconds")

    args = parser.parse_args(argv)
    handlers = {
        "probe": _cmd_probe,
        "health": _cmd_health,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
