"""CLI: ``python -m codescribe_train.harness <subcommand> ...``.

Subcommands:

* ``run``  — launch a harness session against a backend URL.
* ``health`` — probe the backend with the default harness's health check.

Lazy imports keep ``import codescribe_train.harness.cli`` cheap: yaml / httpx and
the concrete harness implementations are not loaded until the relevant
subcommand actually fires.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KNOWN_HARNESSES: tuple[str, ...] = ("claw", "noop")
KNOWN_SANDBOXES: tuple[str, ...] = ("none", "docker")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # noqa: PLC0415 — lazy import, see module docstring

    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"harness config at {path} did not parse as a mapping")
    return loaded


def _build_harness(name: str, *, sandbox: str):
    if name == "claw":
        from codescribe_train.harness.claw import ClawCodeHarness  # noqa: PLC0415

        return ClawCodeHarness(sandbox=sandbox)
    if name == "noop":
        from codescribe_train.harness.noop import NoOpHarness  # noqa: PLC0415

        return NoOpHarness()
    raise ValueError(f"unknown harness: {name!r} (known: {KNOWN_HARNESSES})")


def _cmd_run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    if not workdir.is_dir():
        raise FileNotFoundError(f"workdir does not exist: {workdir}")

    config: dict[str, Any] = {}
    if args.config:
        config = _load_yaml(Path(args.config))

    harness = _build_harness(args.harness, sandbox=args.sandbox)

    if not args.skip_health_check:
        ok = harness.health_check(args.backend)
        if not ok and not args.allow_unhealthy:
            logger.error(
                "backend %s failed health check; pass --allow-unhealthy to ignore", args.backend
            )
            return 3
        if not ok:
            logger.warning(
                "backend %s failed health check (continuing because --allow-unhealthy)",
                args.backend,
            )

    harness.prepare(workdir, args.backend, args.model, harness_config=config)
    return harness.start_session(sys.stdin, sys.stdout)


def _cmd_health(args: argparse.Namespace) -> int:
    harness = _build_harness(args.harness, sandbox="none")
    ok = harness.health_check(args.backend)
    print("ok" if ok else "fail")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="codescribe_train.harness",
        description=(
            "Launch a pluggable agent harness (default: vendored claw-code) "
            "against a local OpenAI-compatible backend. Strictly local."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="launch a harness session")
    p_run.add_argument("--backend", required=True, help="OpenAI-compat backend URL")
    p_run.add_argument("--model", required=True, help="model id to send to the backend")
    p_run.add_argument("--workdir", required=True, help="target git repo / working directory")
    p_run.add_argument(
        "--harness", default="claw", choices=KNOWN_HARNESSES, help="harness to launch"
    )
    p_run.add_argument(
        "--sandbox",
        default="none",
        choices=KNOWN_SANDBOXES,
        help="sandbox mode (claw harness only)",
    )
    p_run.add_argument("--config", default=None, help="harness YAML config path")
    p_run.add_argument(
        "--skip-health-check",
        action="store_true",
        help="don't probe the backend before launching",
    )
    p_run.add_argument(
        "--allow-unhealthy",
        action="store_true",
        help="continue even if the backend health check fails",
    )

    p_health = sub.add_parser("health", help="probe a backend's /v1/models endpoint")
    p_health.add_argument("--backend", required=True)
    p_health.add_argument("--harness", default="claw", choices=KNOWN_HARNESSES)

    args = parser.parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "health": _cmd_health,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
