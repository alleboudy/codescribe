"""``ClawCodeHarness`` — adapter for the vendored ``ultraworkers/claw-code``.

Wraps the Rust binary at ``vendor/claw-code/rust/target/release/claw`` as
an opaque, low-trust subprocess. Renders ``.claude.json`` / ``.claw.json``
into the target workdir from a YAML harness config, forwards a curated set
of env vars (no secrets are injected from defaults), and launches the
binary with an optional sandbox prefix.

Trust posture:

* The submodule SHA is pinned (see ``.gitmodules`` and
  ``vendor/SECURITY-NOTES.md``); the wrapper never auto-pulls.
* No network egress in the runtime path. The only network op is a
  one-time, optional ``health_check`` against the local backend.
* ``ANTHROPIC_API_KEY`` is **never** set from any built-in default. If the
  user has it set in their shell already, it propagates as part of
  ``os.environ`` — by design — but this wrapper does not invent one.
* Default model name should use an ``openai/`` or ``gpt-`` prefix so
  claw-code routes to the OpenAI-compat client and honours
  ``OPENAI_BASE_URL``. ``qwen-*`` and ``qwen/*`` model names route to
  DashScope by default in claw-code; see SECURITY-NOTES.md for details.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from codescribe_train.harness.base import Harness

logger = logging.getLogger(__name__)

# Repo-relative location of the vendored Rust binary. The repo root is
# resolved from this file's location: codescribe_train/harness/claw.py → ../../.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLAW_BINARY = _REPO_ROOT / "vendor" / "claw-code" / "rust" / "target" / "release" / "claw"

# Env vars we forward into the subprocess. These are the *only* env vars
# this wrapper ever sets in the child env from runtime args. Anything else
# (ANTHROPIC_API_KEY, custom proxies, etc.) is inherited from the parent
# process unchanged — never invented from a default.
_FORWARDED_ENV_KEYS_FROM_ARGS: tuple[str, ...] = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "ANTHROPIC_BASE_URL",
)


@dataclass(slots=True)
class ClawCodeHarness(Harness):
    """Adapter for the vendored Rust ``claw`` binary.

    Construct with optional overrides; defaults point at the vendored
    binary at the pinned SHA. ``prepare`` writes ``.claude.json`` /
    ``.claw.json`` into the workdir; ``start_session`` spawns the binary.
    """

    binary_path: Path = field(default_factory=lambda: DEFAULT_CLAW_BINARY)
    sandbox: str = "none"
    docker_image: str = "claw-code:vendored"
    workdir: Path | None = None
    backend_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    harness_config: dict[str, Any] = field(default_factory=dict)
    prepared: bool = False

    # ------------------------------------------------------------------ #
    # Harness interface
    # ------------------------------------------------------------------ #

    def prepare(
        self,
        workdir: Path,
        backend_url: str,
        model: str,
        *,
        harness_config: dict[str, Any],
    ) -> None:
        workdir = Path(workdir).resolve()
        if not workdir.is_dir():
            raise FileNotFoundError(f"workdir does not exist: {workdir}")

        self.workdir = workdir
        self.backend_url = backend_url
        self.model = model
        self.harness_config = dict(harness_config or {})
        # API key, if any, is plumbed via env — never written to a config file.
        self.api_key = self.harness_config.get("api_key")

        claude_json = render_claude_json(self.harness_config, backend_url=backend_url, model=model)
        claw_json = render_claw_json(self.harness_config)

        (workdir / ".claude.json").write_text(
            json.dumps(claude_json, indent=2) + "\n", encoding="utf-8"
        )
        (workdir / ".claw.json").write_text(
            json.dumps(claw_json, indent=2) + "\n", encoding="utf-8"
        )

        prompt_overlay = self._maybe_overlay_path()
        if prompt_overlay is not None and not prompt_overlay.exists():
            logger.warning(
                "harness prompt overlay configured at %s but file is missing", prompt_overlay
            )
        self.prepared = True
        logger.info(
            "ClawCodeHarness prepared (workdir=%s, model=%s, sandbox=%s)",
            workdir,
            model,
            self.sandbox,
        )

    def start_session(self, stdin: IO[str] | IO[bytes], stdout: IO[str] | IO[bytes]) -> int:
        if not self.prepared or self.workdir is None:
            raise RuntimeError("ClawCodeHarness.start_session called before prepare()")
        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"claw binary not found at {self.binary_path}; run scripts/build_claw_code.sh first"
            )

        cmd = [*self.sandbox_args(), *self._launch_args()]
        env = self._build_env()
        logger.info("launching claw: %s", " ".join(cmd))
        try:
            completed = subprocess.run(  # noqa: S603 — args are constructed locally, not from user shell
                cmd,
                cwd=str(self.workdir),
                env=env,
                stdin=stdin,
                stdout=stdout,
                stderr=sys.stderr,
                check=False,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"failed to launch claw harness ({cmd[0]} not on PATH): {e}"
            ) from e
        return int(completed.returncode)

    def health_check(self, backend_url: str) -> bool:
        """Probe the OpenAI-compat ``/v1/models`` endpoint of ``backend_url``."""
        try:
            import httpx  # noqa: PLC0415 — lazy import keeps base CLI import light
        except ImportError:
            logger.warning("httpx not installed; health_check returning False")
            return False

        url = backend_url.rstrip("/") + "/v1/models"
        try:
            response = httpx.get(url, timeout=2.0)
        except httpx.HTTPError as e:
            logger.info("health_check(%s) failed: %s", url, e)
            return False
        return response.status_code < 500

    def sandbox_args(self) -> list[str]:
        if self.sandbox == "none":
            return []
        if self.sandbox == "docker":
            if self.workdir is None:
                raise RuntimeError("sandbox_args() called before prepare(); workdir unknown")
            return [
                "docker",
                "run",
                "--rm",
                "-i",
                "--network",
                "none",
                "-v",
                f"{self.workdir}:/work",
                "-w",
                "/work",
                self.docker_image,
            ]
        raise ValueError(f"unknown sandbox mode: {self.sandbox!r}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _launch_args(self) -> list[str]:
        # Pass --model as a CLI flag, not just via OPENAI_MODEL env var.
        # claw's provider routing keys off the prefix of the CLI flag's
        # value (e.g. `openai/...` selects the OpenAI-compatible client).
        # Setting OPENAI_MODEL alone yields a "missing Anthropic credentials"
        # error because the runtime falls back to the Anthropic default.
        model_args: list[str] = []
        if self.model is not None:
            model_args = ["--model", self.model]
        if self.sandbox == "none":
            return [str(self.binary_path), *model_args]
        # Inside a docker sandbox the entrypoint is the binary inside the
        # image; we pass no host-side path.
        return [*model_args]

    def _build_env(self) -> dict[str, str]:
        """Compose the child env: parent env + curated runtime overrides."""
        env = dict(os.environ)
        if self.backend_url is not None:
            env["OPENAI_BASE_URL"] = self.backend_url
            # Some claw paths read ANTHROPIC_BASE_URL when the model resolves
            # to the Anthropic provider. We pin both to the local backend
            # so a model-name typo can't trigger an outbound request.
            env["ANTHROPIC_BASE_URL"] = self.backend_url
        if self.model is not None:
            env["OPENAI_MODEL"] = self.model
            env["ANTHROPIC_MODEL"] = self.model
        if self.api_key is not None:
            env["OPENAI_API_KEY"] = self.api_key
        else:
            # claw-code's OpenAI-compat client requires *some* value here
            # even for local inference — most local servers ignore it. A
            # real key must come from the user, never from a built-in default.
            env.setdefault("OPENAI_API_KEY", "local-no-auth")
        for key, value in self.extra_env.items():
            if value is None:
                continue
            env[key] = str(value)
        return env

    def _maybe_overlay_path(self) -> Path | None:
        path = self.harness_config.get("prompt_overlay")
        if not path:
            return None
        return Path(path).expanduser().resolve()


# ---------------------------------------------------------------------- #
# Config rendering — pure functions for testability
# ---------------------------------------------------------------------- #


def render_claude_json(config: dict[str, Any], *, backend_url: str, model: str) -> dict[str, Any]:
    """Render a ``.claude.json`` payload from a harness YAML config dict.

    Pure: no I/O, no env reads. The resulting dict is the exact JSON we
    write next to the workdir.
    """
    permissions = dict(config.get("permissions") or {})
    permissions.setdefault("defaultMode", config.get("default_permission_mode", "ask"))
    if "allowedTools" not in permissions and config.get("allowed_tools") is not None:
        permissions["allowedTools"] = list(config["allowed_tools"])
    if "deniedTools" not in permissions and config.get("denied_tools") is not None:
        permissions["deniedTools"] = list(config["denied_tools"])

    payload: dict[str, Any] = {
        "model": model,
        "permissions": permissions,
    }

    overlay = config.get("prompt_overlay")
    if overlay:
        payload["promptOverlay"] = str(overlay)

    mcp_servers = config.get("mcp_servers") or {}
    if mcp_servers:
        payload["mcpServers"] = dict(mcp_servers)

    slash_commands = config.get("slash_commands")
    if slash_commands:
        payload["slashCommands"] = list(slash_commands)

    extra = config.get("claude_json_extra") or {}
    for key, value in extra.items():
        payload.setdefault(key, value)

    # `backend_url` is forwarded via env, not the JSON; it lives here only
    # for diagnostics and doesn't influence the binary's HTTP routing.
    payload["_metadata"] = {"backendUrl": backend_url}
    return payload


def render_claw_json(config: dict[str, Any]) -> dict[str, Any]:
    """Render a ``.claw.json`` payload from a harness YAML config dict."""
    payload: dict[str, Any] = {}
    aliases = config.get("aliases") or {}
    if aliases:
        payload["aliases"] = dict(aliases)
    extras = config.get("claw_json_extra") or {}
    for key, value in extras.items():
        payload.setdefault(key, value)
    return payload


def docker_image_present(image: str = "claw-code:vendored") -> bool:
    """Check whether the local Docker has the sandbox image built.

    Best-effort — returns ``False`` if Docker isn't installed or the image
    isn't built. Callers can use this to warn before launching with
    ``--sandbox docker``.
    """
    docker = shutil.which("docker")
    if docker is None:
        return False
    try:
        result = subprocess.run(  # noqa: S603 — fixed args, no shell
            [docker, "image", "inspect", image],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0
