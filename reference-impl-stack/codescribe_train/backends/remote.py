"""``RemoteBackend`` — interface stub for a future LAN-only inference topology.

Per the strictly-local privacy posture of codescribe-train, this backend is
**deliberately disabled by default**. It exists only so the
:class:`~codescribe_train.backends.base.Backend` ABC has parity coverage for a
future "another machine on your LAN" topology (e.g. a second trusted
machine on a private LAN serving a larger model).

If you want to enable it, set the environment variable
``CODESCRIBE_ENABLE_REMOTE_BACKEND=1`` at process start. There is no
shipped config that wires this backend in by default — the user has to
opt in twice (env var + their own config file).

The remote URL is **never** allowed to be a public-internet host: a
private-network heuristic check rejects anything that isn't loopback or
RFC 1918 / RFC 4193 / link-local. This is a defence-in-depth measure on
top of the fact that the wider codescribe-train system never sets a remote URL
in any default config — see ``docs/STRICTLY-LOCAL-POSTURE.md``.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from codescribe_train.backends.base import Backend

logger = logging.getLogger(__name__)


# Flip this env var to ``1`` (or any truthy value) to enable the backend.
# The default is **disabled**: any ``start()`` call raises
# ``RemoteBackendDisabled`` regardless of how the dataclass was constructed.
ENABLE_ENV_VAR = "CODESCRIBE_ENABLE_REMOTE_BACKEND"


class RemoteBackendDisabled(RuntimeError):
    """Raised by :class:`RemoteBackend` when it's used without explicit opt-in."""


class RemoteBackendNotPrivate(RuntimeError):
    """Raised when the configured URL points outside loopback / private ranges."""


@dataclass(slots=True)
class RemoteBackend(Backend):
    """Interface stub. Disabled by default. **Strictly LAN-only when enabled.**

    Even when enabled, the URL is sanity-checked: only loopback and
    RFC 1918 / RFC 4193 / link-local addresses are accepted. This is a
    backstop against a misconfigured DNS pointing the URL at a public
    host; the primary guarantee is that no shipped config wires this
    backend in at all.
    """

    url: str = ""
    require_health_on_start: bool = True
    health_timeout_s: float = 2.0
    # Allow tests / future LAN-aware deployments to override the
    # strict-private check. Not exposed in any shipped config.
    allow_non_private: bool = False

    _started: bool = field(default=False, init=False)

    # ------------------------------------------------------------------ #
    # Backend interface
    # ------------------------------------------------------------------ #

    def start(self, *, model_id: str, adapter: Path | None, port: int) -> str:
        del adapter, port  # remote backend doesn't manage either
        if not _is_truthy(os.environ.get(ENABLE_ENV_VAR)):
            raise RemoteBackendDisabled(
                "RemoteBackend is disabled per the strictly-local posture. "
                f"Set {ENABLE_ENV_VAR}=1 to enable, and read "
                "docs/STRICTLY-LOCAL-POSTURE.md before doing so."
            )
        if not self.url:
            raise ValueError("RemoteBackend.start: no `url` configured")

        if not self.allow_non_private and not _looks_private(self.url):
            raise RemoteBackendNotPrivate(
                f"RemoteBackend refuses to use {self.url!r}: it does not look "
                "like a loopback / RFC 1918 / RFC 4193 / link-local address. "
                "Strictly-local deployments only — set allow_non_private=True "
                "in code if you genuinely understand the implications."
            )

        if self.require_health_on_start and not self.health_check(
            self.url, timeout_s=self.health_timeout_s
        ):
            raise RuntimeError(f"RemoteBackend: {self.url} did not respond to /v1/models")
        logger.info("RemoteBackend opted-in and ready at %s (model=%s)", self.url, model_id)
        self._started = True
        return self.url

    def stop(self) -> None:
        self._started = False

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


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _looks_private(url: str) -> bool:
    """Return ``True`` if the host in ``url`` is loopback / RFC 1918 / RFC 4193 / link-local.

    Refuses to do a DNS lookup against the public internet — only resolves
    via the system resolver, which on a strictly-local box should be
    answering off ``/etc/hosts`` or a local stub. We treat resolution
    failure as "not private", erring on the side of refusing to start.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Hostname, not literal IP. Resolve once.
        try:
            resolved = socket.gethostbyname(host)
            ip = ipaddress.ip_address(resolved)
        except (OSError, ValueError):
            return False
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_unspecified  # 0.0.0.0 / :: — bound to all local ifaces
    )
