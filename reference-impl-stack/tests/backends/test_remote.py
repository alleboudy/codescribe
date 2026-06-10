"""Tests for ``RemoteBackend`` — disabled by default per strictly-local posture.

Two layers of opt-in:

1. ``CODESCRIBE_ENABLE_REMOTE_BACKEND=1`` env var must be set.
2. The configured URL must be loopback / RFC 1918 / RFC 4193 / link-local.

Both must hold before ``start`` will even try to make a request. The
default config (``configs/backends/local-default.yaml``) does not point
at this backend; only ``configs/backends/lan-remote.yaml`` does, and
it carries a header comment about the env-var requirement.
"""

from __future__ import annotations

from unittest import mock

import pytest

from codescribe_train.backends.base import Backend
from codescribe_train.backends.remote import (
    ENABLE_ENV_VAR,
    RemoteBackend,
    RemoteBackendDisabled,
    RemoteBackendNotPrivate,
    _looks_private,
)


def test_is_a_backend() -> None:
    assert isinstance(RemoteBackend(), Backend)


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    backend = RemoteBackend(url="http://127.0.0.1:8080")
    with pytest.raises(RemoteBackendDisabled):
        backend.start(model_id="x", adapter=None, port=8080)


def test_enabled_with_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    backend = RemoteBackend(url="http://127.0.0.1:8080", require_health_on_start=False)
    endpoint = backend.start(model_id="x", adapter=None, port=8080)
    assert endpoint == "http://127.0.0.1:8080"


def test_truthy_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = RemoteBackend(url="http://127.0.0.1:8080", require_health_on_start=False)
    for value in ("1", "true", "yes", "on", "TRUE", " 1 "):
        monkeypatch.setenv(ENABLE_ENV_VAR, value)
        backend.start(model_id="x", adapter=None, port=8080)


def test_falsy_env_values_remain_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = RemoteBackend(url="http://127.0.0.1:8080")
    for value in ("0", "false", "no", "", "off"):
        monkeypatch.setenv(ENABLE_ENV_VAR, value)
        with pytest.raises(RemoteBackendDisabled):
            backend.start(model_id="x", adapter=None, port=8080)


def test_rejects_public_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    backend = RemoteBackend(url="http://example.com:8080")
    # Force the resolver to return a public IP regardless of the host's DNS.
    with (
        mock.patch(
            "codescribe_train.backends.remote.socket.gethostbyname",
            return_value="93.184.215.14",  # example.com
        ),
        pytest.raises(RemoteBackendNotPrivate),
    ):
        backend.start(model_id="x", adapter=None, port=8080)


def test_accepts_rfc1918(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    backend = RemoteBackend(url="http://192.168.1.50:8080", require_health_on_start=False)
    endpoint = backend.start(model_id="x", adapter=None, port=8080)
    assert endpoint == "http://192.168.1.50:8080"


def test_looks_private_helpers() -> None:
    assert _looks_private("http://127.0.0.1:8080")
    assert _looks_private("http://10.0.0.1:8080")
    assert _looks_private("http://192.168.1.1:8080")
    assert _looks_private("http://172.16.0.1:8080")
    assert _looks_private("http://[::1]:8080")
    assert _looks_private("http://[fc00::1]:8080")  # RFC 4193


def test_looks_private_rejects_public_ip() -> None:
    # 8.8.8.8 is public — must be rejected.
    assert not _looks_private("http://8.8.8.8:8080")


def test_allow_non_private_escape_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """allow_non_private exists for tests / future LAN setups; not in shipped configs."""
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    backend = RemoteBackend(
        url="http://example.com:8080",
        require_health_on_start=False,
        allow_non_private=True,
    )
    # No private-IP check — start() proceeds.
    endpoint = backend.start(model_id="x", adapter=None, port=8080)
    assert endpoint == "http://example.com:8080"


def test_stop_is_idempotent() -> None:
    backend = RemoteBackend(url="http://127.0.0.1:8080")
    backend.stop()
    backend.stop()
