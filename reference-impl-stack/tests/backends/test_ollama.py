"""Tests for ``OllamaBackend`` — daemon lifecycle is OUT-of-band.

The user manages ``ollama serve`` themselves; our backend only validates
that ``GET /api/tags`` answers and passes the configured URL through.
"""

from __future__ import annotations

from unittest import mock

import pytest

from codescribe_train.backends.base import Backend
from codescribe_train.backends.ollama import DEFAULT_OLLAMA_URL, OllamaBackend


def test_is_a_backend() -> None:
    assert isinstance(OllamaBackend(), Backend)


def test_default_url_is_loopback() -> None:
    assert OllamaBackend().url.startswith("http://127.0.0.1")
    assert DEFAULT_OLLAMA_URL.startswith("http://127.0.0.1")


def test_health_check_probes_api_tags() -> None:
    backend = OllamaBackend()
    fake_response = mock.Mock(status_code=200)
    with mock.patch("httpx.get", return_value=fake_response) as get_mock:
        assert backend.health_check("http://127.0.0.1:11434", timeout_s=1.0)
    # The probe must hit /api/tags, not /v1/models.
    called_url = get_mock.call_args.args[0]
    assert called_url.endswith("/api/tags")


def test_health_check_returns_false_on_5xx() -> None:
    backend = OllamaBackend()
    fake_response = mock.Mock(status_code=503)
    with mock.patch("httpx.get", return_value=fake_response):
        assert not backend.health_check("http://127.0.0.1:11434", timeout_s=1.0)


def test_health_check_returns_false_on_http_error() -> None:
    import httpx

    backend = OllamaBackend()
    with mock.patch("httpx.get", side_effect=httpx.ConnectError("nope")):
        assert not backend.health_check("http://127.0.0.1:11434", timeout_s=1.0)


def test_start_returns_url_when_healthy() -> None:
    backend = OllamaBackend()
    with mock.patch.object(backend, "health_check", return_value=True):
        endpoint = backend.start(model_id="qwen2.5-coder:7b", adapter=None, port=11434)
    assert endpoint == backend.url


def test_start_raises_when_daemon_down() -> None:
    backend = OllamaBackend()
    with (
        mock.patch.object(backend, "health_check", return_value=False),
        pytest.raises(RuntimeError, match="not reachable"),
    ):
        backend.start(model_id="x", adapter=None, port=11434)


def test_start_skips_health_when_disabled() -> None:
    backend = OllamaBackend(require_health_on_start=False)
    # No health_check patch — must not be called.
    with mock.patch.object(backend, "health_check") as hc:
        endpoint = backend.start(model_id="x", adapter=None, port=11434)
    hc.assert_not_called()
    assert endpoint == backend.url


def test_stop_is_no_op() -> None:
    backend = OllamaBackend()
    backend.stop()
    backend.stop()
    assert backend._started is False
