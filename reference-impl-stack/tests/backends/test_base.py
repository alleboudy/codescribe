"""Confirm the ``Backend`` ABC has the documented interface.

Pure-introspection tests — no subprocesses, no network. Mirrors
``tests/harness/test_base.py``: catches accidental signature drift early
since every concrete backend inherits from this ABC and the wider system
depends on the signature being stable.
"""

from __future__ import annotations

import inspect
from abc import ABC

from codescribe_train.backends.base import Backend


def test_backend_is_abc() -> None:
    assert inspect.isabstract(Backend)
    assert issubclass(Backend, ABC)


def test_required_methods_exist() -> None:
    for name in ("start", "stop", "health_check"):
        assert callable(getattr(Backend, name))


def test_start_signature() -> None:
    sig = inspect.signature(Backend.start)
    params = sig.parameters
    assert list(params)[0] == "self"
    # All three operational args are keyword-only by design.
    for kw in ("model_id", "adapter", "port"):
        assert params[kw].kind is inspect.Parameter.KEYWORD_ONLY


def test_stop_signature() -> None:
    sig = inspect.signature(Backend.stop)
    assert list(sig.parameters) == ["self"]


def test_health_check_signature() -> None:
    sig = inspect.signature(Backend.health_check)
    params = sig.parameters
    assert list(params)[:2] == ["self", "endpoint"]
    assert params["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY


def test_instantiating_abc_directly_fails() -> None:
    """Sanity check: the ABC genuinely refuses direct instantiation."""
    try:
        Backend()  # type: ignore[abstract]
    except TypeError:
        return
    raise AssertionError("Backend() should have raised TypeError")
