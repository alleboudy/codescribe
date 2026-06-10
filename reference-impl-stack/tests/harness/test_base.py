"""Confirm the ``Harness`` ABC has the documented interface.

Pure-introspection tests — no subprocesses, no network. The point is to
catch accidental signature drift early, since both concrete harnesses
inherit from this ABC and the wider system depends on these signatures
being stable.
"""

from __future__ import annotations

import inspect
from abc import ABC

from codescribe_train.harness.base import Harness


def test_harness_is_abc() -> None:
    assert inspect.isabstract(Harness)
    assert issubclass(Harness, ABC)


def test_required_methods_exist() -> None:
    for name in ("prepare", "start_session", "health_check", "sandbox_args"):
        assert callable(getattr(Harness, name))


def test_prepare_signature() -> None:
    sig = inspect.signature(Harness.prepare)
    params = sig.parameters
    assert list(params)[:4] == ["self", "workdir", "backend_url", "model"]
    assert params["harness_config"].kind is inspect.Parameter.KEYWORD_ONLY


def test_start_session_signature() -> None:
    sig = inspect.signature(Harness.start_session)
    assert list(sig.parameters) == ["self", "stdin", "stdout"]


def test_health_check_signature() -> None:
    sig = inspect.signature(Harness.health_check)
    assert list(sig.parameters) == ["self", "backend_url"]


def test_sandbox_args_signature() -> None:
    sig = inspect.signature(Harness.sandbox_args)
    assert list(sig.parameters) == ["self"]


def test_instantiating_abc_directly_fails() -> None:
    """Sanity check: the ABC genuinely refuses direct instantiation."""
    try:
        Harness()  # type: ignore[abstract]
    except TypeError:
        return
    raise AssertionError("Harness() should have raised TypeError")
