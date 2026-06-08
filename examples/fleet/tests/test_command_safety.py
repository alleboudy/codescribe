"""Static safety checks — mirror the discipline of reference-impl's suite.

These guard the invariants that keep the orchestrator's command-building safe and
the package strictly-local:

* no ``shell=True`` / ``create_subprocess_shell`` anywhere;
* subprocess creation is confined to ``transport.py`` (the single audited boundary);
* command-building modules ``shlex.quote`` interpolated values;
* no ``print(...)`` in library code (CLI/__main__ and scripts/ may print);
* no ``0.0.0.0`` default (network binds to loopback).

The substring checks run against *code only* — comments and string/f-string
literals are blanked first, so a comment like ``(NOT shell=True)`` or a docstring
that merely mentions "subprocess" does not trip them.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "codescribe_fleet"

_BLANK_TYPES = {tokenize.STRING, tokenize.COMMENT}
for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):  # py3.12 PEP 701
    _t = getattr(tokenize, _name, None)
    if _t is not None:
        _BLANK_TYPES.add(_t)


def _code_only(src: str) -> str:
    """Return ``src`` with comments and string-literal *content* blanked to spaces.

    Newlines are preserved so line structure (and multi-line constructs) stay
    intact; f-string interpolation code (e.g. ``shlex.quote(x)``) is preserved
    because PEP 701 tokenizes it as ordinary tokens, not as the literal.
    """
    chars = list(src)
    line_starts = [0]
    for line in src.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def offset(row: int, col: int) -> int:
        return line_starts[row - 1] + col

    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in _BLANK_TYPES:
            for i in range(offset(*tok.start), offset(*tok.end)):
                if chars[i] != "\n":
                    chars[i] = " "
    return "".join(chars)


def _modules() -> list[Path]:
    return sorted(PKG.rglob("*.py"))


def _code(f: Path) -> str:
    return _code_only(f.read_text(encoding="utf-8"))


def test_no_shell_true() -> None:
    for f in _modules():
        assert "shell=True" not in _code(f), f


def test_no_subprocess_shell() -> None:
    for f in _modules():
        assert "create_subprocess_shell" not in _code(f), f


def test_subprocess_confined_to_transport() -> None:
    for f in _modules():
        code = _code(f)
        if "create_subprocess_exec" in code or "subprocess" in code:
            assert f.name == "transport.py", f"subprocess use leaked into {f.name}"


def test_command_builders_quote() -> None:
    # the modules that interpolate paths into command strings must use shlex.
    for name in ("worker.py", "ddp.py", "config.py"):
        assert "shlex.quote" in (PKG / name).read_text(encoding="utf-8"), name


def test_no_print_in_library() -> None:
    for f in _modules():
        if f.name == "__main__.py":
            continue  # entry point may print
        assert "print(" not in _code(f), f"library module {f.name} uses print()"


def test_no_zero_bind_default() -> None:
    for f in _modules():
        assert "0.0.0.0" not in _code(f), f
