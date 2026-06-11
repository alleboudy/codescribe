"""Static gate: no ``subprocess.*(..., shell=True)`` in the sources package.

Implemented as both an AST walk (catches the canonical form) and a regex
sweep (catches awkward formatting and string literals).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[3] / "codescribe_train" / "rag" / "sources"


def _source_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _is_shell_true_keyword(kw: ast.keyword) -> bool:
    if kw.arg != "shell":
        return False
    value = kw.value
    if isinstance(value, ast.Constant) and value.value is True:
        return True
    # `shell=1` / `shell="yes"` would also be truthy at runtime; flag any
    # non-literal-False to be safe. Anything dynamic (name reference,
    # function call) is suspicious — flag.
    return not (isinstance(value, ast.Constant) and value.value is False)


def test_no_subprocess_shell_true_ast() -> None:
    offenders: list[tuple[Path, int, str]] = []
    for path in _source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # subprocess.run / subprocess.Popen / subprocess.call etc.
            is_subprocess_callsite = (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
            )
            if not is_subprocess_callsite:
                continue
            for kw in node.keywords:
                if _is_shell_true_keyword(kw):
                    offenders.append((path, node.lineno, ast.unparse(node)))

    assert not offenders, (
        "subprocess.*(..., shell=True) found in sources package:\n"
        + "\n".join(f"  {p}:{lineno}: {snippet}" for p, lineno, snippet in offenders)
    )


def test_no_shell_true_substring_regex() -> None:
    """Belt-and-braces: catch `shell=True` even outside an AST-parseable call."""
    pattern = re.compile(r"shell\s*=\s*True\b")
    offenders: list[tuple[Path, int, str]] = []
    for path in _source_files():
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.search(line):
                offenders.append((path, i, line.strip()))
    assert not offenders, (
        "literal `shell=True` found in sources package:\n"
        + "\n".join(f"  {p}:{i}: {snippet}" for p, i, snippet in offenders)
    )
