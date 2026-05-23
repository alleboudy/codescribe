from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).parents[3] / "codescribe_rag" / "rag" / "sources"


def test_no_shell_true():
    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant):
                        assert kw.value.value is not True, f"shell=True in {py}"
