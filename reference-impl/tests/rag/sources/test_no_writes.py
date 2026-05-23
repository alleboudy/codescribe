from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).parents[3] / "codescribe_rag" / "rag" / "sources"
BANNED = re.compile(
    r"p4\s+(submit|edit|add|delete|reopen)\b"
    r"|\.(post|put|patch|delete)\s*\(",
    re.IGNORECASE,
)


def test_no_write_operations():
    for py in SRC.rglob("*.py"):
        hits = [m.group(0) for m in BANNED.finditer(py.read_text())]
        assert not hits, f"{py} contains write op(s): {hits}"
