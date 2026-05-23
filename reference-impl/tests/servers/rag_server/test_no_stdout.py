from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).parents[3] / "codescribe_rag" / "servers" / "rag_server"


def test_no_print_in_package():
    for py in PKG.rglob("*.py"):
        for ln in py.read_text().splitlines():
            assert not re.match(r"\s*print\(", ln), f"print() in {py}: {ln}"
