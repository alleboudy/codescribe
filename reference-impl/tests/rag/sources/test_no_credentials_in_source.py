from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).parents[3] / "codescribe_rag" / "rag" / "sources"
KEYISH = re.compile(r"\b[0-9a-fA-F]{32,}\b|eyJ[A-Za-z0-9_-]{10,}")  # hex keys / JWT


def test_no_inline_credentials():
    for py in SRC.rglob("*.py"):
        assert not KEYISH.search(py.read_text()), f"key-shaped literal in {py}"
