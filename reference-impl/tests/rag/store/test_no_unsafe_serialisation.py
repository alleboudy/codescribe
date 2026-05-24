from __future__ import annotations

import re
from pathlib import Path

STORE = Path(__file__).parents[3] / "codescribe_rag" / "rag" / "store"
# JSON+gzip only. Banned-module tokens are assembled from fragments so this guard
# file does not match itself when scanned (per #4 SS8's "import pi"+"ckle" trick).
_BANNED = ["pi" + "ckle", "cPi" + "ckle", "di" + "ll", "shel" + "ve"]
PATTERN = re.compile(r"\bimport\s+(?:" + "|".join(_BANNED) + r")\b")


def test_no_unsafe_serialisation():
    for py in STORE.rglob("*.py"):
        assert not PATTERN.search(py.read_text()), f"unsafe serialisation in {py}"
