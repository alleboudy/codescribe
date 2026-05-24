from __future__ import annotations

import re
from pathlib import Path

PIPE = Path(__file__).parents[3] / "codescribe_rag" / "rag" / "pipelines"


def test_no_print_in_pipelines():
    for py in PIPE.rglob("*.py"):
        for ln in py.read_text().splitlines():
            assert not re.match(r"\s*print\(", ln), f"print() in {py}: {ln}"
