from __future__ import annotations

import sys


def test_p4python_not_imported():
    import codescribe_rag.rag.sources  # noqa: F401
    import codescribe_rag.rag.sources.perforce  # noqa: F401

    assert "P4" not in sys.modules and "p4python" not in sys.modules
