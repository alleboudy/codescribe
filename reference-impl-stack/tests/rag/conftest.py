"""Local conftest for ``tests/rag/``.

Re-exports the e2e test's shared fixtures (``seeded_db`` +
``stub_embedder``) from ``tests/servers/rag_server/conftest.py`` so the
spec-mandated test location (``tests/rag/test_e2e_with_claw.py``) can
use them without duplicating the fixture wiring.

We DON'T use ``pytest_plugins = (...)`` here because that registers
the target conftest twice (once as a plugin under its dotted module
name, once as a conftest under its file path), tripping pytest's
"Plugin already registered" error at collection time. Importing the
fixtures + re-binding them at module scope is the official escape
hatch for sharing fixtures across conftest scopes.
"""

from __future__ import annotations

from tests.servers.rag_server.conftest import (
    StubEmbedder,  # noqa: F401  # used in type hints by tests/rag/test_e2e_with_claw.py
    query_vec,
    seeded_db,
    stub_embedder,
)

# Re-export so pytest picks these up as fixtures defined in this conftest.
__all__ = ("StubEmbedder", "query_vec", "seeded_db", "stub_embedder")
