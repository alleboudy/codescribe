from __future__ import annotations

import pytest

from codescribe_rag.rag.store.retrieve import Retriever
from codescribe_rag.servers.rag_server.tools import (
    FIND_SIMILAR_BUGS_SCHEMA, GET_FIX_DIFF_SCHEMA, RagTools)


@pytest.fixture
def tools(populated_store, embedder):
    return RagTools(populated_store, Retriever(populated_store._conn, embedder))


def test_schemas_shape():
    assert FIND_SIMILAR_BUGS_SCHEMA["required"] == ["query"]
    assert GET_FIX_DIFF_SCHEMA["properties"]["cl_number"]["minimum"] == 1


def test_find_similar_bugs_markdown(tools):
    out = tools.find_similar_bugs("NPE on startup", k=3, min_confidence=0.8)
    assert "Similar bugs" in out and "Bug 1001" in out


def test_invalid_k_raises(tools):
    with pytest.raises(ValueError):
        tools.find_similar_bugs("x", k=999, min_confidence=0.8)


def test_get_fix_diff_truncation(tools):
    out = tools.get_fix_diff(12345, max_chars=10)
    assert "truncated" in out and out.startswith("## CL 12345")
