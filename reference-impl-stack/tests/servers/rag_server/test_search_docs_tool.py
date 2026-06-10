"""The rag MCP server lists + dispatches the search_docs tool."""
from __future__ import annotations

from codescribe_train.rag.store.types import RetrievedDocChunk
from codescribe_train.servers.rag_server.tools import SEARCH_DOCS_SCHEMA, RagTools


class _FakeRetriever:
    def search_docs(self, query: str, k: int = 5):
        return [
            RetrievedDocChunk(
                doc_path="README.md", heading="Run", text_excerpt="make up", score=0.5
            )
        ]


def test_search_docs_schema_shape() -> None:
    assert SEARCH_DOCS_SCHEMA["type"] == "object"
    assert "query" in SEARCH_DOCS_SCHEMA["properties"]
    assert SEARCH_DOCS_SCHEMA["required"] == ["query"]


def test_search_docs_tool_renders_markdown() -> None:
    tools = RagTools(store=None, embedder=None, retriever=_FakeRetriever())
    out = tools.search_docs(query="docker", k=3)
    assert isinstance(out, str)
    # Output must be Markdown (starts with a heading), not a JSON blob
    assert out.startswith("#"), "expected Markdown heading, got: " + out[:80]
    assert "README.md" in out
    assert "Run" in out
    assert "make up" in out
