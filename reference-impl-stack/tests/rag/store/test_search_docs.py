from __future__ import annotations

from pathlib import Path

import numpy as np

from codescribe_train.rag.embed.doc_chunker import DocChunk
from codescribe_train.rag.store.retrieve import Retriever
from codescribe_train.rag.store.types import RetrieveConfig, RetrievedDocChunk
from codescribe_train.rag.store.writer import Store


class _StubEmbedder:
    """Deterministic 1024-d embedder: hashes text to a fixed vector."""

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 1024), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, hash(t) % 1024] = 1.0
        return out


def test_search_docs_returns_doc_chunks(tmp_path: Path) -> None:
    db = tmp_path / "rag.db"
    emb = _StubEmbedder()
    with Store.open(db) as store:
        for path, chunk_text in [
            ("README.md", "bring the stack up with docker compose up"),
            ("docs/DEV.md", "running the test suite with pytest"),
        ]:
            chunk = DocChunk("Run", chunk_text, 0)
            vec = emb.embed([f"{chunk.heading}\n{chunk.text}"])[0]
            store.upsert_doc(path, [chunk], [vec])

        retriever = Retriever(store=store, embedder=emb, config=RetrieveConfig())
        results = retriever.search_docs("docker compose", k=5)

    assert results
    assert all(isinstance(r, RetrievedDocChunk) for r in results)
    # The docker-themed chunk should be retrievable.
    assert any("docker" in r.text_excerpt.lower() for r in results)
    assert all(r.doc_path for r in results)
