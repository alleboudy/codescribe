from __future__ import annotations

import logging
import os
from pathlib import Path

from codescribe_rag.rag.config import RagConfig
from codescribe_rag.rag.embed.embedder import make_embedder
from codescribe_rag.rag.store.retrieve import Retriever
from codescribe_rag.rag.store.writer import Store

logger = logging.getLogger(__name__)


FIND_SIMILAR_BUGS_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural-language description of the issue or feature.",
        },
        "k": {
            "type": "integer",
            "description": "Number of similar bugs to return.",
            "minimum": 1, "maximum": 20, "default": 5,
        },
        "min_confidence": {
            "type": "number",
            "description": "Filter pairs by minimum bug<->CL confidence.",
            "minimum": 0.0, "maximum": 1.0, "default": 0.8,
        },
    },
    "required": ["query"],
}

GET_FIX_DIFF_SCHEMA = {
    "type": "object",
    "properties": {
        "cl_number": {"type": "integer", "minimum": 1},
        "max_chars": {"type": "integer", "minimum": 100, "default": 8000},
    },
    "required": ["cl_number"],
}


class RagTools:
    """Read-only retrieval tools backing the MCP server."""

    def __init__(self, store: Store, retriever: Retriever) -> None:
        self._store = store
        self._retriever = retriever

    @classmethod
    def load(cls, db_path: Path) -> "RagTools":
        config_path = Path(os.environ.get("RAG_CONFIG", "configs/rag.yaml"))
        cfg = RagConfig.from_yaml(config_path)
        embed_cfg = cfg.embed
        backend = os.environ.get("RAG_EMBED_BACKEND")
        if backend:                       # lets the server run on 'hashing' without a model
            embed_cfg = embed_cfg.model_copy(update={"backend": backend})
        store = Store.open(db_path)
        return cls(store, Retriever(store._conn, make_embedder(embed_cfg)))

    def close(self) -> None:
        self._store.close()

    # ------------------------------------------------------------------
    def find_similar_bugs(self, query: str, k: int = 5, min_confidence: float = 0.8) -> str:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= k <= 20:
            raise ValueError(f"k out of range: {k}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence out of range: {min_confidence}")
        results = self._retriever.find_similar_bugs(query=query, k=k, confidence_threshold=min_confidence)
        return self._format_results(results)

    def get_fix_diff(self, cl_number: int, max_chars: int = 8000) -> str:
        text = self._store.get_diff_text(cl_number)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... (truncated; CL {cl_number})"
        return f"## CL {cl_number}\n\n```diff\n{text}\n```"

    @staticmethod
    def _format_results(results: list) -> str:
        if not results:
            return "## Similar bugs (none found)\n\nNo matching bugs in the index."
        lines = [f"## Similar bugs (top {len(results)})\n"]
        for r in results:
            lines.append(f"### Bug {r.bug_id} — {r.status} (score {r.score:.3f})")
            lines.append(f"**Summary:** {r.summary}")
            if r.fix_cl is not None:
                lines.append(f"\n**Fix CL {r.fix_cl}** (confidence {r.confidence:.2f}):")
                if r.fix_diff_excerpt:
                    lines.append("```diff")
                    lines.append(r.fix_diff_excerpt)
                    lines.append("```")
                    lines.append(f"(excerpt; full diff via `get_fix_diff({r.fix_cl})`)")
            else:
                lines.append("\n_No high-confidence fix linked._")
            lines.append("")
        return "\n".join(lines)
