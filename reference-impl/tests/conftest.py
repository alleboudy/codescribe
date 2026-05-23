from __future__ import annotations

from datetime import datetime, timezone

import pytest

from codescribe_rag.rag.embed.embedder import HashingEmbedder
from codescribe_rag.rag.store.writer import Store

UTC = timezone.utc


class _Bug:
    def __init__(self, i, summary, desc):
        self.id = i
        self.summary = summary
        self.description = desc
        self.component = "core"
        self.severity = "major"
        self.status = "RESOLVED"
        self.resolution = "FIXED"
        self.creation_time = datetime(2026, 1, 1, tzinfo=UTC)
        self.last_change_time = datetime(2026, 1, 2, tzinfo=UTC)
        self.assigned_to = "alice"
        self.raw_json = {"id": i}


class _CL:
    def __init__(self, n, diff):
        self.cl = n
        self.author = "alice"
        self.submitted_at = datetime(2026, 1, 3, tzinfo=UTC)
        self.description = f"fix for {n}"
        self.files = ()
        self.diff_text = diff


class _Link:
    def __init__(self, bug_id, cl, conf):
        self.bug_id = bug_id
        self.cl_number = cl
        self.confidence = conf
        self.signals = {"x": conf}


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def populated_store(tmp_path, embedder):
    store = Store.open(tmp_path / "rag.db")
    bugs = [
        _Bug(1001, "NullPointerException during startup", "NPE in ConfigLoader on boot"),
        _Bug(1002, "memory leak in cache", "heap grows unbounded under load"),
        _Bug(1003, "race condition in scheduler", "threads deadlock occasionally"),
    ]
    with store.batch():
        for b in bugs:
            store.upsert_bug(b, embedder.embed_one(f"{b.summary} {b.description}"))
        store.upsert_cl(_CL(12345, "@@ -1 +1 @@\n-bad\n+good\n"), [], [], embedder.embed_one("fix"))
        store.upsert_fix_link(_Link(1001, 12345, 0.9))
        store.upsert_fix_link(_Link(1002, 12345, 0.7))   # below default threshold
    yield store
    store.close()
