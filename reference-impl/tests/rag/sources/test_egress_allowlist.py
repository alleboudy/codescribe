from __future__ import annotations

from datetime import datetime, timezone

import httpx

from codescribe_rag.rag.sources._ratelimit import TokenBucket
from codescribe_rag.rag.sources.bugzilla import BugzillaSource


def test_only_configured_host_contacted():
    seen = []

    def handler(req):
        seen.append(req.url.host)
        return httpx.Response(200, json={"bugs": []})

    src = BugzillaSource("https://bz.example.com", "k", TokenBucket(1000.0))
    src._client = httpx.Client(
        base_url="https://bz.example.com/rest",
        headers={"X-BUGZILLA-API-KEY": "k"},
        transport=httpx.MockTransport(handler))
    list(src.iter_bugs_changed_since(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    assert set(seen) == {"bz.example.com"}
