from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from codescribe_rag.rag.sources._ratelimit import TokenBucket
from codescribe_rag.rag.sources.bugzilla import Bug, BugzillaSource


def make_source(handler) -> BugzillaSource:
    src = BugzillaSource(base_url="https://bz.example.com", api_key="stub-key",
                         rate_limiter=TokenBucket(1000.0), page_size=2)
    src._client = httpx.Client(
        base_url="https://bz.example.com/rest",
        headers={"X-BUGZILLA-API-KEY": "stub-key", "Accept": "application/json"},
        transport=httpx.MockTransport(handler))
    return src


def test_iter_bugs_pagination():
    pages = [
        [{"id": i, "summary": f"b{i}", "component": "c", "severity": "major",
          "status": "RESOLVED", "resolution": "FIXED",
          "creation_time": "2026-01-01T00:00:00Z", "last_change_time": f"2026-01-0{i}T00:00:00Z",
          "assigned_to": "alice"} for i in (1, 2)],
        [{"id": 3, "summary": "b3", "component": "c", "severity": "minor",
          "status": "RESOLVED", "resolution": "FIXED",
          "creation_time": "2026-01-01T00:00:00Z", "last_change_time": "2026-01-03T00:00:00Z",
          "assigned_to": "bob"}],
    ]
    calls = {"n": 0}

    def handler(req):
        if req.url.path == "/rest/bug":
            page = pages[min(calls["n"], len(pages) - 1)]
            calls["n"] += 1
            return httpx.Response(200, json={"bugs": page})
        return httpx.Response(404)

    bugs = list(make_source(handler).iter_bugs_changed_since(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    assert [b.id for b in bugs] == [1, 2, 3]
    assert isinstance(bugs[0], Bug)


def test_safety_margin_subtracted():
    seen = {}

    def handler(req):
        seen["lct"] = req.url.params.get("last_change_time")
        return httpx.Response(200, json={"bugs": []})

    list(make_source(handler).iter_bugs_changed_since(datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)))
    assert seen["lct"] == "2026-05-20T11:59:59Z"  # minus 1s


def test_retry_on_5xx_not_on_404():
    state = {"n": 0}

    def handler(req):
        if req.url.path == "/rest/bug":
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"bugs": []})
        return httpx.Response(404)

    list(make_source(handler).iter_bugs_changed_since(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    assert state["n"] == 2  # one retry on 503

    def handler404(req):
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        list(make_source(handler404).iter_bugs_changed_since(datetime(2026, 1, 1, tzinfo=timezone.utc)))


def test_get_comments():
    def handler(req):
        if req.url.path == "/rest/bug/1/comment":
            return httpx.Response(200, json={"bugs": {"1": {"comments": [
                {"id": 1, "creator": "alice", "text": "Fixed in CL 12345",
                 "creation_time": "2026-01-05T00:00:00Z"}]}}})
        return httpx.Response(404)

    comments = make_source(handler).get_comments(1)
    assert comments[0].text == "Fixed in CL 12345"


def test_parse_iso_z_suffix():
    dt = BugzillaSource._parse_iso("2026-05-20T18:00:00Z")
    assert dt == datetime(2026, 5, 20, 18, 0, 0, tzinfo=timezone.utc)
