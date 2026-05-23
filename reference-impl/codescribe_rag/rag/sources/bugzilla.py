from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator
from urllib.parse import urlparse

import httpx
import tenacity

from ._ratelimit import TokenBucket

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BugComment:
    id: int
    creator: str
    text: str
    creation_time: datetime


@dataclass(frozen=True, slots=True)
class Bug:
    id: int
    summary: str
    description: str            # Bugzilla's "comment 0"; filled by the pipeline
    component: str
    severity: str
    status: str
    resolution: str
    creation_time: datetime
    last_change_time: datetime
    assigned_to: str
    raw_json: dict


def _retryable(e: BaseException) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        return 500 <= e.response.status_code < 600
    return isinstance(e, (httpx.ConnectError, httpx.ReadTimeout))


class BugzillaSource:
    """Read-only Bugzilla REST client."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        rate_limiter: TokenBucket,
        page_size: int = 100,
        request_timeout_s: float = 30.0,
        connect_timeout_s: float = 5.0,
    ) -> None:
        hostname = urlparse(base_url).hostname
        if not hostname:
            raise ValueError(f"Invalid base_url: {base_url}")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/rest",
            headers={
                "X-BUGZILLA-API-KEY": api_key,
                "Accept": "application/json",
                "User-Agent": "codescribe_rag-indexer/1.0",
            },
            timeout=httpx.Timeout(request_timeout_s, connect=connect_timeout_s),
        )
        self._rate = rate_limiter
        self._page_size = page_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def iter_bugs_changed_since(self, since: datetime) -> Iterator[Bug]:
        """Yield bugs modified since ``since``, ascending by last_change_time."""
        # Bugzilla's last_change_time precision is 1s; subtract a safety margin
        # so we don't miss bugs that changed in the same second as the watermark.
        watermark = since - timedelta(seconds=1)
        offset = 0
        while True:
            self._rate.acquire()
            resp = self._get_with_retry("/bug", params={
                "last_change_time": watermark.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": self._page_size,
                "offset": offset,
                "order": "last_change_time ASC",
                "include_fields": "id,summary,component,severity,status,resolution,"
                                  "creation_time,last_change_time,assigned_to,_default",
            })
            bugs = resp.json().get("bugs", [])
            if not bugs:
                return
            for raw in bugs:
                yield self._parse_bug(raw)
            if len(bugs) < self._page_size:
                return
            offset += self._page_size

    def get_comments(self, bug_id: int) -> list[BugComment]:
        """Fetch comments (comment 0 = the description) for a bug."""
        self._rate.acquire()
        resp = self._get_with_retry(f"/bug/{bug_id}/comment")
        comments = resp.json().get("bugs", {}).get(str(bug_id), {}).get("comments", [])
        return [
            BugComment(
                id=int(c["id"]),
                creator=c.get("creator", ""),
                text=c.get("text", ""),
                creation_time=self._parse_iso(c["creation_time"]),
            )
            for c in comments
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=0, max=2),
        retry=tenacity.retry_if_exception(_retryable),
        reraise=True,
    )
    def _get_with_retry(self, path: str, params: dict | None = None) -> httpx.Response:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp

    def _parse_bug(self, raw: dict) -> Bug:
        return Bug(
            id=int(raw["id"]),
            summary=raw.get("summary", ""),
            description="",
            component=raw.get("component", ""),
            severity=raw.get("severity", ""),
            status=raw.get("status", ""),
            resolution=raw.get("resolution", ""),
            creation_time=self._parse_iso(raw["creation_time"]),
            last_change_time=self._parse_iso(raw["last_change_time"]),
            assigned_to=raw.get("assigned_to", ""),
            raw_json=raw,
        )

    @staticmethod
    def _parse_iso(s: str) -> datetime:
        # Bugzilla emits "2026-05-20T18:00:00Z"; strip Z and tag UTC.
        s = s.rstrip("Z")
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

    def close(self) -> None:
        self._client.close()
