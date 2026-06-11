"""Read-only GitHub source via the ``gh`` CLI subprocess.

All network egress goes through ``gh`` (which the operator authenticates
with ``gh auth login``). The egress allowlist is ``api.github.com``;
``tests/rag/sources/test_egress_allowlist.py`` enforces it.

No write verbs: no ``POST``/``PUT``/``PATCH``/``DELETE`` and no GitHub
"create" subcommands of any kind. ``tests/rag/sources/test_no_writes.py``
enforces that statically.

A shared token-bucket caps every call site at 5 rps. Tenacity wraps every
``gh`` call in a bounded retry that only fires on rate-limit / 5xx
markers — never on 401/404.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._ratelimit import TokenBucket

logger = logging.getLogger(__name__)

# Timeout for every gh subprocess (seconds). gh is allowed plenty of time
# to paginate the issues/pulls endpoints over a slow link, but never enough
# to wedge the indexer.
_GH_TIMEOUT = 120

# Earliest `since` value the Issues REST endpoint accepts. GitHub silently
# returns an empty array when `since` is significantly earlier than the
# platform existed; the founding year (2008) is a safe sentinel that
# guarantees a first-time bootstrap pulls every issue. See the docstring on
# `iter_issues_changed_since` for the deploy-time bug this clamps around.
_GH_EPOCH = datetime(2008, 1, 1, tzinfo=UTC)

# stderr markers that justify a retry. Anything else (notably 401 / 404 /
# malformed input) fails fast.
_RETRYABLE_STDERR_MARKERS = (
    "rate limit",
    "rate-limit",
    "abuse",  # secondary rate limit; gh emits "secondary rate limit"
    "secondary",
    "500 Internal",
    "502 Bad Gateway",
    "503 Service Unavailable",
    "504 Gateway Timeout",
    "HTTP 500",
    "HTTP 502",
    "HTTP 503",
    "HTTP 504",
    "could not resolve host",  # transient DNS / proxy hiccup
)


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    body: str | None
    state: str
    state_reason: str | None
    labels: list[str]
    assignees: list[str]
    author: str | None
    created_at: str
    updated_at: str
    closed_at: str | None
    raw_json: str


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    body: str | None
    state: str  # "open" | "closed" | "merged"
    head_sha: str | None
    base_branch: str | None
    author: str | None
    draft: bool
    created_at: str
    updated_at: str
    merged_at: str | None
    closed_at: str | None
    raw_json: str


def _is_retryable(stderr: str) -> bool:
    """Return True when a CalledProcessError stderr signals a transient fault."""
    if not stderr:
        return False
    lowered = stderr.lower()
    return any(marker.lower() in lowered for marker in _RETRYABLE_STDERR_MARKERS)


class _RetryableGhError(subprocess.CalledProcessError):
    """Marker subclass — only this triggers a tenacity retry."""


def _classify(exc: subprocess.CalledProcessError) -> Exception:
    """Promote retryable CalledProcessErrors to ``_RetryableGhError``."""
    if _is_retryable(exc.stderr or ""):
        retryable = _RetryableGhError(
            returncode=exc.returncode,
            cmd=exc.cmd,
            output=exc.output,
            stderr=exc.stderr,
        )
        # Preserve the traceback chain for debugging.
        retryable.__cause__ = exc
        return retryable
    return exc


def _gh_iso(ts: datetime) -> str:
    """Format a datetime as the ISO-8601 ``Z`` form that GitHub expects."""
    if ts.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    # GitHub accepts an offset like +00:00 but the canonical form is "Z".
    iso = ts.astimezone(tz=ts.tzinfo).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{iso}Z"


_GH_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=20),
    retry=retry_if_exception_type(_RetryableGhError),
    reraise=True,
)


class GitHubSource:
    """Read-only enumerator over ``example-org/sample`` (or any GH repo)."""

    def __init__(
        self,
        owner: str,
        repo: str,
        rate_limit_rps: float = 5.0,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner):
            raise ValueError(f"invalid owner: {owner!r}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
            raise ValueError(f"invalid repo: {repo!r}")
        self.owner = owner
        self.repo = repo
        self._bucket = TokenBucket(rate_per_second=rate_limit_rps)

    # --- low-level gh wrappers ----------------------------------------

    def _run_gh(self, args: list[str]) -> str:
        """Run ``gh`` with the shared rate-limit + retry guard."""
        self._bucket.acquire()
        return self._run_gh_with_retry(args)

    @_GH_RETRY
    def _run_gh_with_retry(self, args: list[str]) -> str:
        try:
            proc = subprocess.run(  # noqa: PLW1510 - check=True set explicitly
                ["gh", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=_GH_TIMEOUT,
            )
        except subprocess.CalledProcessError as exc:
            classified = _classify(exc)
            if isinstance(classified, _RetryableGhError):
                logger.warning(
                    "gh transient failure (will retry): %s",
                    (exc.stderr or "").strip(),
                )
            raise classified from exc
        return proc.stdout

    # --- issues --------------------------------------------------------

    def iter_issues_changed_since(self, since: datetime) -> Iterator[Issue]:
        """Yield every issue (NOT PR) updated at or after ``since``.

        Order is ascending by ``updated_at`` so a pipeline can checkpoint
        the watermark after each yielded row.

        The GitHub Issues REST endpoint silently returns an empty array
        when ``since`` is earlier than GitHub itself existed (≤2008). A
        first-time bootstrap defaults the watermark to the Unix
        epoch (1970-01-01), which trips this — we'd index zero issues
        and never know. Clamp the wire value to ``_GH_EPOCH`` (the
        founding year, well before the first commit anywhere in
        production GitHub data) so a fresh bootstrap pulls every issue.
        """
        wire_since = max(since, _GH_EPOCH)
        path = (
            f"repos/{self.owner}/{self.repo}/issues"
            f"?state=all&since={_gh_iso(wire_since)}&per_page=100"
        )
        raw = self._run_gh(["api", "--paginate", path])
        items = self._parse_paginated_json(raw)

        issues: list[Issue] = []
        for item in items:
            # GitHub returns PRs in the issues endpoint with a `pull_request`
            # key — skip those; PRs come from the pulls endpoint.
            if item.get("pull_request"):
                continue
            issues.append(_issue_from_json(item))

        issues.sort(key=lambda i: i.updated_at)
        yield from issues

    # --- pulls ---------------------------------------------------------

    def iter_pulls_changed_since(self, since: datetime) -> Iterator[PullRequest]:
        """Yield every pull request updated at or after ``since``.

        ``since`` is enforced client-side because GitHub's pulls endpoint
        does not natively accept it — we ask for newest-first and stop
        once we cross the watermark. The yielded rows are then re-sorted
        ASC by ``updated_at``.
        """
        path = (
            f"repos/{self.owner}/{self.repo}/pulls"
            "?state=all&sort=updated&direction=desc&per_page=100"
        )
        raw = self._run_gh(["api", "--paginate", path])
        items = self._parse_paginated_json(raw)

        cutoff = _gh_iso(since)
        pulls: list[PullRequest] = []
        for item in items:
            if (item.get("updated_at") or "") < cutoff:
                continue
            pulls.append(_pull_from_json(item))

        pulls.sort(key=lambda p: p.updated_at)
        yield from pulls

    # --- pr diff -------------------------------------------------------

    def get_pr_diff(self, pr_number: int) -> str:
        """Return the unified diff for ``pr_number`` via ``gh pr diff``.

        Returns an empty string on ANY per-PR diff failure — the indexer
        still upserts the PR row's metadata, it just has no chunks /
        embeddings for the diff. This graceful degradation matters because
        the diff fetch is one ``gh`` call per PR across a whole-repo
        bootstrap, and a single failure must not abort the whole
        multi-minute run:

        * **Diff too large** (HTTP 406 / "diff exceeded the maximum number
          of lines") — large PRs on big repos routinely trip this.
        * **Transient gh/network hiccup** — e.g. a flaky ``gh pr diff`` that
          exits 1 mid-run but succeeds on a later retry. Aborting here would
          also skip the docs-indexing stream that runs after pulls, so the
          per-PR diff must soft-fail rather than propagate.
        * **Unavailable head ref** — a squashed/deleted branch whose diff
          gh can no longer materialise.

        Any skipped PR can be backfilled once its diff is fetchable again
        with ``index --refresh-pr <N>``. Errors are logged at WARNING so a
        systematic problem (e.g. broken auth) is still visible — and the
        issues/closing-refs streams, which don't soft-fail, would surface a
        total ``gh`` outage independently.
        """
        if pr_number <= 0:
            raise ValueError(f"pr_number must be positive: {pr_number!r}")
        try:
            return self._run_gh(
                [
                    "pr",
                    "diff",
                    str(pr_number),
                    "--repo",
                    f"{self.owner}/{self.repo}",
                ]
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stderr_lower = stderr.lower()
            too_large = (
                "too_large" in stderr_lower
                or "diff exceeded the maximum" in stderr_lower
                or "http 406" in stderr_lower
            )
            reason = "diff too large for the GitHub API" if too_large else (
                f"gh pr diff failed (exit {exc.returncode})"
            )
            logger.warning(
                "PR #%d: %s; indexing PR metadata only (no chunks). "
                "Backfill later with `index --refresh-pr %d`. gh stderr=%s",
                pr_number,
                reason,
                pr_number,
                stderr,
            )
            return ""

    # --- closing references (GraphQL) ---------------------------------

    def iter_closing_references(self) -> Iterator[tuple[int, int]]:
        """Yield ``(pr_number, issue_number)`` for every closing reference.

        Paginates the GraphQL ``pullRequests`` connection with ``first:100``
        cursors; for each PR, walks the ``closingIssuesReferences`` nodes.
        This is the gold-confidence pairing signal (confidence=1.0 in the
        store).
        """
        cursor: str | None = None
        while True:
            cursor_literal = f'"{cursor}"' if cursor else "null"
            pulls_clause = (
                f"pullRequests(states: [MERGED, CLOSED], first: 100, after: {cursor_literal})"
            )
            query = (
                f'query {{ repository(owner: "{self.owner}", name: "{self.repo}") {{'
                f"  {pulls_clause} {{"
                "    pageInfo { hasNextPage endCursor }"
                "    nodes { number closingIssuesReferences(first: 20) { nodes { number } } }"
                "  }"
                "}}"
            )
            raw = self._run_gh(["api", "graphql", "-f", f"query={query}"])
            data = json.loads(raw) if raw.strip() else {}
            connection = (
                data.get("data", {})
                .get("repository", {})
                .get("pullRequests", {})
            )
            for node in connection.get("nodes") or []:
                pr_number = node.get("number")
                if pr_number is None:
                    continue
                refs = (node.get("closingIssuesReferences") or {}).get("nodes") or []
                for ref in refs:
                    issue_number = ref.get("number")
                    if issue_number is None:
                        continue
                    yield int(pr_number), int(issue_number)
            page_info = connection.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                return
            cursor = page_info.get("endCursor")
            if not cursor:
                return

    # --- helpers -------------------------------------------------------

    @staticmethod
    def _parse_paginated_json(raw: str) -> list[dict[str, Any]]:
        """Parse the output of ``gh api --paginate``.

        ``gh api --paginate`` concatenates each page's JSON array into a
        single stream. The two shapes seen in the wild are:

        * One JSON array (no pagination needed) — ``json.loads`` works.
        * Multiple back-to-back JSON arrays — we walk them with
          ``json.JSONDecoder.raw_decode`` and flatten.
        """
        text = raw.strip()
        if not text:
            return []
        decoder = json.JSONDecoder()
        idx = 0
        items: list[dict[str, Any]] = []
        while idx < len(text):
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text):
                break
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, list):
                items.extend(obj)
            elif isinstance(obj, dict):
                items.append(obj)
            idx = end
        return items


def _issue_from_json(item: dict[str, Any]) -> Issue:
    user = item.get("user") or {}
    return Issue(
        number=int(item["number"]),
        title=item.get("title") or "",
        body=item.get("body"),
        state=item.get("state") or "",
        state_reason=item.get("state_reason"),
        labels=[lbl.get("name", "") for lbl in (item.get("labels") or [])],
        assignees=[a.get("login", "") for a in (item.get("assignees") or [])],
        author=user.get("login"),
        created_at=item.get("created_at") or "",
        updated_at=item.get("updated_at") or "",
        closed_at=item.get("closed_at"),
        raw_json=json.dumps(item, sort_keys=True),
    )


def _pull_from_json(item: dict[str, Any]) -> PullRequest:
    user = item.get("user") or {}
    head = item.get("head") or {}
    base = item.get("base") or {}
    merged_at = item.get("merged_at")
    state = item.get("state") or ""
    # GitHub's pulls endpoint uses state="closed" for merged PRs too; the
    # `merged_at` field is the authoritative signal.
    if merged_at:
        state = "merged"
    return PullRequest(
        number=int(item["number"]),
        title=item.get("title") or "",
        body=item.get("body"),
        state=state,
        head_sha=head.get("sha"),
        base_branch=base.get("ref"),
        author=user.get("login"),
        draft=bool(item.get("draft", False)),
        created_at=item.get("created_at") or "",
        updated_at=item.get("updated_at") or "",
        merged_at=merged_at,
        closed_at=item.get("closed_at"),
        raw_json=json.dumps(item, sort_keys=True),
    )
