"""Issue↔PR pairing — three signal paths + a merge step.

Pure functional, no I/O. Consumed by the pipeline.

Signals (per the design spec):

  * `closingIssuesReferences` (GraphQL) → confidence 1.0
  * PR-body regex (`Closes #N` / `Fixes #N` / `Resolves #N`) → confidence 0.85
  * Temporal+author heuristic (PR merged within N days of issue close,
    author == assignee) → confidence 0.5

`merge_links` collapses duplicates by `(issue_number, pr_number)`, keeping
the MAX confidence and a comma-separated `source` field (sorted, so the
output is deterministic for a given multiset of inputs).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from dateutil.parser import isoparse

from codescribe_train.rag.extract.links import extract_closing_issue_refs_from_pr_body
from codescribe_train.rag.sources.github_source import Issue, PullRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IssuePRLink:
    issue_number: int
    pr_number: int
    source: str
    confidence: float
    signals: dict[str, float] = field(default_factory=dict)


def pair_via_closing_refs(refs: Iterable[tuple[int, int]]) -> list[IssuePRLink]:
    """Path (a): GraphQL `closingIssuesReferences` — confidence 1.0.

    ``refs`` is the iterable yielded by
    ``GitHubSource.iter_closing_references()``, i.e. ``(pr_number,
    issue_number)`` tuples. Duplicates collapse so the pipeline never
    double-writes a row.
    """
    seen: set[tuple[int, int]] = set()
    links: list[IssuePRLink] = []
    for pr_number, issue_number in refs:
        key = (issue_number, pr_number)
        if key in seen:
            continue
        seen.add(key)
        links.append(
            IssuePRLink(
                issue_number=issue_number,
                pr_number=pr_number,
                source="closingIssuesReferences",
                confidence=1.0,
                signals={"closing_issues_reference": 1.0},
            )
        )
    return links


def pair_via_body_keyword(prs: Iterable[PullRequest]) -> list[IssuePRLink]:
    """Path (b): PR body regex (`Closes #N` / `Fixes #N` / `Resolves #N`).

    Confidence 0.85 per the design spec (canonical; reconciled against an
    earlier draft's 0.7 in `configs/rag.yaml`).

    Drafts are excluded per `codescribe_train/rag/sources/AGENTS.md`
    anti-pattern #5.
    """
    seen: set[tuple[int, int]] = set()
    links: list[IssuePRLink] = []
    for pr in prs:
        if pr.draft:
            logger.debug("skipping draft PR #%s in body-keyword pairing", pr.number)
            continue
        for issue_number in extract_closing_issue_refs_from_pr_body(pr.body):
            key = (issue_number, pr.number)
            if key in seen:
                continue
            seen.add(key)
            links.append(
                IssuePRLink(
                    issue_number=issue_number,
                    pr_number=pr.number,
                    source="body_closes_keyword",
                    confidence=0.85,
                    signals={"body_closes_keyword": 0.85},
                )
            )
    return links


def _parse_ts(ts: str | None) -> datetime | None:
    """Parse a GitHub ISO-8601 timestamp; return ``None`` on falsy input."""
    if not ts:
        return None
    return isoparse(ts)


def pair_via_temporal(
    issues: Iterable[Issue],
    prs: Iterable[PullRequest],
    window_days: int = 7,
) -> list[IssuePRLink]:
    """Path (c): temporal+author match — confidence 0.5.

    For every (issue, PR) pair where:

      * the issue has ``closed_at`` set,
      * the PR has ``merged_at`` set,
      * the absolute time delta between ``closed_at`` and ``merged_at``
        is within ``window_days``,
      * the PR is NOT a draft,
      * the PR author is non-empty AND is one of the issue's
        non-empty assignees,

    emit one ``IssuePRLink`` with confidence 0.5.
    """
    issue_list = list(issues)
    pr_list = [pr for pr in prs if not pr.draft]
    window = timedelta(days=window_days)

    # Precompute (closed_at_ts, non_empty_assignees) for each issue.
    issue_index: list[tuple[Issue, datetime, set[str]]] = []
    for issue in issue_list:
        closed = _parse_ts(issue.closed_at)
        if closed is None:
            continue
        assignees = {a for a in issue.assignees if a}
        if not assignees:
            continue
        issue_index.append((issue, closed, assignees))

    seen: set[tuple[int, int]] = set()
    links: list[IssuePRLink] = []
    for pr in pr_list:
        merged = _parse_ts(pr.merged_at)
        if merged is None:
            continue
        author = pr.author
        if not author:
            continue
        for issue, closed, assignees in issue_index:
            if author not in assignees:
                continue
            if abs(merged - closed) > window:
                continue
            key = (issue.number, pr.number)
            if key in seen:
                continue
            seen.add(key)
            links.append(
                IssuePRLink(
                    issue_number=issue.number,
                    pr_number=pr.number,
                    source="temporal_author_match",
                    confidence=0.5,
                    signals={"temporal_author_match": 0.5},
                )
            )
    return links


def merge_links(*sources: Iterable[IssuePRLink]) -> list[IssuePRLink]:
    """Collapse duplicates by ``(issue_number, pr_number)``.

    For each unique pair, keep the MAX confidence and merge:
      * ``source``: comma-separated, sorted, deduped — deterministic for
        any multiset of inputs.
      * ``signals``: union of all contributing ``signals`` dicts (keys
        from each path are disjoint by construction).

    Accepts any number of iterables (lists from `pair_via_*` calls), so
    the pipeline can write ``merge_links(refs, body, temporal)`` directly.
    """
    bucket: dict[tuple[int, int], list[IssuePRLink]] = {}
    for group in sources:
        for link in group:
            key = (link.issue_number, link.pr_number)
            bucket.setdefault(key, []).append(link)

    merged: list[IssuePRLink] = []
    for (issue_number, pr_number), group_links in bucket.items():
        max_confidence = max(link.confidence for link in group_links)
        # Deterministic sort across runs: sources in alphabetical order,
        # deduped via dict-key uniqueness.
        joined_sources = ",".join(sorted({link.source for link in group_links}))
        signals: dict[str, float] = {}
        for link in group_links:
            for k, v in link.signals.items():
                signals[k] = max(signals.get(k, v), v)
        merged.append(
            IssuePRLink(
                issue_number=issue_number,
                pr_number=pr_number,
                source=joined_sources,
                confidence=max_confidence,
                signals=signals,
            )
        )
    return merged


def filter_by_confidence(
    links: Iterable[IssuePRLink],
    *,
    threshold: float,
) -> list[IssuePRLink]:
    """Keep links whose confidence is ``>= threshold``.

    The strict default threshold for the sample corpus is 0.8 (see
    `configs/rag.yaml`'s ``pairing.strict_threshold``). At 0.8 this keeps
    the closing-refs (1.0) and body-keyword (0.85) signals, drops the
    temporal-author heuristic (0.5).
    """
    return [link for link in links if link.confidence >= threshold]
