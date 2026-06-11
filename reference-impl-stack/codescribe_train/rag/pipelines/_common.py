"""Shared inner loop for the bootstrap + incremental indexers.

The two top-level pipelines (``run_bootstrap`` and ``run_incremental``)
differ only in *which* watermarks they start from and how they advance
them. Everything else — embedder batching, store transactions, pairing
post-pass, SIGTERM handling — is shared and lives here.

Hard rules carried over from ``codescribe_train/rag/AGENTS.md``:

* Batch upserts ≤256 records per call.
* Embedder called once per batch, not once per record.
* State watermarks advanced *inside* the same transaction as the batch
  they correspond to (so a SIGTERM between batches never loses data).
* No stdout writes — JSON logging only.
* GitHub responses never cached longer than one pipeline run.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from codescribe_train.rag.embed.chunker import (
    chunk_pr_diff,
    summarise_pr_for_embedding,
)
from codescribe_train.rag.extract.pairing import (
    merge_links,
    pair_via_body_keyword,
    pair_via_closing_refs,
    pair_via_temporal,
)
from codescribe_train.rag.store.writer import Store

if TYPE_CHECKING:
    from codescribe_train.rag.sources.git_source import Commit
    from codescribe_train.rag.sources.github_source import Issue, PullRequest
    from codescribe_train.rag.sources.worktree_docs_source import Doc

logger = logging.getLogger(__name__)

# Per AGENTS.md: batch upserts ≤256 records.
DEFAULT_BATCH_SIZE = 256

# 1-second safety margin when reading back the issue watermark.
WATERMARK_SAFETY_MARGIN_SECONDS = 1

# State keys (per AGENTS.md). Centralised so callers never typo a key.
STATE_LAST_COMMIT_SHA = "last_seen_commit_sha"
STATE_LAST_ISSUE_UPDATED_AT = "last_seen_issue_updated_at"


class StopRequested(Exception):  # noqa: N818 - intentional non-Error suffix; control-flow marker
    """Raised between batches when SIGTERM has been observed.

    Caught at the top of the pipeline; the outer ``run_*`` returns
    cleanly so the operator-visible exit code is 0 (data already
    committed via per-batch transactions).
    """


# ---------------------------------------------------------------------------
# Source / dep injection
# ---------------------------------------------------------------------------


class _GitSourceLike(Protocol):
    def iter_commits_since(self, last_sha: str | None) -> Iterable[Commit]: ...

    def describe(self, sha: str) -> Commit: ...


class _GitHubSourceLike(Protocol):
    def iter_issues_changed_since(self, since: datetime) -> Iterable[Issue]: ...

    def iter_pulls_changed_since(self, since: datetime) -> Iterable[PullRequest]: ...

    def get_pr_diff(self, pr_number: int) -> str: ...

    def iter_closing_references(self) -> Iterable[tuple[int, int]]: ...


class _DocsSourceLike(Protocol):
    def iter_docs(self) -> Iterable[Doc]: ...


class _EmbedderLike(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class IndexerDeps:
    """Injected dependency bundle.

    Lives separately from :class:`RagConfig` so the test suite can swap
    in :class:`FakeGitSource` / :class:`FakeGitHubSource` / cheap
    deterministic embedders without touching the config layer.
    """

    git: _GitSourceLike
    github: _GitHubSourceLike
    embedder: _EmbedderLike
    store_path: Path
    docs: _DocsSourceLike | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    stop_event: asyncio.Event | None = None
    # Test injection hook: a list of ints (each int = "stop after writing N
    # records of any kind"). When the running record count reaches a value
    # in this list, ``stop_event`` is set so the next batch boundary aborts.
    stop_after_records: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SIGTERM handling
# ---------------------------------------------------------------------------


@contextmanager
def install_sigterm_handler(stop_event: asyncio.Event) -> Iterator[None]:
    """Install a SIGTERM handler that sets ``stop_event``.

    Used by ``__main__.py`` so the operator-visible run honours OS
    signals. Tests bypass this and set ``stop_event`` directly to keep
    the test suite signal-free.

    Restores the previous handler on exit so the test suite never leaks
    a custom handler into the next test.
    """
    prior = signal.getsignal(signal.SIGTERM)

    def _on_sigterm(_signum: int, _frame: Any) -> None:
        logger.info("SIGTERM received; will stop at next batch boundary")
        stop_event.set()

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
        yield
    finally:
        signal.signal(signal.SIGTERM, prior)


def check_stop(deps: IndexerDeps, indexed_so_far: int) -> None:
    """Raise :class:`StopRequested` if a stop has been signalled."""
    if (
        deps.stop_after_records
        and indexed_so_far in deps.stop_after_records
        and deps.stop_event is not None
    ):
        deps.stop_event.set()
    if deps.stop_event is not None and deps.stop_event.is_set():
        raise StopRequested()


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------


def _batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield ``items`` in lists of length ≤ ``size``."""
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Per-kind batch upserters
# ---------------------------------------------------------------------------


def index_issues_batch(
    store: Store,
    embedder: _EmbedderLike,
    issues: list[Issue],
) -> str | None:
    """Upsert one batch of issues; return the max ``updated_at`` in the batch."""
    if not issues:
        return None
    texts = [f"{i.title}\n{i.body or ''}" for i in issues]
    vectors = embedder.embed(texts)
    max_updated_at: str | None = None
    with store.batch():
        for issue, vec in zip(issues, vectors, strict=True):
            store.upsert_issue(issue, vec)
            if max_updated_at is None or issue.updated_at > max_updated_at:
                max_updated_at = issue.updated_at
        if max_updated_at is not None:
            store.set_state(STATE_LAST_ISSUE_UPDATED_AT, max_updated_at)
    return max_updated_at


def index_commits_batch(
    store: Store,
    embedder: _EmbedderLike,
    commits: list[Commit],
) -> str | None:
    """Upsert one batch of commits; return the SHA of the last commit indexed."""
    if not commits:
        return None
    texts = [f"{c.message}\n{c.diff_text}" for c in commits]
    vectors = embedder.embed(texts)
    last_sha: str | None = None
    with store.batch():
        for commit, vec in zip(commits, vectors, strict=True):
            store.upsert_commit(commit, vec)
            last_sha = commit.sha
        if last_sha is not None:
            store.set_state(STATE_LAST_COMMIT_SHA, last_sha)
    return last_sha


def index_pulls_batch(
    store: Store,
    embedder: _EmbedderLike,
    github: _GitHubSourceLike,
    pulls: list[PullRequest],
) -> int:
    """Upsert one batch of pulls + their chunks. Returns count indexed."""
    if not pulls:
        return 0

    # Per-PR: fetch diff (≤1 gh call/PR), chunk it, build summary text.
    pr_diffs: list[str] = []
    pr_chunk_lists = []
    summary_texts: list[str] = []
    for pr in pulls:
        diff = github.get_pr_diff(pr.number)
        pr_diffs.append(diff)
        pr_chunk_lists.append(chunk_pr_diff(pr, diff))
        summary_texts.append(summarise_pr_for_embedding(pr, diff))

    # Embed all summaries in a single call.
    summary_vectors = embedder.embed(summary_texts)

    # Flatten all chunks → one embed call across the batch.
    all_chunks_flat: list[Any] = []
    chunk_owner_indices: list[int] = []
    for pr_idx, chunks in enumerate(pr_chunk_lists):
        for chunk in chunks:
            all_chunks_flat.append(chunk)
            chunk_owner_indices.append(pr_idx)
    chunk_texts = [c.chunk_text for c in all_chunks_flat]
    chunk_vectors = (
        embedder.embed(chunk_texts)
        if chunk_texts
        else np.zeros((0, 1024), dtype=np.float32)
    )

    # Slice flat chunk vectors back into per-PR lists.
    per_pr_chunk_vectors: list[list[np.ndarray]] = [[] for _ in pulls]
    for owner_idx, vec in zip(chunk_owner_indices, chunk_vectors, strict=True):
        per_pr_chunk_vectors[owner_idx].append(vec)

    with store.batch():
        for pr, diff, chunks, summary_vec, chunk_vecs in zip(
            pulls,
            pr_diffs,
            pr_chunk_lists,
            summary_vectors,
            per_pr_chunk_vectors,
            strict=True,
        ):
            store.upsert_pull(
                pr=pr,
                chunks=chunks,
                chunk_embeddings=chunk_vecs,
                pr_summary_embedding=summary_vec,
                diff_text=diff,
            )
    return len(pulls)


def index_docs_batch(
    store: Store,
    embedder: _EmbedderLike,
    docs: list[Doc],
) -> int:
    """Upsert one batch of docs (each with its chunks). Returns docs indexed.

    Embeds all chunks across the batch in a single embedder call (one
    call per batch, per AGENTS.md), then slices the flat vectors back
    per-doc.
    """
    if not docs:
        return 0
    all_chunks_flat: list[Any] = []
    owner_indices: list[int] = []
    for doc_idx, doc in enumerate(docs):
        for chunk in doc.chunks:
            all_chunks_flat.append(chunk)
            owner_indices.append(doc_idx)
    chunk_texts = [f"{c.heading}\n{c.text}" for c in all_chunks_flat]
    chunk_vectors = (
        embedder.embed(chunk_texts)
        if chunk_texts
        else np.zeros((0, 1024), dtype=np.float32)
    )
    per_doc_vectors: list[list[np.ndarray]] = [[] for _ in docs]
    for owner_idx, vec in zip(owner_indices, chunk_vectors, strict=True):
        per_doc_vectors[owner_idx].append(vec)

    with store.batch():
        for doc, vecs in zip(docs, per_doc_vectors, strict=True):
            store.upsert_doc(doc.path, doc.chunks, vecs)
    return len(docs)


# ---------------------------------------------------------------------------
# Pairing post-pass
# ---------------------------------------------------------------------------


def run_pairing_post_pass(
    store: Store,
    github: _GitHubSourceLike,
    issues: list[Issue],
    pulls: list[PullRequest],
) -> int:
    """Drive the three pairing paths + merge_links + upsert.

    Returns the number of (issue, PR) links upserted. The
    "higher-confidence-wins" semantic on
    :meth:`Store.upsert_issue_pr_link` protects against the body-keyword
    pass downgrading a closing-ref pass (it never can — same pair, lower
    confidence is a no-op).
    """
    refs = list(github.iter_closing_references())
    refs_links = pair_via_closing_refs(refs)
    body_links = pair_via_body_keyword(pulls)
    temporal_links = pair_via_temporal(issues, pulls)
    merged = merge_links(refs_links, body_links, temporal_links)
    for link in merged:
        store.upsert_issue_pr_link(link)
    return len(merged)


# ---------------------------------------------------------------------------
# Inner loop — shared by bootstrap and incremental
# ---------------------------------------------------------------------------


@dataclass
class _RunResult:
    commits: int = 0
    issues: int = 0
    pulls: int = 0
    links: int = 0
    docs: int = 0
    last_commit_sha: str | None = None
    last_issue_updated_at: str | None = None
    stopped_early: bool = False
    indexed_issues: list[Issue] = field(default_factory=list)
    indexed_pulls: list[PullRequest] = field(default_factory=list)


async def run_inner_loop(
    deps: IndexerDeps,
    *,
    since_sha: str | None,
    since_issue_updated_at: datetime,
    since_pull_updated_at: datetime,
) -> _RunResult:
    """Run the three-stream indexer end-to-end, then the pairing pass.

    The three streams run concurrently (``asyncio.gather`` over
    ``asyncio.to_thread``) so the sync subprocess wrappers (git, gh)
    overlap. Writes are serialised through a single Store instance.
    """
    result = _RunResult()

    loop = asyncio.get_running_loop()

    # Spin the three sync sources off to threads in parallel; collect
    # everything first, then write in batches. The "collect everything"
    # step happens in a thread, NOT on the event loop. We still respect
    # the AGENTS.md "do NOT hold all records in memory" rule by
    # streaming each thread's output through a generator + batching at
    # write time. The fakes are in-memory; the real sources are an
    # already-paginated `gh api --paginate` blob — both bounded by
    # GitHub's per-page caps.

    def _drain_commits() -> list[Commit]:
        return list(deps.git.iter_commits_since(since_sha))

    def _drain_issues() -> list[Issue]:
        return list(deps.github.iter_issues_changed_since(since_issue_updated_at))

    def _drain_pulls() -> list[PullRequest]:
        return list(deps.github.iter_pulls_changed_since(since_pull_updated_at))

    def _drain_docs() -> list[Doc]:
        if deps.docs is None:
            return []
        return list(deps.docs.iter_docs())

    commits_task = loop.run_in_executor(None, _drain_commits)
    issues_task = loop.run_in_executor(None, _drain_issues)
    pulls_task = loop.run_in_executor(None, _drain_pulls)
    docs_task = loop.run_in_executor(None, _drain_docs)
    commits_raw, issues_raw, pulls_raw, docs_raw = await asyncio.gather(
        commits_task, issues_task, pulls_task, docs_task
    )

    # The git source emits Commit objects without diff text (cheap log
    # walk). The pipeline needs the full diff for embedding, so we
    # describe() each one. The describe() call is per-commit subprocess;
    # we batch the describes through a thread pool so the wall-clock cost
    # is bearable on real repos.
    def _describe_all(shas: list[str]) -> list[Commit]:
        return [deps.git.describe(sha) for sha in shas]

    if commits_raw:
        commit_shas = [c.sha for c in commits_raw]
        commits_full = await loop.run_in_executor(
            None, _describe_all, commit_shas
        )
    else:
        commits_full = []

    indexed_count = 0

    # --- commits batches ---
    try:
        for batch in _batched(commits_full, deps.batch_size):
            check_stop(deps, indexed_count)
            last_sha = await loop.run_in_executor(
                None,
                _index_commits_batch_in_store,
                deps,
                batch,
            )
            result.commits += len(batch)
            if last_sha is not None:
                result.last_commit_sha = last_sha
            indexed_count += len(batch)

        # --- issues batches ---
        for batch in _batched(issues_raw, deps.batch_size):
            check_stop(deps, indexed_count)
            max_updated = await loop.run_in_executor(
                None,
                _index_issues_batch_in_store,
                deps,
                batch,
            )
            result.issues += len(batch)
            if max_updated is not None and (
                result.last_issue_updated_at is None
                or max_updated > result.last_issue_updated_at
            ):
                result.last_issue_updated_at = max_updated
            result.indexed_issues.extend(batch)
            indexed_count += len(batch)

        # --- pulls batches ---
        for batch in _batched(pulls_raw, deps.batch_size):
            check_stop(deps, indexed_count)
            n = await loop.run_in_executor(
                None,
                _index_pulls_batch_in_store,
                deps,
                batch,
            )
            result.pulls += n
            result.indexed_pulls.extend(batch)
            indexed_count += n

        # --- docs batches ---
        for batch in _batched(docs_raw, deps.batch_size):
            check_stop(deps, indexed_count)
            n = await loop.run_in_executor(
                None,
                _index_docs_batch_in_store,
                deps,
                batch,
            )
            result.docs += n
            indexed_count += n

        # --- pairing post-pass ---
        n_links = await loop.run_in_executor(
            None,
            _run_pairing_in_store,
            deps,
            result.indexed_issues + _load_all_issues_from_store(deps.store_path),
            result.indexed_pulls + _load_all_pulls_from_store(deps.store_path),
        )
        result.links = n_links
    except StopRequested:
        result.stopped_early = True
        logger.info(
            "indexer stopped early at indexed_count=%d (per stop_event)",
            indexed_count,
        )

    return result


def _index_commits_batch_in_store(
    deps: IndexerDeps, batch: list[Commit]
) -> str | None:
    with Store.open(deps.store_path) as store:
        return index_commits_batch(store, deps.embedder, batch)


def _index_issues_batch_in_store(
    deps: IndexerDeps, batch: list[Issue]
) -> str | None:
    with Store.open(deps.store_path) as store:
        return index_issues_batch(store, deps.embedder, batch)


def _index_pulls_batch_in_store(
    deps: IndexerDeps, batch: list[PullRequest]
) -> int:
    with Store.open(deps.store_path) as store:
        return index_pulls_batch(store, deps.embedder, deps.github, batch)


def _index_docs_batch_in_store(deps: IndexerDeps, batch: list[Doc]) -> int:
    with Store.open(deps.store_path) as store:
        return index_docs_batch(store, deps.embedder, batch)


def _run_pairing_in_store(
    deps: IndexerDeps,
    issues: list[Issue],
    pulls: list[PullRequest],
) -> int:
    """Run the pairing post-pass with deduped inputs."""
    # Dedupe by primary key in case the same record came in via both the
    # "just indexed" path and the "load from store for context" path.
    seen_issue_numbers: set[int] = set()
    unique_issues: list[Issue] = []
    for issue in issues:
        if issue.number in seen_issue_numbers:
            continue
        seen_issue_numbers.add(issue.number)
        unique_issues.append(issue)
    seen_pr_numbers: set[int] = set()
    unique_pulls: list[PullRequest] = []
    for pr in pulls:
        if pr.number in seen_pr_numbers:
            continue
        seen_pr_numbers.add(pr.number)
        unique_pulls.append(pr)

    with Store.open(deps.store_path) as store:
        return run_pairing_post_pass(store, deps.github, unique_issues, unique_pulls)


def _load_all_issues_from_store(store_path: Path) -> list[Issue]:
    """Re-hydrate Issue objects from the store for the pairing pass.

    On an incremental run, the pairing pass needs to consider all stored
    issues (not just the ones in this delta) so a late-arriving PR can
    pair against an old issue. We load minimal fields — closed_at,
    updated_at, assignees — that the temporal heuristic actually reads.
    """
    import json

    from codescribe_train.rag.sources.github_source import Issue

    out: list[Issue] = []
    with Store.open(store_path) as store:
        rows = store.conn.execute(
            "SELECT issue_number, title, body, state, state_reason, "
            "labels, assignees, created_at, updated_at, closed_at, raw_json "
            "FROM issues"
        ).fetchall()
    for row in rows:
        (
            number,
            title,
            body,
            state,
            state_reason,
            labels_json,
            assignees_json,
            created_at,
            updated_at,
            closed_at,
            raw_json,
        ) = row
        out.append(
            Issue(
                number=number,
                title=title or "",
                body=body,
                state=state or "",
                state_reason=state_reason,
                labels=json.loads(labels_json) if labels_json else [],
                assignees=json.loads(assignees_json) if assignees_json else [],
                author=None,
                created_at=created_at or "",
                updated_at=updated_at or "",
                closed_at=closed_at,
                raw_json=raw_json or "",
            )
        )
    return out


def _load_all_pulls_from_store(store_path: Path) -> list[PullRequest]:
    """Re-hydrate PullRequest objects from the store for the pairing pass.

    We do NOT decompress diff_text — pairing only needs metadata. The
    ``draft`` flag is not in the store schema, so we conservatively set
    it to False; the pairing functions already dedupe with the in-memory
    just-indexed-this-run pulls (which carry the real ``draft`` flag).
    """
    from codescribe_train.rag.sources.github_source import PullRequest

    out: list[PullRequest] = []
    with Store.open(store_path) as store:
        rows = store.conn.execute(
            "SELECT pr_number, title, body, state, head_sha, base_branch, "
            "merged_at, author, raw_json FROM pulls"
        ).fetchall()
    for row in rows:
        (
            number,
            title,
            body,
            state,
            head_sha,
            base_branch,
            merged_at,
            author,
            raw_json,
        ) = row
        out.append(
            PullRequest(
                number=number,
                title=title or "",
                body=body,
                state=state or "",
                head_sha=head_sha,
                base_branch=base_branch,
                author=author,
                draft=False,
                created_at="",
                updated_at="",
                merged_at=merged_at,
                closed_at=None,
                raw_json=raw_json or "",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Watermark helpers
# ---------------------------------------------------------------------------


def epoch_datetime() -> datetime:
    """The "indexed nothing yet" sentinel for the GH watermarks."""
    return datetime(1970, 1, 1, tzinfo=UTC)


def parse_watermark(value: str | None) -> datetime:
    """Parse a stored ISO-8601 ``updated_at``; epoch on missing/empty."""
    if not value:
        return epoch_datetime()
    from dateutil.parser import isoparse

    parsed = isoparse(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def apply_safety_margin(ts: datetime) -> datetime:
    """Subtract :data:`WATERMARK_SAFETY_MARGIN_SECONDS` from ``ts``.

    the watermark is stored at the max ``updated_at`` of the
    last indexed batch, but a record updated within the same second could
    be missed across run boundaries. Subtract 1 s when *reading* the
    watermark back to widen the next-run query window.
    """
    return ts - timedelta(seconds=WATERMARK_SAFETY_MARGIN_SECONDS)


# ---------------------------------------------------------------------------
# Refresh helpers (CLI --refresh-pr / --refresh-issue)
# ---------------------------------------------------------------------------


async def refresh_pr(deps: IndexerDeps, pr_number: int) -> bool:
    """Re-index a single PR by number. Returns True if it was found."""
    loop = asyncio.get_running_loop()

    def _pull_matching() -> PullRequest | None:
        for pr in deps.github.iter_pulls_changed_since(epoch_datetime()):
            if pr.number == pr_number:
                return pr
        return None

    pr = await loop.run_in_executor(None, _pull_matching)
    if pr is None:
        return False
    await loop.run_in_executor(
        None, _index_pulls_batch_in_store, deps, [pr]
    )
    return True


async def refresh_issue(deps: IndexerDeps, issue_number: int) -> bool:
    """Re-index a single issue by number. Returns True if it was found."""
    loop = asyncio.get_running_loop()

    def _issue_matching() -> Issue | None:
        for issue in deps.github.iter_issues_changed_since(epoch_datetime()):
            if issue.number == issue_number:
                return issue
        return None

    issue = await loop.run_in_executor(None, _issue_matching)
    if issue is None:
        return False
    await loop.run_in_executor(
        None, _index_issues_batch_in_store, deps, [issue]
    )
    return True
