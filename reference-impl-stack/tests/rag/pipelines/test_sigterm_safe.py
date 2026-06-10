"""SIGTERM-safe pipeline restart — no duplicate rows.

Acceptance: spec test #5 from the design spec.

Per the implementation notes in the spec:

    "An acceptable simplification: in-process test that signals the
    stop_requested flag mid-batch (bypassing the OS signal handler),
    restarts the loop, and asserts no duplicates."

This file implements the simplification. The pipeline's
:func:`install_sigterm_handler` and :func:`check_stop` paths are still
the same paths the OS handler triggers, so the in-process simulation
covers the same code that an OS SIGTERM would.

The full subprocess shape (Popen, sleep, kill -TERM, restart, count) is
not used in CI because it is sensitive to timing (the fakes have no IO
latency and a fresh-process startup eats more than the indexer's
runtime). The PR body documents the subprocess approach for the
operator who runs the indexer against the live ``example-org/sample``
corpus.
"""

from __future__ import annotations

import asyncio

from codescribe_train.rag.pipelines._common import IndexerDeps
from codescribe_train.rag.pipelines.bootstrap import run_bootstrap
from codescribe_train.rag.pipelines.incremental import run_incremental
from codescribe_train.rag.store.writer import Store
from tests.rag.pipelines.fakes import (
    FakeEmbedder,
    FakeGitHubSource,
    FakeGitSource,
    make_issue,
)


def test_sigterm_mid_run_then_restart_no_duplicates(tmp_path) -> None:
    """Stop mid-bootstrap, restart with incremental, assert no duplicate rows."""
    issues = [
        make_issue(n, updated_at=f"2026-05-0{n}T10:00:00Z") for n in range(1, 5)
    ]
    gh = FakeGitHubSource(issues=list(issues))
    git = FakeGitSource()

    stop_event = asyncio.Event()
    deps_first = IndexerDeps(
        git=git,
        github=gh,
        embedder=FakeEmbedder(),
        store_path=tmp_path / "rag.db",
        batch_size=1,
        stop_event=stop_event,
        stop_after_records=[2],  # stop after writing 2 issues
    )
    stats = asyncio.run(run_bootstrap(config=None, deps=deps_first))

    assert stats.stopped_early is True
    assert stats.issues == 2

    with Store.open(deps_first.store_path) as store:
        n_after_first = store.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
    assert n_after_first == 2

    # Simulate operator restart — fresh stop_event, no stop_after.
    deps_second = IndexerDeps(
        git=git,
        github=gh,
        embedder=FakeEmbedder(),
        store_path=deps_first.store_path,
    )
    asyncio.run(run_incremental(config=None, deps=deps_second))

    with Store.open(deps_first.store_path) as store:
        n_after_second = store.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
        numbers = sorted(
            r[0]
            for r in store.conn.execute(
                "SELECT issue_number FROM issues"
            ).fetchall()
        )

    assert n_after_second == 4
    assert numbers == [1, 2, 3, 4]  # no duplicates, no gaps


def test_sigterm_handler_restores_prior_handler(tmp_path) -> None:
    """``install_sigterm_handler`` is a clean-up-on-exit context manager.

    Tests run in arbitrary order; an installed handler that leaks across
    test cases would silently break unrelated tests that send signals
    (none today, but the contract matters).
    """
    import signal

    from codescribe_train.rag.pipelines._common import install_sigterm_handler

    prior = signal.getsignal(signal.SIGTERM)
    stop_event = asyncio.Event()
    with install_sigterm_handler(stop_event):
        installed = signal.getsignal(signal.SIGTERM)
        assert installed is not prior
    after = signal.getsignal(signal.SIGTERM)
    assert after is prior
