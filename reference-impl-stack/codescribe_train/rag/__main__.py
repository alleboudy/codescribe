"""CLI entrypoint for the rag package.

Subcommand wiring:

* ``index`` — bootstrap / incremental / refresh.
* ``status`` — DB summary.
* ``query`` — retrieval CLI. Three sub-actions:
  - ``query issues   --query ... --k N``
  - ``query pr-diff  --pr N``
  - ``query commits  --query ... --k N``

The CLI is intentionally thin — it parses flags, sets up JSON logging
to ``logs/rag-indexer-<unix_ts>.log``, builds the
:class:`IndexerDeps`, and hands off to
:func:`run_bootstrap` / :func:`run_incremental` / :func:`refresh_pr` /
:func:`refresh_issue`. Heavy lifting lives in
``codescribe_train/rag/pipelines/_common.py`` (for ``index``) or
``codescribe_train/rag/store/retrieve.py`` (for ``query``).

The ``query`` subcommand prints human-readable Markdown to stdout. The
"no print to stdout" rule from ``codescribe_train/rag/AGENTS.md`` applies to
the *indexer* pipeline; the query CLI is a human-facing read tool and
stdout is the right channel.

Test-only injection hooks
-------------------------

The env vars ``RAG_TEST_DEPS_FACTORY=module:callable`` (index path) and
``RAG_TEST_RETRIEVER_FACTORY=module:callable`` (query path) override
the default "build live deps from config" path with a test-supplied
factory. The retriever hook's callable takes no arguments and returns
the embedder to use (the store and config are derived from
``RAG_CONFIG`` regardless). The hooks are only honoured when the env
vars are present; production runs never touch them.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codescribe_train.rag.config import RagConfig
    from codescribe_train.rag.pipelines._common import IndexerDeps

logger = logging.getLogger("codescribe_train.rag")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m codescribe_train.rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="run the indexer pipeline")
    p_index.add_argument("--bootstrap", action="store_true")
    p_index.add_argument(
        "--since-sha",
        default=None,
        help="resume bootstrap from this commit SHA (exclusive)",
    )
    p_index.add_argument(
        "--since-issue-updated-at",
        default=None,
        help="ISO-8601; only used with --bootstrap as a resume hint",
    )
    p_index.add_argument("--refresh-pr", type=int, default=None)
    p_index.add_argument("--refresh-issue", type=int, default=None)

    p_query = sub.add_parser("query", help="run the retriever from the CLI")
    q_sub = p_query.add_subparsers(dest="query_cmd", required=True)
    q_issues = q_sub.add_parser(
        "issues", help="hybrid issue search (semantic + BM25 + RRF)",
    )
    q_issues.add_argument("--query", required=True, help="natural-language query")
    q_issues.add_argument("--k", type=int, default=5)
    q_issues.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="confidence floor for attaching the fix PR (default 0.8)",
    )
    q_pr = q_sub.add_parser("pr-diff", help="print one PR's decoded diff")
    q_pr.add_argument("--pr", type=int, required=True, help="PR number")
    q_pr.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="truncate the diff to N chars with a marker (default: no truncation)",
    )
    q_commits = q_sub.add_parser(
        "commits", help="hybrid commit search (semantic + BM25 + RRF)",
    )
    q_commits.add_argument("--query", required=True, help="natural-language query")
    q_commits.add_argument("--k", type=int, default=5)

    sub.add_parser("status", help="print DB + state summary")

    return parser


class _JsonLineFormatter(logging.Formatter):
    """Format each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def _setup_json_logging(cwd: Path) -> Path:
    """Configure root logger to write JSON lines to ``logs/rag-indexer-<ts>.log``."""
    log_dir = cwd / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"rag-indexer-{int(time.time())}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(_JsonLineFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Avoid duplicating handlers if main() is called more than once
    # (e.g. from tests). We tag our handler so subsequent calls replace it.
    for existing in list(root.handlers):
        if getattr(existing, "_rag_json", False):
            root.removeHandler(existing)
    handler._rag_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    return log_path


def _load_config(env: dict[str, str] | None = None) -> RagConfig:
    """Load RagConfig from the path in ``$RAG_CONFIG`` or ``configs/rag.yaml``."""
    from codescribe_train.rag.config import load

    env = env or dict(os.environ)
    cfg_path = env.get("RAG_CONFIG")
    if cfg_path:
        return load(Path(cfg_path))
    # Default to the bundled configs/rag.yaml relative to CWD.
    default = Path.cwd() / "configs" / "rag.yaml"
    if default.exists():
        return load(default)
    # Fall back to all defaults if no config is found — the live binary
    # always ships configs/rag.yaml so this branch is only hit in tests
    # that explicitly want default values.
    from codescribe_train.rag.config import RagConfig

    return RagConfig()


def _resolve_db_path(config: RagConfig) -> Path:
    """Resolve ``config.store.db_path`` against CWD if relative."""
    db_path = Path(config.store.db_path)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _build_deps(config: RagConfig, db_path: Path) -> IndexerDeps:
    """Build live :class:`IndexerDeps` from config, honouring the test hook."""
    from codescribe_train.rag.pipelines._common import IndexerDeps

    factory_spec = os.environ.get("RAG_TEST_DEPS_FACTORY")
    if factory_spec:
        module_name, attr_name = factory_spec.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, attr_name)
        return factory(db_path)

    # Live wiring — lazy imports so `--help` doesn't pull torch.
    from codescribe_train.rag.embed.embedder import Embedder
    from codescribe_train.rag.sources.git_source import GitSource
    from codescribe_train.rag.sources.github_source import GitHubSource
    from codescribe_train.rag.sources.worktree_docs_source import WorktreeDocsSource

    git = GitSource(config.sources.git.repo_path)
    github = GitHubSource(
        owner=config.sources.github.owner,
        repo=config.sources.github.repo,
        rate_limit_rps=config.sources.rate_limit_rps,
    )
    embedder = Embedder(
        model_path=config.embed.model_path,
        device=config.embed.device,
        batch_size=config.embed.batch_size,
        max_length=config.embed.max_length,
    )
    docs = WorktreeDocsSource(config.sources.worktree_docs.repo_path)
    return IndexerDeps(
        git=git,
        github=github,
        embedder=embedder,
        store_path=db_path,
        docs=docs,
    )


def _print_status(db_path: Path) -> None:
    """Print human-readable counts + link-source distribution to stdout."""
    from codescribe_train.rag.store.writer import Store

    with Store.open(db_path) as store:
        n_issues = store.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
        n_pulls = store.conn.execute("SELECT COUNT(*) FROM pulls").fetchone()[0]
        n_commits = store.conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
        n_links = store.conn.execute(
            "SELECT COUNT(*) FROM issue_pr_links"
        ).fetchone()[0]
        n_doc_chunks = store.conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
        link_dist = store.conn.execute(
            "SELECT source, COUNT(*) FROM issue_pr_links GROUP BY source"
        ).fetchall()
        last_sha = store.get_state("last_seen_commit_sha")
        last_issue_at = store.get_state("last_seen_issue_updated_at")

    print(f"db_path: {db_path}")
    print(f"issues:  {n_issues}")
    print(f"pulls:   {n_pulls}")
    print(f"commits: {n_commits}")
    print(f"links:   {n_links}")
    print(f"doc_chunks: {n_doc_chunks}")
    if link_dist:
        print("link-source distribution:")
        for source, count in sorted(link_dist):
            print(f"  {source}: {count}")
    else:
        print("link-source distribution: (none)")
    print(f"last_seen_commit_sha:        {last_sha or '(unset)'}")
    print(f"last_seen_issue_updated_at:  {last_issue_at or '(unset)'}")


def _build_retriever_embedder(config: RagConfig):
    """Build the :class:`Embedder` used by the ``query`` CLI subcommands.

    Honours the ``RAG_TEST_RETRIEVER_FACTORY=module:callable`` env hook
    so tests can swap in a stub embedder; otherwise builds a live
    :class:`~codescribe_train.rag.embed.embedder.Embedder` from the config
    just like the indexer does.
    """
    factory_spec = os.environ.get("RAG_TEST_RETRIEVER_FACTORY")
    if factory_spec:
        module_name, attr_name = factory_spec.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)()

    # Live wiring — lazy import so `--help` doesn't pull torch.
    from codescribe_train.rag.embed.embedder import Embedder

    return Embedder(
        model_path=config.embed.model_path,
        device=config.embed.device,
        batch_size=config.embed.batch_size,
        max_length=config.embed.max_length,
    )


def _format_issues(hits) -> str:  # noqa: ANN001 — TYPE_CHECKING import dance
    """Render a list of :class:`RetrievedIssue` as Markdown for stdout."""
    if not hits:
        return "_no issues matched._\n"
    parts: list[str] = []
    for h in hits:
        parts.append(f"## #{h.issue_number} — {h.title}")
        parts.append(f"*state: {h.state}; score: {h.score:.4f}*")
        if h.fix_pr_number is not None:
            parts.append(f"**fix PR:** #{h.fix_pr_number} (confidence {h.confidence:.2f})")
            if h.fix_excerpt:
                parts.append("```diff")
                parts.append(h.fix_excerpt)
                parts.append("```")
        if h.body_excerpt:
            parts.append("> " + h.body_excerpt.replace("\n", "\n> "))
        parts.append("")
    return "\n".join(parts)


def _format_pr_diff(out) -> str:  # noqa: ANN001 — TYPE_CHECKING import dance
    """Render a :class:`PRDiff` as Markdown for stdout."""
    header = (
        f"# PR #{out.pr_number} — {out.title}\n"
        f"*author: {out.author or '(unknown)'}; "
        f"merged: {out.merged_at or '(unmerged)'}; files: {out.files}*\n"
    )
    if out.truncated:
        header += "*(diff truncated)*\n"
    return f"{header}\n```diff\n{out.diff_text}\n```\n"


def _format_commits(hits) -> str:  # noqa: ANN001 — TYPE_CHECKING import dance
    """Render a list of :class:`RetrievedCommit` as Markdown for stdout."""
    if not hits:
        return "_no commits matched._\n"
    parts: list[str] = []
    for h in hits:
        parts.append(f"## {h.sha[:12]} — {h.message}")
        parts.append(
            f"*author: {h.author or '(unknown)'}; "
            f"authored: {h.authored_at or '(unknown)'}; "
            f"files: {h.files}; score: {h.score:.4f}*",
        )
        if h.diff_excerpt:
            parts.append("```diff")
            parts.append(h.diff_excerpt)
            parts.append("```")
        parts.append("")
    return "\n".join(parts)


def _dispatch_query(
    args: argparse.Namespace, config: RagConfig, db_path: Path,
) -> int:
    """Run the right ``query`` sub-action. Returns the exit code."""
    from codescribe_train.rag.store.retrieve import Retriever
    from codescribe_train.rag.store.types import RetrieveConfig
    from codescribe_train.rag.store.writer import Store

    with Store.open(db_path) as store:
        if args.query_cmd == "pr-diff":
            # pr-diff doesn't need an embedder; skip the load cost.
            retriever = Retriever(store=store, embedder=None, config=RetrieveConfig())
            out = retriever.get_pr_diff(pr_number=args.pr, max_chars=args.max_chars)
            print(_format_pr_diff(out))
            return 0

        embedder = _build_retriever_embedder(config)
        retriever = Retriever(store=store, embedder=embedder, config=RetrieveConfig())
        if args.query_cmd == "issues":
            hits = retriever.find_similar_issues(
                query=args.query, k=args.k, min_confidence=args.min_confidence,
            )
            print(_format_issues(hits))
            return 0
        if args.query_cmd == "commits":
            hits = retriever.search_commits(query=args.query, k=args.k)
            print(_format_commits(hits))
            return 0

    raise ValueError(f"unknown query subcommand {args.query_cmd!r}")


def _dispatch_index(args: argparse.Namespace, config: RagConfig, deps: IndexerDeps) -> int:
    """Run the right ``index`` flavour. Returns the exit code."""
    from codescribe_train.rag.pipelines._common import (
        install_sigterm_handler,
        refresh_issue,
        refresh_pr,
    )

    if args.refresh_pr is not None:
        found = asyncio.run(refresh_pr(deps, args.refresh_pr))
        if not found:
            logger.warning("refresh-pr %d: PR not found in source", args.refresh_pr)
            return 1
        return 0
    if args.refresh_issue is not None:
        found = asyncio.run(refresh_issue(deps, args.refresh_issue))
        if not found:
            logger.warning(
                "refresh-issue %d: issue not found in source", args.refresh_issue,
            )
            return 1
        return 0

    stop_event = asyncio.Event()
    deps.stop_event = stop_event
    with install_sigterm_handler(stop_event):
        if args.bootstrap:
            from codescribe_train.rag.pipelines.bootstrap import run_bootstrap

            since_issues = None
            if args.since_issue_updated_at:
                from dateutil.parser import isoparse

                since_issues = isoparse(args.since_issue_updated_at)
            stats = asyncio.run(
                run_bootstrap(
                    config=config,
                    deps=deps,
                    since_sha=args.since_sha,
                    since_issue_updated_at=since_issues,
                )
            )
            logger.info(
                "bootstrap stats commits=%d issues=%d pulls=%d links=%d",
                stats.commits, stats.issues, stats.pulls, stats.links,
            )
        else:
            from codescribe_train.rag.pipelines.incremental import run_incremental

            stats = asyncio.run(run_incremental(config=config, deps=deps))
            logger.info(
                "incremental stats commits=%d issues=%d pulls=%d links=%d",
                stats.commits, stats.issues, stats.pulls, stats.links,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cwd = Path.cwd()
    log_path = _setup_json_logging(cwd)
    logger.info("rag CLI start cmd=%s log=%s", args.cmd, log_path)

    try:
        config = _load_config()
        db_path = _resolve_db_path(config)

        if args.cmd == "status":
            _print_status(db_path)
            return 0

        if args.cmd == "query":
            return _dispatch_query(args, config, db_path)

        # args.cmd == "index"
        deps = _build_deps(config, db_path)
        return _dispatch_index(args, config, deps)
    except Exception:
        logger.exception("rag CLI failed cmd=%s", args.cmd)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
