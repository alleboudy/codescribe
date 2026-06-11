"""Operator-verified probe CLI for the sources package.

Three subcommands map directly to the operator-verified gates in
the design spec:

* ``git-probe --repo <p> [--last-sha <s>]`` — list commits enumerated by
  :class:`codescribe_train.rag.sources.git_source.GitSource`.
* ``gh-probe --owner <o> --repo <r> --since <YYYY-MM-DD>`` — list issues
  and PRs that have changed since the watermark.
* ``gh-closing-refs --owner <o> --repo <r>`` — list every closing reference
  edge from the GraphQL surface.

Output is human-readable; the indexer pipeline consumes the same
source methods directly, never via this CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from .git_source import GitSource
from .github_source import GitHubSource

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m codescribe_train.rag.sources",
        description="Operator probes for the read-only Git/GitHub source clients.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_git = sub.add_parser("git-probe", help="list commits since --last-sha")
    p_git.add_argument("--repo", type=Path, required=True, help="path to git checkout")
    p_git.add_argument(
        "--last-sha",
        type=str,
        default=None,
        help="exclusive watermark sha; default = full history",
    )

    p_gh = sub.add_parser("gh-probe", help="list issues + PRs changed since --since")
    p_gh.add_argument("--owner", type=str, required=True)
    p_gh.add_argument("--repo", type=str, required=True)
    p_gh.add_argument(
        "--since",
        type=str,
        required=True,
        help="ISO-8601 date or datetime (e.g. 2026-01-01)",
    )

    p_refs = sub.add_parser(
        "gh-closing-refs",
        help="list (pr_number, issue_number) pairs via GraphQL",
    )
    p_refs.add_argument("--owner", type=str, required=True)
    p_refs.add_argument("--repo", type=str, required=True)

    return parser


def _parse_since(raw: str) -> datetime:
    """Parse ``--since`` as an ISO-8601 date or datetime; return UTC-aware."""
    # `datetime.fromisoformat` handles `YYYY-MM-DD` and `YYYY-MM-DDTHH:MM:SS`.
    ts = datetime.fromisoformat(raw)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def _run_git_probe(args: argparse.Namespace) -> int:
    src = GitSource(repo_path=args.repo)
    count = 0
    for commit in src.iter_commits_since(last_sha=args.last_sha):
        pr_marker = f"  (PR #{commit.pr_number})" if commit.pr_number else ""
        sys.stdout.write(
            f"{commit.sha[:12]}  {commit.authored_at}  {commit.author}: "
            f"{commit.message}{pr_marker}\n"
        )
        count += 1
    sys.stdout.write(f"\n{count} commit(s) listed.\n")
    return 0


def _run_gh_probe(args: argparse.Namespace) -> int:
    since = _parse_since(args.since)
    src = GitHubSource(owner=args.owner, repo=args.repo)

    issue_count = 0
    sys.stdout.write(f"=== issues updated >= {since.isoformat()} ===\n")
    for issue in src.iter_issues_changed_since(since):
        sys.stdout.write(
            f"#{issue.number}\t{issue.updated_at}\t{issue.title}\n"
        )
        issue_count += 1

    pr_count = 0
    sys.stdout.write(f"\n=== pulls updated >= {since.isoformat()} ===\n")
    for pr in src.iter_pulls_changed_since(since):
        sys.stdout.write(f"#{pr.number}\t{pr.updated_at}\t{pr.title}\n")
        pr_count += 1

    sys.stdout.write(f"\n{issue_count} issue(s), {pr_count} pull(s) listed.\n")
    return 0


def _run_gh_closing_refs(args: argparse.Namespace) -> int:
    src = GitHubSource(owner=args.owner, repo=args.repo)
    pair_count = 0
    for pr_number, issue_number in src.iter_closing_references():
        sys.stdout.write(f"PR #{pr_number}\tcloses issue #{issue_number}\n")
        pair_count += 1
    sys.stdout.write(f"\n{pair_count} closing-reference pair(s) listed.\n")
    return 0


_DISPATCH = {
    "git-probe": _run_git_probe,
    "gh-probe": _run_gh_probe,
    "gh-closing-refs": _run_gh_closing_refs,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    args = _build_parser().parse_args(argv)
    handler = _DISPATCH[args.cmd]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
