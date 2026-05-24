from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from ..config import RagConfig
from ._ratelimit import TokenBucket
from .bugzilla import BugzillaSource
from .perforce import P4Source


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="codescribe_rag.rag.sources")
    p.add_argument("--config", default="configs/rag.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("p4-probe")
    pp.add_argument("--last-cl", type=int, default=0)
    bp = sub.add_parser("bz-probe")
    bp.add_argument("--since", default="2026-01-01")
    args = p.parse_args(argv)

    cfg = RagConfig.from_yaml(Path(args.config))
    rl = TokenBucket(cfg.rate_limit_rps)

    if args.cmd == "p4-probe":
        src = P4Source(cfg.perforce.p4port, cfg.perforce.p4user, cfg.perforce.ticket_path,
                       rl, binary=cfg.perforce.binary)
        for i, cl in enumerate(src.iter_changes_since(args.last_cl, cfg.perforce.depot_path)):
            print(f"CL {cl.cl} by {cl.author}: {cl.description[:60]}")
            if i >= 4:
                break
    else:
        if urlparse(cfg.bugzilla.base_url).hostname != cfg.bugzilla.allowlist_hostname:
            print("egress allowlist violation", file=sys.stderr)
            return 2
        key = cfg.bugzilla.api_key_path.read_text().strip()
        src = BugzillaSource(cfg.bugzilla.base_url, key, rl, page_size=cfg.bugzilla.page_size)
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        for i, bug in enumerate(src.iter_bugs_changed_since(since)):
            print(f"Bug {bug.id} [{bug.status}]: {bug.summary[:60]}")
            if i >= 4:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
