from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .config import RagConfig
from .embed.embedder import make_embedder
from .pipelines.bootstrap import run_bootstrap
from .pipelines.incremental import read_watermarks, run_incremental
from .sources._ratelimit import TokenBucket
from .sources.bugzilla import BugzillaSource
from .sources.perforce import P4Source
from .store.retrieve import Retriever
from .store.writer import Store


class _LiveSources:
    """Adapts the real Perforce/Bugzilla clients to the pipeline's source API."""

    def __init__(self, cfg: RagConfig, since_cl: int, since_modtime: datetime) -> None:
        rl = TokenBucket(cfg.rate_limit_rps)
        self._p4 = P4Source(cfg.perforce.p4port, cfg.perforce.p4user,
                            cfg.perforce.ticket_path, rl, binary=cfg.perforce.binary)
        key = cfg.bugzilla.api_key_path.read_text().strip()
        self._bz = BugzillaSource(cfg.bugzilla.base_url, key, rl, page_size=cfg.bugzilla.page_size)
        self._depot = cfg.perforce.depot_path
        self._since_cl = since_cl
        self._since_modtime = since_modtime

    def iter_bugs(self):
        yield from self._bz.iter_bugs_changed_since(self._since_modtime)

    def comments_for(self, bug_id):
        return self._bz.get_comments(bug_id)

    def iter_cls(self):
        yield from self._p4.iter_changes_since(self._since_cl, self._depot)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="codescribe_rag.rag")
    p.add_argument("--config", default="configs/rag.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index")
    ix.add_argument("--bootstrap", action="store_true")
    ix.add_argument("--since-cl", type=int, default=1)
    ix.add_argument("--since-bug-modtime", default=None)
    q = sub.add_parser("query")
    q.add_argument("--query", required=True)
    q.add_argument("--k", type=int, default=5)
    sub.add_parser("status")
    args = p.parse_args(argv)

    cfg = RagConfig.from_yaml(Path(args.config))
    store = Store.open(cfg.store.db_path)
    embedder = make_embedder(cfg.embed)

    if args.cmd == "index":
        if args.bootstrap:
            since_cl = args.since_cl
            since = (datetime.fromisoformat(args.since_bug_modtime).replace(tzinfo=timezone.utc)
                     if args.since_bug_modtime else datetime(1970, 1, 1, tzinfo=timezone.utc))
        else:
            since_cl, since = read_watermarks(store)
        src = _LiveSources(cfg, since_cl, since)
        runner = run_bootstrap if args.bootstrap else run_incremental
        print(runner(cfg, src, store, embedder))
    elif args.cmd == "query":
        for res in Retriever(store._conn, embedder).find_similar_bugs(args.query, k=args.k):
            print(f"Bug {res.bug_id} [{res.status}] score={res.score:.3f} "
                  f"fix_cl={res.fix_cl}: {res.summary}")
    else:  # status
        c = store._conn
        print("bugs:", c.execute("SELECT COUNT(*) FROM bugs").fetchone()[0])
        print("changes:", c.execute("SELECT COUNT(*) FROM changes").fetchone()[0])
        print("fix_links:", c.execute("SELECT COUNT(*) FROM fix_links").fetchone()[0])
        print("last_seen_cl:", store.get_state("last_seen_cl"))
        print("last_seen_bug_modtime:", store.get_state("last_seen_bug_modtime"))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
