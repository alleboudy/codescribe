"""CLI entrypoint for the rag store package.

Operator-verified post-merge per the design spec acceptance:

    python -m codescribe_train.rag.store init --db indices/rag.db

Initialises a fresh DB by opening it through :meth:`Store.open`, which
loads the sqlite-vec extension, applies the verbatim schema from
``schema.sql``, and sets the WAL + foreign-keys pragmas. Re-running
``init`` against an existing DB is a no-op (the schema is only applied
to a fresh file). The command prints one line of structured-ish status
to stderr and exits 0 on success, non-zero on error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from codescribe_train.rag.store.writer import Store

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m codescribe_train.rag.store")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="initialise (or no-op open) the rag DB")
    p_init.add_argument(
        "--db",
        type=Path,
        required=True,
        help="path to the sqlite-vec DB file (e.g. indices/rag.db)",
    )
    return parser


def _cmd_init(db_path: Path) -> int:
    was_new = not db_path.exists()
    with Store.open(db_path) as store:
        # Touch the connection so the WAL pragma actually applies before
        # exit (the schema was applied inside open() on a fresh DB).
        store.conn.execute("SELECT 1").fetchone()
    state = "created" if was_new else "opened existing"
    logger.info("rag store init: %s db=%s", state, db_path)
    sys.stdout.write(f"rag.store init: {state} db={db_path}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = _build_parser().parse_args(argv)
    if args.cmd == "init":
        return _cmd_init(args.db)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
