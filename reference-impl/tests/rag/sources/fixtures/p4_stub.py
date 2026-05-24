#!/usr/bin/env python3
"""Tiny stub of the ``p4`` CLI for tests.

Supports the subset the client invokes: ``p4 -G changes ...@<low>,#head``,
``p4 -G describe -s <cl>``, ``p4 describe -du <cl>``.

Unlike a naive stub, ``changes`` parses the ``@<low>,#head`` floor and returns
DESC order (like real ``p4 changes``), so ``P4Source.iter_changes_since`` — which
reverses each page to ascend and advances its cursor — terminates correctly.
"""
from __future__ import annotations

import marshal
import sys

_ALL_CHANGES = [
    {b"change": b"1001", b"user": b"alice", b"time": b"1735689600"},
    {b"change": b"1002", b"user": b"bob", b"time": b"1735776000"},
]


def _emit_marshal(records) -> None:
    for r in records:
        sys.stdout.buffer.write(marshal.dumps(r))
    sys.stdout.buffer.flush()


def _floor_from_args(args) -> int:
    for a in args:
        if "@" in a and ",#head" in a:
            try:
                return int(a.split("@", 1)[1].split(",", 1)[0])
            except ValueError:
                return 0
    return 0


def main() -> int:
    args = sys.argv[1:]
    use_marshal = "-G" in args

    if "changes" in args and use_marshal:
        low = _floor_from_args(args)
        picked = [c for c in _ALL_CHANGES if int(c[b"change"]) >= low]
        picked.sort(key=lambda c: int(c[b"change"]), reverse=True)  # DESC like real p4
        _emit_marshal(picked)
        return 0

    if "describe" in args:
        cl = args[-1]
        if "-du" in args:
            sys.stdout.write(
                f"Change {cl} by alice on 2026-01-01 12:00:00\n\n"
                f"\tFix NPE on startup\n\n"
                f"Affected files ...\n"
                f"... //depot/main/src/Foo.java#5 edit\n\n"
                f"Differences ...\n"
                f"==== //depot/main/src/Foo.java#5 (text) ====\n"
                f"@@ -10,3 +10,5 @@\n"
                f"-    return parse(read());\n"
                f"+    if (!exists()) return defaults();\n"
                f"+    return parse(read());\n"
            )
            return 0
        if use_marshal:
            _emit_marshal([{
                b"change": cl.encode(),
                b"user": b"alice",
                b"time": b"1735689600",
                b"desc": b"Fix NPE on startup",
                b"depotFile0": b"//depot/main/src/Foo.java",
                b"action0": b"edit",
                b"rev0": b"5",
            }])
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
