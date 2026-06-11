"""Static grep gate: the sources package must remain read-only.

If any of the patterns below ever appear in
``codescribe_train/rag/sources/*.py`` we fail loudly — this is the static check
the spec requires.

Patterns checked (verbatim from the design spec acceptance):

* ``git push``
* ``git commit``
* ``gh issue create``
* ``gh pr create``
* ``httpx.(post|put|patch|delete)``
* ``gh api ... -X POST`` (any HTTP verb implies an explicit method override)
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[3] / "codescribe_train" / "rag" / "sources"

# Each pattern is (label, compiled regex). Labels appear in failure messages.
FORBIDDEN_PATTERNS = [
    ("git push", re.compile(r"\bgit\s+push\b")),
    ("git commit", re.compile(r"\bgit\s+commit\b")),
    ("gh issue create", re.compile(r"\bgh\s+issue\s+create\b")),
    ("gh pr create", re.compile(r"\bgh\s+pr\s+create\b")),
    (
        "httpx mutating verb",
        re.compile(r"\bhttpx\s*\.\s*(?:post|put|patch|delete)\b", re.IGNORECASE),
    ),
    (
        "gh api -X <verb>",
        re.compile(r"\bgh\s+api\b[^\n]*\s-X\s+(?:POST|PUT|PATCH|DELETE)\b", re.IGNORECASE),
    ),
]


def _source_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_no_writes_static_grep() -> None:
    files = _source_files()
    assert files, f"no source files found under {PACKAGE}"

    hits: list[tuple[Path, str, str]] = []
    for path in files:
        text = path.read_text()
        for label, pattern in FORBIDDEN_PATTERNS:
            for match in pattern.finditer(text):
                hits.append((path, label, match.group(0)))

    assert not hits, (
        "forbidden write patterns found in sources package:\n"
        + "\n".join(f"  {p}: {label!r} → {snippet!r}" for p, label, snippet in hits)
    )
