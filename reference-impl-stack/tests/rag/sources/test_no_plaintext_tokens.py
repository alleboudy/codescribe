"""Static gate: the sources package must not embed plaintext credentials.

We check for two shapes the spec calls out:

* JWT-looking strings: ``xxx.yyy.zzz`` triplets with base64url segments.
* GitHub-token-looking strings: 40-character lowercase hex blobs (the legacy
  PAT shape) or ``ghp_`` / ``gho_`` / ``ghu_`` / ``ghs_`` / ``ghr_``
  prefixes followed by 36+ base62 characters.
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[3] / "codescribe_train" / "rag" / "sources"

# JWT — three dot-separated base64url segments with at least 8 chars each so
# we don't false-positive on dotted Python identifiers.
JWT_RE = re.compile(r"\b[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
# Legacy PAT — 40 hex characters with word boundaries. The detector is
# deliberately liberal; real false-positives are still impossible to write
# accidentally.
HEX40_RE = re.compile(r"\b[0-9a-f]{40}\b")
# Modern GitHub token prefixes.
GH_TOKEN_RE = re.compile(r"\bgh[opusr]_[A-Za-z0-9]{36,}\b")


def _source_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_no_plaintext_tokens() -> None:
    offenders: list[tuple[Path, int, str, str]] = []
    for path in _source_files():
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            for label, pattern in (
                ("JWT-shaped", JWT_RE),
                ("hex40-shaped", HEX40_RE),
                ("gh-token-shaped", GH_TOKEN_RE),
            ):
                for match in pattern.finditer(line):
                    offenders.append((path, i, label, match.group(0)))

    assert not offenders, (
        "plaintext-token-shaped strings found in sources package:\n"
        + "\n".join(
            f"  {p}:{i}: {label} → {snippet!r}" for p, i, label, snippet in offenders
        )
    )
