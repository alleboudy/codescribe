"""Static check: no unsafe binary-serialisation modules under rag/store/.

Acceptance: spec test #7 from the design spec — "grep store/ for unsafe
binary-serialisation module imports → 0 hits". Backed by
``codescribe_train/rag/store/AGENTS.md``'s anti-pattern rule
("Do NOT use Python's unsafe binary serialisation modules for any stored
field"). JSON + gzip is the only persistence vocabulary the store may use.
"""

from __future__ import annotations

from pathlib import Path

STORE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "codescribe_train" / "rag" / "store"

# Modules forbidden from any rag/store/ source file. Each is built into
# the stdlib; ``shelve`` is included because it wraps the first one and
# imports it transitively at the C level.
FORBIDDEN_MODULES: tuple[str, ...] = (
    # The two unsafe binary-serialisation stdlib modules from the AGENTS.md
    # rule. The literal names are spelled with str.swapcase() / split tricks
    # only so this test file itself does not trip a grep-against-self.
    "pi" + "ckle",
    "ma" + "rshal",
    "sh" + "elve",
)


def _python_files() -> list[Path]:
    return sorted(STORE_DIR.rglob("*.py"))


def test_store_dir_exists() -> None:
    assert STORE_DIR.is_dir(), f"expected store dir at {STORE_DIR}"


def test_no_forbidden_module_imports() -> None:
    hits: list[tuple[Path, int, str]] = []
    for path in _python_files():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.lstrip()
            # Only check actual import statements; comments / docstrings
            # that mention the module name (as in writer.py's AGENTS-rule
            # citation) are allowed.
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            for mod in FORBIDDEN_MODULES:
                if (
                    f"import {mod}" in stripped
                    or f"from {mod}" in stripped
                ):
                    hits.append((path, line_no, line.rstrip()))

    assert hits == [], (
        "unsafe binary-serialisation imports under rag/store/ — JSON+gzip only:\n"
        + "\n".join(f"  {path}:{lineno}: {line}" for path, lineno, line in hits)
    )
