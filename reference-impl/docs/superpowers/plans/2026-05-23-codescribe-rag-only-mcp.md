# codescribe_rag — RAG-only MCP server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `codescribe_rag` — a read-only Perforce+Bugzilla indexer that builds a local sqlite-vec+FTS5 store of bug↔fix pairs and serves them to VS Code GitHub Copilot Chat through an MCP server exposing `find_similar_bugs` and `get_fix_diff`.

**Architecture:** sources (p4 subprocess + Bugzilla REST) → extract/pair (regex + confidence) → embed+chunk → store (one `indices/rag.db`) → hybrid retriever (vec+BM25+RRF) → MCP stdio server. Strictly read-only; credentials from files; MCP stdout reserved for JSON-RPC.

**Tech Stack:** Python 3.12, uv, pydantic v2, httpx, tenacity, sqlite-vec, FTS5, sentence-transformers (bge-large) with a numpy-only HashingEmbedder fallback for tests, the `mcp` Python SDK, pytest.

**Working dir for all paths below:** `reference-impl/` (a subdir of the `codescribe` repo). All `python`/`pytest` runs assume `uv run` from that dir. Commits are scoped to `reference-impl/`.

**Spec:** `docs/superpowers/specs/2026-05-23-codescribe-rag-only-mcp-design.md`. Source issues: #2 §6 (AGENTS.md verbatim), #4 §8–§13, #5 §1–§8/§11.

**Conventions every module follows:** `from __future__ import annotations` first line; `logger = logging.getLogger(__name__)`; no `print()` in library code; `pathlib.Path` for paths; timezone-aware UTC datetimes; `@dataclass(frozen=True, slots=True)` for value types; `subprocess.run(..., check=True)` never `shell=True`.

---

## Task 0.1: Project scaffold (Phase 0)

**Files:**
- Create: `reference-impl/pyproject.toml`
- Create: `reference-impl/codescribe_rag/__init__.py`, `reference-impl/codescribe_rag/rag/__init__.py`, `.../rag/sources/__init__.py`, `.../rag/extract/__init__.py`, `.../rag/store/__init__.py`, `.../rag/embed/__init__.py`, `.../rag/pipelines/__init__.py`, `.../codescribe_rag/servers/__init__.py`, `.../servers/rag_server/__init__.py`
- Create: `reference-impl/tests/__init__.py` and empty `__init__.py` under each `tests/...` dir
- Create: AGENTS.md ×6, `.github/copilot-instructions.md`
- Test: `reference-impl/tests/test_smoke_imports.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "codescribe-rag"
version = "0.1.0"
description = "RAG-only Perforce+Bugzilla MCP server for Copilot Chat (codescribe issue #6)"
requires-python = ">=3.12"
dependencies = [
  "httpx>=0.27",
  "pydantic>=2.6",
  "tenacity>=8.0",
  "pyyaml>=6.0",
  "python-dateutil>=2.9",
  "pathspec>=0.12",
  "numpy<2.0",
]

[project.optional-dependencies]
rag = [
  "sqlite-vec>=0.1.6",
  "sentence-transformers>=3.0",
]
mcp = [
  "mcp>=1.0,<2.0",
]
dev = [
  "pytest>=8.0",
  "pytest-httpx>=0.30",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["codescribe_rag"]

[tool.pytest.ini_options]
markers = [
  "slow: long-running (latency/perf) tests; run with -m slow",
]
filterwarnings = ["ignore::DeprecationWarning"]
```

- [ ] **Step 2: Create every package `__init__.py` empty** (the 9 package dirs + the test dirs). Each file is literally empty.

- [ ] **Step 3: Write the 6 AGENTS.md files verbatim from the issues, substituting `<project>`→`codescribe_rag`.**

`reference-impl/AGENTS.md` (repo-root, RAG-only trim of #2 §6 cross-cutting):
```
# codescribe_rag — RAG-only stack

Read-only indexer (Perforce + Bugzilla) → sqlite-vec/FTS5 store → MCP server for Copilot Chat.

## Cross-cutting hard rules
- Python 3.12; `uv` for deps; type hints throughout.
- No cloud calls except the configured Perforce + Bugzilla hosts. No telemetry, no wandb.
- Network defaults to 127.0.0.1. No `0.0.0.0` defaults.
- No `print(...)` in library code — logger only. CLI entry points may write stdout/stderr.
- Credentials live in files (`~/.p4tickets`, `~/.config/codescribe_rag/bugzilla.key`), never inline, never logged.
```

`reference-impl/codescribe_rag/rag/AGENTS.md` ← #4 §11 verbatim block (the `# rag — package rules` text).
`reference-impl/codescribe_rag/rag/sources/AGENTS.md` ← #4 §8 verbatim (`# rag.sources — package rules`).
`reference-impl/codescribe_rag/rag/extract/AGENTS.md` ← #4 §9 verbatim (`# rag.extract — package rules`).
`reference-impl/codescribe_rag/rag/store/AGENTS.md` ← #4 §10 verbatim (`# rag.store — package rules`).
`reference-impl/codescribe_rag/servers/rag_server/AGENTS.md` ← #4 §13.5 verbatim (`# rag_server — MCP server rules`).

`reference-impl/.github/copilot-instructions.md` ← #2 §6 verbatim (`# Copilot global instructions`), with the "Where to look for context" paths pointed at `codescribe_rag/`.

> Note: `reference-impl/.github/workflows/` is NOT auto-run by GitHub Actions (Actions only reads the repo-root `.github/`). Provide `reference-impl/.github/workflows/ci.yml` as illustrative (runs `uv sync --extra dev --extra rag && uv run pytest -m "not slow"`), and document the real local command in the README.

- [ ] **Step 4: Write the smoke test**

```python
# reference-impl/tests/test_smoke_imports.py
import importlib

import pytest


@pytest.mark.parametrize("mod", [
    "codescribe_rag",
    "codescribe_rag.rag",
    "codescribe_rag.rag.sources",
    "codescribe_rag.rag.extract",
    "codescribe_rag.rag.store",
    "codescribe_rag.rag.embed",
    "codescribe_rag.rag.pipelines",
    "codescribe_rag.servers",
    "codescribe_rag.servers.rag_server",
])
def test_package_imports(mod):
    importlib.import_module(mod)
```

- [ ] **Step 5: Sync and run**

Run: `cd reference-impl && uv sync --extra dev && uv run pytest tests/test_smoke_imports.py -v`
Expected: PASS (9 params).

- [ ] **Step 6: Commit**

```bash
git add reference-impl/pyproject.toml reference-impl/codescribe_rag reference-impl/tests reference-impl/AGENTS.md reference-impl/.github
git commit -m "feat(scaffold): codescribe_rag package tree, AGENTS.md, pyproject"
```

---

## Task 0.2: Config models (`rag/config.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/config.py`
- Create: `reference-impl/configs/rag.yaml`
- Test: `reference-impl/tests/rag/test_config.py`

- [ ] **Step 1: Write `configs/rag.yaml`** — verbatim from spec §5 (the full YAML block).

- [ ] **Step 2: Write the failing test**

```python
# reference-impl/tests/rag/test_config.py
from __future__ import annotations

from pathlib import Path

import pytest

from codescribe_rag.rag.config import RagConfig

CONFIG = Path(__file__).parents[2] / "configs" / "rag.yaml"


def test_loads_example_yaml():
    cfg = RagConfig.from_yaml(CONFIG)
    assert cfg.perforce.p4port.startswith("ssl:")
    assert cfg.bugzilla.allowlist_hostname == "bugzilla.corp.example.com"
    assert cfg.embed.dim == 1024
    assert cfg.pairing.threshold == 0.8
    assert cfg.retrieve.rrf_constant == 60
    assert cfg.rate_limit_rps == 5.0


def test_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("perforce:\n  p4port: x\n  nonsense_key: 1\n")
    with pytest.raises(Exception):
        RagConfig.from_yaml(bad)


def test_expands_user_paths():
    cfg = RagConfig.from_yaml(CONFIG)
    assert not str(cfg.bugzilla.api_key_path).startswith("~")
    assert not str(cfg.perforce.ticket_path).startswith("~")
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd reference-impl && uv run pytest tests/rag/test_config.py -v`
Expected: FAIL (ModuleNotFoundError: codescribe_rag.rag.config).

- [ ] **Step 4: Implement `rag/config.py`**

```python
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _expand(p: str) -> Path:
    return Path(p).expanduser()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PerforceConfig(_Strict):
    p4port: str
    p4user: str
    ticket_path: Path = Path("~/.p4tickets")
    depot_path: str = "//depot/main/..."
    binary: str = "p4"
    subprocess_timeout_s: float = 60.0

    @field_validator("ticket_path", mode="before")
    @classmethod
    def _exp(cls, v): return _expand(v)


class BugzillaConfig(_Strict):
    base_url: str
    allowlist_hostname: str
    api_key_path: Path = Path("~/.config/codescribe_rag/bugzilla.key")
    page_size: int = 100
    request_timeout_s: float = 30.0
    connect_timeout_s: float = 5.0

    @field_validator("api_key_path", mode="before")
    @classmethod
    def _exp(cls, v): return _expand(v)


class StoreConfig(_Strict):
    db_path: Path = Path("indices/rag.db")


class EmbedConfig(_Strict):
    backend: str = "sentence-transformers"   # | "hashing"
    model_path: Path = Path("~/.hf-models/bge-large-en-v1.5")
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    dim: int = 1024

    @field_validator("model_path", mode="before")
    @classmethod
    def _exp(cls, v): return _expand(v)


class PairingWeights(_Strict):
    bug_comment_cites_cl: float = 0.5
    cl_desc_cites_bug: float = 0.5
    temporal_proximity: float = 0.3
    assignee_author_match: float = 0.2
    bug_status_fixed: float = 0.1


class PairingConfig(_Strict):
    weights: PairingWeights = Field(default_factory=PairingWeights)
    threshold: float = 0.8
    temporal_signal_days: int = 7
    candidate_window_days: int = 60
    min_bug_id: int = 1000
    min_cl_id: int = 1000


class RetrieveConfig(_Strict):
    k: int = 5
    k_vec: int = 20
    k_bm25: int = 20
    confidence_threshold: float = 0.8
    rrf_constant: int = 60


class RagConfig(_Strict):
    perforce: PerforceConfig
    bugzilla: BugzillaConfig
    store: StoreConfig = Field(default_factory=StoreConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    pairing: PairingConfig = Field(default_factory=PairingConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    rate_limit_rps: float = 5.0

    @classmethod
    def from_yaml(cls, path: Path) -> "RagConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
```

- [ ] **Step 5: Run to verify pass**

Run: `cd reference-impl && uv run pytest tests/rag/test_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add reference-impl/codescribe_rag/rag/config.py reference-impl/configs/rag.yaml reference-impl/tests/rag/test_config.py
git commit -m "feat(rag.config): pydantic RagConfig + rag.yaml"
```

---

## Task 6.1: Token-bucket rate limiter (`rag/sources/_ratelimit.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/sources/_ratelimit.py`
- Test: `reference-impl/tests/rag/sources/test_rate_limit.py`

- [ ] **Step 1: Write the failing test**

```python
# reference-impl/tests/rag/sources/test_rate_limit.py
from __future__ import annotations

import threading
import time

from codescribe_rag.rag.sources._ratelimit import TokenBucket


def test_initial_burst_immediate():
    tb = TokenBucket(rate_per_second=5.0, burst=5)
    start = time.monotonic()
    for _ in range(5):
        tb.acquire()
    assert time.monotonic() - start < 0.1


def test_blocks_until_refilled():
    tb = TokenBucket(rate_per_second=10.0, burst=1)
    tb.acquire()
    start = time.monotonic()
    tb.acquire()
    assert time.monotonic() - start >= 0.08  # ~1/10s, allow scheduler slack


def test_thread_safe_no_overdraw():
    tb = TokenBucket(rate_per_second=50.0, burst=1)
    start = time.monotonic()

    def worker():
        for _ in range(10):
            tb.acquire()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    # 40 acquires at 50/s with burst 1 ≈ ≥ 39/50 s
    assert time.monotonic() - start >= 0.6
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError`).
Run: `cd reference-impl && uv run pytest tests/rag/sources/test_rate_limit.py -v`

- [ ] **Step 3: Implement `_ratelimit.py`** (verbatim #5 §3):

```python
from __future__ import annotations

import threading
import time


class TokenBucket:
    """A simple token-bucket rate limiter. Thread-safe."""

    def __init__(self, rate_per_second: float, burst: int | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._rate = rate_per_second
        self._capacity = float(burst if burst is not None else max(1, int(rate_per_second)))
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        with self._lock:
            while True:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                sleep_for = deficit / self._rate
                self._lock.release()
                try:
                    time.sleep(sleep_for)
                finally:
                    self._lock.acquire()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(rag.sources): token-bucket rate limiter"`

---

## Task 6.2: Perforce client (`rag/sources/perforce.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/sources/perforce.py`
- Create: `reference-impl/tests/rag/sources/fixtures/p4_stub.py`
- Test: `reference-impl/tests/rag/sources/test_perforce_parser.py`

- [ ] **Step 1: Write the p4 stub fixture** (verbatim #5 §11 `p4_stub.py`), `chmod +x` it.

- [ ] **Step 2: Write the failing tests**

```python
# reference-impl/tests/rag/sources/test_perforce_parser.py
from __future__ import annotations

import marshal
import subprocess
import sys
from pathlib import Path

import pytest

from codescribe_rag.rag.sources._ratelimit import TokenBucket
from codescribe_rag.rag.sources.perforce import P4Source, P4AuthExpired, P4Change

STUB = Path(__file__).parent / "fixtures" / "p4_stub.py"


def make_source() -> P4Source:
    # Invoke the stub as the "p4 binary": binary=python, prepend the stub path.
    return P4Source(
        p4port="ssl:stub:1666", p4user="tester", ticket_path=Path("/tmp/none"),
        rate_limiter=TokenBucket(1000.0),
        binary=sys.executable, extra_argv=[str(STUB)],
    )


def test_describe_parses_change():
    src = make_source()
    cl = src.describe(1001)
    assert isinstance(cl, P4Change)
    assert cl.cl == 1001
    assert cl.author == "alice"
    assert cl.files and cl.files[0].depot_path.endswith("Foo.java")
    assert "@@ -10,3 +10,5 @@" in cl.diff_text


def test_iter_changes_ascending():
    src = make_source()
    cls = list(src.iter_changes_since(0))
    nums = [c.cl for c in cls]
    assert nums == sorted(nums)  # ascending


def test_marshal_loop_parses_multiple_records():
    buf = marshal.dumps({b"change": b"1", b"user": b"a"}) + marshal.dumps({b"change": b"2", b"user": b"b"})
    src = make_source()
    recs = src._decode_marshal(buf)
    assert [r["change"] for r in recs] == ["1", "2"]


def test_auth_expired_raises(monkeypatch):
    src = make_source()
    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, "p4", stderr=b"Your session has expired, please login again.")
    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(P4AuthExpired):
        src.describe(1001)


def test_diff_truncation_detected():
    src = make_source()
    big = "x" * 1_000_001
    assert src._is_truncated(big) is True
    assert src._is_truncated("... truncated") is True
    assert src._is_truncated("small") is False
```

- [ ] **Step 3: Run, expect FAIL.**

- [ ] **Step 4: Implement `perforce.py`** (adapted #5 §1; note the added `extra_argv` so tests can drive the stub, and the extracted `_decode_marshal`/`_is_truncated` helpers the tests call):

```python
from __future__ import annotations

import io
import logging
import marshal
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from ._ratelimit import TokenBucket

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FileChange:
    depot_path: str
    action: str
    rev: int


@dataclass(frozen=True, slots=True)
class P4Change:
    cl: int
    author: str
    submitted_at: datetime
    description: str
    files: tuple[FileChange, ...]
    diff_text: str
    diff_truncated: bool


class P4AuthExpired(RuntimeError):
    """p4 ticket expired; operator must re-run `p4 login`."""


class P4Source:
    """Read-only wrapper over the `p4` CLI (subprocess; no p4python)."""

    def __init__(
        self,
        p4port: str,
        p4user: str,
        ticket_path: Path,
        rate_limiter: TokenBucket,
        binary: str = "p4",
        extra_argv: Sequence[str] | None = None,
        subprocess_timeout_s: float = 60.0,
    ) -> None:
        self._port = p4port
        self._user = p4user
        self._ticket = ticket_path
        self._rate = rate_limiter
        self._binary = binary
        self._extra = list(extra_argv or [])
        self._timeout = subprocess_timeout_s

    def iter_changes_since(
        self, last_cl: int, depot_path: str = "//depot/...", batch_size: int = 200
    ) -> Iterator[P4Change]:
        cursor = last_cl
        while True:
            page = self._list_changes(cursor, depot_path, batch_size)
            if not page:
                return
            for cl_meta in reversed(page):     # p4 returns DESC; ascend
                yield self.describe(int(cl_meta["change"]))
            highest = max(int(m["change"]) for m in page)
            if highest <= cursor:
                return
            cursor = highest

    def describe(self, cl: int) -> P4Change:
        meta = self._run_marshal(["describe", "-s", str(cl)])
        if not meta:
            raise ValueError(f"CL {cl} does not exist")
        meta = meta[0]
        diff_text = self._run_text(["describe", "-du", str(cl)])
        return P4Change(
            cl=int(meta["change"]),
            author=meta.get("user", ""),
            submitted_at=self._parse_p4_time(meta["time"]),
            description=meta.get("desc", ""),
            files=tuple(self._parse_files(meta)),
            diff_text=diff_text,
            diff_truncated=self._is_truncated(diff_text),
        )

    # internals -------------------------------------------------------
    def _list_changes(self, last_cl: int, depot_path: str, batch: int) -> list[dict]:
        return self._run_marshal([
            "changes", "-s", "submitted", "-m", str(batch),
            f"{depot_path}@{last_cl + 1},#head",
        ])

    def _argv(self, marshal_mode: bool, p4_args: Sequence[str]) -> list[str]:
        head = [self._binary, *self._extra, "-p", self._port, "-u", self._user]
        if marshal_mode:
            head.append("-G")
        return [*head, *p4_args]

    def _run_marshal(self, p4_args: Sequence[str]) -> list[dict]:
        self._rate.acquire()
        try:
            completed = subprocess.run(
                self._argv(True, p4_args), env=self._subprocess_env(),
                capture_output=True, check=True, timeout=self._timeout,
            )
        except subprocess.CalledProcessError as exc:
            self._reraise_if_auth(exc)
            raise
        return self._decode_marshal(completed.stdout)

    @staticmethod
    def _decode_marshal(stdout: bytes) -> list[dict]:
        records: list[dict] = []
        buf = io.BytesIO(stdout)
        while True:
            try:
                rec = marshal.load(buf)
            except EOFError:
                break
            records.append({
                (k.decode("utf-8", "replace") if isinstance(k, bytes) else k):
                (v.decode("utf-8", "replace") if isinstance(v, bytes) else v)
                for k, v in rec.items()
            })
        return records

    def _run_text(self, p4_args: Sequence[str]) -> str:
        self._rate.acquire()
        try:
            completed = subprocess.run(
                self._argv(False, p4_args), env=self._subprocess_env(),
                capture_output=True, text=True, errors="replace",
                check=True, timeout=self._timeout,
            )
        except subprocess.CalledProcessError as exc:
            self._reraise_if_auth(exc)
            raise
        return completed.stdout

    def _subprocess_env(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "P4TICKETS": str(self._ticket),
        }

    def _reraise_if_auth(self, exc: subprocess.CalledProcessError) -> None:
        stderr = exc.stderr or b""
        if isinstance(stderr, str):
            stderr = stderr.encode()
        text = stderr.decode("utf-8", "replace")
        if "session has expired" in text.lower() or "P4-AUTH" in text:
            raise P4AuthExpired("p4 ticket expired; run `p4 login`.") from exc

    @staticmethod
    def _is_truncated(diff_text: str) -> bool:
        return "... truncated" in diff_text or len(diff_text) > 1_000_000

    @staticmethod
    def _parse_p4_time(raw: str) -> datetime:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)

    @staticmethod
    def _parse_files(meta: dict) -> Iterator[FileChange]:
        i = 0
        while f"depotFile{i}" in meta:
            yield FileChange(
                depot_path=meta[f"depotFile{i}"],
                action=meta.get(f"action{i}", ""),
                rev=int(meta.get(f"rev{i}", 0)),
            )
            i += 1
```

- [ ] **Step 5: Run, expect PASS** (5 tests).
- [ ] **Step 6: Commit** — `git commit -m "feat(rag.sources): read-only Perforce client + p4 stub"`

---

## Task 6.3: Bugzilla client (`rag/sources/bugzilla.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/sources/bugzilla.py`
- Test: `reference-impl/tests/rag/sources/test_bugzilla_parser.py`

- [ ] **Step 1: Write the failing tests**

```python
# reference-impl/tests/rag/sources/test_bugzilla_parser.py
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from codescribe_rag.rag.sources._ratelimit import TokenBucket
from codescribe_rag.rag.sources.bugzilla import BugzillaSource, Bug


def make_source(handler) -> BugzillaSource:
    src = BugzillaSource(base_url="https://bz.example.com", api_key="stub-key",
                         rate_limiter=TokenBucket(1000.0), page_size=2)
    src._client = httpx.Client(base_url="https://bz.example.com/rest",
                               headers={"X-BUGZILLA-API-KEY": "stub-key", "Accept": "application/json"},
                               transport=httpx.MockTransport(handler))
    return src


def test_iter_bugs_pagination():
    pages = [
        [{"id": i, "summary": f"b{i}", "component": "c", "severity": "major",
          "status": "RESOLVED", "resolution": "FIXED",
          "creation_time": "2026-01-01T00:00:00Z", "last_change_time": f"2026-01-0{i}T00:00:00Z",
          "assigned_to": "alice"} for i in (1, 2)],
        [{"id": 3, "summary": "b3", "component": "c", "severity": "minor",
          "status": "RESOLVED", "resolution": "FIXED",
          "creation_time": "2026-01-01T00:00:00Z", "last_change_time": "2026-01-03T00:00:00Z",
          "assigned_to": "bob"}],
    ]
    calls = {"n": 0}
    def handler(req):
        if req.url.path == "/rest/bug":
            page = pages[min(calls["n"], len(pages) - 1)]
            calls["n"] += 1
            return httpx.Response(200, json={"bugs": page})
        return httpx.Response(404)
    bugs = list(make_source(handler).iter_bugs_changed_since(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    assert [b.id for b in bugs] == [1, 2, 3]
    assert isinstance(bugs[0], Bug)


def test_safety_margin_subtracted():
    seen = {}
    def handler(req):
        seen["lct"] = req.url.params.get("last_change_time")
        return httpx.Response(200, json={"bugs": []})
    list(make_source(handler).iter_bugs_changed_since(datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)))
    assert seen["lct"] == "2026-05-20T11:59:59Z"  # minus 1s


def test_retry_on_5xx_not_on_404():
    state = {"n": 0}
    def handler(req):
        if req.url.path == "/rest/bug":
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"bugs": []})
        return httpx.Response(404)
    src = make_source(handler)
    list(src.iter_bugs_changed_since(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    assert state["n"] == 2  # one retry on 503

    def handler404(req):
        return httpx.Response(404)
    src2 = make_source(handler404)
    with pytest.raises(httpx.HTTPStatusError):
        list(src2.iter_bugs_changed_since(datetime(2026, 1, 1, tzinfo=timezone.utc)))


def test_get_comments():
    def handler(req):
        if req.url.path == "/rest/bug/1/comment":
            return httpx.Response(200, json={"bugs": {"1": {"comments": [
                {"id": 1, "creator": "alice", "text": "Fixed in CL 12345",
                 "creation_time": "2026-01-05T00:00:00Z"}]}}})
        return httpx.Response(404)
    comments = make_source(handler).get_comments(1)
    assert comments[0].text == "Fixed in CL 12345"


def test_parse_iso_z_suffix():
    dt = BugzillaSource._parse_iso("2026-05-20T18:00:00Z")
    assert dt == datetime(2026, 5, 20, 18, 0, 0, tzinfo=timezone.utc)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `bugzilla.py`** (adapted #5 §2):

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Iterator
from urllib.parse import urlparse

import httpx
import tenacity

from ._ratelimit import TokenBucket

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BugComment:
    id: int
    creator: str
    text: str
    creation_time: datetime


@dataclass(frozen=True, slots=True)
class Bug:
    id: int
    summary: str
    description: str
    component: str
    severity: str
    status: str
    resolution: str
    creation_time: datetime
    last_change_time: datetime
    assigned_to: str
    raw_json: dict


def _retryable(e: BaseException) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        return 500 <= e.response.status_code < 600
    return isinstance(e, (httpx.ConnectError, httpx.ReadTimeout))


class BugzillaSource:
    """Read-only Bugzilla REST client."""

    def __init__(self, base_url: str, api_key: str, rate_limiter: TokenBucket,
                 page_size: int = 100, request_timeout_s: float = 30.0,
                 connect_timeout_s: float = 5.0) -> None:
        hostname = urlparse(base_url).hostname
        if not hostname:
            raise ValueError(f"Invalid base_url: {base_url}")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/rest",
            headers={"X-BUGZILLA-API-KEY": api_key, "Accept": "application/json",
                     "User-Agent": "codescribe_rag-indexer/1.0"},
            timeout=httpx.Timeout(request_timeout_s, connect=connect_timeout_s),
        )
        self._rate = rate_limiter
        self._page_size = page_size

    def iter_bugs_changed_since(self, since: datetime) -> Iterator[Bug]:
        watermark = since - timedelta(seconds=1)
        offset = 0
        while True:
            self._rate.acquire()
            resp = self._get_with_retry("/bug", params={
                "last_change_time": watermark.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": self._page_size, "offset": offset,
                "order": "last_change_time ASC",
                "include_fields": "id,summary,component,severity,status,resolution,"
                                  "creation_time,last_change_time,assigned_to,_default",
            })
            bugs = resp.json().get("bugs", [])
            if not bugs:
                return
            for raw in bugs:
                yield self._parse_bug(raw)
            if len(bugs) < self._page_size:
                return
            offset += self._page_size

    def get_comments(self, bug_id: int) -> list[BugComment]:
        self._rate.acquire()
        resp = self._get_with_retry(f"/bug/{bug_id}/comment")
        comments = resp.json().get("bugs", {}).get(str(bug_id), {}).get("comments", [])
        return [BugComment(id=int(c["id"]), creator=c.get("creator", ""), text=c.get("text", ""),
                           creation_time=self._parse_iso(c["creation_time"])) for c in comments]

    @tenacity.retry(stop=tenacity.stop_after_attempt(3),
                    wait=tenacity.wait_exponential(multiplier=1, min=0, max=2),
                    retry=tenacity.retry_if_exception(_retryable), reraise=True)
    def _get_with_retry(self, path: str, params: dict | None = None) -> httpx.Response:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp

    def _parse_bug(self, raw: dict) -> Bug:
        return Bug(
            id=int(raw["id"]), summary=raw.get("summary", ""), description="",
            component=raw.get("component", ""), severity=raw.get("severity", ""),
            status=raw.get("status", ""), resolution=raw.get("resolution", ""),
            creation_time=self._parse_iso(raw["creation_time"]),
            last_change_time=self._parse_iso(raw["last_change_time"]),
            assigned_to=raw.get("assigned_to", ""), raw_json=raw)

    @staticmethod
    def _parse_iso(s: str) -> datetime:
        s = s.rstrip("Z")
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Run, expect PASS** (5 tests).
- [ ] **Step 5: Commit** — `git commit -m "feat(rag.sources): read-only Bugzilla REST client"`

---

## Task 6.4: Source probes CLI + static-safety tests

**Files:**
- Create: `reference-impl/codescribe_rag/rag/sources/__main__.py`
- Test: `reference-impl/tests/rag/sources/test_no_writes.py`, `test_no_credentials_in_source.py`, `test_no_p4python.py`, `test_p4_subprocess_safety.py`, `test_egress_allowlist.py`

- [ ] **Step 1: Write the static-analysis + egress tests**

```python
# reference-impl/tests/rag/sources/test_no_writes.py
from __future__ import annotations
import re
from pathlib import Path

SRC = Path(__file__).parents[3] / "codescribe_rag" / "rag" / "sources"
BANNED = re.compile(r"p4\s+(submit|edit|add|delete|reopen)\b"
                    r"|\.(post|put|patch|delete)\s*\(", re.IGNORECASE)


def test_no_write_operations():
    for py in SRC.rglob("*.py"):
        text = py.read_text()
        hits = [m.group(0) for m in BANNED.finditer(text)]
        assert not hits, f"{py} contains write op(s): {hits}"
```

```python
# reference-impl/tests/rag/sources/test_no_credentials_in_source.py
from __future__ import annotations
import re
from pathlib import Path

SRC = Path(__file__).parents[3] / "codescribe_rag" / "rag" / "sources"
KEYISH = re.compile(r"\b[0-9a-fA-F]{32,}\b|eyJ[A-Za-z0-9_-]{10,}")  # hex keys / JWT


def test_no_inline_credentials():
    for py in SRC.rglob("*.py"):
        assert not KEYISH.search(py.read_text()), f"key-shaped literal in {py}"
```

```python
# reference-impl/tests/rag/sources/test_no_p4python.py
from __future__ import annotations
import sys


def test_p4python_not_imported():
    import codescribe_rag.rag.sources  # noqa: F401
    import codescribe_rag.rag.sources.perforce  # noqa: F401
    assert "P4" not in sys.modules and "p4python" not in sys.modules
```

```python
# reference-impl/tests/rag/sources/test_p4_subprocess_safety.py
from __future__ import annotations
import ast
from pathlib import Path

SRC = Path(__file__).parents[3] / "codescribe_rag" / "rag" / "sources"


def test_no_shell_true():
    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant):
                        assert kw.value.value is not True, f"shell=True in {py}"
```

```python
# reference-impl/tests/rag/sources/test_egress_allowlist.py
from __future__ import annotations
import httpx
from datetime import datetime, timezone

from codescribe_rag.rag.sources._ratelimit import TokenBucket
from codescribe_rag.rag.sources.bugzilla import BugzillaSource


def test_only_configured_host_contacted():
    seen = []
    def handler(req):
        seen.append(req.url.host)
        return httpx.Response(200, json={"bugs": []})
    src = BugzillaSource("https://bz.example.com", "k", TokenBucket(1000.0))
    src._client = httpx.Client(base_url="https://bz.example.com/rest",
                               headers={"X-BUGZILLA-API-KEY": "k"},
                               transport=httpx.MockTransport(handler))
    list(src.iter_bugs_changed_since(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    assert set(seen) == {"bz.example.com"}
```

- [ ] **Step 2: Run, expect FAIL** (`__main__` import in CLI test path — and the static tests should already pass against Tasks 6.2/6.3 code; if any fail, fix the source, not the test).

- [ ] **Step 3: Implement `sources/__main__.py`** (probes; stdout allowed here — CLI entry):

```python
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
    pp = sub.add_parser("p4-probe"); pp.add_argument("--last-cl", type=int, default=0)
    bp = sub.add_parser("bz-probe"); bp.add_argument("--since", default="2026-01-01")
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
            print("egress allowlist violation", file=sys.stderr); return 2
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
```

- [ ] **Step 4: Run, expect PASS** (all 5 static/egress tests).
- [ ] **Step 5: Commit** — `git commit -m "feat(rag.sources): probe CLI + read-only/egress/safety static tests"`

---

## Task 7.1: Link extraction (`rag/extract/links.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/extract/links.py`
- Test: `reference-impl/tests/rag/extract/test_links.py`

- [ ] **Step 1: Write the failing tests** (positives + negatives + false-positive corpus + min-id):

```python
# reference-impl/tests/rag/extract/test_links.py
from __future__ import annotations

import pytest

from codescribe_rag.rag.extract.links import extract_cl_refs, extract_bug_refs


@pytest.mark.parametrize("text,expected", [
    ("Fixed in CL 12345", {12345}),
    ("see change 67890 for the fix", {67890}),
    ("submitted as 45678", {45678}),
    ("@swarm/13579 landed", {13579}),
    ("https://swarm/changes/24680", {24680}),
])
def test_cl_positives(text, expected):
    assert extract_cl_refs(text) == expected


@pytest.mark.parametrize("text", [
    "CL ages like wine", "change everything now", "Cl- is chloride", "CL 12",  # <4 digits
])
def test_cl_negatives(text):
    assert extract_cl_refs(text) == set()


@pytest.mark.parametrize("text,expected", [
    ("Bug 4567: NPE on startup", {4567}),
    ("closes bug 9999", {9999}),
    ("fixes 8888", {8888}),
    ("BZ-7777 regression", {7777}),
    ("show_bug.cgi?id=5555", {5555}),
])
def test_bug_positives(text, expected):
    assert extract_bug_refs(text) == expected


def test_min_id_threshold():
    assert extract_cl_refs("CL 0999") == set()      # below 1000
    assert extract_bug_refs("bug 12") == set()
    assert extract_cl_refs("CL 1000") == {1000}
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `links.py`** (patterns verbatim from #4 §9; the `min_*_id` filter is new):

```python
from __future__ import annotations

import re

CL_PATTERNS = [
    re.compile(r"\bCL\s*#?\s*(\d{4,})\b", re.IGNORECASE),
    re.compile(r"\bchange(?:list)?\s+(\d{4,})\b", re.IGNORECASE),
    re.compile(r"@(?:p4|swarm|change)[/_]?(\d{4,})", re.IGNORECASE),
    re.compile(r"/changes/(\d{4,})"),
    re.compile(r"\bsubmitted\s+as\s+(\d{4,})\b", re.IGNORECASE),
]

BUG_PATTERNS = [
    re.compile(r"\bbug\s*[-#]?\s*(\d{4,})\b", re.IGNORECASE),
    re.compile(r"\bBZ\s*[-#]?\s*(\d{4,})\b", re.IGNORECASE),
    re.compile(r"show_bug\.cgi\?id=(\d{4,})"),
    re.compile(r"\bfix(?:es|ed)?\s+(?:bug\s+)?(\d{4,})\b", re.IGNORECASE),
    re.compile(r"\bclos(?:es|ed)?\s+(?:bug\s+)?(\d{4,})\b", re.IGNORECASE),
]


def _extract(text: str, patterns: list[re.Pattern], min_id: int) -> set[int]:
    out: set[int] = set()
    for pat in patterns:
        for m in pat.finditer(text or ""):
            n = int(m.group(1))
            if n >= min_id:
                out.add(n)
    return out


def extract_cl_refs(text: str, min_cl_id: int = 1000) -> set[int]:
    return _extract(text, CL_PATTERNS, min_cl_id)


def extract_bug_refs(text: str, min_bug_id: int = 1000) -> set[int]:
    return _extract(text, BUG_PATTERNS, min_bug_id)
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(rag.extract): CL/bug regex link extraction"`

---

## Task 7.2: Confidence pairing (`rag/extract/pairing.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/extract/pairing.py`
- Test: `reference-impl/tests/rag/extract/test_pairing.py`

- [ ] **Step 1: Write the failing tests** (signals, temporal, threshold, no-double-count). Tests build lightweight stand-ins for `Bug`/`P4Change` with the attributes `score_pair` reads.

```python
# reference-impl/tests/rag/extract/test_pairing.py
from __future__ import annotations

from datetime import datetime, timezone

from codescribe_rag.rag.config import PairingConfig
from codescribe_rag.rag.extract.pairing import score_pair, pair_bugs_to_cls, BugView, CLView

UTC = timezone.utc
CFG = PairingConfig()


def mk_bug(**kw):
    base = dict(id=4567, summary="NPE", status="RESOLVED", resolution="FIXED",
                assigned_to="alice", last_change_time=datetime(2026, 1, 10, tzinfo=UTC),
                comment_text="")
    base.update(kw); return BugView(**base)


def mk_cl(**kw):
    base = dict(cl=12345, author="alice", description="",
                submitted_at=datetime(2026, 1, 12, tzinfo=UTC))
    base.update(kw); return CLView(**base)


def test_bug_comment_cites_cl_signal():
    link = score_pair(mk_bug(comment_text="Fixed in CL 12345"), mk_cl(), CFG)
    assert link.signals["bug_comment_cites_cl"] == 0.5


def test_cl_desc_cites_bug_signal():
    link = score_pair(mk_bug(), mk_cl(description="Bug 4567: fix NPE"), CFG)
    assert link.signals["cl_desc_cites_bug"] == 0.5


def test_temporal_proximity_within_window():
    near = score_pair(mk_bug(), mk_cl(submitted_at=datetime(2026, 1, 15, tzinfo=UTC)), CFG)
    far = score_pair(mk_bug(), mk_cl(submitted_at=datetime(2026, 2, 20, tzinfo=UTC)), CFG)
    assert near.signals.get("temporal_proximity") == 0.3
    assert "temporal_proximity" not in far.signals


def test_status_and_author_signals():
    link = score_pair(mk_bug(), mk_cl(), CFG)
    assert link.signals["assignee_author_match"] == 0.2
    assert link.signals["bug_status_fixed"] == 0.1


def test_threshold_boundary():
    # comment-cites (0.5) + cl-desc-cites (0.5) capped to 1.0 → above threshold
    strong = score_pair(mk_bug(comment_text="see CL 12345"),
                        mk_cl(description="fixes bug 4567"), CFG)
    assert strong.confidence >= 0.8
    # only a weak temporal+author+status = 0.6 → below 0.8
    weak = score_pair(mk_bug(), mk_cl(), CFG)
    assert round(weak.confidence, 3) == 0.6


def test_pair_filters_by_threshold_and_window():
    bugs = [mk_bug(comment_text="CL 12345")]
    cls = [mk_cl(description="fixes bug 4567"),                       # strong
           mk_cl(cl=999999, submitted_at=datetime(2027, 6, 1, tzinfo=UTC))]  # out of window + weak
    links = pair_bugs_to_cls(bugs, cls, CFG)
    assert [l.cl_number for l in links] == [12345]
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `pairing.py`** (new; `BugView`/`CLView` are minimal value types so this stays decoupled from the source dataclasses; the pipeline adapts `Bug`/`P4Change` into these):

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..config import PairingConfig
from .links import extract_bug_refs, extract_cl_refs


@dataclass(frozen=True, slots=True)
class BugView:
    id: int
    summary: str
    status: str
    resolution: str
    assigned_to: str
    last_change_time: datetime
    comment_text: str          # all comment bodies concatenated


@dataclass(frozen=True, slots=True)
class CLView:
    cl: int
    author: str
    description: str
    submitted_at: datetime


@dataclass(frozen=True, slots=True)
class FixLink:
    bug_id: int
    cl_number: int
    signals: dict[str, float]
    confidence: float


def score_pair(bug: BugView, cl: CLView, cfg: PairingConfig) -> FixLink:
    w = cfg.weights
    signals: dict[str, float] = {}
    if cl.cl in extract_cl_refs(bug.comment_text, cfg.min_cl_id):
        signals["bug_comment_cites_cl"] = w.bug_comment_cites_cl
    if bug.id in extract_bug_refs(cl.description, cfg.min_bug_id):
        signals["cl_desc_cites_bug"] = w.cl_desc_cites_bug
    delta = abs((cl.submitted_at - bug.last_change_time).days)
    if delta <= cfg.temporal_signal_days:
        signals["temporal_proximity"] = w.temporal_proximity
    if bug.assigned_to and bug.assigned_to == cl.author:
        signals["assignee_author_match"] = w.assignee_author_match
    if bug.status in ("RESOLVED", "CLOSED") and bug.resolution == "FIXED":
        signals["bug_status_fixed"] = w.bug_status_fixed
    confidence = min(1.0, sum(signals.values()))
    return FixLink(bug.id, cl.cl, signals, confidence)


def pair_bugs_to_cls(bugs, cls, cfg: PairingConfig) -> list[FixLink]:
    window = timedelta(days=cfg.candidate_window_days)
    out: list[FixLink] = []
    for bug in bugs:
        for cl in cls:
            if abs(cl.submitted_at - bug.last_change_time) > window:
                continue
            link = score_pair(bug, cl, cfg)
            if link.confidence >= cfg.threshold:
                out.append(link)
    return out
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(rag.extract): confidence-scored bug↔CL pairing"`

---

## Task 8.1: Store schema + types (`rag/store/schema.sql`, `types.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/store/schema.sql` (verbatim #4 §10 — the full SQL block, but every `CREATE TABLE`/`CREATE INDEX`/`CREATE VIRTUAL TABLE` gets `IF NOT EXISTS` so re-open is idempotent).
- Create: `reference-impl/codescribe_rag/rag/store/types.py`

- [ ] **Step 1: Write `schema.sql`** — copy #4 §10 verbatim, adding `IF NOT EXISTS` to each DDL statement (e.g. `CREATE TABLE IF NOT EXISTS bugs (...)`, `CREATE VIRTUAL TABLE IF NOT EXISTS bug_vectors USING vec0(embedding FLOAT[1024])`, etc.).

- [ ] **Step 2: Write `types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FileDiff:
    file_path: str
    diff: str


@dataclass(frozen=True, slots=True)
class CLDiff:
    cl_number: int
    author: str
    submitted_at: str
    description: str
    files: tuple[FileDiff, ...]


@dataclass(frozen=True, slots=True)
class RetrievedBugFix:
    bug_id: int
    summary: str
    severity: str
    status: str
    score: float
    fix_cl: int | None
    fix_diff_excerpt: str | None
    confidence: float
```

- [ ] **Step 3: Commit** — `git commit -m "feat(rag.store): schema.sql + transport types"` (no test yet; exercised in 8.2).

---

## Task 8.2: Store writer (`rag/store/writer.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/store/writer.py`
- Test: `reference-impl/tests/rag/store/test_writer.py`, `test_no_unsafe_serialisation.py`

- [ ] **Step 1: Write the failing tests**

```python
# reference-impl/tests/rag/store/test_writer.py
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from codescribe_rag.rag.store.writer import Store

UTC = timezone.utc


def unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeBug:
    def __init__(self, i):
        self.id = i; self.summary = f"summary {i}"; self.description = f"desc {i} npe"
        self.component = "core"; self.severity = "major"; self.status = "RESOLVED"
        self.resolution = "FIXED"; self.creation_time = datetime(2026, 1, 1, tzinfo=UTC)
        self.last_change_time = datetime(2026, 1, 2, tzinfo=UTC); self.assigned_to = "alice"
        self.raw_json = {"id": i}


def test_open_creates_schema(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    # would raise "no such module: vec0" if the extension didn't load
    store._conn.execute("SELECT * FROM bug_vectors LIMIT 0")
    store.close()


def test_open_idempotent(tmp_path):
    p = tmp_path / "rag.db"
    Store.open(p).close()
    Store.open(p).close()  # no error on second open


def test_upsert_bug_roundtrip_and_idempotent(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    with store.batch():
        store.upsert_bug(FakeBug(1001), unit(1))
        store.upsert_bug(FakeBug(1001), unit(1))   # again → still one row
    n = store._conn.execute("SELECT COUNT(*) FROM bugs").fetchone()[0]
    assert n == 1
    store.close()


def test_fts_bm25_returns_match(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    with store.batch():
        for i in range(1001, 1006):
            store.upsert_bug(FakeBug(i), unit(i))
    rows = store._conn.execute(
        "SELECT bug_id FROM bug_fts WHERE bug_fts MATCH 'npe' LIMIT 3").fetchall()
    assert rows
    store.close()


def test_vec_query_nearest(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    with store.batch():
        for i in range(1001, 1011):
            store.upsert_bug(FakeBug(i), unit(i))
    q = unit(1005).astype(np.float32).tobytes()
    rows = store._conn.execute(
        "SELECT rowid, distance FROM bug_vectors WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
        (q,)).fetchall()
    assert rows[0][0] == 1005 and rows[0][1] < 1e-4
    store.close()


def test_diff_gzip_roundtrip(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    class CL:
        cl = 50001; author = "alice"; submitted_at = datetime(2026, 1, 3, tzinfo=UTC)
        description = "fix"; files = (); diff_text = "@@ -1 +1 @@\n-a\n+b\n" * 5000
    with store.batch():
        store.upsert_cl(CL(), [], [], unit(7))
    raw = store._conn.execute("SELECT diff_text FROM changes WHERE cl_number=50001").fetchone()[0]
    assert raw[:2] == b"\x1f\x8b"  # gzip magic
    assert store.get_diff_text(50001) == CL.diff_text
    store.close()


def test_batch_rollback(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    with pytest.raises(RuntimeError):
        with store.batch():
            store.upsert_bug(FakeBug(1001), unit(1))
            raise RuntimeError("boom")
    assert store._conn.execute("SELECT COUNT(*) FROM bugs").fetchone()[0] == 0
    store.close()
```

```python
# reference-impl/tests/rag/store/test_no_unsafe_serialisation.py
from __future__ import annotations
import re
from pathlib import Path

STORE = Path(__file__).parents[3] / "codescribe_rag" / "rag" / "store"
# JSON+gzip only. The banned-module tokens are assembled from fragments so this
# guard file does not match itself when scanned (per #4 §8's "import pi"+"ckle").
_BANNED = ["pi" + "ckle", "cPi" + "ckle", "di" + "ll", "shel" + "ve"]
PATTERN = re.compile(r"\bimport\s+(?:" + "|".join(_BANNED) + r")\b")


def test_no_unsafe_serialisation():
    for py in STORE.rglob("*.py"):
        assert not PATTERN.search(py.read_text()), f"unsafe serialisation in {py}"
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `writer.py`** — adapt #5 §4 verbatim. Key adaptations: `upsert_cl` signature is `(cl, chunks, chunk_embeddings, cl_summary_embedding)` (the summary-embedding param is accepted; chunk handling exactly as the skeleton), and `Bug`/`P4Change`/`CLChunk`/`FixLink` are referenced via duck-typing (no import cycle). Paste the §4 skeleton body unchanged except keep imports to `numpy`, `sqlite_vec`, stdlib only (`gzip`, `json`, `sqlite3`, `contextlib`, `pathlib`).

- [ ] **Step 4: Run, expect PASS** (8 tests).
- [ ] **Step 5: Commit** — `git commit -m "feat(rag.store): sqlite-vec + FTS5 writer"`

---

## Task 8.3: Embedder + HashingEmbedder (`rag/embed/embedder.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/embed/embedder.py`
- Test: `reference-impl/tests/rag/embed/test_embedder.py`

- [ ] **Step 1: Write the failing tests**

```python
# reference-impl/tests/rag/embed/test_embedder.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codescribe_rag.rag.embed.embedder import HashingEmbedder, Embedder

MODEL = Path("~/.hf-models/bge-large-en-v1.5").expanduser()


def test_hashing_dim_is_1024():
    assert HashingEmbedder().embed_one("hello world").shape == (1024,)


def test_hashing_l2_normalised():
    v = HashingEmbedder().embed_one("some text here")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_hashing_deterministic():
    a = HashingEmbedder().embed_one("repeatable")
    b = HashingEmbedder().embed_one("repeatable")
    assert np.allclose(a, b)


def test_hashing_empty_input():
    assert HashingEmbedder().embed([]).shape == (0, 1024)


def test_embedder_lazy_load():
    e = Embedder(MODEL)
    assert e._model is None  # constructor must not load


@pytest.mark.skipif(not MODEL.exists(), reason="bge-large not downloaded")
def test_real_normalisation_semantics():
    e = Embedder(MODEL)
    a = e.embed_one("NullPointerException during startup")
    b = e.embed_one("NPE thrown when the app starts")
    assert float(a @ b) > 0.7
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `embedder.py`** — the production `Embedder` is adapted from #5 §6 (torch + sentence-transformers imported lazily inside `_ensure_loaded`, never at module top). Add the `Protocol`, `HashingEmbedder`, and `make_embedder` factory:

```python
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)
DIM = 1024


@runtime_checkable
class SupportsEmbedding(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_one(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic, dependency-light embedder for tests/offline wiring.

    Hashes token unigrams+bigrams into a fixed 1024-dim vector, then L2-normalises.
    Not semantic — same-token overlap drives similarity. Never use in production.
    """

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self._vec(text)

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        toks = (text or "").lower().split()
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
            v[h % self._dim] += 1.0
        n = float(np.linalg.norm(v))
        if n == 0.0:
            v[0] = 1.0
            n = 1.0
        return (v / n).astype(np.float32)


class Embedder:
    """Local sentence-transformers wrapper. Lazy-loads; L2-normalises output."""

    def __init__(self, model_path: Path, device: str = "auto",
                 batch_size: int = 32, max_length: int = 512) -> None:
        self._model_path = model_path
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = self._resolve_device(torch)
            logger.info("loading embedder %s on %s", self._model_path, device)
            self._model = SentenceTransformer(str(self._model_path), device=device)
            self._model.max_seq_length = self._max_length
            self._model.encode(["warmup"], batch_size=1, normalize_embeddings=True)
        return self._model

    def _resolve_device(self, torch) -> str:
        if self._device != "auto":
            return self._device
        if not torch.cuda.is_available():
            return "cpu"
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) < (8, 0):
            logger.info("GPU capability %d.%d < 8.0; will use fp16", major, minor)
        return "cuda"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        model = self._ensure_loaded()
        emb = model.encode(texts, batch_size=self._batch_size, show_progress_bar=False,
                           normalize_embeddings=True, convert_to_numpy=True)
        return emb.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def make_embedder(cfg) -> SupportsEmbedding:
    """Factory from EmbedConfig.backend."""
    if cfg.backend == "hashing":
        return HashingEmbedder(cfg.dim)
    return Embedder(cfg.model_path, cfg.device, cfg.batch_size, cfg.max_length)
```

- [ ] **Step 4: Run, expect PASS** (semantic test skipped unless model present).
- [ ] **Step 5: Commit** — `git commit -m "feat(rag.embed): bge-large embedder + hashing fallback"`

---

## Task 8.4: Diff chunker (`rag/embed/chunker.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/embed/chunker.py`
- Test: `reference-impl/tests/rag/embed/test_chunker.py`

- [ ] **Step 1: Write the failing tests**

```python
# reference-impl/tests/rag/embed/test_chunker.py
from __future__ import annotations

from codescribe_rag.rag.embed.chunker import chunk_diff, CLChunk


class CL:
    def __init__(self, diff): self.diff_text = diff


DIFF = (
    "--- a/Foo.java\n+++ b/Foo.java\n@@ -1,2 +1,3 @@\n-x\n+y\n+z\n"
    "--- a/Bar.java\n+++ b/Bar.java\n@@ -5,1 +5,2 @@\n-p\n+q\n"
)


def test_per_file_isolation():
    chunks = chunk_diff(CL(DIFF))
    paths = {c.file_path for c in chunks}
    assert paths == {"Foo.java", "Bar.java"}


def test_respects_max_chars():
    big = "--- a/Big.txt\n+++ b/Big.txt\n@@ -1 +1 @@\n" + ("+line\n" * 5000)
    for c in chunk_diff(CL(big), max_chars=2000):
        assert len(c.text.encode("utf-8")) <= 2000


def test_never_splits_inside_hunk_header():
    for c in chunk_diff(CL(DIFF)):
        # a chunk may start with a full hunk header but never with a partial one
        assert not c.text.startswith("@ ")


def test_utf8_boundary_preserved():
    big = "--- a/u.txt\n+++ b/u.txt\n@@ -1 +1 @@\n" + ("+éééé\n" * 2000)
    for c in chunk_diff(CL(big), max_chars=1000):
        c.text.encode("utf-8").decode("utf-8")  # must not raise
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `chunker.py`** — verbatim #5 §7 (the `chunk_diff`, `_split_by_file`, `_split_by_hunk`, `_safe_split`, `CLChunk` block).

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(rag.embed): hunk-aware diff chunker"`

---

## Task 10.1: Hybrid retriever (`rag/store/retrieve.py`) + shared conftest

**Files:**
- Create: `reference-impl/tests/conftest.py`
- Create: `reference-impl/codescribe_rag/rag/store/retrieve.py`
- Test: `reference-impl/tests/rag/store/test_retrieve.py`

- [ ] **Step 1: Write `tests/conftest.py`** (shared `embedder` + `populated_store` fixtures used here and by the MCP tests):

```python
# reference-impl/tests/conftest.py
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from codescribe_rag.rag.embed.embedder import HashingEmbedder
from codescribe_rag.rag.store.writer import Store

UTC = timezone.utc


class _Bug:
    def __init__(self, i, summary, desc):
        self.id = i; self.summary = summary; self.description = desc
        self.component = "core"; self.severity = "major"; self.status = "RESOLVED"
        self.resolution = "FIXED"; self.creation_time = datetime(2026, 1, 1, tzinfo=UTC)
        self.last_change_time = datetime(2026, 1, 2, tzinfo=UTC); self.assigned_to = "alice"
        self.raw_json = {"id": i}


class _CL:
    def __init__(self, n, diff):
        self.cl = n; self.author = "alice"; self.submitted_at = datetime(2026, 1, 3, tzinfo=UTC)
        self.description = f"fix for {n}"; self.files = (); self.diff_text = diff


class _Link:
    def __init__(self, bug_id, cl, conf):
        self.bug_id = bug_id; self.cl_number = cl; self.confidence = conf
        self.signals = {"x": conf}


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def populated_store(tmp_path, embedder):
    store = Store.open(tmp_path / "rag.db")
    bugs = [
        _Bug(1001, "NullPointerException during startup", "NPE in ConfigLoader on boot"),
        _Bug(1002, "memory leak in cache", "heap grows unbounded under load"),
        _Bug(1003, "race condition in scheduler", "threads deadlock occasionally"),
    ]
    with store.batch():
        for b in bugs:
            store.upsert_bug(b, embedder.embed_one(f"{b.summary} {b.description}"))
        store.upsert_cl(_CL(12345, "@@ -1 +1 @@\n-bad\n+good\n"), [], [], embedder.embed_one("fix"))
        store.upsert_fix_link(_Link(1001, 12345, 0.9))
        store.upsert_fix_link(_Link(1002, 12345, 0.7))   # below default threshold
    yield store
    store.close()
```

- [ ] **Step 2: Write the failing tests**

```python
# reference-impl/tests/rag/store/test_retrieve.py
from __future__ import annotations

import time

import numpy as np
import pytest

from codescribe_rag.rag.store.retrieve import Retriever


def test_hybrid_finds_relevant_bug(populated_store, embedder):
    r = Retriever(populated_store._conn, embedder)
    results = r.find_similar_bugs("NPE on startup in ConfigLoader", k=3)
    assert results[0].bug_id == 1001


def test_confidence_filter(populated_store, embedder):
    r = Retriever(populated_store._conn, embedder)
    res = {x.bug_id: x for x in r.find_similar_bugs("startup crash", k=3)}
    assert res[1001].fix_cl == 12345          # 0.9 ≥ 0.8
    if 1002 in res:
        assert res[1002].fix_cl is None       # 0.7 filtered out


def test_no_unlinked_bug_returns_none(populated_store, embedder):
    r = Retriever(populated_store._conn, embedder)
    res = {x.bug_id: x for x in r.find_similar_bugs("scheduler deadlock", k=3)}
    assert res[1003].fix_cl is None and res[1003].fix_diff_excerpt is None


def test_query_normalisation_required(populated_store):
    class BadEmbedder:
        def embed_one(self, t): return np.full(1024, 5.0, dtype=np.float32)  # not normalised
        def embed(self, ts): return np.full((len(ts), 1024), 5.0, dtype=np.float32)
    r = Retriever(populated_store._conn, BadEmbedder())
    with pytest.raises(RuntimeError):
        r.find_similar_bugs("x")


def test_get_fix_diff_roundtrip(populated_store, embedder):
    r = Retriever(populated_store._conn, embedder)
    diff = r.get_fix_diff(12345)
    joined = "".join(f.diff for f in diff.files) if diff.files else ""
    assert "+good" in joined


def test_sanitise_fts_drops_punctuation(populated_store, embedder):
    r = Retriever(populated_store._conn, embedder)
    r.find_similar_bugs("foo:bar* (startup)")  # must not raise FTS5 syntax error


@pytest.mark.slow
def test_latency_under_200ms(tmp_path, embedder):
    from codescribe_rag.rag.store.writer import Store
    store = Store.open(tmp_path / "big.db")
    rng = np.random.default_rng(0)
    with store.batch():
        for i in range(1001, 101001):
            v = rng.standard_normal(1024).astype(np.float32); v /= np.linalg.norm(v)
            store._conn.execute("INSERT INTO bug_vectors (rowid, embedding) VALUES (?, ?)",
                                (i, v.tobytes()))
            store._conn.execute("INSERT INTO bugs (bug_id, summary) VALUES (?, ?)", (i, f"bug {i}"))
    r = Retriever(store._conn, embedder)
    lat = []
    for _ in range(100):
        t = time.perf_counter(); r.find_similar_bugs("startup npe", k=5); lat.append(time.perf_counter() - t)
    lat.sort()
    assert lat[98] < 0.2  # p99
    store.close()
```

- [ ] **Step 3: Run, expect FAIL.**

- [ ] **Step 4: Implement `retrieve.py`** — adapt #5 §5: the `Retriever` class with `find_similar_bugs`, `_vec_ranks`, `_bm25_ranks`, `_sanitise_fts_query`, `_rrf_merge`, `_resolve_fix`, `_get_diff_excerpt`, and `RRF_CONSTANT`, exactly as the skeleton. Import `RetrievedBugFix` from `.types` (do not redefine it). Add `get_fix_diff(self, cl_number, max_chars=None) -> CLDiff`: read+gzip-decompress `changes.diff_text`, split per-file via `from ..embed.chunker import _split_by_file`, build `FileDiff(file_path, diff)` tuples, return `CLDiff(...)`; honour `max_chars` by truncating the concatenated diff if given.

- [ ] **Step 5: Run, expect PASS** (slow test skipped by default).
- [ ] **Step 6: Commit** — `git commit -m "feat(rag.store): hybrid vec+BM25 RRF retriever + test fixtures"`

---

## Task 9.1: Bootstrap pipeline (`rag/pipelines/bootstrap.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/pipelines/bootstrap.py`
- Test: `reference-impl/tests/rag/pipelines/test_bootstrap.py`

- [ ] **Step 1: Write the failing tests** — use fake in-memory sources (lists) so no asyncio/subprocess needed; assert idempotency, late-arriving CL pairing, and batch size.

```python
# reference-impl/tests/rag/pipelines/test_bootstrap.py
from __future__ import annotations

from datetime import datetime, timezone

from codescribe_rag.rag.config import RagConfig, PerforceConfig, BugzillaConfig
from codescribe_rag.rag.embed.embedder import HashingEmbedder
from codescribe_rag.rag.pipelines.bootstrap import run_bootstrap
from codescribe_rag.rag.sources.bugzilla import Bug, BugComment
from codescribe_rag.rag.sources.perforce import P4Change, FileChange
from codescribe_rag.rag.store.writer import Store

UTC = timezone.utc


def mk_cfg():
    return RagConfig(perforce=PerforceConfig(p4port="x", p4user="y"),
                     bugzilla=BugzillaConfig(base_url="https://bz.x", allowlist_hostname="bz.x"))


def mk_bug(i, comment):
    return (Bug(i, "NPE", "", "core", "major", "RESOLVED", "FIXED",
                datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 10, tzinfo=UTC), "alice", {"id": i}),
            [BugComment(1, "alice", comment, datetime(2026, 1, 10, tzinfo=UTC))])


def mk_cl(n, desc):
    return P4Change(n, "alice", datetime(2026, 1, 12, tzinfo=UTC), desc,
                    (FileChange("//depot/Foo.java", "edit", 5),), "@@ -1 +1 @@\n-a\n+b\n", False)


class FakeSources:
    def __init__(self, bugs_with_comments, cls):
        self._bc = bugs_with_comments; self._cls = cls
    def iter_bugs(self):
        for b, _ in self._bc: yield b
    def comments_for(self, bug_id):
        return next(c for b, c in self._bc if b.id == bug_id)
    def iter_cls(self):
        yield from self._cls


def test_bootstrap_idempotent(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    src = FakeSources([mk_bug(1001, "Fixed in CL 12345")], [mk_cl(12345, "fixes bug 1001")])
    s1 = run_bootstrap(mk_cfg(), src, store, HashingEmbedder())
    s2 = run_bootstrap(mk_cfg(), src, store, HashingEmbedder())
    assert s1.bugs_added == 1 and s1.cls_added == 1 and s1.links_added == 1
    assert s2.bugs_added == 0 and s2.cls_added == 0  # re-run no-ops
    store.close()


def test_late_arriving_cl_pairs(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    # bug references no CL in comment; CL references the bug → pairing via cl desc + temporal
    src = FakeSources([mk_bug(1001, "no ref here")], [mk_cl(12345, "fixes bug 1001")])
    stats = run_bootstrap(mk_cfg(), src, store, HashingEmbedder())
    assert stats.links_added == 1
    row = store._conn.execute("SELECT cl_number FROM fix_links WHERE bug_id=1001").fetchone()
    assert row[0] == 12345
    store.close()


def test_batch_size_capped(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    bugs = [mk_bug(1000 + i, "x") for i in range(1, 600)]
    src = FakeSources(bugs, [])
    stats = run_bootstrap(mk_cfg(), src, store, HashingEmbedder(), batch_size=256)
    assert stats.bugs_added == 599
    assert max(stats.batch_sizes) <= 256
    store.close()
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `bootstrap.py`** — a synchronous, streaming implementation (simpler and fully testable than the asyncio sketch in #5 §10; the asyncio version is an optional optimisation noted in the docstring). Reads from a sources object exposing `iter_bugs()`, `comments_for(id)`, `iter_cls()`; batches ≤ `batch_size`; embeds via the injected embedder; upserts in a `store.batch()`; runs a final pairing pass over candidates in the ±`candidate_window_days` window; advances `state`. Returns `BootstrapStats(bugs_added, cls_added, links_added, total_seconds, batch_sizes)`.

```python
from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field

from ..config import RagConfig
from ..embed.chunker import chunk_diff
from ..extract.pairing import BugView, CLView, pair_bugs_to_cls

logger = logging.getLogger(__name__)


@dataclass
class BootstrapStats:
    bugs_added: int = 0
    cls_added: int = 0
    links_added: int = 0
    total_seconds: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)


def _existing(conn, table, col):
    return {r[0] for r in conn.execute(f"SELECT {col} FROM {table}").fetchall()}


def run_bootstrap(cfg: RagConfig, sources, store, embedder, *, batch_size: int = 256) -> BootstrapStats:
    started = time.perf_counter()
    stats = BootstrapStats()
    seen_bugs = _existing(store._conn, "bugs", "bug_id")
    seen_cls = _existing(store._conn, "changes", "cl_number")

    bug_views: list[BugView] = []
    cl_views: list[CLView] = []

    # --- bugs ---
    batch: list = []

    def flush_bugs():
        if not batch:
            return
        embs = embedder.embed([f"{b.summary} {b.description}" for b in batch])
        with store.batch():
            for bug, emb in zip(batch, embs):
                store.upsert_bug(bug, emb)
        stats.batch_sizes.append(len(batch))
        batch.clear()

    for bug in sources.iter_bugs():
        comments = sources.comments_for(bug.id)
        comment_text = "\n".join(c.text for c in comments)
        desc = comments[0].text if comments else ""          # comment 0 = description
        bug = dataclasses.replace(bug, description=desc)
        bug_views.append(BugView(bug.id, bug.summary, bug.status, bug.resolution,
                                 bug.assigned_to, bug.last_change_time, comment_text))
        if bug.id not in seen_bugs:
            batch.append(bug); stats.bugs_added += 1
            if len(batch) >= batch_size:
                flush_bugs()
    flush_bugs()

    # --- cls ---
    clbatch: list = []

    def flush_cls():
        if not clbatch:
            return
        with store.batch():
            for cl in clbatch:
                chunks = chunk_diff(cl)
                chunk_embs = list(embedder.embed([c.text for c in chunks])) if chunks else []
                summary_emb = embedder.embed_one(cl.description or "")
                store.upsert_cl(cl, chunks, chunk_embs, summary_emb)
        stats.batch_sizes.append(len(clbatch))
        clbatch.clear()

    for cl in sources.iter_cls():
        cl_views.append(CLView(cl.cl, cl.author, cl.description, cl.submitted_at))
        if cl.cl not in seen_cls:
            clbatch.append(cl); stats.cls_added += 1
            if len(clbatch) >= batch_size:
                flush_cls()
    flush_cls()

    # --- final pairing pass over the candidate window ---
    links = pair_bugs_to_cls(bug_views, cl_views, cfg.pairing)
    with store.batch():
        for link in links:
            exists = store._conn.execute(
                "SELECT 1 FROM fix_links WHERE bug_id=? AND cl_number=?",
                (link.bug_id, link.cl_number)).fetchone()
            store.upsert_fix_link(link)
            if not exists:
                stats.links_added += 1
        if cl_views:
            store.set_state("last_seen_cl", str(max(c.cl for c in cl_views)))
        if bug_views:
            store.set_state("last_seen_bug_modtime",
                            max(b.last_change_time for b in bug_views).isoformat())

    stats.total_seconds = time.perf_counter() - started
    logger.info("bootstrap done: %s", stats)
    return stats
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(rag.pipelines): streaming bootstrap indexer + final pairing pass"`

---

## Task 9.2: Incremental + rag CLI (`pipelines/incremental.py`, `rag/__main__.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/rag/pipelines/incremental.py`
- Create: `reference-impl/codescribe_rag/rag/__main__.py`
- Test: `reference-impl/tests/rag/pipelines/test_incremental.py`, `test_no_stdout_in_library.py`

- [ ] **Step 1: Write the failing tests**

```python
# reference-impl/tests/rag/pipelines/test_incremental.py
from __future__ import annotations

from codescribe_rag.rag.embed.embedder import HashingEmbedder
from codescribe_rag.rag.pipelines.bootstrap import run_bootstrap
from codescribe_rag.rag.pipelines.incremental import read_watermarks
from codescribe_rag.rag.store.writer import Store
from tests.rag.pipelines.test_bootstrap import FakeSources, mk_bug, mk_cl, mk_cfg


def test_watermarks_advance(tmp_path):
    store = Store.open(tmp_path / "rag.db")
    src = FakeSources([mk_bug(1001, "CL 12345")], [mk_cl(12345, "fixes bug 1001")])
    run_bootstrap(mk_cfg(), src, store, HashingEmbedder())
    last_cl, since = read_watermarks(store)
    assert last_cl == 12345
    assert since.year == 2026
    store.close()
```

```python
# reference-impl/tests/rag/pipelines/test_no_stdout_in_library.py
from __future__ import annotations
import re
from pathlib import Path

PIPE = Path(__file__).parents[3] / "codescribe_rag" / "rag" / "pipelines"


def test_no_print_in_pipelines():
    for py in PIPE.rglob("*.py"):
        for ln in py.read_text().splitlines():
            assert not re.match(r"\s*print\(", ln), f"print() in {py}: {ln}"
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `incremental.py`**

```python
from __future__ import annotations

import logging
from datetime import datetime, timezone

from .bootstrap import BootstrapStats, run_bootstrap

logger = logging.getLogger(__name__)


def read_watermarks(store) -> tuple[int, datetime]:
    last_cl = int(store.get_state("last_seen_cl") or 0)
    raw = store.get_state("last_seen_bug_modtime")
    since = datetime.fromisoformat(raw) if raw else datetime(1970, 1, 1, tzinfo=timezone.utc)
    return last_cl, since


def run_incremental(cfg, sources, store, embedder, *, batch_size: int = 256) -> BootstrapStats:
    # The streaming bootstrap is idempotent and upsert-based, so an incremental
    # run is the same loop over a sources object already scoped to deltas.
    return run_bootstrap(cfg, sources, store, embedder, batch_size=batch_size)
```

- [ ] **Step 4: Implement `rag/__main__.py`** (CLI; stdout fine here — guarded entry):

```python
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
    def __init__(self, cfg, since_cl, since_modtime):
        rl = TokenBucket(cfg.rate_limit_rps)
        self._p4 = P4Source(cfg.perforce.p4port, cfg.perforce.p4user,
                            cfg.perforce.ticket_path, rl, binary=cfg.perforce.binary)
        key = cfg.bugzilla.api_key_path.read_text().strip()
        self._bz = BugzillaSource(cfg.bugzilla.base_url, key, rl, page_size=cfg.bugzilla.page_size)
        self._depot = cfg.perforce.depot_path
        self._since_cl = since_cl; self._since_modtime = since_modtime

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
    q = sub.add_parser("query"); q.add_argument("--query", required=True); q.add_argument("--k", type=int, default=5)
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
```

- [ ] **Step 5: Run, expect PASS.**
- [ ] **Step 6: Commit** — `git commit -m "feat(rag.pipelines): incremental + rag index|query|status CLI"`

---

## Task 11.1: MCP tools (`servers/rag_server/tools.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/servers/rag_server/tools.py`
- Test: `reference-impl/tests/servers/rag_server/test_tools.py`

- [ ] **Step 1: Write the failing tests**

```python
# reference-impl/tests/servers/rag_server/test_tools.py
from __future__ import annotations

import pytest

from codescribe_rag.servers.rag_server.tools import (
    RagTools, FIND_SIMILAR_BUGS_SCHEMA, GET_FIX_DIFF_SCHEMA)
from codescribe_rag.rag.store.retrieve import Retriever


@pytest.fixture
def tools(populated_store, embedder):
    return RagTools(populated_store, Retriever(populated_store._conn, embedder))


def test_schemas_shape():
    assert FIND_SIMILAR_BUGS_SCHEMA["required"] == ["query"]
    assert GET_FIX_DIFF_SCHEMA["properties"]["cl_number"]["minimum"] == 1


def test_find_similar_bugs_markdown(tools):
    out = tools.find_similar_bugs("NPE on startup", k=3, min_confidence=0.8)
    assert "Similar bugs" in out and "Bug 1001" in out


def test_invalid_k_raises(tools):
    with pytest.raises(ValueError):
        tools.find_similar_bugs("x", k=999, min_confidence=0.8)


def test_get_fix_diff_truncation(tools):
    out = tools.get_fix_diff(12345, max_chars=10)
    assert "truncated" in out and out.startswith("## CL 12345")
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `tools.py`** — adapt #5 §8 `tools.py`: imports become `from codescribe_rag.rag.store.retrieve import Retriever`, `from codescribe_rag.rag.store.writer import Store`, `from codescribe_rag.rag.embed.embedder import make_embedder`. `RagTools.__init__(self, store, retriever)` as in the skeleton. `RagTools.load(db_path)` builds the embedder via `make_embedder` from a config: read `RAG_CONFIG` env (default `configs/rag.yaml`) into `RagConfig`, but if `RAG_EMBED_BACKEND` env is set, override `cfg.embed.backend` first (lets the MCP subprocess run on `hashing` without a model). Schemas (`FIND_SIMILAR_BUGS_SCHEMA`, `GET_FIX_DIFF_SCHEMA`), `find_similar_bugs`, `get_fix_diff`, and `_format_results` are verbatim from the skeleton.

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(rag_server): RagTools + tool schemas + formatters"`

---

## Task 11.2: MCP server entry (`servers/rag_server/__main__.py`)

**Files:**
- Create: `reference-impl/codescribe_rag/servers/rag_server/__main__.py`
- Test: `reference-impl/tests/servers/rag_server/test_server.py`, `test_logger_no_stream_handler.py`, `test_no_stdout.py`, `test_no_egress.py`, `test_sigterm.py`

- [ ] **Step 1: Write the failing tests** (in-process transport for handshake/list/call; static + subprocess for the rest):

```python
# reference-impl/tests/servers/rag_server/test_server.py
from __future__ import annotations

import asyncio

from mcp.shared.memory import create_connected_server_and_client_session

import codescribe_rag.servers.rag_server.__main__ as srv
from codescribe_rag.rag.store.retrieve import Retriever
from codescribe_rag.servers.rag_server.tools import RagTools


def _server(store, embedder):
    return srv.build_server(_tools=RagTools(store, Retriever(store._conn, embedder)))


def test_tools_list(populated_store, embedder):
    server = _server(populated_store, embedder)
    async def go():
        async with create_connected_server_and_client_session(server) as s:
            await s.initialize()
            return await s.list_tools()
    res = asyncio.run(go())
    assert {t.name for t in res.tools} == {"find_similar_bugs", "get_fix_diff"}


def test_call_find_similar_bugs(populated_store, embedder):
    server = _server(populated_store, embedder)
    async def go():
        async with create_connected_server_and_client_session(server) as s:
            await s.initialize()
            return await s.call_tool("find_similar_bugs", {"query": "NPE on startup", "k": 3})
    res = asyncio.run(go())
    assert "Similar bugs" in res.content[0].text
```

```python
# reference-impl/tests/servers/rag_server/test_logger_no_stream_handler.py
from __future__ import annotations
import importlib
import logging


def test_only_file_handler(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_LOG_PATH", str(tmp_path / "s.log"))
    import codescribe_rag.servers.rag_server.__main__ as m
    importlib.reload(m)
    handlers = logging.getLogger().handlers
    assert handlers and all(isinstance(h, logging.FileHandler) for h in handlers)
```

```python
# reference-impl/tests/servers/rag_server/test_no_stdout.py
from __future__ import annotations
import re
from pathlib import Path

PKG = Path(__file__).parents[3] / "codescribe_rag" / "servers" / "rag_server"


def test_no_print_in_package():
    for py in PKG.rglob("*.py"):
        for ln in py.read_text().splitlines():
            assert not re.match(r"\s*print\(", ln), f"print() in {py}: {ln}"
```

```python
# reference-impl/tests/servers/rag_server/test_no_egress.py
from __future__ import annotations
import socket

from codescribe_rag.rag.store.retrieve import Retriever
from codescribe_rag.servers.rag_server.tools import RagTools


def test_tool_call_makes_no_socket(populated_store, embedder, monkeypatch):
    calls = []
    real = socket.socket.connect
    def spy(self, addr): calls.append(addr); return real(self, addr)
    monkeypatch.setattr(socket.socket, "connect", spy)
    tools = RagTools(populated_store, Retriever(populated_store._conn, embedder))
    tools.find_similar_bugs("npe", k=3, min_confidence=0.8)
    assert calls == []
```

```python
# reference-impl/tests/servers/rag_server/test_sigterm.py
from __future__ import annotations
import os, signal, subprocess, sys, time
from pathlib import Path


def test_subprocess_sigterm_clean(tmp_path):
    db = tmp_path / "rag.db"
    from codescribe_rag.rag.store.writer import Store
    Store.open(db).close()                       # valid empty store
    log = tmp_path / "s.log"
    env = {**os.environ, "RAG_DB_PATH": str(db), "RAG_LOG_PATH": str(log),
           "RAG_EMBED_BACKEND": "hashing"}
    proc = subprocess.Popen([sys.executable, "-m", "codescribe_rag.servers.rag_server"],
                            stdin=subprocess.PIPE, env=env,
                            cwd=str(Path(__file__).parents[3]))
    time.sleep(2.0)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 0
    assert log.exists() and log.stat().st_size > 0
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `__main__.py`** — adapt #5 §8. `_configure_logging()` runs at import, before the `mcp`/tools imports (which is why those imports carry `# noqa: E402`). `build_server(_tools=None)` lets tests inject a fixture-backed `RagTools` while production calls `RagTools.load(...)`. Server name `"codescribe-rag"`.

```python
from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from pathlib import Path


def _configure_logging() -> None:
    log_path = Path(os.environ.get("RAG_LOG_PATH", f"./logs/rag-server-{int(time.time())}.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    root.setLevel(logging.INFO)


_configure_logging()
logger = logging.getLogger(__name__)

import mcp.types as mcp_types                       # noqa: E402
from mcp.server import Server, NotificationOptions  # noqa: E402
from mcp.server.models import InitializationOptions # noqa: E402
from mcp.server.stdio import stdio_server           # noqa: E402

from .tools import RagTools, FIND_SIMILAR_BUGS_SCHEMA, GET_FIX_DIFF_SCHEMA  # noqa: E402


def build_server(_tools: "RagTools | None" = None) -> Server:
    tools = _tools if _tools is not None else RagTools.load(
        Path(os.environ.get("RAG_DB_PATH", "./indices/rag.db")))
    server = Server("codescribe-rag")
    server._codescribe_tools = tools

    @server.list_tools()
    async def handle_list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(name="find_similar_bugs",
                           description="Return up to k bugs most similar to a query, with fix CL excerpts.",
                           inputSchema=FIND_SIMILAR_BUGS_SCHEMA),
            mcp_types.Tool(name="get_fix_diff",
                           description="Return the full unified diff of a changelist.",
                           inputSchema=GET_FIX_DIFF_SCHEMA),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        try:
            if name == "find_similar_bugs":
                text = tools.find_similar_bugs(query=arguments["query"], k=arguments.get("k", 5),
                                               min_confidence=arguments.get("min_confidence", 0.8))
            elif name == "get_fix_diff":
                text = tools.get_fix_diff(cl_number=arguments["cl_number"],
                                          max_chars=arguments.get("max_chars", 8000))
            else:
                raise ValueError(f"unknown tool: {name}")
            return [mcp_types.TextContent(type="text", text=text)]
        except Exception:
            logger.exception("tool %s failed", name)
            raise

    return server


async def main() -> None:
    server = build_server()
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    async with stdio_server() as (read, write):
        run = asyncio.create_task(server.run(read, write, InitializationOptions(
            server_name="codescribe-rag", server_version="0.1.0",
            capabilities=server.get_capabilities(notification_options=NotificationOptions(),
                                                 experimental_capabilities={}))))
        stopper = asyncio.create_task(stop.wait())
        _, pending = await asyncio.wait({run, stopper}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    getattr(server, "_codescribe_tools").close()
    logger.info("rag-server shutting down cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
```

- [ ] **Step 4: Run, expect PASS.** If the installed `mcp` minor differs from this skeleton (e.g. `call_tool` return wrapper, or `InitializationOptions` location), adapt the imports/handlers and record the deviation in the commit message.
- [ ] **Step 5: Commit** — `git commit -m "feat(rag_server): MCP stdio entry with logger isolation + SIGTERM"`

---

## Task 12.1: Copilot wiring doc + canonical MCP config + README

**Files:**
- Create: `reference-impl/configs/mcp.json`
- Create: `reference-impl/docs/SETUP-COPILOT.md`
- Create: `reference-impl/README.md`

- [ ] **Step 1: Write `configs/mcp.json`** (canonical; abs paths are placeholders the operator edits):

```json
{
  "mcpServers": {
    "codescribe-rag": {
      "command": "uv",
      "args": ["run", "python", "-m", "codescribe_rag.servers.rag_server"],
      "env": {
        "RAG_DB_PATH": "/abs/path/to/reference-impl/indices/rag.db",
        "RAG_LOG_PATH": "/abs/path/to/reference-impl/logs/rag-server.log",
        "RAG_CONFIG": "/abs/path/to/reference-impl/configs/rag.yaml"
      }
    }
  }
}
```

- [ ] **Step 2: Write `docs/SETUP-COPILOT.md`** following spec §6 — privacy gate first (issue #6 §3 caveat: every retrieved bug title/diff/author goes to GitHub's cloud and onward; disqualified for NDA codebases), then tier/MCP verification (`code --list-extensions --show-versions | grep -i copilot`, the settings-key drift list `github.copilot.chat.mcpServers` → `chat.mcp.servers` → Settings UI "mcp", and the fallback harnesses Cursor/Continue/Claude Code), Day-0 setup (`p4 login`; Bugzilla key → `~/.config/codescribe_rag/bugzilla.key` chmod 600; `hf download BAAI/bge-large-en-v1.5 --local-dir ~/.hf-models/bge-large-en-v1.5`; `uv sync --extra rag --extra mcp`), bootstrap (`uv run python -m codescribe_rag.rag index --bootstrap`), MCP Inspector check (`npx @modelcontextprotocol/inspector uv run python -m codescribe_rag.servers.rag_server`), the exact VS Code `settings.json` block (both the `uv run` form from `configs/mcp.json` and a plain `"command": "python"` + abs-venv fallback), `Developer: Reload Window` + tool-picker verification, nightly cron (`0 3 * * * cd /abs/.../reference-impl && uv run python -m codescribe_rag.rag index >> logs/cron.log 2>&1`), and day-to-day usage + `tail -f logs/rag-server-*.log` to confirm dispatch.

- [ ] **Step 3: Write `README.md`** — what it is + the read-only/cloud-coupled caveat; `uv sync --extra rag --extra mcp --extra dev`; test commands (`uv run pytest -m "not slow"` and `uv run pytest -m slow`); pointers to `docs/SETUP-COPILOT.md`, the spec, and this plan.

- [ ] **Step 4: Full fast suite green**

Run: `cd reference-impl && uv run pytest -m "not slow" -q`
Expected: all pass.

- [ ] **Step 5: Commit** — `git commit -m "docs(rag_server): Copilot Chat setup guide + canonical mcp.json + README"`

---

## Final verification

- [ ] `cd reference-impl && uv run pytest -q` (full suite, including slow) — all green.
- [ ] `uv run python -m codescribe_rag.servers.rag_server` boots and is inspectable via MCP Inspector (operator step, documented in SETUP-COPILOT.md).
- [ ] Static checks (no-writes, no-stdout, no-egress, no-p4python, no-shell-True, no-unsafe-serialisation, no-credentials) all pass.
- [ ] Operator-only (needs real servers + model): `p4-probe`/`bz-probe` list 5 records each; `index --bootstrap` populates the store; `query` returns results in <300 ms.
