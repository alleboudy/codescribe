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
    submitted_at: datetime          # timezone-aware UTC
    description: str
    files: tuple[FileChange, ...]
    diff_text: str
    diff_truncated: bool


class P4AuthExpired(RuntimeError):
    """p4 ticket expired; operator must re-run ``p4 login``."""


class P4Source:
    """Read-only wrapper over the ``p4`` CLI (subprocess; no p4python).

    Each method spawns a fresh ``p4`` subprocess. ``extra_argv`` lets tests drive
    a stub interpreter (e.g. ``binary=sys.executable, extra_argv=[stub_path]``).
    """

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def iter_changes_since(
        self, last_cl: int, depot_path: str = "//depot/...", batch_size: int = 200
    ) -> Iterator[P4Change]:
        """Yield submitted changes with ``change > last_cl``, ascending."""
        cursor = last_cl
        while True:
            page = self._list_changes(cursor, depot_path, batch_size)
            if not page:
                return
            # ``p4 changes`` returns DESC order; reverse to ascend.
            for cl_meta in reversed(page):
                yield self.describe(int(cl_meta["change"]))
            highest = max(int(m["change"]) for m in page)
            if highest <= cursor:
                return
            cursor = highest

    def describe(self, cl: int) -> P4Change:
        """Return a fully-populated P4Change for CL number ``cl``."""
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
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
        # Minimal environment; pass only what p4 needs (notably P4TICKETS).
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
            raise P4AuthExpired("p4 ticket expired; operator must re-run `p4 login`.") from exc

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
