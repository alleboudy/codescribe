"""Transports — the *only* place a command actually runs on a host.

A :class:`Transport` runs a shell command on a host and moves files to/from it.
Three implementations:

* :class:`LocalTransport`  — runs on localhost via a subprocess. Powers the
  no-GPU, no-SSH demo (point the train launcher at ``scripts/fake_train.py``).
* :class:`SSHTransport`    — drives a real worker over SSH (asyncssh). Production.
* (tests inject their own in-memory fake implementing the same three methods.)

Security note: a transport runs a *command string* by design — that is the SSH
execution model (``ssh host 'cmd'``). Injection safety therefore lives one layer
up: callers in :mod:`codescribe_fleet.worker` / :mod:`codescribe_fleet.monitor`
``shlex.quote`` every interpolated path/value before it reaches here. This module
is the single audited subprocess boundary; ``tests/test_command_safety.py`` pins
that invariant.
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class TransportError(RuntimeError):
    """A transport-level failure (connection lost, copy failed, timeout)."""


@runtime_checkable
class Transport(Protocol):
    async def run(self, command: str, *, timeout: float | None = None) -> CommandResult: ...
    async def push(self, local: str | Path, remote: str | Path) -> None: ...
    async def pull(self, remote: str | Path, local: str | Path) -> None: ...
    async def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Local                                                                        #
# --------------------------------------------------------------------------- #
class LocalTransport:
    """Run commands on localhost. Used by the demo; faithful to the SSH contract.

    ``push``/``pull`` are local filesystem copies (a localhost "remote" shares the
    coordinator's filesystem), so a job's worker-side checkpoint dir and the
    coordinator's pulled copy are genuinely distinct directories — the demo
    exercises the real copy path, not a no-op.
    """

    def __init__(self, name: str = "local") -> None:
        self.name = name

    async def run(self, command: str, *, timeout: float | None = None) -> CommandResult:
        # We run through `bash -c <command>` (NOT shell=True) because the command
        # is an emulated remote shell line (it may contain `cd ... && ...`).
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise TransportError(f"command timed out after {timeout}s: {command!r}") from exc
        return CommandResult(
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
        )

    async def push(self, local: str | Path, remote: str | Path) -> None:
        await asyncio.to_thread(_copy_tree_or_file, Path(local), Path(remote))

    async def pull(self, remote: str | Path, local: str | Path) -> None:
        await asyncio.to_thread(_copy_tree_or_file, Path(remote), Path(local))

    async def close(self) -> None:  # nothing to tear down
        return None


def _copy_tree_or_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise TransportError(f"source path does not exist: {src}")
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# --------------------------------------------------------------------------- #
# SSH (production)                                                             #
# --------------------------------------------------------------------------- #
class SSHTransport:
    """Drive a real worker over SSH. Requires the ``ssh`` extra (asyncssh).

    asyncssh is imported lazily so the package (and its test suite / local demo)
    works without it installed. Call :meth:`connect` before use, or let
    :func:`connect_ssh` do it for you.
    """

    def __init__(
        self,
        host: str,
        *,
        username: str | None = None,
        port: int = 22,
        client_keys: list[str] | None = None,
        known_hosts: str | None = None,
    ) -> None:
        self.host = host
        self.username = username
        self.port = port
        self.client_keys = client_keys
        # known_hosts=None means "use the user's ~/.ssh/known_hosts" in asyncssh,
        # which still verifies the host key. We never disable host-key checking.
        self.known_hosts = known_hosts
        self._conn = None  # asyncssh.SSHClientConnection once connected

    async def connect(self) -> None:
        try:
            import asyncssh
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise TransportError(
                "SSHTransport requires the `ssh` extra: `uv sync --extra ssh`"
            ) from exc
        self._conn = await asyncssh.connect(
            self.host,
            port=self.port,
            username=self.username,
            client_keys=self.client_keys,
            known_hosts=self.known_hosts,
        )
        logger.info("ssh connected: %s@%s:%d", self.username or "", self.host, self.port)

    def _require_conn(self):
        if self._conn is None:
            raise TransportError(f"SSHTransport to {self.host} is not connected")
        return self._conn

    async def run(self, command: str, *, timeout: float | None = None) -> CommandResult:
        conn = self._require_conn()
        try:
            result = await conn.run(command, timeout=timeout, check=False)
        except Exception as exc:  # asyncssh.ProcessError, TimeoutError, ...
            raise TransportError(f"ssh run failed on {self.host}: {exc}") from exc
        return CommandResult(
            returncode=result.exit_status if result.exit_status is not None else -1,
            stdout=_as_text(result.stdout),
            stderr=_as_text(result.stderr),
        )

    async def push(self, local: str | Path, remote: str | Path) -> None:
        # Contract (matches LocalTransport): the children of a directory `local`
        # end up directly under `remote` — NOT nested one level deeper. asyncssh's
        # scp nests the source under `remote` when `remote` already exists as a
        # directory, so for a directory we scp into `remote`'s PARENT (which the
        # caller has created) and rely on basename(local) == basename(remote)
        # (callers guarantee this). For a file we scp to the full remote path.
        import asyncssh

        conn = self._require_conn()
        dest = _scp_push_dest(Path(local), str(remote))
        try:
            await asyncssh.scp(str(local), (conn, dest), recurse=True)
        except Exception as exc:
            raise TransportError(f"scp push to {self.host} failed: {exc}") from exc

    async def pull(self, remote: str | Path, local: str | Path) -> None:
        # Contract (matches LocalTransport): the children of ``remote`` end up
        # directly under ``local``. scp creates ``<dest>/<basename(remote)>``, so we
        # scp into ``local``'s PARENT and rely on basename(remote) == basename(local)
        # (the Worker guarantees this: both are the job id).
        import asyncssh

        conn = self._require_conn()
        dest_parent = Path(local).parent
        dest_parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncssh.scp((conn, str(remote)), str(dest_parent), recurse=True)
        except Exception as exc:
            raise TransportError(f"scp pull from {self.host} failed: {exc}") from exc

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            await self._conn.wait_closed()
            self._conn = None


async def connect_ssh(
    host: str,
    *,
    username: str | None = None,
    port: int = 22,
    client_keys: list[str] | None = None,
    known_hosts: str | None = None,
) -> SSHTransport:
    t = SSHTransport(
        host,
        username=username,
        port=port,
        client_keys=client_keys,
        known_hosts=known_hosts,
    )
    await t.connect()
    return t


def _scp_push_dest(local_path: Path, remote: str) -> str:
    """Return the scp destination for pushing ``local`` to ``remote``.

    Directory → ``remote``'s parent (so scp recreates the dir AS ``remote``, with
    children directly under it, matching LocalTransport). File → the full
    ``remote`` path. The caller ensures the returned dir exists and, for
    directories, that ``basename(local) == basename(remote)``.
    """
    if local_path.is_dir():
        return posixpath.dirname(remote)
    return remote


def _as_text(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)
