"""``Harness`` ABC — the pluggable agent-harness interface.

A harness owns the user-facing agent loop: slash commands, tool execution,
streaming output. Concrete implementations (e.g.
:class:`~codescribe_train.harness.claw.ClawCodeHarness`,
:class:`~codescribe_train.harness.noop.NoOpHarness`) wrap a specific binary or
in-process loop behind this small surface.

Design notes
------------

* **Network-free runtime by contract.** ``health_check`` is the *only*
  method permitted to make a network call; everything else is local
  filesystem + subprocess only.
* **Sandbox composition by prefix.** ``sandbox_args`` returns a command-line
  prefix the caller prepends to whatever launch command the concrete
  harness produces. Default ``[]`` runs the harness in the user shell;
  ``--sandbox docker`` returns a ``docker run --network none`` prefix.
* **Config is data, not code.** ``prepare`` renders YAML into whatever
  on-disk config files the harness expects (``.claude.json`` /
  ``.claw.json`` for claw-code, nothing for noop).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any


class Harness(ABC):
    """Abstract base class for agent-harness adapters.

    Concrete subclasses must implement all four methods. The interface is
    deliberately small so a future native harness can satisfy it without a
    sprawling Python integration.
    """

    @abstractmethod
    def prepare(
        self,
        workdir: Path,
        backend_url: str,
        model: str,
        *,
        harness_config: dict[str, Any],
    ) -> None:
        """Render on-disk config + set up env for one session.

        Implementations write whatever config files the underlying binary
        expects (e.g. ``.claude.json`` / ``.claw.json``) into ``workdir``,
        and stash any per-session state on ``self`` for ``start_session``
        to consume.

        ``harness_config`` is the parsed YAML dict from
        ``configs/harness/*.yaml``. Implementations must not mutate it —
        they should treat it as read-only input data.

        Must not reach the network.
        """

    @abstractmethod
    def start_session(self, stdin: IO[str] | IO[bytes], stdout: IO[str] | IO[bytes]) -> int:
        """Launch the harness as a subprocess and stream IO.

        ``stdin`` / ``stdout`` are the streams the harness should attach to
        — typically ``sys.stdin`` and ``sys.stdout``, but tests pass
        in-memory streams. Returns the subprocess exit code.

        ``prepare`` must have been called first.
        """

    @abstractmethod
    def health_check(self, backend_url: str) -> bool:
        """Return ``True`` if the backend at ``backend_url`` is reachable.

        Implementations probe an OpenAI-compatible endpoint (e.g.
        ``GET {backend_url}/v1/models``). This is the *only* method allowed
        to touch the network.
        """

    @abstractmethod
    def sandbox_args(self) -> list[str]:
        """Return the command-line prefix used to wrap the launch.

        Default (no sandbox) is ``[]``. The Docker variant returns the
        ``["docker", "run", "--rm", ..., "--network", "none", ...]``
        prefix. The caller composes the final command as
        ``[*sandbox_args(), *launch_args]``.
        """
