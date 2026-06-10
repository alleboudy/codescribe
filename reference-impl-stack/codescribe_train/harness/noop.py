"""``NoOpHarness`` — minimal in-process harness used as a swap test.

Reads lines from stdin, echoes each one back to stdout (with an ``echo:``
prefix so the round-trip is visible in tests), exits cleanly on EOF. Used
to demonstrate that the :class:`~codescribe_train.harness.base.Harness` interface
is honestly pluggable — switching to ``--harness noop`` does not touch
claw-code or any vendored binary at all.

This harness never reaches the network. ``health_check`` returns ``True``
unconditionally because there is no backend to talk to.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from codescribe_train.harness.base import Harness

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NoOpHarness(Harness):
    """In-process echo harness. Useful as a swap-test target."""

    workdir: Path | None = None
    prepared: bool = False
    backend_url: str | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def prepare(
        self,
        workdir: Path,
        backend_url: str,
        model: str,
        *,
        harness_config: dict[str, Any],
    ) -> None:
        self.workdir = Path(workdir)
        self.backend_url = backend_url
        self.model = model
        self.extra = dict(harness_config or {})
        self.prepared = True
        logger.info("NoOpHarness prepared (workdir=%s)", self.workdir)

    def start_session(self, stdin: IO[str] | IO[bytes], stdout: IO[str] | IO[bytes]) -> int:
        if not self.prepared:
            raise RuntimeError("NoOpHarness.start_session called before prepare()")
        for raw in stdin:
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            line = line.rstrip("\n")
            if not line:
                continue
            response = f"echo: {line}\n"
            payload: bytes | str = (
                response.encode("utf-8") if "b" in getattr(stdout, "mode", "") else response
            )
            stdout.write(payload)
            # Some test streams don't support flush; best-effort.
            with contextlib.suppress(AttributeError, ValueError):
                stdout.flush()
        return 0

    def health_check(self, backend_url: str) -> bool:
        # No backend to probe; the noop harness is for swap-testing the
        # abstraction itself, not exercising the inference layer.
        return True

    def sandbox_args(self) -> list[str]:
        return []
