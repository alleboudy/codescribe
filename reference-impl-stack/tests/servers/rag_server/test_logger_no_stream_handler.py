"""``_configure_logging`` leaves the root logger with FileHandler ONLY.

Spec test from the design spec:

  > test_logger_no_stream_handler.py — root logger has only
  > FileHandler.

The stdio JSON-RPC channel breaks the moment any logger writes to
``stdout`` (or ``stderr`` — clients may also parse those). The server
must aggressively reset the root logger to a single FileHandler at
startup, even when the parent process (the harness) has already
installed a console handler.

``FileHandler`` itself inherits from :class:`logging.StreamHandler`,
so the precise contract is:

* exactly one handler on the root logger after ``_configure_logging``;
* that handler **is** a :class:`logging.FileHandler` (whose ``stream``
  attribute points at the log file, not stdout/stderr).
"""

from __future__ import annotations

import logging
from pathlib import Path


def test_configure_logging_replaces_all_handlers_with_file_handler(
    tmp_path: Path,
) -> None:
    """A pre-installed StreamHandler is removed; one FileHandler remains."""
    from codescribe_train.servers.rag_server.__main__ import _configure_logging

    root = logging.getLogger()
    saved = list(root.handlers)
    # Simulate a parent harness that installed a console handler.
    import sys

    pollutant = logging.StreamHandler(sys.stdout)
    root.addHandler(pollutant)
    try:
        log_path = tmp_path / "rag-server.log"
        _configure_logging(log_path)
        handlers = list(root.handlers)
        assert len(handlers) == 1, (
            f"expected exactly one root handler; got {len(handlers)}: {handlers!r}"
        )
        only = handlers[0]
        assert isinstance(only, logging.FileHandler), (
            f"expected a FileHandler on the root logger; got {type(only).__name__}"
        )
        # The handler must point at the requested log file, not stdio.
        assert Path(only.baseFilename).resolve() == log_path.resolve()
        # And the pollutant we installed must be gone.
        assert pollutant not in handlers
    finally:
        # Restore the original handler list so we don't pollute later tests.
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved:
            root.addHandler(h)
