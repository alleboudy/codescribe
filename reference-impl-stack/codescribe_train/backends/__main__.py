"""Entry point for ``python -m codescribe_train.backends``.

Delegates to :func:`codescribe_train.backends.cli.main`. Kept as a thin shim so
the module-as-script invocation works without forcing eager imports of
the concrete backend implementations.
"""

from __future__ import annotations

import sys

from codescribe_train.backends.cli import main

if __name__ == "__main__":
    sys.exit(main())
