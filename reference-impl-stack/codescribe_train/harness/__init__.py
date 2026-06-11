"""``codescribe_train.harness`` — pluggable agent-harness layer.

The harness is the user-facing CLI agent loop (slash commands, tool calls,
streaming render). The default implementation, :mod:`codescribe_train.harness.claw`,
wraps the **vendored** ``ultraworkers/claw-code`` Rust binary at a pinned
commit SHA. The :class:`~codescribe_train.harness.base.Harness` ABC exists so a
future native harness can be slotted in without touching ``data/``,
``train/``, or ``backends/``.

Trust posture: the default harness is treated as an **opaque, not-fully-
trusted subprocess**. The Python wrapper:

* never auto-updates the submodule;
* never reaches the network in its runtime path;
* generates ``.claude.json`` / ``.claw.json`` config files only — no other
  IPC with the binary;
* defaults to ``sandbox_args() == []`` for direct execution, with an opt-in
  ``--sandbox docker`` path that wraps the binary in a ``--network none``
  container.

See :class:`~codescribe_train.harness.base.Harness` for the interface contract.
"""

from __future__ import annotations
