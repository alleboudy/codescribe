"""codescribe-fleet — parallel & distributed QLoRA fleet orchestrator.

Reference implementation for codescribe issue #3 ("Fleet guide: parallel &
distributed QLoRA fine-tuning across N workstation laptops"). It turns the
pseudocode in §5.3 / §6 of that issue into runnable code.

Two paths, mirroring the issue:

* **Path A** — parallel hyperparameter sweep. One independent training job per
  worker, no gradient sync. This is the recommended path for an 8 GB-VRAM laptop
  fleet on a gigabit LAN. See :mod:`codescribe_fleet.orchestrator`.
* **Path B** — distributed data-parallel (`torchrun` + NCCL). Rarely worth it on
  this hardware class; see :mod:`codescribe_fleet.ddp` for the launcher/env
  builders and ``scripts/run_ddp.sh`` for the per-node entry point.

The orchestrator is transport-agnostic: it drives workers through a
:class:`~codescribe_fleet.transport.Transport`. Production uses
``SSHTransport`` (asyncssh); the local demo uses ``LocalTransport`` (subprocess);
tests inject an in-memory fake. That is what makes the whole thing runnable on a
single laptop with no GPU — point it at ``scripts/fake_train.py``.
"""

__version__ = "0.1.0"
