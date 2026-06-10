"""Sample formatters for the data pipeline.

Three orthogonal formatters that take filtered/deduped :class:`FileRecord`
streams (and, for diff-instr, a git repo path) and emit text-only training
:class:`Sample` objects suitable for SFTTrainer-style training.
"""

from codescribe_train.data.formatters.base import Sample
from codescribe_train.data.formatters.diff_instr import format_diff_instructions
from codescribe_train.data.formatters.document import format_document
from codescribe_train.data.formatters.fim import format_fim

__all__ = [
    "Sample",
    "format_diff_instructions",
    "format_document",
    "format_fim",
]
