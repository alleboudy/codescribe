"""Common types for sample formatters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Sample:
    """One training sample, ready to be written to JSONL.

    ``text`` is the final text — formatters apply any required special tokens
    (FIM markers, file separators, chat template) themselves. ``metadata``
    carries diagnostic info that does **not** end up in the model context but
    is useful for sanity-checking the dataset and tracing samples back to the
    source files / commits they came from.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
