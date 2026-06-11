"""Fill-in-the-middle (FIM) formatter for Qwen 2.5 Coder.

Format (Prefix-Suffix-Middle order, Qwen-native):

    <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}

For each surviving file, mask ``samples_per_file`` random spans of
``min_middle_lines``..``max_middle_lines`` consecutive lines and emit one
Sample per masked span. Splitting is line-based, not token-based — we don't
need a tokenizer at the formatter stage; chunk-length enforcement happens
later when the dataset is fed into the trainer.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Iterator

from codescribe_train.data.formatters.base import Sample
from codescribe_train.data.walker import FileRecord

logger = logging.getLogger(__name__)


def format_fim(
    records: Iterable[FileRecord],
    *,
    rng: random.Random,
    samples_per_file: int = 2,
    min_middle_lines: int = 1,
    max_middle_lines: int = 8,
) -> Iterator[Sample]:
    """Emit FIM samples by masking random line spans inside each file.

    Files with fewer than ``min_middle_lines + 2`` lines (i.e. not enough to
    leave at least one prefix line and one suffix line around the masked span)
    are skipped.
    """
    if min_middle_lines < 1:
        raise ValueError("min_middle_lines must be >= 1")
    if max_middle_lines < min_middle_lines:
        raise ValueError("max_middle_lines must be >= min_middle_lines")

    for record in records:
        try:
            content = record.abspath.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("fim: cannot read %s: %s", record.relpath, e)
            continue
        lines = content.splitlines(keepends=True)
        # Need at least 1 prefix + min_middle + 1 suffix.
        if len(lines) < min_middle_lines + 2:
            continue
        for _ in range(samples_per_file):
            middle_len = rng.randint(min_middle_lines, max_middle_lines)
            # Cap the middle length by what the file can afford (at least 1
            # prefix, 1 suffix). If max would overflow, shrink it.
            max_possible = len(lines) - 2
            middle_len = min(middle_len, max_possible)
            max_start = len(lines) - middle_len - 1  # leave >=1 suffix line
            start = rng.randint(1, max_start)  # leave >=1 prefix line
            end = start + middle_len
            prefix = "".join(lines[:start])
            middle = "".join(lines[start:end])
            suffix = "".join(lines[end:])
            text = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"
            yield Sample(
                text=text,
                metadata={
                    "format": "fim",
                    "src_path": record.relpath,
                    "middle_lines": middle_len,
                    "start_line": start,
                },
            )
