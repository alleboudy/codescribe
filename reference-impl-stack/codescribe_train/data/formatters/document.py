"""File-as-document formatter.

Wraps each file's content with the Qwen 2.5 Coder ``<|repo_name|>`` /
``<|file_sep|>`` markers and the file's relpath. Best used for
continued-pretraining-style examples where the model reads whole files and
learns project-specific structure.

These markers are part of Qwen 2.5 Coder's pretraining vocabulary; in Instruct
mode they're treated as ordinary text but they bracket files cleanly when
continued-pretraining the base.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

from codescribe_train.data.formatters.base import Sample
from codescribe_train.data.walker import FileRecord

logger = logging.getLogger(__name__)


def format_document(
    records: Iterable[FileRecord],
    *,
    repo_name: str,
) -> Iterator[Sample]:
    """Emit one Sample per file: ``<|repo_name|>...\\n<|file_sep|>relpath\\n{content}``."""
    repo_header = f"<|repo_name|>{repo_name}\n"
    for record in records:
        try:
            content = record.abspath.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("document: cannot read %s: %s", record.relpath, e)
            continue
        text = f"{repo_header}<|file_sep|>{record.relpath}\n{content}"
        yield Sample(
            text=text,
            metadata={
                "format": "document",
                "src_path": record.relpath,
                "size_bytes": record.size_bytes,
            },
        )
