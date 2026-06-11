"""PRChunk dataclass shape mirrors the pr_chunks schema columns.

A regression here (e.g. renaming ``hunk_index`` to ``hunk_id``) would
silently break ``Store.upsert_pull``'s chunk fan-out; this test pins the
contract between the chunker and the store.
"""

from __future__ import annotations

import dataclasses

from codescribe_train.rag.embed.chunker import PRChunk


def test_prchunk_is_a_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(PRChunk)
    fields = {f.name: f.type for f in dataclasses.fields(PRChunk)}
    # Match the pr_chunks schema columns (minus the auto-assigned chunk_id PK).
    assert set(fields) == {"pr_number", "file_path", "hunk_index", "chunk_text"}
    # frozen=True: PRChunks are write-once values handed to the store.
    chunk = PRChunk(
        pr_number=1, file_path="a.py", hunk_index=0, chunk_text="diff"
    )
    try:
        chunk.pr_number = 2  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("PRChunk must be frozen so it can't mutate post-insert")
