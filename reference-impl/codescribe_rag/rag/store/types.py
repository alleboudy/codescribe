from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FileDiff:
    file_path: str
    diff: str


@dataclass(frozen=True, slots=True)
class CLDiff:
    cl_number: int
    author: str
    submitted_at: str
    description: str
    files: tuple[FileDiff, ...]


@dataclass(frozen=True, slots=True)
class RetrievedBugFix:
    bug_id: int
    summary: str
    severity: str
    status: str
    score: float                        # RRF score
    fix_cl: int | None                  # None if no fix above the confidence threshold
    fix_diff_excerpt: str | None
    confidence: float                   # confidence of the linked fix (NaN if unlinked)
