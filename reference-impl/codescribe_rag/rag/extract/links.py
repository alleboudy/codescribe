from __future__ import annotations

import re

# CL reference patterns -- matched in bug COMMENT text.
CL_PATTERNS = [
    re.compile(r"\bCL\s*#?\s*(\d{4,})\b", re.IGNORECASE),
    re.compile(r"\bchange(?:list)?\s+(\d{4,})\b", re.IGNORECASE),
    re.compile(r"@(?:p4|swarm|change)[/_]?(\d{4,})", re.IGNORECASE),
    re.compile(r"/changes/(\d{4,})"),                  # Swarm URLs
    re.compile(r"\bsubmitted\s+as\s+(\d{4,})\b", re.IGNORECASE),
]

# Bug reference patterns -- matched in CL DESCRIPTION text.
BUG_PATTERNS = [
    re.compile(r"\bbug\s*[-#]?\s*(\d{4,})\b", re.IGNORECASE),
    re.compile(r"\bBZ\s*[-#]?\s*(\d{4,})\b", re.IGNORECASE),
    re.compile(r"show_bug\.cgi\?id=(\d{4,})"),
    re.compile(r"\bfix(?:es|ed)?\s+(?:bug\s+)?(\d{4,})\b", re.IGNORECASE),
    re.compile(r"\bclos(?:es|ed)?\s+(?:bug\s+)?(\d{4,})\b", re.IGNORECASE),
]


def _extract(text: str, patterns: list[re.Pattern], min_id: int) -> set[int]:
    out: set[int] = set()
    for pat in patterns:
        for m in pat.finditer(text or ""):
            n = int(m.group(1))
            if n >= min_id:
                out.add(n)
    return out


def extract_cl_refs(text: str, min_cl_id: int = 1000) -> set[int]:
    """CL numbers referenced in ``text`` (e.g. a bug comment). IDs < min_cl_id dropped."""
    return _extract(text, CL_PATTERNS, min_cl_id)


def extract_bug_refs(text: str, min_bug_id: int = 1000) -> set[int]:
    """Bug IDs referenced in ``text`` (e.g. a CL description). IDs < min_bug_id dropped."""
    return _extract(text, BUG_PATTERNS, min_bug_id)
