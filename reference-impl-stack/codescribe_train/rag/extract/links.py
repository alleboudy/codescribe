"""Link-extraction regexes for PR bodies → closing issue references.

Pure functional, no I/O. See the design spec for the spec.

The regex itself is verbatim from the design spec:

    \\b(?:close[sd]?|fix(?:es|ed)?|resolve[sd]?)\\s+#(\\d{1,6})\\b

— case-insensitive. The pipeline feeds the captured issue numbers
into `pair_via_body_keyword` to produce `IssuePRLink` rows.
"""

from __future__ import annotations

import re

BODY_CLOSES_PATTERN = re.compile(
    r"\b(?:close[sd]?|fix(?:es|ed)?|resolve[sd]?)\s+#(\d{1,6})\b",
    re.IGNORECASE,
)


def extract_closing_issue_refs_from_pr_body(body: str | None) -> set[int]:
    """Return every distinct issue number referenced as a closing keyword.

    ``body`` is the PR description (may be ``None`` for PRs created with
    an empty body). The result is a set so duplicates collapse — the
    pipeline only cares about distinct (issue, PR) pairs.
    """
    if not body:
        return set()
    return {int(match.group(1)) for match in BODY_CLOSES_PATTERN.finditer(body)}
