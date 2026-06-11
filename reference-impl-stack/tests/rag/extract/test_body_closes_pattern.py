"""BODY_CLOSES_PATTERN regex: positive and negative cases from the design spec.

The spec pattern is:

    r"\\b(?:close[sd]?|fix(?:es|ed)?|resolve[sd]?)\\s+#(\\d{1,6})\\b"

with `re.IGNORECASE`. These tests pin both the matcher and the convenience
extractor `extract_closing_issue_refs_from_pr_body`.
"""

from __future__ import annotations

from codescribe_train.rag.extract.links import (
    BODY_CLOSES_PATTERN,
    extract_closing_issue_refs_from_pr_body,
)

# --- positives (spec) ---------------------------------------------------


def test_closes_lowercase_single_space() -> None:
    assert BODY_CLOSES_PATTERN.search("Closes #42") is not None


def test_fixes_double_space() -> None:
    # "Fixes  #1337" — note the two spaces between the keyword and the #.
    assert BODY_CLOSES_PATTERN.search("Fixes  #1337") is not None


def test_resolves_lowercase() -> None:
    assert BODY_CLOSES_PATTERN.search("resolves #7") is not None


# --- positives (other keyword forms in the spec alternation) -----------


def test_close_no_suffix() -> None:
    assert BODY_CLOSES_PATTERN.search("close #1") is not None


def test_closed_past_tense() -> None:
    assert BODY_CLOSES_PATTERN.search("Closed #99") is not None


def test_fix_no_suffix() -> None:
    assert BODY_CLOSES_PATTERN.search("fix #2") is not None


def test_fixed_past_tense() -> None:
    assert BODY_CLOSES_PATTERN.search("Fixed #200") is not None


def test_resolve_no_suffix() -> None:
    assert BODY_CLOSES_PATTERN.search("resolve #3") is not None


def test_resolved_past_tense() -> None:
    assert BODY_CLOSES_PATTERN.search("Resolved #50") is not None


# --- negatives (spec) --------------------------------------------------


def test_negative_close_that_file() -> None:
    """`close that file` must not match — there is no `#NNN` after it."""
    assert BODY_CLOSES_PATTERN.search("close that file") is None


def test_negative_fixing_a_typo() -> None:
    """`fixing a typo` must not match — `fixing` is not in the alternation."""
    assert BODY_CLOSES_PATTERN.search("fixing a typo") is None


# --- extractor: groups + multiplicities --------------------------------


def test_extractor_returns_empty_for_none_body() -> None:
    assert extract_closing_issue_refs_from_pr_body(None) == set()  # type: ignore[arg-type]


def test_extractor_returns_empty_for_empty_body() -> None:
    assert extract_closing_issue_refs_from_pr_body("") == set()


def test_extractor_single_match() -> None:
    assert extract_closing_issue_refs_from_pr_body("Closes #42") == {42}


def test_extractor_multiple_matches_dedupes() -> None:
    body = "Closes #1\nFixes #2\nresolves #1\n"
    assert extract_closing_issue_refs_from_pr_body(body) == {1, 2}


def test_extractor_ignores_in_word_match() -> None:
    """A `#42` not preceded by a closing keyword must not be captured."""
    assert extract_closing_issue_refs_from_pr_body("See #42 for context.") == set()
