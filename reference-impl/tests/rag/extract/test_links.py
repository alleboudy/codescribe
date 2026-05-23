from __future__ import annotations

import pytest

from codescribe_rag.rag.extract.links import extract_bug_refs, extract_cl_refs


@pytest.mark.parametrize("text,expected", [
    ("Fixed in CL 12345", {12345}),
    ("see change 67890 for the fix", {67890}),
    ("submitted as 45678", {45678}),
    ("@swarm/13579 landed", {13579}),
    ("https://swarm/changes/24680", {24680}),
])
def test_cl_positives(text, expected):
    assert extract_cl_refs(text) == expected


@pytest.mark.parametrize("text", [
    "CL ages like wine", "change everything now", "Cl- is chloride", "CL 12",
])
def test_cl_negatives(text):
    assert extract_cl_refs(text) == set()


@pytest.mark.parametrize("text,expected", [
    ("Bug 4567: NPE on startup", {4567}),
    ("closes bug 9999", {9999}),
    ("fixes 8888", {8888}),
    ("BZ-7777 regression", {7777}),
    ("show_bug.cgi?id=5555", {5555}),
])
def test_bug_positives(text, expected):
    assert extract_bug_refs(text) == expected


def test_min_id_threshold():
    assert extract_cl_refs("CL 0999") == set()      # below 1000
    assert extract_bug_refs("bug 12") == set()
    assert extract_cl_refs("CL 1000") == {1000}
