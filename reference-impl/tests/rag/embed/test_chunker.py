from __future__ import annotations

from codescribe_rag.rag.embed.chunker import CLChunk, chunk_diff


class CL:
    def __init__(self, diff):
        self.diff_text = diff


DIFF = (
    "--- a/Foo.java\n+++ b/Foo.java\n@@ -1,2 +1,3 @@\n-x\n+y\n+z\n"
    "--- a/Bar.java\n+++ b/Bar.java\n@@ -5,1 +5,2 @@\n-p\n+q\n"
)


def test_per_file_isolation():
    paths = {c.file_path for c in chunk_diff(CL(DIFF))}
    assert paths == {"Foo.java", "Bar.java"}


def test_respects_max_chars():
    big = "--- a/Big.txt\n+++ b/Big.txt\n@@ -1 +1 @@\n" + ("+line\n" * 5000)
    for c in chunk_diff(CL(big), max_chars=2000):
        assert len(c.text.encode("utf-8")) <= 2000


def test_never_splits_inside_hunk_header():
    for c in chunk_diff(CL(DIFF)):
        assert not c.text.startswith("@ ")


def test_utf8_boundary_preserved():
    big = "--- a/u.txt\n+++ b/u.txt\n@@ -1 +1 @@\n" + ("+éééé\n" * 2000)
    for c in chunk_diff(CL(big), max_chars=1000):
        c.text.encode("utf-8").decode("utf-8")  # must not raise
    assert isinstance(CLChunk("f", 0, "t").text, str)
