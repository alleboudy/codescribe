from __future__ import annotations

from datetime import datetime, timezone

from codescribe_rag.rag.config import PairingConfig
from codescribe_rag.rag.extract.pairing import BugView, CLView, pair_bugs_to_cls, score_pair

UTC = timezone.utc
CFG = PairingConfig()


def mk_bug(**kw):
    base = dict(id=4567, summary="NPE", status="RESOLVED", resolution="FIXED",
                assigned_to="alice", last_change_time=datetime(2026, 1, 10, tzinfo=UTC),
                comment_text="")
    base.update(kw)
    return BugView(**base)


def mk_cl(**kw):
    base = dict(cl=12345, author="alice", description="",
                submitted_at=datetime(2026, 1, 12, tzinfo=UTC))
    base.update(kw)
    return CLView(**base)


def test_bug_comment_cites_cl_signal():
    link = score_pair(mk_bug(comment_text="Fixed in CL 12345"), mk_cl(), CFG)
    assert link.signals["bug_comment_cites_cl"] == 0.5


def test_cl_desc_cites_bug_signal():
    link = score_pair(mk_bug(), mk_cl(description="Bug 4567: fix NPE"), CFG)
    assert link.signals["cl_desc_cites_bug"] == 0.5


def test_temporal_proximity_within_window():
    near = score_pair(mk_bug(), mk_cl(submitted_at=datetime(2026, 1, 15, tzinfo=UTC)), CFG)
    far = score_pair(mk_bug(), mk_cl(submitted_at=datetime(2026, 2, 20, tzinfo=UTC)), CFG)
    assert near.signals.get("temporal_proximity") == 0.3
    assert "temporal_proximity" not in far.signals


def test_status_and_author_signals():
    link = score_pair(mk_bug(), mk_cl(), CFG)
    assert link.signals["assignee_author_match"] == 0.2
    assert link.signals["bug_status_fixed"] == 0.1


def test_threshold_boundary():
    strong = score_pair(mk_bug(comment_text="see CL 12345"),
                        mk_cl(description="fixes bug 4567"), CFG)
    assert strong.confidence >= 0.8
    weak = score_pair(mk_bug(), mk_cl(), CFG)  # temporal+author+status = 0.6
    assert round(weak.confidence, 3) == 0.6


def test_pair_filters_by_threshold_and_window():
    bugs = [mk_bug(comment_text="CL 12345")]
    cls = [mk_cl(description="fixes bug 4567"),
           mk_cl(cl=999999, submitted_at=datetime(2027, 6, 1, tzinfo=UTC))]
    links = pair_bugs_to_cls(bugs, cls, CFG)
    assert [l.cl_number for l in links] == [12345]
