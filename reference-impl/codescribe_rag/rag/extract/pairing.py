"""Confidence-scored bug<->CL pairing.

Strict default threshold is 0.8 (see ``PairingConfig.threshold``): typically a
single explicit cross-reference (0.5) plus corroborating signals, or a
bidirectional cross-reference. Below threshold, retrieval would surface
unrelated fixes, so we do not record the link.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..config import PairingConfig
from .links import extract_bug_refs, extract_cl_refs


@dataclass(frozen=True, slots=True)
class BugView:
    id: int
    summary: str
    status: str
    resolution: str
    assigned_to: str
    last_change_time: datetime
    comment_text: str          # all comment bodies concatenated


@dataclass(frozen=True, slots=True)
class CLView:
    cl: int
    author: str
    description: str
    submitted_at: datetime


@dataclass(frozen=True, slots=True)
class FixLink:
    bug_id: int
    cl_number: int
    signals: dict[str, float]
    confidence: float


def score_pair(bug: BugView, cl: CLView, cfg: PairingConfig) -> FixLink:
    w = cfg.weights
    signals: dict[str, float] = {}
    if cl.cl in extract_cl_refs(bug.comment_text, cfg.min_cl_id):
        signals["bug_comment_cites_cl"] = w.bug_comment_cites_cl
    if bug.id in extract_bug_refs(cl.description, cfg.min_bug_id):
        signals["cl_desc_cites_bug"] = w.cl_desc_cites_bug
    if abs((cl.submitted_at - bug.last_change_time).days) <= cfg.temporal_signal_days:
        signals["temporal_proximity"] = w.temporal_proximity
    if bug.assigned_to and bug.assigned_to == cl.author:
        signals["assignee_author_match"] = w.assignee_author_match
    if bug.status in ("RESOLVED", "CLOSED") and bug.resolution == "FIXED":
        signals["bug_status_fixed"] = w.bug_status_fixed
    confidence = min(1.0, sum(signals.values()))
    return FixLink(bug.id, cl.cl, signals, confidence)


def pair_bugs_to_cls(bugs, cls, cfg: PairingConfig) -> list[FixLink]:
    """Score every bug x CL pair within the temporal candidate window; keep links
    at or above ``cfg.threshold``."""
    window = timedelta(days=cfg.candidate_window_days)
    out: list[FixLink] = []
    for bug in bugs:
        for cl in cls:
            if abs(cl.submitted_at - bug.last_change_time) > window:
                continue
            link = score_pair(bug, cl, cfg)
            if link.confidence >= cfg.threshold:
                out.append(link)
    return out
