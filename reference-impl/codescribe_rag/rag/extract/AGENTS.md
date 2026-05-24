# rag.extract — package rules

Pure-functional regex extraction + scoring. No I/O. Deterministic. Cheap to re-run.

## Hard rules

- Regex patterns live in `links.py` as module-level constants, each with a unit test.
- `extract_cl_refs(text: str) -> set[int]` and `extract_bug_refs(text: str) -> set[int]` are the canonical entry points.
- Confidence scoring lives in `pairing.py` with weights from `RagConfig.pairing` so they're tunable without code changes.
- Strict default threshold: 0.8. Document this prominently in the module docstring.

## Anti-patterns

- Do NOT match obvious false positives: "CL ages" (style spec), "change everything" (English prose). Patterns require digit count ≥4 to avoid this.
- Do NOT extract bug refs from auto-generated CL descriptions (Swarm bots that say "Bug 0: placeholder"). Filter bug IDs below a configured `min_bug_id` (default 1000).
- Do NOT pair a bug to a CL on a single weak signal. Threshold the SUM; never accept a single 0.3 signal as a link.
