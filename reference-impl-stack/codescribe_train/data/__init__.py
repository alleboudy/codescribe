"""Generic git-repo → train/val/test JSONL pipeline.

Phase 1 of codescribe-train. Walks any git repository, applies a YAML config,
deduplicates, formats samples (FIM / document / diff-instruction), and writes
deterministic splits ready for QLoRA fine-tuning.
"""
