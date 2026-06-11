"""QLoRA training pipeline.

Phase 2 of codescribe-train. Loads a fine-tuned base coder model via Unsloth, attaches
a LoRA adapter, runs SFTTrainer over the JSONL splits emitted by `data/`, and
exports a Q4_K_M GGUF ready for `llama-server`.

Strictly local — no W&B, no Hugging Face Hub push, no telemetry.
"""
