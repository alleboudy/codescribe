"""Fallback export: LoRA → fp16 merged dir via Unsloth, then GGUF + quantize via our vendored llama.cpp.

Three failures to work around in `python -m codescribe_train.train export`:

1. The standard PEFT path (`AutoModelForCausalLM` with `device_map='auto'`
   + `PeftModel.from_pretrained`) doesn't work on 8 GB VRAM because of a
   compounding `device_map` overflow and a PEFT key-mismatch on
   Unsloth-saved adapters. See issue #3.

2. Unsloth's `save_pretrained_gguf` tries to apt-install llama.cpp
   itself, asking for interactive confirmation. Background processes
   crash with EOFError. We don't want it apt-installing llama.cpp anyway
   — we have a vendored, pinned-SHA build at vendor/llama.cpp/.

3. Unsloth's GGUF exporter doesn't see fp16 shards already cached by
   transformers, re-downloads them. ~15 GB redundant traffic. See
   issue #3.

Solution: use Unsloth ONLY for the LoRA merge (which it does well + with
low memory), then call our vendored llama.cpp directly for the convert
+ quantize. No interactive prompts, no apt install, no extra download.

Run from the repo root with the train extras + flash-attn installed:

    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-} \
    .venv/bin/python scripts/export_via_unsloth.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Strictly-local kill switches before any HF / torch imports.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ADAPTER_DIR = Path("checkpoints/sample-qwen7b-lora-v1")
MERGED_DIR = Path("checkpoints/sample-qwen7b-merged-v1")
GGUF_F16 = Path("checkpoints/sample-qwen7b-f16.gguf")
GGUF_Q4 = Path("checkpoints/sample-qwen7b-q4_k_m.gguf")
LLAMA_CPP = Path("vendor/llama.cpp")


def step_merge() -> None:
    """LoRA → fp16 merged HuggingFace model dir via Unsloth."""
    if MERGED_DIR.is_dir() and any(MERGED_DIR.glob("*.safetensors")):
        print(f"[skip] merged model already at {MERGED_DIR}")
        return
    print(f"[merge] loading {ADAPTER_DIR} via Unsloth (4-bit, seq_len=1024)...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )
    print(f"[merge] save_pretrained_merged → {MERGED_DIR} (16-bit fp16)...")
    MERGED_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"[merge] OK")


def step_convert_to_gguf_f16() -> None:
    """Merged HF dir → fp16 GGUF via vendor/llama.cpp/convert_hf_to_gguf.py."""
    if GGUF_F16.is_file():
        print(f"[skip] fp16 GGUF already at {GGUF_F16}")
        return
    converter = LLAMA_CPP / "convert_hf_to_gguf.py"
    if not converter.is_file():
        print(f"ERROR: {converter} not found", file=sys.stderr)
        sys.exit(1)
    print(f"[convert] {converter} → {GGUF_F16}")
    subprocess.run(
        [
            sys.executable,
            str(converter),
            str(MERGED_DIR),
            "--outfile",
            str(GGUF_F16),
            "--outtype",
            "f16",
        ],
        check=True,
    )
    print(f"[convert] OK ({GGUF_F16.stat().st_size / 1e9:.2f} GB)")


def step_quantize_to_q4_k_m() -> None:
    """fp16 GGUF → Q4_K_M GGUF via vendor/llama.cpp/build/bin/llama-quantize."""
    if GGUF_Q4.is_file():
        print(f"[skip] Q4_K_M GGUF already at {GGUF_Q4}")
        return
    quantize_bin = LLAMA_CPP / "build" / "bin" / "llama-quantize"
    if not quantize_bin.is_file():
        print(f"ERROR: {quantize_bin} not built — run scripts/build_llama_cpp.sh", file=sys.stderr)
        sys.exit(1)
    print(f"[quantize] {quantize_bin} → {GGUF_Q4} (Q4_K_M)")
    subprocess.run(
        [str(quantize_bin), str(GGUF_F16), str(GGUF_Q4), "Q4_K_M"],
        check=True,
    )
    print(f"[quantize] OK ({GGUF_Q4.stat().st_size / 1e9:.2f} GB)")


def main() -> int:
    if not ADAPTER_DIR.is_dir():
        print(f"ERROR: adapter dir not found: {ADAPTER_DIR}", file=sys.stderr)
        return 1
    step_merge()
    step_convert_to_gguf_f16()
    step_quantize_to_q4_k_m()
    print()
    print(f"DONE. Final GGUF: {GGUF_Q4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
