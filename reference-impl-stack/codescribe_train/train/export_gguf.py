"""Merge a LoRA adapter into the base model and export to GGUF.

Two-step process:

1. **Merge**: load base + LoRA, merge weights, save the merged model in
   HuggingFace ``safetensors`` format.  For Unsloth-saved adapters
   (``adapter_config.json`` has ``auto_mapping.unsloth_fixed: true``), the
   merge uses Unsloth's 4-bit-aware path which fits in 8 GB VRAM.  For
   vanilla PEFT adapters, the standard ``transformers + peft`` path is used
   with CPU-only placement (slower but avoids OOM on small GPUs).

2. **Quantize to GGUF**: convert the merged model to ``Q4_K_M`` GGUF using
   llama.cpp's ``convert_hf_to_gguf.py`` + ``llama-quantize``.  Requires a
   vendored llama.cpp build under ``vendor/llama.cpp/``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExportPaths:
    merged_dir: Path
    gguf_path: Path | None


def is_unsloth_adapter(adapter_dir: Path) -> bool:
    """Check whether *adapter_dir* was saved by Unsloth."""
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        return False
    cfg = json.loads(cfg_path.read_text())
    return bool(cfg.get("auto_mapping", {}).get("unsloth_fixed"))


def _unsloth_merge(adapter_dir: Path, out_dir: Path, seq_len: int = 1024) -> Path:
    """Merge via Unsloth's 4-bit-aware path (fits in 8 GB VRAM)."""
    import os

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from unsloth import FastLanguageModel

    logger.info("loading %s via Unsloth (4-bit, seq_len=%d)", adapter_dir, seq_len)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("save_pretrained_merged → %s (16-bit)", out_dir)
    model.save_pretrained_merged(
        str(out_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    return out_dir


def _peft_merge(adapter_dir: Path, base_model_id: str, out_dir: Path) -> Path:
    """Merge via standard transformers + PEFT (CPU-only to avoid OOM)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("loading base %s (bf16, cpu-only)", base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    logger.info("loading LoRA adapter %s", adapter_dir)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    logger.info("merging adapter weights into base...")
    merged = model.merge_and_unload()
    logger.info("saving merged model to %s", out_dir)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def merge_lora(adapter_dir: Path, base_model_id: str, out_dir: Path) -> Path:
    """Merge the LoRA adapter into a fp16 base and save to *out_dir*."""
    if is_unsloth_adapter(adapter_dir):
        logger.info("Unsloth adapter detected — using 4-bit merge path")
        return _unsloth_merge(adapter_dir, out_dir)
    logger.info("vanilla PEFT adapter — using CPU merge path")
    return _peft_merge(adapter_dir, base_model_id, out_dir)


def quantize_to_gguf(
    merged_dir: Path,
    out_path: Path,
    *,
    llama_cpp_dir: Path | None = None,
    quant_type: str = "Q4_K_M",
) -> Path | None:
    """Convert the merged model to GGUF.  Requires llama.cpp."""
    if llama_cpp_dir is None or not llama_cpp_dir.is_dir():
        logger.warning(
            "llama.cpp not available. Run manually after building:\n"
            "    python <llama.cpp>/convert_hf_to_gguf.py %s --outfile %s --outtype f16",
            merged_dir,
            out_path,
        )
        return None
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.is_file():
        logger.error("convert_hf_to_gguf.py not found at %s", converter)
        return None
    quant_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not quant_bin.is_file():
        logger.warning("llama-quantize not built; single-step q8_0 fallback")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(converter),
                str(merged_dir),
                "--outfile",
                str(out_path),
                "--outtype",
                "q8_0",
            ],
            check=True,
        )
    else:
        f16_path = out_path.with_suffix(".f16.gguf")
        subprocess.run(
            [
                sys.executable,
                str(converter),
                str(merged_dir),
                "--outfile",
                str(f16_path),
                "--outtype",
                "f16",
            ],
            check=True,
        )
        subprocess.run(
            [str(quant_bin), str(f16_path), str(out_path), quant_type],
            check=True,
        )
        f16_path.unlink(missing_ok=True)
    return out_path


def export(
    *,
    adapter_dir: Path,
    base_model_id: str,
    merged_dir: Path,
    gguf_path: Path,
    llama_cpp_dir: Path | None = None,
) -> ExportPaths:
    """End-to-end: merge + (optional) GGUF quantize."""
    merge_lora(adapter_dir, base_model_id, merged_dir)
    out_gguf = quantize_to_gguf(merged_dir, gguf_path, llama_cpp_dir=llama_cpp_dir)
    return ExportPaths(merged_dir=merged_dir, gguf_path=out_gguf)
