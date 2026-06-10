# `codescribe_train/train/`

QLoRA fine-tuning + evaluation + GGUF export. Phase 2 of `codescribe-train`.

Loads Qwen 2.5 Coder 7B Instruct in 4-bit NF4 via Unsloth, attaches a LoRA adapter, runs `trl.SFTTrainer` on the JSONL splits from [`codescribe_train/data/`](../data/README.md), then optionally merges the LoRA back into the base and converts to a `Q4_K_M` GGUF for `llama-server`. Strictly local — `report_to="none"`, `WANDB_DISABLED`, `HF_HUB_DISABLE_TELEMETRY` all set defensively.

> Full background:
> - [`docs/QLoRA-AND-UNSLOTH.md`](../../docs/QLoRA-AND-UNSLOTH.md) — LoRA, QLoRA, Unsloth, hyperparameters
> - [`docs/FLASH-ATTENTION.md`](../../docs/FLASH-ATTENTION.md) — why FA2 is cross-built and what it does
> - [`docs/HARDWARE-AND-PERFORMANCE.md`](../../docs/HARDWARE-AND-PERFORMANCE.md) — the seq_len 2048→1024 decision

---

## Public surface

Four CLI subcommands, all under `python -m codescribe_train.train`:

### `smoke` — Phase 2 verification gate

```bash
python -m codescribe_train.train smoke \
    --config configs/train/qwen7b_qlora.yaml \
    --dataset datasets/sample/ \
    --out checkpoints/sample-qwen7b-lora-smoke/
```

10 steps, no eval, no save. Used to confirm the training loop works end-to-end on your hardware. Should report `train_loss` finite and decreasing.

### `run` — full training

```bash
python -m codescribe_train.train run \
    --config configs/train/qwen7b_qlora.yaml \
    --dataset datasets/sample/ \
    --out checkpoints/sample-qwen7b-lora-v1/
```

Full 3-epoch run. Saves a LoRA adapter at `--out`. ETA on the reference 8 GB-VRAM laptop GPU with FA2 + seq_len=1024: **~3.4 hours** for the 3-epoch / 582-step config.

### `eval` — score a saved adapter

```bash
python -m codescribe_train.train eval \
    --adapter checkpoints/sample-qwen7b-lora-v1/ \
    --dataset datasets/sample/ \
    --tasks evals/sample_tasks.json \
    --report eval-report.json
```

Two scores:
- **Held-out perplexity** on `test.jsonl` (capped at 200 samples).
- **Sample task suite** — 20 handwritten prompts probing project-specific knowledge. Each task has `expected_signals` (substrings that should appear) and `forbidden_signals` (must NOT appear); the score is positive_hits/expected_total - forbidden_hits/forbidden_total. Naive substring matching is a fast filter only — qualitative review of `--report eval-report.json`'s response field is the real evaluation.

### `export` — LoRA → fp16 → GGUF

```bash
python -m codescribe_train.train export \
    --adapter checkpoints/sample-qwen7b-lora-v1/ \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
    --merged checkpoints/sample-qwen7b-merged-v1/ \
    --gguf checkpoints/sample-qwen7b-q4_k_m.gguf \
    --llama-cpp vendor/llama.cpp
```

Three steps:
1. Load base + LoRA, call `peft.merge_and_unload()` → fp16 HuggingFace model dir at `--merged`.
2. `convert_hf_to_gguf.py` → fp16 GGUF.
3. `llama-quantize` → `Q4_K_M` GGUF at `--gguf`.

If `--llama-cpp` is omitted, only the merge step runs — useful before `vendor/llama.cpp/build/bin/llama-quantize` exists.

---

## File layout

```
train/
├── __init__.py
├── __main__.py        # → python -m codescribe_train.train <subcommand>
├── cli.py             # argparse for smoke/run/eval/export; lazy-imports per subcommand
├── sft.py             # the QLoRA training loop (Unsloth + bitsandbytes + trl.SFTTrainer)
├── eval.py            # perplexity + task-suite scoring
└── export_gguf.py     # peft.merge_and_unload + llama.cpp convert + quantize
```

`__init__.py` and `cli.py` are import-cheap: they don't pull in `torch` / `unsloth` / `peft` unless a subcommand handler actually runs. This is verified by `tests/train/test_config_and_cli.py::test_train_cli_help_does_not_import_torch` — don't break this discipline.

---

## Configs

`configs/train/qwen7b_qlora.yaml` is the production config. Locked decisions:

| Field | Value | Rationale |
|---|---|---|
| `base_model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Best 8 GB-VRAM-friendly coder; Qwen2 tokenizer; native FIM support |
| `lora.r` / `lora.alpha` | 16 / 32 | Standard ratio (alpha/r=2.0); 40 M trainable params (0.53% of base) |
| `lora.target_modules` | All 7 of {q,k,v,o}_proj + {gate,up,down}_proj | Standard QLoRA recipe |
| `quantization.bnb_4bit_quant_type` | `nf4` | NormalFloat 4-bit; better than uniform quant for normal-distributed weights |
| `quantization.bnb_4bit_use_double_quant` | true | Quantize the dequantization scales too — saves another ~0.4 bits/weight |
| `training.seq_len` | **1024** | NOT 2048 — see [`docs/HARDWARE-AND-PERFORMANCE.md § 3`](../../docs/HARDWARE-AND-PERFORMANCE.md#3-the-seq_len-20481024-decision). Caps per-step VRAM. |
| `training.batch_size` | 1 | 8 GB VRAM cap |
| `training.grad_accum` | 16 | Effective batch 16 — emulates "real" batch size |
| `training.optimizer` | `paged_adamw_8bit` | 8-bit Adam state with QLoRA paging — saves VRAM during gradient spikes |
| `training.bf16` / `fp16` | true / false | Blackwell native bf16; fp16 is forbidden on Blackwell |
| `training.gradient_checkpointing` | true | Trade compute for memory — required at 8 GB |

The `smoke` block at the bottom overrides `max_steps` and `eval_steps` / `save_steps` for the verification subcommand.

---

## Strictly-local enforcement

`sft.py:_enforce_local_only_env()` runs at training start:

```python
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE: opt-in via shell env (default 0)
```

Plus `SFTConfig(report_to="none", ...)`. Multiple kill switches, defence in depth.

The first training run pulls the base model from HF Hub (`unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`, ~5 GB). After it's cached locally, you can run with `HF_HUB_OFFLINE=1` for fully air-gapped operation. See [`docs/STRICTLY-LOCAL-POSTURE.md`](../../docs/STRICTLY-LOCAL-POSTURE.md).

---

## Conventions

- **Loss must be finite and decreasing.** A NaN loss midway through is a bug — most likely cause is fp16 in the config (it's not allowed on Blackwell). Check `bf16: true, fp16: false`.
- **Don't change `report_to`** to anything other than `"none"`. Code review will reject this.
- **Save artefacts only to `--out`.** Don't write to global state like `~/.cache/huggingface/<your-org>/`.
- **Lazy imports for everything torch-flavoured.** `import codescribe_train.train.cli` should not trigger torch initialisation. The dispatch dict in `cli.main` does the heavy import inside each subcommand handler.

---

## Hardware prereqs

Before running anything in this package:

1. **CUDA toolkit 12.8** at `/usr/local/cuda-12.8/` — `scripts/setup_wsl.sh` adds it to PATH.
2. **`uv sync --extra train`** to install torch + unsloth + bitsandbytes + peft + trl + accelerate + transformers + datasets + sentencepiece + protobuf + einops.
3. **Flash Attention 2 wheel** installed (cross-built on a second (builder) machine). See [`docs/FLASH-ATTENTION.md`](../../docs/FLASH-ATTENTION.md). Without it, Unsloth falls back to Xformers and step time roughly doubles.
4. **At least one Phase 1 dataset** at `datasets/<repo>/`. See [`codescribe_train/data/README.md`](../data/README.md).

---

## Tests

`tests/train/test_config_and_cli.py` — torch-free tests:
- YAML config parses, expected fields present
- Sample task JSON has 10+ tasks, no duplicate IDs, valid categories
- `import codescribe_train.train.cli` doesn't pull torch / unsloth / peft (lazy-import discipline)
- `--help` shows all 4 subcommands
- `smoke` subcommand requires its args (argparse error)

5 tests pass. Real verification of the training loop is the `smoke` subcommand against actual hardware — no hardware-mocking in pytest.

---

## See also

- [`codescribe_train/data/README.md`](../data/README.md) — produces the JSONL splits this package consumes
- [`codescribe_train/backends/README.md`](../backends/README.md) — serves the GGUF this package produces
- [`evals/sample_tasks.json`](../../evals/sample_tasks.json) — the qualitative task suite scored by `eval`
