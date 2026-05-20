# 02 — Fine-tuning a model on a codebase

## What "fine-tuning" actually means

A language model is a giant matrix of weights — for a 7-billion-parameter model, roughly 7 billion floating-point numbers arranged in layers. Pre-training on the internet produced those weights so they encode general knowledge of language and code. **Fine-tuning is the process of nudging those weights** with additional training data so the model is better at a specific task or domain — in our case, your codebase.

There are three sensible levels of fine-tuning, in order of cost:

| Method | What changes | Memory needed | When to use |
|---|---|---|---|
| **Full-parameter fine-tuning** | Every weight in the model | ~6× model size in VRAM (model + optimizer state + grads). For 7B: ~80+ GB. | Never on a laptop. Only on dedicated GPUs (A100/H100). |
| **LoRA** | A tiny "adapter" matrix added alongside the base; base stays frozen | ~1.2× base size. For 7B: ~20+ GB. | Mid-range workstations. Still impractical on 8 GB. |
| **QLoRA** | LoRA on top of a 4-bit-quantised base | Base ~4.4 GB + adapter ~80 MB + activations. For 7B: ~7 GB. | **This is our setup.** Works on 8 GB. |

So we use **QLoRA**.

## LoRA: the low-rank insight

The full-parameter weight update for a single training step looks like `W = W + ΔW` where `ΔW` is the same shape as `W` — gigantic. The LoRA paper (Hu et al., 2021) observed that for fine-tuning tasks, `ΔW` tends to have very low intrinsic rank — most of the information lives in a tiny number of directions. So you can factor `ΔW ≈ B · A` where `B` and `A` are skinny matrices. If `W` is 4096×4096, you might pick rank `r=16`, so `B` is 4096×16 and `A` is 16×4096 — together they have ~131K parameters instead of ~16M. **Same expressive power for fine-tuning, ~100× fewer trainable params.**

Concrete hyperparameters that matter (per [`02-fine-tuning.md § config schema`](#config-schema-canonical-yaml) below):

- **`r` (rank)** — How "wide" the adapter is. Higher = more capacity, more VRAM, more risk of overfitting. We default to `16`. Going up to `32` or `64` is common; going below `8` rarely worth it.
- **`alpha`** — A scaling factor: the adapter contributes `(alpha / r) · B · A`. We set `alpha = r = 16` so the scaling is 1.0. Some recipes set `alpha = 2 * r`.
- **`dropout`** — Regularization on the adapter. Default `0.0` for QLoRA (the 4-bit base provides implicit regularization).
- **`target_modules`** — Which weight matrices in the model get an adapter. Setting `"all-linear"` attaches one to every linear projection in every attention block + every MLP block. This is the modern default; older recipes only adapted q/k/v projections.

## QLoRA: the 4-bit twist

A 7B model in fp16 is 14 GB. In 4-bit (using `bitsandbytes`'s NF4 quantization) it's ~4.4 GB. **You can't train in 4-bit directly** — gradients explode/vanish — but you can:

1. Hold the base in 4-bit (frozen).
2. Hold the LoRA adapter in bf16.
3. During forward pass, dequantise the base into bf16 on-the-fly, apply both base and adapter, compute logits.
4. During backward pass, only the LoRA adapter receives gradients.

The dequantisation is fast because bitsandbytes' kernels are written for this exact pattern. Net VRAM cost for 7B QLoRA on 8 GB is:

| Component | Bytes |
|---|---|
| 4-bit base | ~4.4 GB |
| LoRA adapter (bf16) | ~150 MB |
| Optimizer state (paged_adamw_8bit) | ~300 MB |
| Forward-pass activations (seq_len=1024) | ~1.5 GB |
| Working buffers | ~500 MB |
| **Total** | **~6.8 GB** |

That leaves ~1.2 GB headroom on an 8 GB GPU — tight but workable. Doubling `seq_len` to 2048 pushes activations to ~3.5 GB and you OOM. (Documented hard rule: `seq_len=1024` is the default.)

## Unsloth: why we use it instead of vanilla `transformers.Trainer`

[Unsloth](https://unsloth.ai) is a drop-in replacement for HuggingFace's training loop that ships **fused Triton kernels** for the QLoRA hot path. Practically:

- **2–4× faster** per step on consumer GPUs (sm_61 through sm_120).
- **30–50% less VRAM** for the same hyperparameters (so you can train at `seq_len=2048` on a 12 GB card where vanilla `Trainer` OOMs).
- **Simpler API** for the QLoRA pattern: `unsloth.FastLanguageModel.from_pretrained(...)` + `get_peft_model(...)` and you're done.

The trade-off: Unsloth's optimisations sometimes conflict with multi-GPU DDP wrapping (see issue [#3](https://github.com/alleboudy/codescribe/issues/3) §6.4 — "The Unsloth-vs-DDP tension"). For single-GPU training, Unsloth is uniformly better; for multi-node distributed, you may need to fall back to vanilla `Trainer + accelerate`.

## The optimizer story: `paged_adamw_8bit`

AdamW is the standard transformer optimizer. Its state has two tensors per parameter (first and second moments). For a 7B model in fp32 that's ~56 GB of optimizer state — bigger than the model.

**`paged_adamw_8bit`** (from `bitsandbytes`) stores the optimizer state in 8-bit and *pages it between CPU RAM and GPU VRAM* as needed. The state on the GPU at any moment is ~4–6× smaller, and CPU paging is fast enough that the training loop doesn't stall.

**You cannot use `adamw_torch`** on 8 GB VRAM for QLoRA-7B. It OOMs immediately. This is encoded as a hard rule in the train package's AGENTS.md.

## Training data: completion vs FIM vs instruction

The fine-tune learns from JSONL training samples. We support three formats; you can mix them or pick one per training run.

### `completion` — bare text windows

```jsonl
{"text": "def parse(s):\n    return json.loads(s)\n\ndef format(d):\n    return json.dumps(d, indent=2)\n..."}
```

Sliding windows of `seq_len` tokens with 256-token overlap between windows. Simple, robust, learns "what code in this codebase looks like".

### `fim` — Fill-In-Middle

```jsonl
{"text": "<|fim_prefix|>def parse(s):\n    return <|fim_suffix|>\n\ndef format(d):\n    return json.dumps(d, indent=2)<|fim_middle|>json.loads(s)"}
```

This is what teaches the model to *autocomplete in the middle of a file* — what IDE inline completion actually needs. The Qwen tokenizer reserves three literal tokens (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`) for this format. Critical detail: when picking the random hole boundaries, **never split a multi-byte UTF-8 character in half** — the tokenizer drops the byte and your sample becomes garbage.

### `instruction` — synthesised chat pairs

```jsonl
{"messages": [
  {"role": "user", "content": "Implement the function on line 42 of src/parser.py"},
  {"role": "assistant", "content": "def parse(s):\n    return json.loads(s)"}
]}
```

Synthesised from file context. Good when the model needs to learn the chat-completion conventions of how it'll be used.

For our stack: a typical fine-tune mixes 70% completion + 20% FIM + 10% instruction.

## Splitting the data: by file path, never by chunk

The single textbook mistake to avoid: **don't let chunks of the same file land in different splits**. If `src/parser.py:1-1024` is in `train.jsonl` and `src/parser.py:1024-2048` is in `val.jsonl`, your eval will look amazing (because the model already saw the file) and your real-world performance will be terrible.

The rule encoded in `data/AGENTS.md`: hash by `(repo-relative path + seed salt)` and assign each file to exactly one split. Adding a new file later must not reshuffle existing files between splits — the hash is deterministic.

## Evaluation

After training, the `eval` subcommand scores the adapter against `evals/<repo>_tasks.json` — a JSON list of tasks like:

```json
[
  {
    "id": "T01-metric-explain",
    "category": "qa",
    "prompt": "Explain how the FIT-field parser handles unknown manufacturer IDs.",
    "expected_token_substrings": ["fallback", "unknown_manufacturer", "log_warning"],
    "forbidden_token_substrings": ["random.choice", "raise NotImplementedError"]
  }
]
```

The scorer:
1. Generates ≤256 tokens at temperature 0.2 per task.
2. Counts how many `expected_token_substrings` appear in the response (`expected_hits / expected_total`).
3. Counts how many `forbidden_token_substrings` appear (subtracted from the score).
4. Emits a per-task score + an aggregate `task_mean`.

**No LLM judge.** Just deterministic string matching. This is on purpose: LLM judges are noisy and add a cloud dependency we don't accept.

A `task_mean` of ~0.5 is typical for a small held-out task set; ~0.7 is excellent. <0.3 means something is wrong — likely the wrong adapter loaded.

## Export to GGUF (the historic landmine)

After training you have a LoRA adapter. To serve it via `llama.cpp` you need to:

1. **Load the fp16 base** (not the 4-bit one) — this is where the historic bug bit. If you pass `device_map="auto"` here, HuggingFace's accelerate falls back to CPU+disk offload, then Peft tries to set up offload metadata and crashes with `KeyError: base_model.model.model.model.layers.10.input_layernorm`. **Force `device_map={"": "cuda:0"}`** or do the merge on CPU. If you have to redownload the fp16 base because the local one isn't recognised, set `local_files_only=True` to suppress the redownload (cost a full day of bandwidth the first time this bug was hit).
2. **Merge the adapter into the base** via `PeftModel.from_pretrained(...).merge_and_unload()`. Outputs ~14 GB fp16 model.
3. **Convert HF → GGUF** via `vendor/llama.cpp/convert_hf_to_gguf.py`. Outputs ~14 GB f16 GGUF.
4. **Quantise to Q4_K_M** via `vendor/llama.cpp/build/bin/llama-quantize <in.gguf> <out.gguf> Q4_K_M`. Outputs ~4.4 GB Q4 GGUF.

The Q4_K_M GGUF is what `llama-server` actually serves at runtime. See [`04-inference.md`](04-inference.md).

## Config schema (canonical YAML)

```yaml
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
seq_len: 1024
lora:
  r: 16
  alpha: 16
  dropout: 0.0
  target: all-linear
optim:
  name: paged_adamw_8bit
  lr: 2.0e-4
  warmup_steps: 5
  scheduler: cosine
train:
  epochs: 3
  batch_size: 1
  grad_accum_steps: 16
  max_grad_norm: 1.0
save:
  steps: 200
  total_limit: 2
log:
  steps: 10
```

Notes:
- Effective batch size = `batch_size × grad_accum_steps` = 16. Larger reduces gradient noise; we're at 16 because `batch_size=1` is the only thing that fits and `grad_accum_steps` is essentially free.
- `lr=2e-4` is the QLoRA reference value. Tune via the sweep tooling in issue [#3](https://github.com/alleboudy/codescribe/issues/3) (Path A).
- `warmup_steps=5` is short because LoRA adapters initialise close to zero — they don't need a long warmup.
- `scheduler: cosine` decays the LR from peak to ~0 over the run. Stable; the alternative `linear` is fine too.

## Smoke vs full run vs eval

The `train` package exposes four CLI subcommands so you can validate the loop before committing to a multi-hour training run:

```bash
python -m <project>.train smoke   --config ...  --dataset ...  --out checkpoints/smoke/
python -m <project>.train run     --config ...  --dataset ...  --out checkpoints/v1/
python -m <project>.train eval    --adapter checkpoints/v1/  --tasks ...  --report eval-report.json
python -m <project>.train export  --adapter checkpoints/v1/  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
                                  --merged checkpoints/merged-v1/ --gguf checkpoints/qwen-coder-7b-q4_k_m.gguf \
                                  --llama-cpp vendor/llama.cpp
```

**Always run `smoke` first on a fresh hardware setup.** It's 10 steps. Either it prints `train_loss=<decreasing>` in <15 minutes or something is wrong; abort and diagnose. Don't kick off a 3-hour `run` until smoke is green.

## Further reading

- Original LoRA paper: https://arxiv.org/abs/2106.09685
- QLoRA paper: https://arxiv.org/abs/2305.14314
- Unsloth docs: https://docs.unsloth.ai
- HuggingFace PEFT: https://huggingface.co/docs/peft
- See [`13-further-reading.md`](13-further-reading.md) for more.
