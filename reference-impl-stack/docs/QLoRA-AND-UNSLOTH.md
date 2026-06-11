# QLoRA and Unsloth — what they are and why we use them here

This doc covers the fine-tuning stack: **LoRA → QLoRA → Unsloth**, in three layers of optimisation, plus the practical decisions specific to this project (hyperparameters, sequence length, the three sample formatters).

> **New to the project?** Start with [`CONCEPTS.md §1`](CONCEPTS.md#1-fine-tuning--baking-project-idioms-into-the-weights) for the 30,000-foot view of where fine-tuning fits alongside RAG and MCP, then come back here for the math and the hyperparameter rationale.

---

## 1. The problem fine-tuning solves

A **base model** (Qwen 2.5 Coder 7B Instruct, in our case) is trained on trillions of tokens of mostly-public code and text. It writes plausible code in any popular language, but it doesn't know your project's specific conventions, naming, library choices, or domain language.

**Fine-tuning** continues training on a smaller, project-specific dataset. The goal isn't to teach the model to code — it can already do that — but to *adjust its style and knowledge* toward your particular target. The result is a model that writes code matching your project's idioms instead of the generic public-corpus average.

The two practical questions are:

1. **Which weights do you change?** Changing all 7.6 billion is expensive and risks "catastrophic forgetting" of the model's general abilities.
2. **In what precision?** Storing 7.6 B weights × 2 bytes (bf16) = 15 GB on the GPU; Adam optimiser state takes 4× that for full fine-tuning (~60 GB). An 8 GB-VRAM GPU has nowhere near that. Something has to give.

LoRA solves the first question. QLoRA solves the second. Unsloth speeds both up dramatically.

---

## 2. LoRA — Low-Rank Adapters

[LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) freezes the base model entirely and inserts small trainable matrices alongside specific layers — typically the attention `q_proj`, `k_proj`, `v_proj`, `o_proj` and the MLP `gate_proj`, `up_proj`, `down_proj`. The math:

For each chosen layer, the original forward pass is `y = W·x` where `W` is a `(d_out, d_in)` matrix in the frozen base. LoRA adds a parallel path:

```
y = W·x + (B·A)·x
```

where `A` is `(r, d_in)` and `B` is `(d_out, r)` with `r` (the **rank**) much smaller than `d_in` or `d_out`. Only `A` and `B` are trained; `W` stays frozen.

**Why this works**:
- The number of trainable parameters drops from `d_in × d_out` (millions per layer) to `r × (d_in + d_out)` (tens of thousands per layer).
- Empirically, fine-tuning's "delta" (how the weights need to move) tends to have low intrinsic rank — meaning a low-rank approximation captures most of the useful change.
- After training, you can either keep the LoRA path separate (lightweight serving — base model + small adapter file) or **merge** `B·A` back into `W` (`W' = W + B·A`) for a single fp16 model.

**For our run**:
- `r=16`, `alpha=32` (the scaling factor `alpha/r = 2.0` — standard convention).
- Targeting the seven projection matrices listed above.
- **40 370 176 trainable parameters out of 7 655 986 688** — that's **0.53%**. The adapter file on disk is ~80 MB; the merged fp16 model is ~15 GB.

What `r` and `alpha` actually mean in practice:

- **Higher `r`** = more capacity to absorb fine-tuning signal, more VRAM during training, slower convergence per step but more total signal absorbed.
- **Higher `alpha`** = scales the LoRA contribution at inference time. With `alpha/r = 2.0`, the adapter's effect is amplified vs the base.
- Common pairs in the wild: `(8, 16)`, `(16, 32)`, `(64, 128)`. We chose `(16, 32)` as the sweet spot for 8 GB VRAM.

The [original LoRA paper](https://arxiv.org/abs/2106.09685) is short and worth reading directly if you want the math.

---

## 3. QLoRA — adding 4-bit quantization

[QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023) takes LoRA further: keep the base frozen *and* store its weights in **4-bit precision** during training, while keeping the small LoRA adapters in bf16.

The key innovations from the QLoRA paper:

### NF4 quantization

A 4-bit value can represent only 16 distinct numbers. Standard 4-bit quantization picks 16 evenly-spaced numbers across the weight distribution's range — wasteful, because real weights are concentrated near zero with long tails. NF4 (NormalFloat 4) picks 16 values that are **information-theoretically optimal for normally-distributed weights** — bunched near zero, sparser at the extremes. Every transformer's pre-trained weights are approximately normal, so NF4 fits naturally.

For 7B weights at 4 bits: `7.6e9 × 0.5 bytes = ~3.8 GB`. (Plus per-block scaling factors, but those are tiny.)

### Double quantization

The scaling factors used to dequantize NF4 back to fp16 are themselves quantized — saves another ~0.4 bits per weight on average.

### Paged optimizers

When sequence length spikes during training (some batches have long sequences, others short), the gradient computation can briefly need a lot of VRAM. QLoRA introduces "paged" optimisers that swap their state between GPU and CPU memory transparently when needed, preventing OOM crashes on the spikes. We use `paged_adamw_8bit` — Adam with 8-bit state plus paging.

### Why it works without losing accuracy

The base model's weights stay in low precision (NF4) **only as a memory store**. During the forward pass, each weight is dequantized to bf16 just before the matmul — math happens in bf16. The frozen base never updates, so the quantization error is constant; it just gets baked into the LoRA's training signal.

The LoRA adapters themselves are still bf16, so the trainable part has full precision. The QLoRA paper shows this preserves performance ≈ identical to a full fp16 fine-tune at a fraction of the memory.

**For our run**:
- Base loaded in 4-bit NF4 + double quantization → ~4.5 GB on GPU
- LoRA adapters in bf16 → ~80 MB
- `paged_adamw_8bit` optimiser state → ~40 MB (8-bit)
- Activations + gradients during forward/backward → ~3 GB peak with grad-checkpointing
- **Total VRAM: ~7.6 GB at peak**, fits in the 8 GB card.

The [QLoRA paper](https://arxiv.org/abs/2305.14314) goes deep on the math; the practical takeaway is that NF4 + bf16 LoRA + paged optimisers lets a 7B model fine-tune in <8 GB VRAM with no measurable quality loss vs full fp16.

---

## 4. Unsloth — kernel-level optimisations on top

[Unsloth](https://github.com/unslothai/unsloth) is a separate project from QLoRA. It takes an already-QLoRA-configured model and rewrites the hot loops in **fused Triton kernels** — significantly faster than the stock PyTorch implementations.

The big wins:

| Technique | What it does | Speedup |
|---|---|---|
| **Custom autograd kernels** for LoRA forward/backward | Avoids re-allocating activations across the LoRA path | ~2× faster than vanilla PEFT |
| **Padding-free batching** | Packs multiple short samples into one batch without explicit padding (no wasted FLOPs on `[PAD]` tokens) | ~1.5× faster on variable-length data |
| **Optimised gradient checkpointing** | Recomputes only the bare minimum during backward | Saves VRAM + slightly faster |
| **4-bit base loading from pre-quantized HF repos** | E.g. `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` — already-quantized weights download instead of fp16 → quantize-on-load | Saves ~10 GB download + a few minutes load time |

Unsloth's marketing claim is "**2× faster training, 70% less VRAM**" vs raw HuggingFace + bitsandbytes + PEFT. In our measurements it's closer to ~1.5× for training compute, with the bigger win being the memory savings that let the run fit at all.

**Important constraint**: Unsloth requires sm_75+ (Turing or newer) for its Triton kernels. Pascal-class cards (sm_61) and older can't use it — so a Pascal (sm_61) card can't serve as an Unsloth trainer; you need a Turing-or-newer (sm_75+) GPU for that role.

**Important runtime check**: Unsloth probes for `flash_attn` at import time. If found and importable, it uses FA2; otherwise it falls back to Xformers. The fallback is the message you'll see if the FA2 wheel isn't installed:

```
Unsloth: Your Flash Attention 2 installation seems to be broken. Using Xformers instead. No performance changes will be seen.
```

The "no performance changes will be seen" part is misleading — Xformers is genuinely ~2× slower than FA2 on this workload. See [`docs/FLASH-ATTENTION.md`](FLASH-ATTENTION.md) for the full story.

---

## 5. Why three formatters?

`codescribe_train/data/formatters/` ships three orthogonal formatters and a per-repo YAML chooses how much weight to give each.

### `document.py` — file as document

```
<|repo_name|>sample
<|file_sep|>backend/api/users.py
{full file content, possibly truncated to seq_len tokens}
```

This format teaches the model **continuation**: given the start of a project file, predict the rest. It's the closest analogue to how the base model was originally pretrained on code, and the markers (`<|repo_name|>`, `<|file_sep|>`) are part of Qwen 2.5 Coder's pretraining vocabulary — it's seen this exact format before.

Best for: file-level coherence (knowing what comes after `from sqlalchemy import` in *this* project's style), top-level structure (imports, class definitions, common patterns).

### `fim.py` — Fill-in-the-Middle

```
<|fim_prefix|>{prefix lines}<|fim_suffix|>{suffix lines}<|fim_middle|>{middle lines}
```

For each file, this formatter randomly picks a contiguous span of 1–8 lines to mask, and trains the model to predict those middle lines given the surrounding context. The PSM (Prefix-Suffix-Middle) ordering is Qwen-native.

Best for: **inline code completion** — the model learns to fill in a function body when given the signature + a return type, fill in a missing branch in an `if/elif`, etc. This is the format that makes editor autocompletion feel smart.

### `diff_instr.py` — git history → ChatML

```
<|im_start|>user
{commit subject}

{commit body}
<|im_end|>
<|im_start|>assistant
{commit diff}
<|im_end|>
```

For each non-trivial commit (subject 12–200 chars, diff ≤ 400 lines, non-merge), this formatter emits a chat-format sample where the user asks for a change (the commit message) and the assistant responds with the diff. The 2 000-commit cap prevents the dataset from being dominated by mass-rename commits.

Best for: **agentic editing** — the model learns to translate natural-language change requests into valid diffs in the style this project uses. This is the format that's most useful when paired with the harness, because the harness presents user intent as messages.

### Why all three?

Each format teaches a different *use* of the model:
- Document → "what code looks like in this project"
- FIM → "fill in this gap"
- Diff_instr → "make this change"

The single-objective alternative would be to pick one (most projects pick FIM-only). We use all three because the agent-driven workflow exercises all three: the harness asks for completions (FIM), shows whole files (document), and requests edits (diff_instr).

---

## 6. The hyperparameters in `configs/train/qwen7b_qlora.yaml`

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05         # small regularisation; Unsloth fast-path requires 0,
                        # but 0.05 produces better generalisation and the slight
                        # speed hit is acceptable on 8 GB VRAM
  target_modules: [q_proj, k_proj, v_proj, o_proj,
                   gate_proj, up_proj, down_proj]
  bias: none

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true

training:
  seq_len: 1024         # see below
  batch_size: 1         # 8 GB VRAM, so micro-batch 1
  grad_accum: 16        # effective batch 16 — emulates "real" batch size
  epochs: 3             # ~582 steps at our dataset size
  learning_rate: 2.0e-4
  lr_schedule: cosine
  warmup_ratio: 0.03    # ~3% of total steps to ramp up LR
  weight_decay: 0.01
  optimizer: paged_adamw_8bit
  bf16: true            # native bf16 on any sm_80+ GPU — prefer it over fp16
  gradient_checkpointing: true
  max_grad_norm: 1.0    # gradient clipping
  seed: 1729            # reproducibility
```

The `seq_len=1024` is the load-bearing number. With FA2 enabled and Unsloth's padding-free batching, **`seq_len=2048` produced wildly variable per-step times** (34s vs 300s on the same 10-step smoke) because padding-free batching packs different shapes per step, and the longest-shape steps blew through the 8 GB VRAM headroom and triggered swap. At 1024, all batches stay comfortably in VRAM and the per-step time is rock-steady at ~21 s.

Document-format samples lose tail content above 1024 tokens — for files that are over ~250 lines, the model only trains on the start. FIM and diff-instr samples are mostly already under 1024 tokens, so they're unaffected. See [`docs/HARDWARE-AND-PERFORMANCE.md`](HARDWARE-AND-PERFORMANCE.md) for the measurements.

---

## 7. Eval — what we score against

`python -m codescribe_train.train eval` runs two checks:

### 7.1 Held-out perplexity on `test.jsonl`

Loads up to 200 samples from the test split (deterministically split-by-file from the same dataset; no overlap with train). For each sample, computes the model's loss on the input as if it were predicting it autoregressively, accumulates total loss × tokens / total tokens, returns `exp(avg_loss)`. Lower is better.

This is a **necessary-but-not-sufficient** metric. A model can have low test perplexity by learning surface-level patterns without actually being more useful for the things you'd ask it to do.

### 7.2 Sample task suite (`evals/sample_tasks.json`)

20 handwritten prompts probing sample-specific knowledge — e.g., "how is `delivery_reliability` scored?", "list the steps to add a new feature flag", "write a SQLAlchemy migration that adds a `gift_wrap_fee_cents` column to `orders`". Each task has an `expected_signals` list (substrings that should appear, case-insensitive, with positive weight) and a `forbidden_signals` list (must NOT appear, with penalty weight).

The score is naive substring matching; it's a fast filter, not a quality measure. **Real evaluation is reading the responses by hand** — but the substring score is good enough to catch obvious regressions (the model dropping a key idiom or fabricating a wrong API).

A `--report eval-report.json` flag dumps the full responses + scores; that's where to look for the qualitative pass.

### What we *don't* run

**HumanEval / MBPP** — standard public coding benchmarks. We deliberately skipped these because:
- Running them needs an internet-fetched dataset, which slightly contradicts the strictly-local posture during a hot training session (the training machine's network may be off when training against a private repo).
- Public benchmarks measure general coding ability, which we *don't expect* fine-tuning on sample to improve. The fine-tune optimises for sample-style fluency, not for HumanEval-style trick-question solving.
- Running them once on the base model and once after fine-tuning to confirm "no regression" would be useful, but it's left as a deferred task.

If you want to validate against HumanEval anyway, it's in `bigcode/humaneval` on the HF Hub; pull it once, cache locally, and run an offline harness.

---

## 8. Export — LoRA → fp16 → GGUF

After training, the LoRA adapter is on disk at `checkpoints/sample-qwen7b-lora-v1/`. To serve it, you need to:

1. **Merge** the LoRA into the base in fp16. `peft.PeftModel.merge_and_unload()` does this; the result is a regular HuggingFace model directory at `checkpoints/sample-qwen7b-merged-v1/` (~15 GB).
2. **Convert** to GGUF — llama.cpp's wire format. Uses `vendor/llama.cpp/convert_hf_to_gguf.py`. Produces an fp16 GGUF (~15 GB).
3. **Quantize** to Q4_K_M — 4-bit per weight with K-means quantisation, the default sweet-spot quant for llama-server. Uses `vendor/llama.cpp/build/bin/llama-quantize`. Produces `checkpoints/sample-qwen7b-q4_k_m.gguf` (~4.5 GB).

The CLI does all three:

```bash
python -m codescribe_train.train export \
    --adapter checkpoints/sample-qwen7b-lora-v1/ \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
    --merged checkpoints/sample-qwen7b-merged-v1/ \
    --gguf checkpoints/sample-qwen7b-q4_k_m.gguf \
    --llama-cpp vendor/llama.cpp
```

If `--llama-cpp` is omitted, only the merge step runs and the export logs the manual command for the convert + quantize.

---

## 9. Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685) — original Hu et al. 2021.
- [QLoRA paper](https://arxiv.org/abs/2305.14314) — Dettmers et al. 2023, with the NF4 + double-quant + paged-optimiser tricks.
- [Unsloth project](https://github.com/unslothai/unsloth) — kernel-level optimisations on top of QLoRA.
- [Qwen 2.5 Coder report](https://arxiv.org/abs/2409.12186) — the model we fine-tune; explains the FIM / file-sep / repo_name vocabulary.
- [Hugging Face PEFT docs](https://huggingface.co/docs/peft/index) — the adapter-management library; works under both stock HF training and Unsloth.
- `transformers` + `bitsandbytes` integration: [HF docs](https://huggingface.co/docs/transformers/quantization).
