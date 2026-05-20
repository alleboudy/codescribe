# 11 — Hardware: GPUs, CUDA, VRAM, all the numeric details

## The vocabulary

| Term | Meaning |
|---|---|
| **VRAM** | Video RAM. The memory on the GPU itself. This is the constraint that dominates LLM work. |
| **CUDA** | NVIDIA's parallel-compute platform. CUDA Toolkit version = the compiler + libraries; CUDA Runtime = the per-app library that ships with PyTorch. They must be compatible (toolkit ≥ runtime). |
| **Compute Capability** (sm_XX) | The "instruction set version" of an NVIDIA GPU. Pascal sm_61, Turing sm_75, Ampere sm_80/86, Ada sm_89, Blackwell sm_120. Determines which features (FA2, bf16, FP8) are available. |
| **TGP** (Total Graphics Power) | For laptop GPUs: configurable max power the OEM allows. Higher = faster but hotter. |
| **TDP** | Thermal Design Power; desktop GPUs. Similar concept. |
| **Tensor Cores** | Specialised matrix-multiplication hardware introduced with Volta (sm_70+). Required for fast bf16/fp16 GEMM. |
| **HBM vs GDDR** | High-Bandwidth Memory (datacentre cards: H100/A100) vs GDDR6/X (consumer cards: 30/40/50 series). Different bandwidth-to-capacity ratios. |

## The compute-capability generation map

| GPU family | Released | sm_XX | bf16 native? | FA2 support? | fp8 support? |
|---|---|---|---|---|---|
| **Pascal** (1080, 1080 Ti, P100) | 2016 | sm_61 (consumer), sm_60 (datacentre) | no (fp16 only) | no | no |
| **Volta** (V100) | 2017 | sm_70 | no | partial | no |
| **Turing** (2060, 2070, 2080, T4) | 2018 | sm_75 | no | partial | no |
| **Ampere** (3060, 3080, 3090, A100) | 2020 | sm_80, sm_86 | yes | yes | no |
| **Ada Lovelace** (4060, 4070, 4080, 4090, L40) | 2022 | sm_89 | yes | yes | yes |
| **Hopper** (H100, H200) | 2022 | sm_90 | yes | yes (FA3) | yes |
| **Blackwell** (5070, 5080, 5090, B100, B200) | 2024–2025 | sm_120 | yes | yes (FA3) | yes |

What this means for us:

- **Pascal (sm_61)** — works for inference but NO FA2 (so slower training), fp16 only (no bf16). The GTX 1080 Ti in the reference setup is Pascal. Acceptable for serving, terrible for training.
- **Turing (sm_75)** — works for inference, partial FA2. RTX 2070-class laptop GPUs are here. Marginal for training.
- **Ampere+ (sm_80+)** — modern. FA2, bf16, paged_adamw_8bit all happy.
- **Ada (sm_89)** — what the fleet's 8 GB laptop GPUs run.
- **Blackwell (sm_120)** — the latest; what the RTX 5070 Laptop reference uses.

## Finding your GPU's compute capability

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

Example output:
```
name, compute_cap, memory.total [MiB]
NVIDIA GeForce RTX 5070 Laptop GPU, 12.0, 8188 MiB
```

`compute_cap = 12.0` → `sm_120`. (The decimal-to-`sm_XX` conversion is `compute_cap * 10` → strip the dot.)

For llama.cpp builds, you pass this to cmake:
```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=120 ...
```

## VRAM math for 7B QLoRA training

| Component | Bytes |
|---|---|
| 4-bit base (NF4 + quant scales) | ~4.4 GB |
| LoRA adapter (bf16) | ~150 MB |
| Optimizer state (paged_adamw_8bit) | ~300 MB |
| Forward activations (seq_len=1024) | ~1.5 GB |
| Working buffers (cuBLAS, gradient compute, FA2) | ~500 MB |
| **Total at seq_len=1024** | **~6.85 GB** |
| Add: seq_len=2048 activations | ~3.5 GB extra |
| **Total at seq_len=2048** | **~10.35 GB** |

On an 8 GB card, only `seq_len=1024` fits with margin. `seq_len=2048` OOMs. The hard rule encoded in `train/AGENTS.md`.

## VRAM math for 7B Q4_K_M inference

| Component | Bytes |
|---|---|
| Q4_K_M model weights (loaded at -ngl -1) | ~4.4 GB |
| KV cache at ctx_size=32768 (fp16) | ~1.8 GB |
| Activation buffers per prompt | ~300 MB |
| llama.cpp internal heap | ~200 MB |
| **Total** | **~6.7 GB** |

On 8 GB: tight but fine. On 6 GB: drop ctx to 16384 OR Q3_K_M.

## Mixed precision (bf16 vs fp16 vs Tensor Cores)

- **fp16** (half-precision float): 16 bits, ~3 decimal digits of precision, smaller dynamic range. Supported since Volta (sm_70).
- **bf16** (brain float): 16 bits, same dynamic range as fp32 (8-bit exponent), less precision. Supported since Ampere (sm_80).
- **fp8** (e5m2 / e4m3): 8 bits. Supported since Hopper/Ada (sm_89+).
- **Tensor Cores**: hardware that does matrix-multiply-and-accumulate (D = A·B + C) in fp16/bf16/fp8 at much higher throughput than scalar ops.

For QLoRA: **bf16 is preferred** on Ampere+. Fewer numerical surprises during long training runs (bf16's wide dynamic range avoids gradient underflow). On Pascal/Turing/older Tesla, you're stuck with fp16; works but slightly more fragile.

## Flash Attention 2 (FA2)

Standard attention requires O(N²) memory for sequence length N — at N=2048, that's a lot. FA2 reorders the computation so it's O(N) memory and ~3× faster. Critical for QLoRA training at non-trivial sequence lengths.

**FA2 requires sm_80+** (Ampere or newer). On Pascal/Turing/Volta, you fall back to Unsloth's xformers backend (much slower per step).

This is why issue [#2](https://github.com/alleboudy/llm-finetuner/issues/2)'s reference hardware is Ada/Blackwell, not Pascal. Pascal is in the project's history as the *desktop* (the 1080 Ti) but only for *inference* — it's not the training target.

Installing FA2 is its own adventure: it ships as wheels per (PyTorch, CUDA, Python) combo, and pre-built wheels don't exist for every cell. The reference setup did a cross-build on the Pascal desktop to produce a wheel for the Blackwell laptop. Documented in this stack's history as "FA2 cross-build".

## CUDA Toolkit vs CUDA Runtime version

- **CUDA Toolkit** = the compiler (`nvcc`) and developer libraries you install system-wide. Pick a version (12.8 for this stack) and stick with it.
- **CUDA Runtime** = the shared library each Python package (PyTorch, bitsandbytes, etc.) ships with. PyTorch wheels are tagged like `cu128`, `cu121`, etc. — the cu-suffix is the CUDA Runtime version they were built against.

**Rule**: Toolkit version ≥ Runtime version. You can have CUDA Toolkit 12.8 installed and use a PyTorch built against CUDA Runtime 12.1 — works. The reverse (Toolkit 12.1, Runtime 12.8) breaks.

`uv sync --extra train` should pull a PyTorch wheel matching your toolkit. If it doesn't, document the wheel you actually want in `pyproject.toml`:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
```

## Power and thermal management for laptops

Laptop GPUs throttle aggressively when:
- On battery (TGP drops from 75 W to 30 W → ~3× slower).
- Hot (the cooling can't sustain 75 W indefinitely; throttles to whatever the chassis dissipates).
- The OEM has set conservative defaults (a "balanced" OEM thermal profile can halve performance versus the maximum one).

For training:
- **Always on AC**, ideally 130 W+ USB-C PD or barrel-jack.
- **Set the BIOS thermal profile to maximum.**
- **Watch `nvidia-smi -q -d POWER`** during training — power draw should be steady at the configured TGP.
- **Watch temperature.gpu** — sustained 95 °C means you're throttled.

[Issue #3](https://github.com/alleboudy/llm-finetuner/issues/3) §2 has the workstation specifics.

## Reference hardware in this project

The canonical configurations documented across the issues:

### Training laptop (issue [#2](https://github.com/alleboudy/llm-finetuner/issues/2) reference)
- **RTX 5070 Laptop GPU** — Blackwell, sm_120, 8 GB GDDR7, configurable TGP up to ~115 W.
- i7-14650HX, 24 threads.
- WSL2 Ubuntu 24.04, 7.6 GB visible RAM (deliberately tight).
- Training perf: ~20.9 s/step at seq_len=1024 with FA2.

### Optional inference desktop
- **GTX 1080 Ti** — Pascal, sm_61, 11 GB GDDR5X.
- i7-8700K, 32 GB DDR4.
- Used historically for FA2 cross-build (now a runtime inference option).
- Inference perf: ~67 tok/s on Q4_K_M Qwen 7B.

### Fleet laptops (issue [#3](https://github.com/alleboudy/llm-finetuner/issues/3))
- **8 GB Ada-class laptop GPU** — Ada, sm_89, 8 GB GDDR6, TGP 35–75 W.
- Recent mobile-workstation lines from the major OEMs offer this GPU option.
- Training perf (est.): ~28–35 s/step.

## When to escalate beyond laptop-class

You've outgrown laptops when:

- A single training run takes more than ~6 hours of GPU time even with FA2 + seq_len=1024.
- You want to fine-tune a model bigger than 7B (Qwen 14B needs ~12 GB even in 4-bit).
- You need 64K+ context — KV cache alone exceeds 8 GB.
- You want to fine-tune multiple models in parallel — laptops don't support multi-GPU sensibly.

Options at the next tier:

| Path | Cost (rough) | Notes |
|---|---|---|
| **Desktop with 1× RTX 4090** (24 GB) | €2K capex | Replaces 4× laptop fleet for most workloads |
| **Desktop with 2× RTX 4090** | €4K capex | Multi-GPU DDP locally |
| **Cloud A100 spot** | $1–1.50/hr | Per-job; better for sporadic workloads |
| **Cloud H100** | $2–4/hr | For bigger models |
| **On-prem H100** | €30K+ capex | Only if you train constantly |

For our strictly-local posture, cloud is disqualified. The escalation path is: laptop → desktop with 4090 → local server with multiple cards.

## Apple Silicon (M1/M2/M3/M4 chips with Metal)

- **Inference**: works via llama.cpp's Metal backend. Performance comparable to mid-range NVIDIA (M3 Max ≈ GTX 1080 Ti for Q4_K_M 7B).
- **Training**: NOT supported by our stack. Unsloth requires CUDA. There's experimental MLX-based training on Apple Silicon but the QLoRA path is much less mature.

Practical use: Mac laptops can serve the fine-tune via llama.cpp Metal but you'll need a CUDA box to *produce* the fine-tune in the first place.

## Disk

Often overlooked. You need:

| What | Size |
|---|---|
| HuggingFace cache (base model + tokenizer) | ~15 GB per model |
| Training datasets | 100 MB – 10 GB per repo |
| LoRA adapter checkpoints | ~100 MB per checkpoint × `save.total_limit` |
| Merged fp16 model | ~14 GB per run |
| Q4_K_M GGUF | ~4.4 GB per run |
| RAG store (sqlite-vec) | ~500 MB – 5 GB per repo indexed |
| llama.cpp build artefacts | ~2 GB |
| logs | sub-GB |

Budget **≥40 GB free** on the partition that holds your project. SSDs are strongly preferred; HDD will make HuggingFace cache loading slow.

## Further reading

- NVIDIA compute capability matrix: https://developer.nvidia.com/cuda-gpus
- CUDA Toolkit downloads: https://developer.nvidia.com/cuda-downloads
- PyTorch installation matrix: https://pytorch.org/get-started/locally/
- Unsloth GPU compatibility: https://docs.unsloth.ai/get-started/system-requirements
- Flash Attention 2: https://github.com/Dao-AILab/flash-attention
- llama.cpp build flags: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- See [issue #3 §2](https://github.com/alleboudy/llm-finetuner/issues/3) for vendor-specific thermal config.
