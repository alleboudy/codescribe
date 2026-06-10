# Flash Attention — background, why it matters here, and the cross-build story

This doc covers three things, in order:

1. **Background**: what attention is, why naïve attention is memory-bound on long sequences, and what Flash Attention (FA) does about it.
2. **The variants**: FA1 / FA2 / FA3 / Xformers — what changed between them, what hardware each needs.
3. **The scenario**: why an 8 GB-VRAM Blackwell-class laptop (sm_120) with a tight WSL2 RAM allocation needs FA2, why you can't `pip install` it, and how a second machine's Pascal-class GPU ends up cross-building the wheel.

---

## 1. Background — why attention needs help

A transformer block has, fundamentally, two operations: a fully-connected MLP and an **attention** operation. The attention operation, in pseudo-math, looks like this for a single head:

```
Q = X @ W_q          # (N, d)
K = X @ W_k          # (N, d)
V = X @ W_v          # (N, d)
S = Q @ Kᵀ           # (N, N)   <- the attention scores matrix
P = softmax(S / √d)  # (N, N)
O = P @ V            # (N, d)
```

`N` is the sequence length, `d` is the head dimension. The trouble is that **`S` and `P` are `N × N`** — quadratic in sequence length. For a 2048-token sequence, that's 4 194 304 entries per head. With 32 heads in a 7B model and `bfloat16` precision (2 bytes), one layer's attention scores alone take ~256 MB. Across 28 layers in the forward pass, then again in the backward pass for gradients, you're materialising **gigabytes** of intermediate tensors that you only really need long enough to multiply against `V` and discard.

Two consequences fall out of this naïve formulation:

1. **Memory-bound, not compute-bound.** GPUs can do an absurd number of FMAs per second; the bottleneck is reading `S` from HBM (high-bandwidth GPU memory), running softmax over it, and writing `P` back. The arithmetic intensity is too low.
2. **The HBM trip dominates wall clock**, especially on long sequences. You spend more time shuffling bytes between HBM and the SRAM caches than you do doing math.

This is what every attention-acceleration technique tries to fix. The cleanest framing: **avoid materialising the `N × N` matrix at all.**

---

## 2. What Flash Attention does

[Flash Attention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) is a **fused, IO-aware** attention kernel. Two key tricks:

### Trick 1 — item along the sequence dimension

Instead of computing all of `S = Q @ Kᵀ` in one go, FA splits `Q`, `K`, and `V` into blocks (items) that fit in the GPU's small but fast SRAM (per-SM cache, ~100 KB), and computes attention block-by-block:

```
for each block of Q (size B_q × d):
    for each block of K, V (size B_kv × d):
        s_block  = Q_block @ K_blockᵀ        # (B_q, B_kv)
        p_block  = softmax_normalised(s_block)  # using a running statistic
        o_block += p_block @ V_block          # accumulate into output
```

The trick is that each `s_block` lives in SRAM, gets used immediately, and is discarded — never written back to HBM. The HBM traffic drops from `O(N²)` (the full scores matrix) to `O(N · d)` (just the inputs and final outputs).

The catch is that `softmax` is non-local: you can't softmax a row of `S` until you've seen all `N` entries. FA gets around this with the **online softmax** algorithm — you keep running statistics (`max` and `sum_exp`) per `Q` row and update them as you process each `K, V` block, then renormalise the accumulated output at the end. This is mathematically equivalent to standard softmax, but item-friendly.

### Trick 2 — recompute, don't store, for the backward pass

Backward through attention normally needs the full `S` and `P` matrices saved from the forward pass — because gradients flow through softmax. FA throws them away during forward and **recomputes them** during backward. It's slightly more compute per backward pass, but the win in memory + HBM bandwidth more than pays for it.

### What you get

| | Standard attention | Flash Attention |
|---|---|---|
| Memory | `O(N²)` per layer | `O(N · d)` per layer |
| HBM reads/writes | `O(N² + N · d)` | `O(N · d)` |
| Numerically equivalent | (baseline) | **yes** (deterministic, same bits up to floating-point accumulation order) |
| Speedup at N=2048 | (1×) | typically 2–4× on Ampere, more on newer archs |
| Long-context viability | bottlenecked at ~4–8K tokens | scales to 64K+ |

The point isn't a small constant-factor win — it's that **memory is no longer the binding constraint**, which lets you train at longer sequences than you otherwise could, on the same hardware.

---

## 3. The variants

| | What's new | Hardware required |
|---|---|---|
| **Flash Attention 1** (May 2022, [paper](https://arxiv.org/abs/2205.14135)) | The item + online-softmax + recompute design above. | sm_75+ (Turing or newer; meant for Ampere V100/A100). |
| **Flash Attention 2** (Jul 2023, [paper](https://arxiv.org/abs/2307.08691)) | Same algorithm, **better parallelism**: distributes work across thread blocks differently and uses fewer non-matmul FLOPs. ~2× faster than FA1 on Ampere. **Adds explicit support for newer arch features** as Tri-Dao adds them — sm_80 (A100), sm_89 (Ada), sm_90 (Hopper), sm_120 (Blackwell). | sm_75+. Pin v2.7+ for sm_120 support; this stack uses v2.8.3. |
| **Flash Attention 3** (Jul 2024, [paper](https://arxiv.org/abs/2407.08608)) | Adds Hopper-only optimisations (warp-specialisation, asynchronous matmul scheduling via TMA, FP8 path). 1.5–2× faster than FA2 *on H100* — same speed as FA2 on everything else. | **sm_90 / Hopper only**. Doesn't exist for Blackwell consumer cards yet. |
| **Xformers `memory_efficient_attention`** | Meta's library. Implements similar item-and-recompute ideas, plus a number of other attention variants. **Pure-Python install** — ships prebuilt wheels covering most torch versions × CUDA × Python combinations. Does NOT match FA2 on speed; typically ~50% slower. | Anything torch supports. |

This stack targets FA2 v2.8.3 because:
- FA3 doesn't exist for Blackwell.
- FA1 is unmaintained — its sm_120 support never landed.
- Xformers is the fallback Unsloth picks if FA2 isn't importable; it measures **~2× slower** in practice.

---

## 4. Why you can't just `pip install flash-attn`

`flash-attn` is distributed as a Python package, but the heavy lifting is C++ + CUDA. To install it, pip either:
1. Pulls a **prebuilt wheel** matching your stack (fastest), or
2. Falls back to a **source build** that compiles the kernels with `nvcc` against your local CUDA toolkit.

The stack that needs wheels:

| Component | Version |
|---|---|
| Python | 3.13 |
| OS | Linux x86_64 |
| Torch | 2.10.0+cu128 |
| GPU arch | sm_120 (Blackwell — RTX 50-series) |
| C++ ABI | cxx11_abi=TRUE (PyTorch's cu128 wheels) |

Tri-Dao's [v2.8.3 release page](https://github.com/Dao-AILab/flash-attention/releases) (the latest stable at the time of writing) ships ~70 wheels indexed by **`cu12torch{2.4..2.9}` × `cxx11abi{TRUE,FALSE}` × `cp{39..313}`**. None target torch 2.10. The closest match (`cu12torch2.8 / cp313`) installs cleanly but **fails at `import flash_attn`** with:

```
ImportError: undefined symbol:
_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

This is a C++ ABI break. Between torch 2.9 and 2.10, the last parameter of `c10::cuda::c10_cuda_check_implementation` changed from signed `int` to **unsigned int**. The mangled symbol's last char went from `b` (bool, in this position; it actually encodes the `int` parameter via Itanium ABI rules) to `j` (`unsigned int`). Any wheel built against torch ≤ 2.9 references the old symbol, which doesn't exist in torch 2.10's `libc10_cuda.so`.

That's a hard blocker for any prebuilt wheel until Tri-Dao ships one against torch 2.10.

So: source build. The build needs:
- `nvcc` (CUDA toolkit) — 78 `.cu` files compiled with `-gencode arch=compute_120,code=sm_120`.
- A matching torch installed in the venv (so `setup.py` can find `torch.utils.cpp_extension`).
- ~6–12 GiB **host RAM per concurrent `nvcc` invocation** because the templates instantiated for each `.cu` blow up at compile-time. With `MAX_JOBS=2`, peak ≈ 12–24 GiB.

The reference machine has a tight (~8 GiB) WSL2 RAM allocation. A source build there OOMs partway through and corrupts the venv — it's a dead end on such hardware.

---

## 5. The cross-build solution

Compilation is **architecture-agnostic** in a way that running isn't:

- `nvcc` is a **cross-compiler**. Tell it `-arch=compute_120,code=sm_120` and it produces sm_120 PTX + SASS even if the build host's GPU is sm_61 (Pascal class) or has no GPU at all. The build *only* needs nvcc itself, the matching toolkit headers, and a CPU.
- The runtime hardware (a Blackwell-class GPU) just executes the resulting `.so`. It never knows or cares which machine compiled it.

This is identical to how you'd build an iOS app on a Mac: the Mac doesn't run iOS, but it produces an iOS binary.

So: **a second machine (Pascal-class GPU, more RAM) builds; the 8 GB-VRAM machine runs.** The builder's GPU is irrelevant — it's chosen for the larger RAM.

### The build pipeline (roughly)

```
                  [builder machine, more RAM]
                                                       
       ┌──────────────────────────────────────────┐
       │  apt: cuda-toolkit-12-8 → nvcc           │
       │  uv: Python 3.13 + torch 2.10+cu128      │
       │                                          │
       │  $ FLASH_ATTN_CUDA_ARCHS=120 \           │
       │    MAX_JOBS=4 \                          │
       │    pip wheel flash-attn --no-build-isol  │
       │    -w /tmp/wheels/                       │
       │                                          │
       │  78 × `nvcc -arch=compute_120,code=sm_120│
       │       *.cu` → 78 .o files                │
       │  linker → flash_attn_2_cuda.cpython-313- │
       │       x86_64-linux-gnu.so                │
       │  packaging → flash_attn-2.8.3-cp313-     │
       │       cp313-linux_x86_64.whl  (68 MB)    │
       │                                          │
       │  build time: ~37 min                     │
       └──────────────────┬───────────────────────┘
                          │ scp
                          ▼
                     [8 GB-VRAM GPU]
       ┌──────────────────────────────────────────┐
       │  $ pip install --no-deps                 │
       │       flash_attn-...-linux_x86_64.whl    │
       │                                          │
       │  At training time, `import flash_attn`   │
       │  loads the .so. CUDA kernels execute on  │
       │  the Blackwell-class GPU's sm_120 hardware.│
       └──────────────────────────────────────────┘
```

Build steps verbatim (and what to do when an upstream wheel finally ships) live in [`docs/known-issues.md` — Flash Attention 2 — INSTALLED](known-issues.md#flash-attention-2--installed-cross-build).

### Why `FLASH_ATTN_CUDA_ARCHS=120` matters

Without it, the build compiles kernels for **every supported architecture** (sm_75, sm_80, sm_89, sm_90, sm_120) — multiplying the compile time and RAM by ~5×. Only sm_120 is needed here; setting the env var skips the rest. Build time dropped from "≥ 2 h, OOM-prone" to "37 min, comfortable".

---

## 6. Why you *also* drop `seq_len` 2048 → 1024

Installing FA2 isn't enough by itself. Flash Attention reduces attention's *peak* memory, but it doesn't shrink the model weights or the optimizer state. With `seq_len=2048` and Unsloth's padding-free batching (which packs short samples into longer effective batches to keep the GPU busy), the 8 GB-VRAM machine's VRAM sits at **97% utilisation** — comfortable when a batch happens to draw mostly short samples, catastrophically slow when it draws a long document and triggers cudaMalloc retries / disk swap.

Measured per-step time for a 10-step smoke training run:

| seq_len | step time | notes |
|---|---|---|
| 2048, no FA2 (Xformers) | 85.8 s consistent | the original baseline |
| 2048, FA2 | 34, 37, 35, 34, **170, 300** s | first 4 steps in headroom; step 5+ hit swap |
| 1024, FA2 | 20.9 s consistent | comfortable VRAM, no swap |

The 1024 cap means document-format samples (whole files wrapped with `<|file_sep|>` markers) get truncated to their first 1024 tokens. Most sample files fit comfortably; only the largest (long generated files, big test fixtures) lose tail content. FIM and diff-instruction samples are mostly already under 1024 tokens, so they're unaffected.

Net effect on a 3-epoch training run over 3 099 train samples (effective batch 16):

| Config | s/step | 582 steps wall clock |
|---|---|---|
| Xformers @ 2048 (pre-cross-build) | 85.8 | ~14 h |
| FA2 @ 1024 (current default) | 20.9 | **~3.4 h** |

That's the **4.1× speedup** you've seen quoted elsewhere in the docs. The two changes (FA2 install + seq_len cap) compound: FA2 is faster per token, and the cap prevents the worst-case batches that were blowing the average up.

---

## 7. When this breaks

### Torch upgrades

The C++ ABI breaks roughly every 6–12 months at major torch boundaries. When you bump torch (2.11+), your installed FA2 wheel will fail to import with the same kind of mangled-symbol error seen at 2.10. Action: re-run the builder-machine cross-build against the new torch.

### Upstream wheels catch up

When Tri-Dao ships a wheel for `cu12torch2.10` × `cp313` × `linux_x86_64`, drop:

```bash
.venv/bin/python -m pip install flash-attn --no-build-isolation
```

into your install path and remove the wheel-vendoring step. Track [Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases) — the rerun command in `docs/known-issues.md` checks programmatically.

### A new `flash-attn` major version

If FA4 lands and you want to migrate, note that **Unsloth's runtime check imports `flash_attn`** (the v2 package name). FA4 doesn't satisfy that check unless aliased, even if it's installed. You'd need both, or wait for Unsloth to support FA4 explicitly.

---

## 8. Further reading

- Original FA paper: [Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- FA2 paper: [Flash Attention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- FA3 paper: [Flash Attention-3: Fast and Accurate Attention with Asynchrony and Low-Precision](https://arxiv.org/abs/2407.08608)
- Tri-Dao's repo: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- Online softmax: ["Online Normalizer Calculation for Softmax"](https://arxiv.org/abs/1805.02867) — the trick that makes item-wise softmax work
- Why HBM bandwidth matters: ["Roofline: An Insightful Visual Performance Model"](https://www.cs.berkeley.edu/~yelick/cs194f07/lectures/lec18-roofline.pdf) — the canonical mental model for memory-bound vs compute-bound kernels.
