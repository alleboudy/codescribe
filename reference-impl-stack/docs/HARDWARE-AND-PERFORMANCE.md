# Hardware and performance

What hardware this project targets, what was measured on it, and the load-bearing performance decisions that came out of those measurements.

The numbers below come from one representative two-machine reference setup. Treat the specific specs as an example class, not a hard requirement — the design rationale generalises to any setup with the same constraint shape.

---

## 1. The two reference machines

The project is designed around two physical machines. Both live on the same LAN; nothing leaves the LAN.

### Machine A — the training machine; primary, runs everything by default

| | Value |
|---|---|
| GPU | a recent consumer GPU with ~8 GB VRAM (e.g. Blackwell-class, sm_120) |
| GPU arch | Blackwell, **compute capability 12.0** (sm_120) |
| VRAM | **8.0 GiB** (≈8150 MiB usable) |
| CUDA runtime | 13.1 (driver 591.44, Windows-side) |
| CUDA toolkit | 12.8 installed in WSL2 |
| CPU | a modern x86 laptop CPU, e.g. ~16-core / 24-thread, DDR5 |
| OS | WSL2 Ubuntu 24.04 (kernel 6.6) |
| Visible RAM | a tight WSL2 RAM allocation (**~8 GiB**; left at the default) |
| Python | 3.13.13 (uv-managed) |
| Role | Trains, runs harness, runs default backend |

The VRAM and the WSL2 RAM are the two binding constraints. Everything else is comfortable.

### Machine B — the builder machine; cross-build only by default, can serve inference on opt-in

| | Value |
|---|---|
| GPU | a Pascal-class (sm_61) GPU with ~11 GB VRAM |
| GPU arch | Pascal, **compute capability 6.1** (sm_61) |
| VRAM | ~11 GiB |
| CPU | a 6-core / 12-thread desktop CPU, DDR4 |
| OS | Ubuntu 24.04.4 LTS Server |
| RAM | **~31 GiB DDR4** |
| Python | 3.13.13 (uv-managed) |
| Role | Cross-builds the FA2 wheel (RAM-bound work); optional 14B-class inference server |

A Pascal-class GPU is **too old** for some of the hot paths — Pascal lacks tensor cores, doesn't support bf16, can't run Flash Attention 2 (sm_75+ required), can't run Unsloth's Triton kernels. So it can't *train*. But its ~32 GiB RAM is exactly what FA2 source builds need, and `nvcc` is a cross-compiler — see [`docs/FLASH-ATTENTION.md`](FLASH-ATTENTION.md) for the build pipeline.

For inference, Pascal works fine in `llama-server`. Its ~11 GiB VRAM > Machine A's 8 GiB, so Machine B is the better fit for running 13–14B-class models that don't fit on Machine A. That's wired up as the optional `RemoteBackend` topology — disabled by default per the strictly-local posture (the second machine is on LAN, but loopback is the default for "local"). See [`configs/backends/lan-remote.yaml`](../configs/backends/lan-remote.yaml) for the opt-in config.

---

## 2. Constraints that show up everywhere

These are the constraints every design decision in this repo has to respect. They're not negotiable without changing the strict-local posture.

### A tight WSL2 RAM allocation (~8 GiB)

The Windows host likely has 32–64 GiB physical RAM, but WSL2's allocation is left at a tight default (~8 GiB). This means:

- **Source-building anything CUDA-heavy on an 8 GB-VRAM GPU OOMs.** This is what blocks `pip install flash-attn` from a fresh source build. The workaround is the cross-build on the second machine.
- **Swap thrash is one bad batch away** during training. Swap is on a 2 GiB partition (so the buffer is small) and writes hit disk, which is 100× slower than RAM. A single oversized batch can stall a step from 21s → 300s. The fix is to keep VRAM + RAM headroom comfortable; see the seq_len decision below.
- **Loading the model at startup is the tightest moment**. Unsloth's bnb-4bit pre-quantized variant (~5 GB on disk) gets memory-mapped, but actually fitting it into the GPU plus initialising the 4-bit dequant tables uses ~6.5 GB peak system RAM. Other processes on the same machine (the target repo's dev server, an agent CLI, etc.) can push the machine into swap before training even starts.

### 8 GiB VRAM

VRAM is the hard cap on:
- Model weight footprint at runtime
- KV cache for attention (grows with sequence length × layers × heads)
- Activations + gradients during backward pass
- Optimiser state

Rough VRAM ledger for the QLoRA training:

| | Bytes |
|---|---|
| Qwen 2.5 Coder 7B in NF4 (4-bit, double-quant) | ~4.5 GB |
| LoRA adapter (~40 M params × 2 bytes bf16) | ~80 MB |
| `paged_adamw_8bit` state | ~40 MB |
| Activations (with grad checkpointing) | ~2.5 GB at seq_len=1024 |
| Forward + backward intermediate buffers, KV cache | ~0.5 GB |
| **Total peak** | **~7.6 GB** (cuts it close) |

Without grad checkpointing, activations alone would blow past 8 GB. Without 4-bit, the base alone takes ~14 GB and there's no point even trying. The whole stack is co-designed to fit.

### Blackwell sm_120

The newness has both upsides and downsides.

**Upsides**:
- Native bf16 (1.5× faster than fp16 on Blackwell and far more numerically stable).
- Native FP4 / FP8 paths (unused here yet, but the hardware is there).
- 5th-gen tensor cores → ~26 TFLOPS FP32, much higher in lower-precision modes.
- ~672 GB/s memory bandwidth.

**Downsides**:
- Brand-new arch (released 2025), so the ML library ecosystem is still catching up. **No prebuilt FA2 wheels for sm_120** at the time of writing — you'll see this dependency more than once across the docs. It's the single biggest pain point.
- Some libraries (torchao's C++ extensions, for instance) refuse to load with a polite warning that they want torch ≥ 2.11 — those are safely ignored because torchao isn't on the hot path.
- CUDA 12.8 is the minimum that compiles for `compute_120,code=sm_120`; older toolkits silently fall back to older arch targets and the resulting kernels are slow.

### sm_61 (Pascal-class) — what works and what doesn't

| Feature | sm_61 status |
|---|---|
| Standard CUDA matmul | ✓ |
| FP16 storage | ✓ |
| FP16 *math acceleration* (tensor cores) | ✗ — Pascal predates tensor cores |
| BF16 | ✗ |
| Flash Attention 2 runtime | ✗ — kernels need sm_75+ |
| Unsloth Triton kernels | ✗ — also need sm_75+ |
| QLoRA NF4 (bitsandbytes) | ✗ at training time — NF4 dequant kernels need sm_75+ |
| `nvcc` cross-compile to sm_120 | ✓ (compilation is arch-agnostic) |
| llama-server inference (cuBLAS path) | ✓ — well-supported, no tensor cores but workable |

The clean line: **Pascal can serve, can compile, cannot train.** That's why the second machine is wired as builder + optional inference server, and the primary machine is wired as trainer + harness. A Turing-or-newer (sm_75+) GPU lifts the training restrictions, and on a datacenter card (e.g. A100 sm_80) or a recent consumer card (e.g. RTX 4090 sm_89) the FA2/Triton/NF4 paths all work natively.

---

## 3. The seq_len 2048→1024 decision

The single most consequential performance tweak in this repo. The smoke training runs that drove it:

### Run 1 — no FA2, seq_len=2048 (the original baseline)

```
10/10 steps: 76, 78, 99, 92, 81, 80, 88, 76, 95, 93 s
mean: 85.8 s/step, very consistent
```

Predictable. With Xformers as the attention impl and seq_len=2048, the machine is comfortable but per-step time is dominated by attention's HBM traffic.

ETA for 3 epochs (582 steps): **~14 hours**.

### Run 2 — FA2 enabled, seq_len=2048 (concerning)

```
Step  1: 34.45 s   (FA2 working as expected)
Step  2: 36.93 s
Step  3: 35.30 s
Step  4: 34.50 s
Step  5: 169.61 s  ← started swapping
Step  6: 299.73 s  ← getting worse
[...]
```

Memory snapshot at step 6:
- GPU: 7765/8150 MiB used (95%)
- RAM: 4.2 GiB used / 235 MiB free
- **Swap: 1.4 GiB used / 648 MiB free**
- Process state: `S` (sleeping on I/O — kernel waiting for swap reads)

What was happening: Unsloth's padding-free batching packs short samples into long batches and lets long samples fill a batch by themselves. At seq_len=2048, a batch with one long document sample needs the full 2048-token attention work + KV cache + activations. Combined with the LoRA gradient state already on the GPU, VRAM was at 95% and any extra cudaMalloc was triggering retries → falling back to host memory → swapping.

The 4-step-then-spike pattern is unintuitive but consistent with the dataset's length distribution: ~58% of samples are >4000 chars (~1000 tokens). Random sample selection means most batches were short for the first 4 steps; step 5 happened to draw a long one.

Naively, FA2's promise is "you can go to longer sequences without paying the memory cost". That's true *at the attention layer*. But the rest of the model (KV cache, activations, optimiser state) still scales with sequence length. On an 8 GB card already 95% full, FA2 doesn't buy you headroom — it just lets you survive at the limit.

### Run 3 — FA2 enabled, seq_len=1024 (current default)

```
10/10 steps: 21.08, 21.42, 20.96, 20.55, 21.13, 20.78, 20.84, 21.20, 20.96, 20.66 s
mean: 20.96 s/step, all within ±0.4s of each other
```

VRAM peaked at ~7.3 GiB — comfortable headroom, no swap.

ETA for 3 epochs (582 steps): **~3.4 hours**.

The cap means document-format samples > ~250 lines lose tail content. Looking at the dataset's char-length distribution:

```
chars   coverage at seq_len=1024 (~4 chars/token, so ~4096 chars)
median  4829   →  truncated
p90    19924   →  heavily truncated (only first ~21% trains)
p99    70796   →  trains on first ~6%
max   142005   →  trains on first ~3%
```

Of 3 099 train samples, **1 309 of them are short enough to fit fully** (every FIM and diff-instruction sample, plus document samples for short files). The remaining 1 790 lose their tails. The trade is acceptable — early-file content (imports, top-level definitions, class signatures) is the most generalisable training signal anyway; deep file bodies are often boilerplate or test fixtures.

### Net effect — the decision matrix

|     | Xformers @ 2048 | FA2 @ 2048 | **FA2 @ 1024 (current)** |
|---|---|---|---|
| s/step | 85.8 | 34–300 (variable) | **20.9** |
| Variance | low | extreme | low |
| Swap thrash | no | yes | no |
| 3-epoch ETA | 14 h | unstable | **3.4 h** |
| Document-sample tail loss | none | none | ~1800 of 3099 samples |

**4.1× faster than the baseline. Acceptable trade-off on tail content.**

---

## 4. Why the reference setup doesn't bump WSL2 RAM

The default here is to keep WSL2's allocation at a tight default (~8 GiB). The `.wslconfig` change is two lines on the Windows side:

```
[wsl2]
memory=24GB
swap=8GB
```

Followed by `wsl --shutdown`. The reference setup doesn't fight the constraint because:
1. The existing performance ceiling is accepted (intern-level results, ~3.4 h training).
2. Heavier WSL2 allocation on a single laptop-class machine would interfere with non-development work on the Windows side.
3. The cross-build escape hatch (the second machine) sidesteps the only situation where the constraint actually blocks the work — building large CUDA libraries.

If you do bump it, you can experiment with `seq_len=1536` or `seq_len=2048` again — there might be a more comfortable sweet spot. The smoke training script is the right place to measure.

---

## 5. The flash-attn cross-build, summarised

The builder machine's role in performance is one-time: produce the wheel that lets the training machine run at FA2 speed.

| Step | Where | Roughly how long |
|---|---|---|
| Apt prereqs (build-essential, cmake, ninja, git, …) + `cuda-toolkit-12-8` | builder | 12–15 min |
| `uv` + Python 3.13 install | builder | 30 s |
| Build venv: `torch==2.10.0+cu128` + numpy + ninja + setuptools | builder | 3–5 min |
| `pip wheel flash-attn` with `FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=4` | builder | **~37 min** |
| `scp` wheel to the trainer | LAN | 2 s |
| `pip install --no-deps <wheel>` on the trainer | trainer | 5 s |
| `pip install einops` on the trainer (FA2 runtime dep) | trainer | 1 s |
| Re-run smoke to confirm FA2 picked up | trainer | ~3.5 min |

Total wall-clock: ~1 hour, mostly nvcc compiling 78 `.cu` files for sm_120.

This is **once per torch-major-version**. The C++ ABI breaks every 6–12 months on torch's release cadence, so plan to re-run when the trainer bumps to 2.11+. The build-script comments document the rerun.

---

## 6. Other observations worth knowing

### llama-server initialises CUDA before the model loads

When `llama-server --help` runs, the binary touches CUDA early. On a Blackwell-class (sm_120) card you'll see something like:

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8150 MiB):
  Device 0: <Blackwell-class GPU>, compute capability 12.0, VMM: yes, VRAM: 8150 MiB
```

This is reassurance that the cross-compile produced binaries that actually run on the target — same way the FA2 import would crash with "no kernel image" if the wheel had been built for the wrong arch.

### `--no-mmap` matters on a tight-RAM machine

`llama-server` defaults to `mmap`-loading the model file. On a system with abundant page cache, this is the fastest path. On the tight-RAM machine, the page cache competes with everything else, and `mmap` ends up causing more swap pressure than direct reads. The default config sets `no_mmap: true`. Verified empirically: with mmap on, the first inference call after a cache eviction hits 5–10s latency; with `no_mmap`, the model is fully resident and inference is instant.

### Background processes on the training machine

The training machine runs more than just the training:
- the target repo's dev server — ~90% of one CPU, ~60 MB RAM. Fine.
- an agent CLI (if running) — varies, ~360 MB RAM at idle.
- a local Postgres (the sample app's DB) — small.

Total ambient load is ~600 MB RAM + ~10% CPU. Training has plenty of headroom for it. Worth knowing if you ever see RAM pressure: the first thing to ask is "what else is running?"

### Why `BUILD_JOBS=2` for llama.cpp on the training machine

The training machine's `scripts/build_llama_cpp.sh` defaults to `BUILD_JOBS=2`. Each `nvcc` invocation peaks ~3–4 GB host RAM during the CUDA-template instantiation phase. With 2 parallel jobs, peak combined ~7 GB — fits in a tight ~8 GiB allocation. With full-core parallelism (the obvious default), the machine OOMs partway through linking.

The builder machine runs `BUILD_JOBS=4` because it has ~32 GiB and can afford it. Tune accordingly for your hardware — the script's header documents the trade.

---

## 7. Ongoing measurements log

Logged here so anyone chasing a regression has a baseline to compare against.

### Baseline snapshot

- Phase 2 smoke (no FA2, seq_len=2048): mean 85.8 s/step, σ ≈ 8 s
- Phase 2 smoke (FA2, seq_len=2048): unstable (range 34–300 s/step)
- Phase 2 smoke (FA2, seq_len=1024): mean **20.9 s/step**, σ ≈ 0.3 s
- llama-server cold start with bundled GGUF on a Blackwell-class (sm_120) card: ~22 s to "HTTP server listening"
- llama-server `/v1/chat/completions` short prompt + 30-token response: 247 ms wall-clock end-to-end
- Egress audit (`strace -f -e connect`): 7 AF_INET connect() during one codescribe-train run session, 0 non-loopback
- Cross-build of flash-attn 2.8.3 on a Pascal-class (sm_61) builder box: 37 min wall-clock with `MAX_JOBS=4`

If anyone is benchmarking a change, please add to this list with date and short context.
