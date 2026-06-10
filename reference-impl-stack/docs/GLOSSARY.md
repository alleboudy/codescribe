# Glossary

Quick lookup for the project's jargon. Hyperlink-rich; longer treatments live in their own docs.

> **New to the project?** Read [`CONCEPTS.md`](CONCEPTS.md) first for the conceptual map of fine-tuning + RAG + MCP and how they compose for `../your-repo`. This glossary is for fast term lookups once you have the shape.

---

### **AF_INET / AF_INET6 / AF_UNIX / AF_NETLINK**
Linux socket address families. AF_INET = IPv4 over network, AF_INET6 = IPv6 over network — these are the ones we audit for non-loopback connections. AF_UNIX = process-to-process IPC on the same host. AF_NETLINK = kernel-userspace queries. The latter two are local by definition. ⇒ [`docs/STRICTLY-LOCAL-POSTURE.md § 6`](STRICTLY-LOCAL-POSTURE.md#6-verification--the-egress-audit)

### **adapter**
Two meanings in this project. (1) A LoRA adapter — the small trainable matrices added on top of a frozen base model. (2) A wrapper class around a vendored binary (`ClawCodeHarness` is an adapter for `vendor/claw-code/`'s Rust binary).

### **bf16 / bfloat16**
A 16-bit floating-point format with the same exponent range as fp32 (~10⁻³⁸ to ~10³⁸) but only 7 mantissa bits. Numerically more stable than fp16 for training because the wider exponent prevents underflow / overflow. Native on Ampere+, Hopper, Blackwell. **Never use fp16 on Blackwell** — it's slower (no native FP16 acceleration was ever a Blackwell promise) and less stable.

### **bnb / bitsandbytes**
Library that implements 8-bit and 4-bit quantization for PyTorch models. NF4 (the 4-bit format we use) is bitsandbytes-specific. Requires sm_75+ for the 4-bit path; older GPUs fall back to fp16 + 8-bit only.

### **Blackwell**
NVIDIA's 2025 consumer + workstation GPU architecture. Compute capability 12.0 (sm_120). Successor to Ada Lovelace. RTX 5080, 5090. Native bf16, FP8, FP4, 5th-gen tensor cores.

### **claw-code**
[`ultraworkers/claw-code`](https://github.com/ultraworkers/claw-code), a Rust CLI agent harness. Vendored under `vendor/claw-code/` at pinned SHA `357629d`. Default harness for `codescribe-train run`. ⇒ [`docs/ARCHITECTURE.md § 2`](ARCHITECTURE.md#2-each-phase-in-one-paragraph), [`vendor/SECURITY-NOTES.md`](../vendor/SECURITY-NOTES.md).

### **compute capability**
NVIDIA's version number for GPU hardware features. Major.minor. sm_61 = Pascal. sm_75 = Turing. sm_80 = Ampere (A100). sm_86 = Ampere consumer (RTX 30-series). sm_89 = Ada (RTX 40-series). sm_90 = Hopper (H100). sm_120 = Blackwell.

### **CUDA toolkit**
NVIDIA's compiler suite for GPU code: `nvcc`, headers, libs, profilers. Different from the **CUDA driver** (the kernel module / `libcuda.so` that lets userspace talk to the GPU). The toolkit is needed for **building** CUDA code; the driver is needed for **running** it.

### **diff_instr**
One of the three sample formatters in `codescribe_train/data/`. Turns each non-trivial git commit into a ChatML user/assistant pair where the user message is the commit subject + body and the assistant response is the diff. ⇒ [`docs/QLoRA-AND-UNSLOTH.md § 5`](QLoRA-AND-UNSLOTH.md#5-why-three-formatters)

### **document (formatter)**
One of the three sample formatters. Wraps each file with `<|repo_name|>` + `<|file_sep|>` markers and emits its content as a single sample. Good for teaching file-level patterns.

### **DoD (Definition of Done)**
The plan's per-phase pass criterion. Phase 4 DoD: `codescribe-train run` opens a session against the local fine-tuned model end-to-end, with the egress audit passing. Tracked in the continuity playbook ([issue #1](https://github.com/example-org/codescribe-train/issues/1)).

### **egress audit**
The end-to-end test that captures every `connect()` syscall via `strace -f` during a `codescribe-train run` session and confirms zero non-loopback destinations. ⇒ [`docs/known-issues.md § Egress audit`](known-issues.md), [`docs/STRICTLY-LOCAL-POSTURE.md § 6`](STRICTLY-LOCAL-POSTURE.md#6-verification--the-egress-audit).

### **FA / FA2 / FA3**
Flash Attention. FA1 (2022) introduced the item + online-softmax approach. FA2 (2023) restructured for better parallelism. FA3 (2024) is Hopper-only. We use **FA2 v2.8.3** on Blackwell. ⇒ [`docs/FLASH-ATTENTION.md`](FLASH-ATTENTION.md).

### **FIM (Fill-in-the-Middle)**
Sample format that masks a span of contiguous text and trains the model to predict it given the surrounding prefix and suffix. Native to Qwen 2.5 Coder. The PSM (Prefix-Suffix-Middle) ordering is what we emit: `<|fim_prefix|>{...}<|fim_suffix|>{...}<|fim_middle|>{...}`. Good for editor autocompletion training.

### **GGUF**
GGML Universal File. The wire format `llama.cpp` consumes for serving. Roughly: a quantized model plus tokenizer plus metadata, in a single self-contained file. Produced by `convert_hf_to_gguf.py` (HuggingFace → GGUF) plus optionally `llama-quantize` (fp16 GGUF → quantized GGUF).

### **gradient checkpointing**
Trade compute for memory: don't save activations during forward; recompute them during backward. Common for fitting big models in tight VRAM. Cost: ~30% slower training step, savings: ~3–5× less activation memory.

### **HBM**
High-Bandwidth Memory. The GPU's main DRAM (the "VRAM" you read about). Distinct from the per-SM SRAM caches that Flash Attention's tiling exploits. HBM bandwidth is what attention ops bottleneck on.

### **HumanEval**
A 164-problem coding benchmark from OpenAI (2021). Asks the model to complete a function given a docstring + signature. We don't currently run it (rationale in [`docs/QLoRA-AND-UNSLOTH.md § 7`](QLoRA-AND-UNSLOTH.md#what-we-dont-run)).

### **sample**
Two things share the name. (1) The project name is a neutral placeholder with no special meaning. (2) [`../your-repo`](../../your-repo/) — the user's existing project, used here as the first training-data target. **Not the same project as `codescribe-train`** despite the name overlap.

### **LoRA**
Low-Rank Adaptation. Inserts small trainable matrices `A` and `B` in parallel with frozen weight matrices `W`. Only `A` and `B` train. ⇒ [`docs/QLoRA-AND-UNSLOTH.md § 2`](QLoRA-AND-UNSLOTH.md#2-lora--low-rank-adapters).

### **manifest.json**
File that the data pipeline writes alongside `train.jsonl` / `val.jsonl` / `test.jsonl`. Records the source repo, repo_name, config used, split weights, salt, seed, and per-split totals. Provenance for reproducibility.

### **MMQ / FA / KV-cache (in llama.cpp)**
Three of the major hot paths in `llama.cpp`'s CUDA backend. MMQ = "matrix-matrix quantized" (the matmul kernels for quantized weights). FA = Flash Attention (yes, llama.cpp has its own). KV-cache = the running attention key+value buffer for the current inference. All three are heavily templated CUDA — which is why building llama.cpp from source is RAM-hungry.

### **NF4**
NormalFloat 4-bit. A 4-bit quantization format whose 16 representable values are placed at quantiles of a normal distribution rather than at uniform intervals. Designed by the QLoRA paper for transformer weights, which are approximately Gaussian. ⇒ [`docs/QLoRA-AND-UNSLOTH.md § 3`](QLoRA-AND-UNSLOTH.md#3-qlora--adding-4-bit-quantization).

### **noop / NoOpHarness**
A harness implementation that just echoes its stdin to stdout and exits cleanly on EOF. Used as a swap-test that the `Harness` ABC actually allows alternative implementations. ⇒ `tests/harness/test_swap.py`.

### **nvcc**
NVIDIA's CUDA compiler. Cross-compiler — can target any supported `compute_*,sm_*` tuple regardless of the build host's GPU. Hot loops in fa-attn, llama.cpp, bitsandbytes are all `nvcc`-compiled.

### **OpenAI-compatible API**
The `/v1/chat/completions` and `/v1/models` HTTP shape originally defined by OpenAI. `llama-server`, `vllm`, `ollama`, OpenRouter, Together, Groq, Mistral and many others speak this protocol. The harness layer talks OpenAI Chat Completions; the backend layer exposes it. The standard makes them swappable.

### **paged_adamw_8bit**
The Adam optimiser with 8-bit state plus QLoRA's "paged" extension that swaps state to host memory on demand. Saves ~3× VRAM vs fp32 Adam.

### **padding-free batching (Unsloth)**
Unsloth's optimisation that packs multiple short samples into the same batch without explicit padding tokens. Different from TRL's `packing` (which is similar but coarser-grained). Good for variable-length data; bad for VRAM predictability when sequence-length variance is high — see the seq_len 2048→1024 decision in [`docs/HARDWARE-AND-PERFORMANCE.md § 3`](HARDWARE-AND-PERFORMANCE.md#3-the-seq_len-20481024-decision).

### **Pascal**
NVIDIA's 2016 GPU architecture. Compute capability 6.0 / 6.1. GTX 1080, Tesla P100. **No tensor cores. No bf16. No FA2/Unsloth.** Useful here only for inference and as an `nvcc` host.

### **PEFT**
Parameter-Efficient Fine-Tuning. The HuggingFace library that implements LoRA, prefix tuning, prompt tuning, etc. We use it via Unsloth's `FastLanguageModel.get_peft_model()`.

### **probe (hardware)**
`codescribe_train.backends.probe` — reads `nvidia-smi` (or `pynvml` if installed) and `psutil` to produce a `HardwareProfile`, then `recommend_backend()` picks defaults based on free VRAM and target model size. ⇒ `codescribe_train/backends/probe.py`.

### **Q4_K_M**
A llama.cpp quantization scheme. ~4.5 bits per weight on average using K-means clustering with mixed precision (some tensors in fp16). The de-facto sweet spot for general-purpose serving — small enough to fit comfortably, big enough to stay close to fp16 quality.

### **QLoRA**
Quantized LoRA. The base model is held in 4-bit during training while the small LoRA adapters stay in bf16. ⇒ [`docs/QLoRA-AND-UNSLOTH.md § 3`](QLoRA-AND-UNSLOTH.md#3-qlora--adding-4-bit-quantization).

### **Qwen 2.5 Coder 7B Instruct**
The base model fine-tuned in this project. Released by Alibaba's Qwen team (Sept 2024). 7.6 B parameters. FIM-trained at pretraining time, so the FIM markers are part of its vocabulary. We use the `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` pre-quantized variant for ~10 GB less first-load download.

### **RemoteBackend**
A `Backend` ABC stub class that exists in code but refuses to start unless `CODESCRIBE_ENABLE_REMOTE_BACKEND=1` is set. Reserved for the optional "second machine on your LAN" topology — e.g., a second machine on your LAN with a larger GPU serving 14B-class models. Never wired into a default config. ⇒ [`docs/STRICTLY-LOCAL-POSTURE.md § 1`](STRICTLY-LOCAL-POSTURE.md#1-the-headline).

### **rk / r / rank (LoRA)**
The "low rank" of the LoRA adapter — controls how much capacity the adapter has to absorb fine-tuning signal. We use `r=16`. Common alternatives: 8 (more conservative), 64 (more capacity, more VRAM).

### **routing trap (claw-code)**
A bug-shaped behaviour we surfaced in the audit: bare `qwen-*` / `qwen/*` model names route to DashScope's API by prefix match, regardless of `OPENAI_BASE_URL`. We avoid it by prefixing model names with `openai/` in default configs. Documented in [`vendor/SECURITY-NOTES.md`](../vendor/SECURITY-NOTES.md).

### **salt (data splitter)**
A string mixed into the per-file hash that determines train/val/test bucket assignment. Same salt + same input → same split, every time. Different salt → different split. We use `sample-v1`. **Don't change the salt without rebuilding the dataset** or train and val will start mixing samples.

### **sm_NN**
NVIDIA's GPU streaming-multiprocessor architecture identifier. `sm_120` = Blackwell (RTX 50-series). `sm_90` = Hopper (H100). `sm_89` = Ada (RTX 40-series). `sm_61` = Pascal. The number prefix to `sm_` is the compute capability major.minor with the dot dropped (so 12.0 → 120). Build target.

### **SFTTrainer**
The supervised fine-tuning trainer from TRL (`trl.SFTTrainer`). Wraps HuggingFace's `Trainer` with conveniences for SFT-specific tasks. We use it via Unsloth, which patches it to use Unsloth's optimised kernels.

### **strace -f -e trace=connect**
The egress audit method. `strace -f` follows children, `-e trace=connect` filters to just the `connect()` syscall. Output: every outbound socket connection attempt by every process in the tree, with destination address. ⇒ [`docs/STRICTLY-LOCAL-POSTURE.md § 6`](STRICTLY-LOCAL-POSTURE.md#6-verification--the-egress-audit).

### **submodule (vendored)**
Both `vendor/claw-code` and `vendor/llama.cpp` are git submodules pinned at exact SHAs. Bumping requires explicit `git submodule update --remote` followed by review of the diff and the audit. Auto-updates impossible.

### **TRL**
[Transformer Reinforcement Learning](https://github.com/huggingface/trl). HuggingFace's library for RLHF / DPO / SFT. We use `SFTTrainer` and `SFTConfig` only.

### **Unsloth**
[unslothai/unsloth](https://github.com/unslothai/unsloth). Library that rewrites the QLoRA training hot path in fused Triton kernels. Runtime requirement: sm_75+. Optionally uses Flash Attention 2 if importable, falls back to Xformers. ⇒ [`docs/QLoRA-AND-UNSLOTH.md § 4`](QLoRA-AND-UNSLOTH.md#4-unsloth--kernel-level-optimisations-on-top).

### **uv**
[astral-sh/uv](https://github.com/astral-sh/uv). The Python project / package manager we use throughout. Replaces `pip`, `virtualenv`, `pyenv`. Single static binary, no Python bootstrap needed. Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

### **VRAM**
Video RAM. The GPU's HBM. For example, 8 GB on an 8 GB-VRAM training GPU, 11 GB on a Pascal-class (sm_61) inference GPU.

### **W&B / Weights & Biases**
A third-party experiment-tracking service. The TRL `SFTConfig.report_to` field can target it, which would phone home metrics during training. We always set `report_to="none"` and `WANDB_DISABLED=true` defensively.

### **Xformers**
Meta's library of memory-efficient attention implementations. Predates Flash Attention 2; broader hardware support but slower than FA2. Unsloth's fallback when FA2 isn't importable.

### **`<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>`**
Special tokens in Qwen 2.5 Coder's vocabulary used for the Fill-in-the-Middle format. Our FIM formatter emits them in PSM (Prefix-Suffix-Middle) order.

### **`<|file_sep|>` / `<|repo_name|>`**
Special tokens in Qwen 2.5 Coder's vocabulary used to bracket files in continued-pretraining-style data. Our document formatter uses them.

### **`<|im_start|>` / `<|im_end|>`**
ChatML role markers. Used in Qwen Instruct models to mark the start and end of `system` / `user` / `assistant` turns. Our diff_instr formatter emits them.
