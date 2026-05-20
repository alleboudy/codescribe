# 12 — Glossary

One-line definitions. For deeper coverage follow the link.

**Adapter** — a small set of weights trained on top of a frozen base model. See LoRA.

**Agent / Agentic** — an LLM-driven program that uses tools to act in the world (read files, run commands, call APIs). See [harness](05-harnesses.md).

**AGENTS.md** — a convention file at any directory level read by AI coding agents (Copilot Coding Agent, Claude Code, Cursor) on every task. Encodes per-package rules.

**ANN** — Approximate Nearest Neighbour. The class of algorithms (HNSW, IVF, ScaNN) used by vector stores to find similar embeddings faster than O(N).

**API key** — a credential token. Used for Bugzilla auth (`X-BUGZILLA-API-KEY` header), HuggingFace, GitHub, etc.

**Attention** — the core operation in a transformer; each token attends to all others to compute its next-layer representation.

**BF16 / bfloat16** — a 16-bit float with fp32's dynamic range and reduced precision. Preferred for modern LLM training. Needs sm_80+.

**Bitsandbytes** — the Python library providing 4-bit quantization (NF4) and the `paged_adamw_8bit` optimizer.

**BM25** — Best Matching 25; the textbook lexical retrieval ranking algorithm. The default ranker in SQLite's FTS5.

**Bugzilla** — open-source bug tracker. See [10-bugzilla.md](10-bugzilla.md).

**Changelist (CL)** — atomic unit of change in Perforce. See [09-perforce.md](09-perforce.md).

**Chunking** — splitting source text into smaller pieces before embedding. See [08-rag.md § Chunking](08-rag.md#chunking-strategies).

**Claw / claw-code** — the Rust agent harness this stack vendors. See [05-harnesses.md](05-harnesses.md).

**CL** — see Changelist.

**closingIssuesReferences** — a field on a GitHub PR exposed via GraphQL; lists the issues a PR explicitly closes. The "gold" pairing signal for the GitHub-based RAG variant.

**Compute capability (sm_XX)** — NVIDIA's GPU instruction-set version. See [11-hardware.md](11-hardware.md).

**Context size / window** — the maximum number of tokens the LLM can attend to per request. Qwen 2.5 Coder 7B trains at 32768.

**Copilot** — GitHub's umbrella brand for AI coding products. See [06-github-copilot.md](06-github-copilot.md).

**Cosine similarity** — `(v1 · v2) / (||v1|| · ||v2||)`. For unit vectors, equals the dot product.

**CUDA** — NVIDIA's parallel-compute platform. Toolkit (compiler + libs) vs Runtime (per-app shared lib) — see [11-hardware.md § CUDA versions](11-hardware.md#cuda-toolkit-vs-cuda-runtime-version).

**Diff** — a textual representation of changes between two versions of a file. We use unified diff format.

**Egress audit** — verifying that a process makes no unexpected outbound network connections. Done with `strace -e trace=connect` or a Python `socket.connect` monkey-patch.

**Embedder** — a model that produces embeddings. We use BAAI/bge-large-en-v1.5.

**Embedding** — a fixed-size dense numeric vector representing the meaning of a text. See [08-rag.md § Embeddings](08-rag.md#embeddings-vectors-of-meaning).

**FA2 / Flash Attention 2** — a memory-efficient O(N) attention implementation. Requires sm_80+. Required for fast QLoRA training at non-trivial sequence lengths.

**FIM / Fill-In-Middle** — a training format where the model learns to fill code given both prefix and suffix. Qwen tokens: `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`.

**Fine-tuning** — adapting a pre-trained model to a specific domain by training on additional data. See [02-fine-tuning.md](02-fine-tuning.md).

**fp16 / float16** — 16-bit float; standard for inference on older GPUs.

**fp32 / float32** — standard 32-bit float; reference precision.

**FTS5** — SQLite's built-in full-text search module; provides BM25 ranking. Same database as our vector store.

**GGUF** — "GGML Universal File"; the binary file format llama.cpp uses for models. Self-contained: weights + tokenizer + metadata.

**Git** — the distributed version control system. Contrast with [Perforce](09-perforce.md).

**`gh` CLI** — GitHub's official command-line tool. Used by our RAG indexer's GitHub source.

**Harness** — see [05-harnesses.md](05-harnesses.md). The program that wraps an LLM as an agent.

**HF / HuggingFace** — the company; their `transformers`, `peft`, `datasets`, and `huggingface_hub` libraries are foundational.

**HNSW** — Hierarchical Navigable Small World; a popular ANN algorithm.

**httpx** — modern Python HTTP client; what we use instead of `requests`.

**Hybrid retrieval** — combining vector search and BM25, merging the rankings. See [08-rag.md § Hybrid retrieval](08-rag.md#hybrid-retrieval-combining-embeddings--bm25).

**Instruction tuning** — fine-tuning a base model to follow instruction-style prompts ("the Instruct variant").

**JSON-RPC** — a simple JSON-based RPC protocol. MCP's wire format.

**KV cache** — Keys and Values cached per token in attention layers during incremental generation. Memory cost is proportional to context size. See [04-inference.md § KV cache](04-inference.md#kv-cache-vram-math-why---ctx-size-32768-costs-18-gb).

**L2 normalisation** — scaling a vector so its Euclidean norm equals 1. Required for cosine search done via dot product.

**llama.cpp** — the C++ inference runtime we use. See [04-inference.md](04-inference.md).

**llama-server** — the OpenAI-compatible HTTP daemon from llama.cpp.

**LLM** — Large Language Model. Generally a transformer with ≥1B parameters.

**LoRA** — Low-Rank Adaptation. A parameter-efficient fine-tuning method. See [02-fine-tuning.md § LoRA](02-fine-tuning.md#lora-the-low-rank-insight).

**Marshal (Python `marshal` module)** — Python's binary object-serialisation format used by `p4 -G` output. Distinct from JSON (textual) and from Python's other (unsafe) binary serialiser that we explicitly forbid via the AGENTS.md anti-patterns.

**MCP** — Model Context Protocol. See [07-mcp.md](07-mcp.md).

**MTEB** — Massive Text Embedding Benchmark; the standard leaderboard for embedding models.

**`-ngl` / `n_gpu_layers`** — llama.cpp flag; how many model layers to offload to GPU. `-ngl -1` = all.

**NF4** — Normalised Float 4; the 4-bit quantization scheme used by QLoRA (via bitsandbytes).

**Optimizer** — algorithm that updates weights from gradients during training. We use `paged_adamw_8bit`.

**`paged_adamw_8bit`** — bitsandbytes' AdamW variant with 8-bit state pageable between CPU and GPU. Required for 7B QLoRA on 8 GB VRAM.

**Peft / PEFT** — HuggingFace's Parameter-Efficient Fine-Tuning library; provides the LoRA adapter implementation.

**Perforce / Helix Core** — centralised VCS. See [09-perforce.md](09-perforce.md).

**Posture (strictly-local)** — design constraint that data must not leave the host. Drives almost every architectural choice in this stack.

**Pydantic** — Python data validation library; used for our config models.

**Q4_K_M** — a llama.cpp quantization scheme; mixed-precision 4-bit average. Our serving default.

**QLoRA** — Quantized LoRA: LoRA training on top of a 4-bit-quantised base. The recipe we use.

**Quantization** — reducing weight precision (fp32→fp16→int8→int4) to save memory at some quality cost. See [04-inference.md § Quantization levels](04-inference.md#quantization-levels).

**RAG** — Retrieval-Augmented Generation. See [08-rag.md](08-rag.md).

**Rate limiting** — capping the rate of outbound requests to a remote service. We use a token bucket; see [issue #5 §3](https://github.com/alleboudy/codescribe/issues/5).

**Rendezvous** (in `torchrun`) — the bootstrap mechanism for distributed training; coordinates the participating nodes. See [issue #3 §6.2](https://github.com/alleboudy/codescribe/issues/3).

**RRF** — Reciprocal Rank Fusion; the textbook way to merge multiple retrieval rankings.

**sm_XX** — see Compute capability.

**Sentence-transformers** — Python library wrapping embedding models. What we use for the embedder.

**seq_len** — sequence length during training; the number of tokens per training example. Our default 1024.

**sqlite-vec** — SQLite extension adding vector-search capabilities. Our vector store.

**Stdio / STDIO** — standard input/standard output; the most common MCP transport.

**Strictly-local** — see Posture.

**SVD** — Singular Value Decomposition; the linear-algebra fact LoRA exploits (low-rank approximation of a matrix).

**Tailscale** — mesh VPN we use to connect hosts. Useful for fleet setups.

**`tenacity`** — Python library for retry logic. Used in our HTTP clients.

**Token** — the atomic unit a model sees. Roughly 0.75 tokens per English word; coder models have lower tokens-per-code-line due to identifier-aware tokenization.

**Token bucket** — a rate-limiting algorithm. See [issue #5 §3](https://github.com/alleboudy/codescribe/issues/5).

**Tokenizer** — the function that maps text → tokens. Qwen 2.5 Coder uses a 152K-vocab BPE tokenizer with FIM tokens.

**Tool call / Tool use** — the model emitting a `tool_calls` JSON in an OpenAI chat-completions response.

**Transformer** — the neural-network architecture underlying modern LLMs (Vaswani et al., 2017).

**Unsloth** — a drop-in faster trainer for QLoRA. See [02-fine-tuning.md § Unsloth](02-fine-tuning.md#unsloth-why-we-use-it-instead-of-vanilla-transformerstrainer).

**`uv`** — modern Python package manager (Astral). Replaces pip + virtualenv + pip-tools. We use it exclusively.

**Vector store** — database optimised for nearest-neighbour search on embeddings. Ours: sqlite-vec.

**VRAM** — Video RAM; GPU memory. The constraint.

**WAL (SQLite journal mode)** — Write-Ahead Log; allows concurrent reads while writes are in progress.

**Wandb / Weights & Biases** — cloud experiment tracker; banned in this stack's hard rules (telemetry violation).

**WSL2** — Windows Subsystem for Linux 2; what runs the Linux side of our msi-1 reference host.

**Workspace (Perforce)** — a local checkout of a depot subset. See [09-perforce.md § Workspace](09-perforce.md#workspace--client).
