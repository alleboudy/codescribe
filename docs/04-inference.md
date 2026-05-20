# 04 — Inference: serving the fine-tuned model

After fine-tuning + GGUF export, you have a `qwen-coder-7b-q4_k_m.gguf` file on disk. This doc explains how it actually serves traffic.

## Why `llama.cpp` (and not HuggingFace `transformers` for inference)

`transformers` is the right library for *training* — it has the optimizer integrations, gradient flow, distributed-training plumbing, and the broadest model coverage. It's a poor fit for production inference:

- Pure-Python overhead: every token costs a Python round-trip.
- VRAM footprint at inference is 2–3× what `llama.cpp` needs for the same model.
- No good way to serve over HTTP without writing your own server wrapper.
- 4-bit `bitsandbytes` is slow at inference (it's optimised for training).

`llama.cpp` (https://github.com/ggml-org/llama.cpp) is a C++ implementation specifically built for fast local inference. Key properties:

- **CPU, CUDA, Metal, ROCm** support in one codebase.
- **Quantization-native**: weights stay in their quantised form on-disk and at inference; no dequantisation in Python.
- **Single self-contained file format**: GGUF.
- **Sub-second startup** even for 7B models.
- **HTTP server (`llama-server`) ships in the same binary set** as the inference runtime.

We vendor `llama.cpp` as a git submodule pinned to a stable tag (so upstream churn doesn't break us). Build via `scripts/build_llama_cpp.sh`.

## The GGUF format

GGUF stands for "GGML Universal File" (GGML was the predecessor library; the format moved on). It's a binary file holding everything needed to load and run a model:

- Architecture metadata (model type, layer count, head dimensions, vocab size, …).
- Tokenizer (vocabulary + merges).
- Generation defaults (BOS, EOS, padding tokens).
- Weights for every layer, possibly quantised to varying precisions per-tensor.

Two key consequences:

1. **You don't need any Python dependencies at inference time.** The `llama-server` binary reads the GGUF and serves it. No HuggingFace, no transformers, no torch.
2. **The file is self-describing.** `llama.cpp` tools can introspect a GGUF (`gguf-dump`) and tell you the architecture, quant levels, etc., without loading the model.

## Quantization levels

GGUF supports a zoo of quantization schemes. The ones you'll see:

| Quant | Bits per weight (avg) | Size for 7B | Quality vs fp16 | When to use |
|---|---|---|---|---|
| `f32` | 32 | ~28 GB | identical | training; never at inference |
| `f16` | 16 | ~14 GB | identical | reference for quality comparisons |
| `Q8_0` | 8 | ~7 GB | virtually identical | when you have the VRAM and want max quality |
| `Q6_K` | ~6 | ~5.5 GB | very high | premium balance; fits on 8 GB with ~2 GB context |
| `Q5_K_M` | ~5 | ~5 GB | high | good middle ground |
| **`Q4_K_M`** | **~4.5** | **~4.4 GB** | **good** | **our default** |
| `Q3_K_M` | ~3 | ~3.5 GB | noticeable quality drop | only if VRAM is desperate |
| `Q2_K` | ~2 | ~2.5 GB | significant quality loss | not recommended |

`K_M` ("K-quants medium") means a mixed-precision scheme where attention weights stay at higher precision and feed-forward weights drop to the named bits. This preserves more quality than uniform quantization. `K_S` is "small" (lower quality) and `K_L` is "large" (higher quality) variants.

**For our 7B fine-tune on 8 GB VRAM, `Q4_K_M` is the sweet spot.** It leaves room for a 32K-token KV cache and the runtime overhead.

## llama-server: the HTTP daemon

`llama-server` is a binary in the same `llama.cpp` build tree (`vendor/llama.cpp/build/bin/llama-server`). It loads a GGUF, opens an HTTP listener, and serves OpenAI-compatible endpoints.

### Canonical invocation

```bash
vendor/llama.cpp/build/bin/llama-server \
    -m checkpoints/qwen-coder-7b-q4_k_m.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    -ngl -1 \
    --ctx-size 32768 \
    --no-mmap
```

Flag-by-flag:

| Flag | Default | Why this value |
|---|---|---|
| `-m` | — | Path to the GGUF. |
| `--host` | `127.0.0.1` | Loopback-only. **Never `0.0.0.0`** in default configs — strictly-local posture. Documented hard rule. |
| `--port` | `8080` | Standard. Change if you have a port conflict. |
| `-ngl <N>` | `0` | Number of layers offloaded to GPU. `-ngl -1` means "all layers"; for Qwen 7B that's ~28 layers. `0` means CPU-only inference. Intermediate values split between CPU and GPU when the model exceeds VRAM. |
| `--ctx-size` | upstream default ~4096 | We use `32768` to match Qwen's `n_ctx_train`. Larger context = more KV cache VRAM. **Anything below 16384 breaks every modern agent harness whose system prompt is ~9K tokens.** |
| `--no-mmap` | mmap on | Disables memory-mapping the GGUF. Mmap is faster startup but confuses VRAM accounting; turning it off gives reliable `nvidia-smi` numbers. |

### KV cache VRAM math (why `--ctx-size 32768` costs ~1.8 GB)

For each token in the context, `llama-server` keeps a key and value tensor per layer per attention head. For Qwen 2.5 Coder 7B:

- 28 layers.
- 4 KV heads (Group-Query Attention; not 28 — that's the *query* head count).
- Head dimension 128.
- fp16 (2 bytes per element by default; can be reduced with `--cache-type-k q8_0` etc.).

Per token: `28 layers × 4 kv-heads × 128 head-dim × 2 (K+V) × 2 bytes = 57344 bytes = 56 KB/token`.

At `ctx_size = 32768`: `32768 × 56 KB = 1.79 GB`.

Add the model itself (~4.4 GB at Q4_K_M, loaded into VRAM at `-ngl -1`) and you're at ~6.2 GB. Leaves ~1.7 GB on an 8 GB card for prompt-processing buffers and headroom.

If you want a larger context (64K), you need to either:
- Drop to Q3_K_M to free 1 GB.
- Use `--cache-type-k q8_0 --cache-type-v q8_0` to compress the KV cache 2×.
- Use a card with more VRAM.

## OpenAI-compatible API

`llama-server` exposes:

| Path | Purpose |
|---|---|
| `GET /v1/models` | Lists the served model. Returns `{"data": [{"id": "<filename>", "meta": {"n_ctx_train": 32768, ...}}]}` |
| `GET /health` | Liveness check. Returns `{"status": "ok"}` once the model is loaded. |
| `POST /v1/chat/completions` | Standard OpenAI chat-completions endpoint. Accepts `messages`, `temperature`, `max_tokens`, etc. |
| `POST /v1/completions` | Legacy raw-text completions (for older clients). |
| `POST /v1/embeddings` | Embedding endpoint (if you started `llama-server` with `--embeddings`). |

This is what makes any harness with an OpenAI client work. You point `OPENAI_BASE_URL=http://127.0.0.1:8080/v1` and `OPENAI_API_KEY=<any-non-empty-string>` (the server ignores the key) and the harness talks to your local model as if it were OpenAI.

### A worked chat-completions example

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "qwen-coder-7b-q4_k_m.gguf",
      "messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function that reverses a string."}
      ],
      "max_tokens": 200,
      "temperature": 0.2
    }' | python3 -m json.tool
```

Response:
```json
{
  "choices": [{"message": {"role": "assistant", "content": "def reverse_string(s):\n    return s[::-1]"}, ...}],
  "created": 1735689600,
  "model": "qwen-coder-7b-q4_k_m.gguf",
  "usage": {"prompt_tokens": 32, "completion_tokens": 18, "total_tokens": 50}
}
```

## Performance numbers (reference)

On the test hardware, Q4_K_M Qwen 7B with `--ctx-size 32768 -ngl -1`:

| Hardware | Prompt processing | Generation |
|---|---|---|
| GTX 1080 Ti (Pascal, 11 GB) | ~190 tok/s | ~67 tok/s |
| RTX 2000 Ada Laptop (8 GB) | ~280 tok/s (est.) | ~80 tok/s (est.) |
| RTX 5070 Laptop (Blackwell, 8 GB) | ~340 tok/s (est.) | ~95 tok/s (est.) |
| Apple M3 Max (Metal, 36 GB unified) | ~120 tok/s | ~38 tok/s |
| CPU only (i9 13th gen, 24 threads) | ~25 tok/s | ~6 tok/s |

These are rough; your numbers will vary with thermal throttling, model variant, OS, etc. Always measure your own.

## The two non-obvious gotchas (why this is documented as a hard rule)

1. **`--ctx-size` defaults are too small for modern harnesses.** The upstream `llama.cpp` default is 4096 (a few releases back; newer ones are 8192). Every modern coding harness sends a system prompt of ~9000 tokens by default. If you start `llama-server` with the upstream default and call `claw` against it, you get `exceed_context_size_error` on the very first message. Always set `--ctx-size 32768`.

2. **Binding `0.0.0.0` exposes the model to your LAN/Tailscale net.** For a strictly-local stack this is a posture violation. Refuse to default `host` to `0.0.0.0` and require an explicit `unsafe_bind_all=True` from the caller before allowing it.

Both encoded as static tests in the `serve` package (see issue [#2 §10](https://github.com/alleboudy/llm-finetuner/issues/2) Phase 3).

## Alternatives: vLLM, Ollama

| Server | When it shines | Why we don't use it |
|---|---|---|
| **vLLM** | High-throughput batched inference for many concurrent users; PagedAttention | Heavy Python dependency; designed for cluster deployment; overkill for one user on a laptop |
| **Ollama** | One-command model installer; nice CLI UX | Wraps `llama.cpp`; adds a daemon layer; less direct control over the binary; their default ctx-size is also too small for our harnesses |
| **text-generation-inference** (HuggingFace) | Production-grade HF-native serving | Requires CUDA + lots of VRAM; not local-friendly |

For our setup, `llama-server` direct is the right level of abstraction: small, fast, single-binary, no daemon, OpenAI-compatible.

## Further reading

- llama.cpp main repo: https://github.com/ggml-org/llama.cpp
- GGUF format spec: https://github.com/ggml-org/llama.cpp/blob/master/docs/gguf.md
- The `convert_hf_to_gguf.py` script (where HF safetensors becomes GGUF): https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py
- `llama-server` README: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- See [`13-further-reading.md`](13-further-reading.md) for benchmarks and quantization theory.
