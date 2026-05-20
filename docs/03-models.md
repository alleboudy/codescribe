# 03 — Open-source models (and why Qwen 2.5 Coder 7B Instruct)

## The lay of the land

As of mid-2026, the open-weights LLM landscape for coding work breaks down roughly:

| Family | Notable variants | License | Strengths |
|---|---|---|---|
| **Qwen** (Alibaba) | Qwen 2.5, Qwen 2.5 Coder, Qwen 3 (general) | Tongyi Qianwen / Apache 2.0 (varies by model) | Strongest coder under 14B; broad language coverage; mature ecosystem |
| **DeepSeek** | DeepSeek-Coder, DeepSeek-V3 | DeepSeek License (free for research+commercial with attribution) | Very strong coder; large variants |
| **Code Llama** (Meta) | 7B / 13B / 34B / 70B | Llama 2 License | Mature but ageing |
| **StarCoder** (HuggingFace + ServiceNow) | StarCoder2 3B / 7B / 15B | BigCode OpenRAIL-M | Trained on permissively-licensed code only — important for some contexts |
| **CodeGemma** (Google) | 2B / 7B | Gemma License | Decent; smaller community |

For this stack we picked **Qwen 2.5 Coder 7B Instruct** as the canonical base. The reasoning below.

## Why Qwen 2.5 Coder 7B Instruct specifically

### Right size

7B parameters at 4-bit quantisation = ~4.4 GB. Comfortably fits on 8 GB VRAM with room for QLoRA training (see [`02-fine-tuning.md § QLoRA`](02-fine-tuning.md#qlora-the-4-bit-twist)). 1B–3B models are noticeably less capable for non-trivial code tasks; 13B and above don't fit in 8 GB even at 4-bit.

### Coder-tuned base

Qwen 2.5 Coder was pre-trained on a code-heavy mix (estimated ~70% code, 30% natural language). The general Qwen 2.5 7B model is fine for many tasks but the coder variant scores meaningfully higher on code benchmarks (HumanEval, MBPP, etc.) and — more importantly for us — has been trained on enough code to *understand the structure of large codebases*, not just generate syntactically correct snippets.

### Instruct, not base

The "Instruct" suffix means the model was further trained to follow instruction-style chat prompts (`<|im_start|>user ... <|im_end|>`). The non-Instruct base would require us to either prompt-engineer around its raw completion behaviour or apply our own instruct-fine-tuning step — extra work for no benefit since our downstream use is conversational (via a coding harness).

### FIM tokens

Qwen reserves three literal tokens for Fill-In-Middle training (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`). Most coder models do this in *some* form but the token names vary. The data pipeline ([`02-fine-tuning.md § Training data`](02-fine-tuning.md#training-data-completion-vs-fim-vs-instruction)) emits these literal tokens; switching base models means re-checking the FIM token format. Documented hard rule.

### Context length

Qwen 2.5 Coder 7B is trained for `n_ctx_train=32768` tokens. That's the context window the model was actually trained on — going beyond it at inference works (RoPE positional encoding extrapolates) but quality degrades. **Setting `--ctx-size 32768` at inference** matches the training distribution AND is enough for any modern agent harness's system prompt + reasonable conversation history. Documented hard rule.

### License

Tongyi Qianwen License — permissive enough for commercial use under most conditions. The full text is at https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/LICENSE. Check with your legal team if your use case is unusual.

## Other models you might consider

### DeepSeek-Coder

Comparable quality; some benchmarks favour DeepSeek slightly. The license is more permissive than Qwen's in some ways. Worth A/B-testing against your downstream tasks if you have the time. The data pipeline + train script in this stack would need only:
- Update `configs/train/<config>.yaml`'s `base_model` field.
- Update the FIM token strings in `data/AGENTS.md` if DeepSeek uses different ones.
- Re-run all of Phase 1 + Phase 2.

### Code Llama 7B

Older (released 2023); less performant on modern benchmarks. Useful if your team is already on the Llama ecosystem and switching costs are high.

### StarCoder2-7B

Notable because StarCoder is trained *only* on permissively-licensed code (the BigCode project's licensing audit). Some legal departments will only allow this. If that's you, swap the base and re-run.

### Larger models (Qwen 14B, DeepSeek V3, etc.)

Don't fit in 8 GB VRAM, even at 4-bit. To use them you need:
- A 12 GB+ GPU for inference (consumer 4070/4080-class) and 24 GB+ for QLoRA training.
- OR an external workstation/server. (See [`11-hardware.md § Escalation`](11-hardware.md#when-to-escalate-beyond-laptop-class).)

## Tokenizer notes

Qwen 2.5 Coder's tokenizer:

- Vocabulary size: 152K (large by GPT-2's 50K standard; gives better compression for code).
- Byte-fallback enabled — handles arbitrary UTF-8 bytes including emoji and non-Latin scripts without producing `<unk>`.
- Special tokens that show up in prompts:
  - `<|im_start|>` / `<|im_end|>` — chat turn boundaries.
  - `<|im_sep|>` — speaker/content separator.
  - `<|endoftext|>` — sequence terminator.
  - `<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>` — FIM (see above).
  - Various role tokens (`<|user|>`, `<|assistant|>`, `<|system|>`).

The Hugging Face `transformers` AutoTokenizer handles all of this transparently — you rarely need to touch tokens directly except when writing the FIM data formatter.

## What "base" vs "Instruct" looks like in practice

Same model architecture, different training mix:

- **Base** (`Qwen/Qwen2.5-Coder-7B`) — pure language modelling on a large code+text corpus. If you prompt it with "What does this function do?" it might respond with another question, repeat the function, or output something irrelevant — it doesn't *understand* "function-do" as a request structure.
- **Instruct** (`Qwen/Qwen2.5-Coder-7B-Instruct`) — same model plus an SFT (supervised fine-tuning) pass on instruction-formatted data. Responds to "What does this function do?" with a coherent answer.

We use Instruct. Our fine-tune is layered on top, retaining its instruction-following while adding codebase-specific knowledge.

## How to obtain the model

```bash
# Via the modern hf CLI (preferred):
hf download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir ~/.hf-models/qwen-coder-7b

# Or via the older syntax:
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
    --local-dir ~/.hf-models/qwen-coder-7b
```

This downloads:
- `*.safetensors` (the bf16 weights, sharded across multiple files — total ~15 GB).
- `tokenizer.json`, `tokenizer_config.json` (the tokenizer).
- `config.json` (model architecture metadata).
- `generation_config.json` (default generation params).

You need `~/.hf-models/qwen-coder-7b` (or the default `~/.cache/huggingface/hub/...`) on disk *before* the train pipeline tries to load it, otherwise QLoRA's `from_pretrained` will hit HuggingFace Hub at training time — possibly triggering the historic re-download bug if anything is misconfigured (see [`02-fine-tuning.md § Export`](02-fine-tuning.md#export-to-gguf-the-historic-landmine)).

## Hugging Face auth (if the model is gated)

Qwen 2.5 Coder 7B Instruct is *not* gated as of 2026-05 — you can download it without authentication. Some models are gated (Llama 3 family, certain Mistral variants); for those you need:

```bash
hf auth login
# enter your HF access token (created at https://huggingface.co/settings/tokens)
```

Store the token at `~/.cache/huggingface/token` (mode 600). Never commit it.

## Further reading

- Qwen 2.5 Coder paper: https://arxiv.org/abs/2409.12186
- Model card on HuggingFace: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- Qwen GitHub: https://github.com/QwenLM/Qwen2.5-Coder
- Comparison benchmarks (HumanEval, MBPP, etc.): https://evalplus.github.io/leaderboard.html
- The bigcode-evaluation-harness repo has reproducible benchmark scripts: https://github.com/bigcode-project/bigcode-evaluation-harness
