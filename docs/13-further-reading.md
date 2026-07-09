# 13 — Further reading

External links for each topic. Use these when the in-tree docs aren't deep enough.

## Foundational papers

- **"Attention Is All You Need"** (Vaswani et al., 2017) — the original transformer paper. The architecture every modern LLM is built on. https://arxiv.org/abs/1706.03762
- **"Language Models are Few-Shot Learners"** (Brown et al., 2020 — GPT-3 paper) — establishes the "scale + prompt" pattern. https://arxiv.org/abs/2005.14165
- **"Training Compute-Optimal Large Language Models"** (Hoffmann et al., 2022 — Chinchilla paper) — established the parameters-vs-tokens trade-off. https://arxiv.org/abs/2203.15556

## Fine-tuning

- **"LoRA: Low-Rank Adaptation of Large Language Models"** (Hu et al., 2021) — the LoRA paper. https://arxiv.org/abs/2106.09685
- **"QLoRA: Efficient Finetuning of Quantized LLMs"** (Dettmers et al., 2023) — the QLoRA paper. https://arxiv.org/abs/2305.14314
- **"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"** (Frantar et al., 2022) — the canonical 4-bit quantization paper. https://arxiv.org/abs/2210.17323
- **HuggingFace PEFT library docs**: https://huggingface.co/docs/peft
- **Unsloth documentation**: https://docs.unsloth.ai
- **bitsandbytes**: https://github.com/bitsandbytes-foundation/bitsandbytes

## Inference

- **llama.cpp** main repo: https://github.com/ggml-org/llama.cpp
- **GGUF format spec**: https://github.com/ggml-org/llama.cpp/blob/master/docs/gguf.md
- **`llama-server` README**: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- **vLLM** (alternative serving): https://github.com/vllm-project/vllm
- **Ollama** (alternative serving): https://github.com/ollama/ollama
- **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (Dao et al., 2022): https://arxiv.org/abs/2205.14135
- **"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"** (Dao, 2023): https://arxiv.org/abs/2307.08691

## Models

- **Qwen 2.5 Coder paper** (Hui et al., 2024): https://arxiv.org/abs/2409.12186
- **Qwen 2.5 Coder GitHub**: https://github.com/QwenLM/Qwen2.5-Coder
- **Qwen 2.5 Coder 7B Instruct model card**: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- **DeepSeek-Coder paper**: https://arxiv.org/abs/2401.14196
- **Code Llama**: https://ai.meta.com/blog/code-llama-large-language-model-coding/
- **StarCoder2 paper**: https://arxiv.org/abs/2402.19173
- **HumanEval benchmark**: https://github.com/openai/human-eval
- **MBPP benchmark**: https://github.com/google-research/google-research/tree/master/mbpp
- **EvalPlus leaderboard** (HumanEval+ and MBPP+, more rigorous): https://evalplus.github.io/leaderboard.html

## RAG

- **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (Lewis et al., 2020) — the original RAG paper. https://arxiv.org/abs/2005.11401
- **"Lost in the Middle: How Language Models Use Long Contexts"** (Liu et al., 2023): https://arxiv.org/abs/2307.03172
- **MTEB leaderboard** (embedding benchmark): https://huggingface.co/spaces/mteb/leaderboard
- **BAAI BGE model family**: https://huggingface.co/BAAI/bge-large-en-v1.5
- **sqlite-vec**: https://github.com/asg017/sqlite-vec
- **SQLite FTS5**: https://www.sqlite.org/fts5.html
- **BM25** original paper (Robertson et al., 1994): https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf
- **Reciprocal Rank Fusion** (Cormack et al., 2009): https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- **"Active Retrieval Augmented Generation"** (Jiang et al., 2023): https://arxiv.org/abs/2305.06983
- **HyDE (Hypothetical Document Embeddings)** (Gao et al., 2022): https://arxiv.org/abs/2212.10496

## MCP (Model Context Protocol)

- **Specification**: https://modelcontextprotocol.io/specification/
- **Python SDK**: https://github.com/modelcontextprotocol/python-sdk
- **TypeScript SDK**: https://github.com/modelcontextprotocol/typescript-sdk
- **MCP Inspector**: https://github.com/modelcontextprotocol/inspector
- **Community server list**: https://github.com/modelcontextprotocol/servers
- **Anthropic's launch announcement**: https://www.anthropic.com/news/model-context-protocol

## Neuro-symbolic, agent memory & reasoning

For [`14-agent-memory-and-dreaming.md`](14-agent-memory-and-dreaming.md) and [`15-deductive-reasoning-and-imagination.md`](15-deductive-reasoning-and-imagination.md).

- **MemGPT / Letta** — context window as OS-managed RAM, paged memory: https://arxiv.org/abs/2310.08560
- **Generative Agents** (Park et al., 2023) — memory stream + `recency × importance × relevance` + reflection: https://arxiv.org/abs/2304.03442
- **Reflexion** (Shinn et al., 2023) — agents reflecting on their own experience: https://arxiv.org/abs/2303.11366
- **"The Curse of Recursion"** (Shumailov et al., *Nature* 2024) — model collapse; why grounded + eval-gated synthesis is non-negotiable: https://www.nature.com/articles/s41586-024-07566-y
- **STaR** (Zelikman et al., 2022) — bootstrap by keeping self-generated outputs that pass a check: https://arxiv.org/abs/2203.14465
- **World Models** (Ha & Schmidhuber, 2018) — training inside imagined rollouts: https://worldmodels.github.io/
- **Dreamer v3** (Hafner et al., 2023) — control by latent imagination: https://arxiv.org/abs/2301.04104
- **AlphaGeometry** (Trinh et al., *Nature* 2024) — neural proposes, symbolic verifies: https://www.nature.com/articles/s41586-023-06747-5
- **Soufflé** — a fast Datalog engine that compiles to C++ (the reasoner "bolt-on"): https://souffle-lang.github.io/
- **RETE** (Forgy, 1982) — incremental cached evaluation over a changing fact base: https://www.csl.sri.com/users/mwfong/technical/rete-forgy82.pdf

## Harnesses

- **claw-code** (vendored in this stack): ultraworkers/claw-code on GitHub (private)
- **aider**: https://github.com/Aider-AI/aider
- **continue.dev**: https://docs.continue.dev/
- **Cursor**: https://cursor.com/
- **Claude Code**: https://docs.claude.com/en/docs/claude-code
- **GitHub Copilot Coding Agent**: https://docs.github.com/copilot/concepts/about-copilot-coding-agent
- **GitHub Copilot Chat**: https://docs.github.com/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide
- **OpenAI function calling**: https://platform.openai.com/docs/guides/function-calling
- **AGENTS.md convention**: https://agents.md/

## Perforce

- **Helix Core documentation**: https://www.perforce.com/manuals/p4guide/
- **`p4` command reference**: https://www.perforce.com/manuals/cmdref/
- **Streams concept**: https://www.perforce.com/manuals/p4guide/Content/P4Guide/chapter.streams.html

## Bugzilla

- **REST API documentation**: https://bugzilla.readthedocs.io/en/latest/api/index.html
- **General Bugzilla docs**: https://bugzilla.readthedocs.io/en/latest/
- **Mozilla Bugzilla** (the largest public install — useful for learning the workflow): https://bugzilla.mozilla.org/

## Hardware / CUDA

- **CUDA Toolkit downloads**: https://developer.nvidia.com/cuda-downloads
- **CUDA GPUs compute-capability matrix**: https://developer.nvidia.com/cuda-gpus
- **PyTorch installation matrix**: https://pytorch.org/get-started/locally/
- **Unsloth GPU compatibility**: https://docs.unsloth.ai/get-started/system-requirements
- **NVIDIA Tensor Core programming guide**: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html

## Distributed training

- **PyTorch DDP overview**: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- **PyTorch FSDP**: https://pytorch.org/docs/stable/fsdp.html
- **`torchrun` docs**: https://pytorch.org/docs/stable/elastic/run.html
- **NCCL documentation**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
- **HuggingFace `accelerate`**: https://huggingface.co/docs/accelerate

## Tooling

- **uv (Python package manager)**: https://docs.astral.sh/uv/
- **ruff (linter/formatter)**: https://docs.astral.sh/ruff/
- **pytest**: https://docs.pytest.org/
- **tenacity (retry library)**: https://tenacity.readthedocs.io/
- **httpx (HTTP client)**: https://www.python-httpx.org/

## Surveys + opinionated writing

- **"Building agentic systems"** (Lilian Weng, 2023) — foundational survey: https://lilianweng.github.io/posts/2023-06-23-agent/
- **"Agentic patterns"** (Chip Huyen, 2025): https://huyenchip.com/2025/01/07/agents.html
- **"State of GPT"** (Karpathy, 2023): https://www.youtube.com/watch?v=bZQun8Y4L2A
- **"How to train your ChatGPT"** (Karpathy, 2024): https://www.youtube.com/watch?v=zjkBMFhNj_g
- **"The bitter lesson"** (Sutton, 2019): http://www.incompleteideas.net/IncIdeas/BitterLesson.html

## Reproducing this stack from scratch

- The **Phase 1–5 sequence** documented in [issue #2](https://github.com/alleboudy/codescribe/issues/2) is the canonical bootstrap path.
- The **runbook in [issue #1](https://github.com/alleboudy/codescribe/issues/1)** is how an operator goes from "GGUF on disk" to "interactive coding session against the fine-tune".
- The **multi-laptop fleet in [issue #3](https://github.com/alleboudy/codescribe/issues/3)** is the scaling path.
- The **RAG plan in [issue #4](https://github.com/alleboudy/codescribe/issues/4)** is the inference-time augmentation.
- The **Python skeletons in [issue #5](https://github.com/alleboudy/codescribe/issues/5)** are the implementer's reference.
