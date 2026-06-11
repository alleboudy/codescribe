# CONCEPTS — what the moving parts are and why they're stacked this way

This doc is the conceptual entry point for `codescribe-train`. It answers:

- What problem are we actually solving?
- What is fine-tuning? What is RAG? What is MCP? What is an embedding? What is chunking?
- What are the trade-offs of each — when does each technique earn its keep, and when is it the wrong tool?
- How does `codescribe-train` apply each one specifically to the `../your-repo` target repo?

It's deliberately a **landing page**. The math, the hyperparameters, the per-platform setup, the operator runbooks — those live in the detail docs linked at the end of each section. Read this once to get the shape, then jump into the detail docs when you need to *do* something.

> Audience: someone new to this repo (or someone returning after months and needing to swap their cache back in). Glossary lives in [`GLOSSARY.md`](GLOSSARY.md); architecture diagram in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 0. The mission in one paragraph

`codescribe-train` is a **strictly-local, intern-level coding agent for one specific git repo (`../your-repo`)**. "Strictly local" means no cloud GPU, no remote inference, no telemetry, no weights ever leaving the box — see [`STRICTLY-LOCAL-POSTURE.md`](STRICTLY-LOCAL-POSTURE.md). "Intern-level" means we accept that a fine-tuned 7B local model isn't going to outperform frontier cloud models on hard problems; the value is **consistent project idioms + low-friction tool access on the developer's own machine**, not raw capability. See [`docs/STRICTLY-LOCAL-POSTURE.md`](../docs/STRICTLY-LOCAL-POSTURE.md) for the capability-ceiling rationale.

To get there, the project stacks **three complementary techniques** on top of a base model (Qwen 2.5 Coder 7B Instruct). Each one fixes a problem the others can't:

| Technique | Fixes what's wrong with… | Lives in this repo as… |
|---|---|---|
| **Fine-tuning** (LoRA / QLoRA / Unsloth) | The base model's generic *style* — it writes "average internet" code, not sample-shaped code | `codescribe_train/data/` + `codescribe_train/train/` |
| **RAG** (embeddings + chunking + hybrid retrieval) | Fine-tuning's *staleness* — weights don't update when someone commits | `codescribe_train/rag/` + `codescribe_train/servers/rag_server/` |
| **MCP** (Model Context Protocol — tools-as-stdio-RPC) | RAG's *one-shot* nature — what if the agent needs to actually *run* something, like grep or pytest? | `codescribe_train/servers/` + `vendor/claw-code/` (harness) |

The rest of this doc walks each layer in turn, then shows how they compose at runtime for `../your-repo`.

---

## 1. Fine-tuning — baking project idioms into the weights

### What it is

The base Qwen 2.5 Coder 7B was trained on trillions of tokens of mostly-public code. It writes plausible code in any popular language, but it doesn't know the conventions, library choices, or naming patterns of any specific repo. **Fine-tuning** continues training on a small, project-specific dataset, nudging the weights toward the target repo's style.

We don't full-fine-tune. We use a stack of three optimisations, each layered on the previous one to shrink the resource footprint without losing quality:

| Layer | What it does | Why we need it |
|---|---|---|
| **LoRA** | Freezes the 7B base; trains small `(rank=16)` matrices alongside the attention + MLP projections. **40 M trainable params vs 7.6 B (0.53 %)** | Full fine-tune would take ~60 GB of optimiser state — we have 8 GB |
| **QLoRA** | Keeps the frozen base in **4-bit NF4** during training; LoRA adapters stay in bf16 | Drops base memory from 15 GB (bf16) to ~4.5 GB (4-bit); brings the full run to ~7.6 GB peak VRAM, fits in 8 GB |
| **Unsloth** | Replaces stock LoRA forward/backward with fused Triton kernels + padding-free batching + 4-bit base loading | ~1.5–2× faster training; mandatory at this memory budget |

The math, the exact hyperparameter choices, and the rationale for each one are in [`QLoRA-AND-UNSLOTH.md`](QLoRA-AND-UNSLOTH.md). What matters for this overview is that all three are **necessary** to run a 7B QLoRA fine-tune on an 8 GB consumer GPU at all — dropping any of them puts you back into "needs an A100" territory.

### Pros

- **Internalises project style.** After 3 epochs, the model writes code that *feels* like `../your-repo`'s code without you having to prompt it with examples. Naming, async patterns, custom helper choices — it absorbs them.
- **Zero runtime cost.** The fine-tune happens once. Serving the merged GGUF on `llama-server` is exactly as fast as serving the base.
- **Works offline.** No tool calls, no network round-trips at inference — just `llama.cpp` running on the local GPU.

### Cons

- **Stale.** The weights are a snapshot. Yesterday's commit isn't in them. Today's PR isn't in them. Re-training takes hours.
- **Confabulates facts.** Ask "what columns does the `orders` table have?" and the model will produce a plausible-looking list that may or may not match reality. Weights are for *idiom*, not *lookup*.
- **Capability ceiling.** A 7B model is ~7B intelligent. Fine-tuning doesn't make it smarter; it makes it more on-brand.
- **Catastrophic forgetting risk.** Push too hard (too many epochs, too high learning rate, too narrow data) and the model forgets how to code outside the project. We deliberately keep the dataset diverse (three formatters — document, FIM, diff_instr — see [`QLoRA-AND-UNSLOTH.md §5`](QLoRA-AND-UNSLOTH.md#5-why-three-formatters)) to avoid this.

### How codescribe-train applies fine-tuning to ../your-repo

The data pipeline in [`codescribe_train/data/`](../codescribe_train/data/README.md) walks `../your-repo` via `git ls-files`, applies a YAML-driven file filter, dedupes by SHA-256, and splits 90/5/5 train/val/test **by file path** (no chunks of the same file leak across splits). The survivors go through three orthogonal **formatters**:

| Formatter | Output shape | What it teaches |
|---|---|---|
| `document` | Full file, wrapped with `<\|repo_name\|>` / `<\|file_sep\|>` | Repo-wide patterns: imports, top-level naming, file structure |
| `fim` | `<\|fim_prefix\|>{prefix}<\|fim_suffix\|>{suffix}<\|fim_middle\|>{middle}` (random 1–8-line span masked) | Local completion — "given this surrounding context, fill the gap" |
| `diff_instr` | ChatML pair: commit subject+body → unified diff | Intent → change — the kind of edit the agent actually has to make |

The training run lives in [`codescribe_train/train/`](../codescribe_train/train/README.md): QLoRA at `r=16`, `alpha=32`, `seq_len=1024` (chosen so we fit in 8 GB VRAM — see [`HARDWARE-AND-PERFORMANCE.md`](HARDWARE-AND-PERFORMANCE.md) for the seq_len=2048 → 1024 decision), three epochs, `paged_adamw_8bit` optimiser, `bf16` mixed precision. Output goes through `python -m codescribe_train.train export` which merges the LoRA adapter into a single fp16 model and quantises to Q4_K_M GGUF (~4.6 GB). That's `checkpoints/sample-qwen7b-q4_k_m.gguf` — the artefact served by `llama-server` and rsync'd to every other host you port to (see [`PORTING.md`](PORTING.md)).

**Detail docs:** [`QLoRA-AND-UNSLOTH.md`](QLoRA-AND-UNSLOTH.md) for the math + hyperparameter rationale, [`FLASH-ATTENTION.md`](FLASH-ATTENTION.md) for the FA2 wheel saga, [`HARDWARE-AND-PERFORMANCE.md`](HARDWARE-AND-PERFORMANCE.md) for measured numbers.

---

## 2. RAG — retrieving project state at inference time

### What it is

Fine-tuning bakes style into weights. **RAG (Retrieval-Augmented Generation)** does the opposite: keeps the weights fluent-and-frozen, but at inference time fetches the *specific facts the current question needs* and pastes them into the prompt as context. The model reads them as plain text and integrates them into its answer.

The flow:

```
user question ──► EMBED query into a vector
                   ──► RETRIEVE top-k similar chunks from a vector store
                        ──► AUGMENT the prompt with the retrieved chunks
                             ──► GENERATE the answer
```

The two building blocks worth understanding in isolation: **embeddings** and **chunking**.

### Embeddings — fixed-length vectors that capture meaning

An **embedding model** is a small neural net (10 M – 1 B params, ~50–2000 MB on disk) that maps a piece of text to a fixed-length vector (typically 384–1024 dimensions). The geometry is designed so that **semantically similar texts end up close** in vector space (cosine similarity → ~1.0), while unrelated texts are far apart.

Key properties:

- **Bi-encoder.** The same model embeds queries and documents into the *same* space. You can pre-compute the document embeddings once at index time and only embed the query at retrieval time — the expensive part is amortised.
- **Off-the-shelf usually wins.** Unlike the main model, you generally don't fine-tune the embedder. Models like `BAAI/bge-large-en-v1.5` (335 M params, ~1.3 GB) are battle-tested on English text + code and work well as-is.
- **Cheap to run.** CPU-only is fine for many use cases; on a GPU it's sub-second for a query.

`codescribe-train` uses **`BAAI/bge-large-en-v1.5`** (1024-dim, MIT licence), chosen over the smaller `bge-small` because code retrieval benefits from the extra capacity. Embedder lives in `codescribe_train/rag/embed/embedder.py`; the model snapshot is mirrored across hosts via the porter (it's part of the ~8.5 GB transfer documented in [`PORTING.md §3`](PORTING.md#3-what-every-porter-transfers)).

### Chunking — splitting source material so retrieval can land on the right thing

A 5000-line Python file is too big to retrieve as a single chunk: the embedding loses specificity, and you'd waste context-window budget pasting the whole file into the prompt. So before indexing, every document is **split into chunks** (typically 256–1024 tokens each).

How you chunk matters a lot. Bad chunking splits in the middle of a function definition; good chunking respects semantic boundaries (paragraphs, function defs, diff hunks):

| Strategy | Good for | How `codescribe-train` uses it |
|---|---|---|
| **Recursive split** (paragraph → sentence → token) | Generic prose, README files | Default fallback in `codescribe_train/rag/embed/chunker.py` |
| **PR-diff aware** (per-file, per-hunk, never break hunk headers) | GitHub PR diffs from `example-org/sample` | The bootstrap indexer's primary chunker for PRs |
| **Commit-message + subject** | Git commit history | Each commit is its own chunk; the message *is* the searchable content |

Note: the training-time formatters (document/FIM/diff_instr) in `codescribe_train/data/` are also a form of chunking, but for a different purpose — they shape what the model *learns* during fine-tuning, not what the retriever fetches at inference time. Same word, different stage of the pipeline.

### Hybrid retrieval — dense + sparse + RRF fusion

Pure dense (embedding-based) retrieval is great for semantic similarity ("find issues about CSV import field parsing") but weaker on exact-term matches ("find the issue numbered #373"). Pure sparse retrieval (BM25 / FTS5 keyword search) is the opposite. `codescribe-train` uses **both, fused with Reciprocal Rank Fusion (RRF)**:

```
dense retriever:  bge-large(q) → top-K by cosine sim   ──┐
                                                          ├──► RRF fusion ──► final top-k
sparse retriever: BM25 / FTS5 over chunk text    ──┘
```

RRF gives each candidate a score of `1 / (rank + 60)` from each retriever, sums them, sorts. Robust to either retriever being wrong on a given query. The implementation is in `codescribe_train/rag/retrieve/`.

### Pros (of RAG over fine-tuning for the same information)

- **Fresh.** The rag store re-indexes on a nightly cron; yesterday's PR shows up in tomorrow's tool call.
- **Sourced.** Retrieved chunks come with provenance (file path, line range, commit SHA, PR number). The agent's answer is grounded in something you can look at.
- **Cheap to update.** Re-indexing a few thousand commits + a few hundred issues takes minutes; re-training the LoRA takes hours.
- **No catastrophic forgetting.** The model's weights never change.

### Cons

- **Latency.** Embedding the query + retrieving + augmenting the prompt adds ~50–500 ms per turn. Not free.
- **Context-window pressure.** Retrieved chunks compete with the user's actual prompt for the LLM's context window. Bad retrieval that pastes irrelevant text wastes budget.
- **Retrieval can fail.** If your chunking is bad, your embedder is wrong for the domain, or your query is ambiguous, the retriever returns nothing useful and the model answers from weights anyway — possibly worse than if you hadn't tried.
- **No style transfer.** RAG can tell the model what files / commits exist, but it can't make the model *write code that fits the project's idioms*. That's fine-tuning's job.

### How codescribe-train applies RAG to ../your-repo

The RAG sidecar in [`codescribe_train/rag/`](../codescribe_train/rag/README.md) indexes:

| Source | Indexer module | Chunks per source | What ends up in the vector store |
|---|---|---|---|
| `../your-repo` commits (git log) | `codescribe_train/rag/sources/git_source.py` | One per commit | Subject + body + diff stat, with `sha`, `author`, `authored_at` metadata |
| `example-org/sample` GitHub issues | `codescribe_train/rag/sources/github_source.py` | One per issue | Title + body, with state, labels, linked PRs metadata |
| `example-org/sample` GitHub PRs | `codescribe_train/rag/sources/github_source.py` | Header chunk + per-hunk chunks of the unified diff | Each chunk carries PR number, file path, hunk header for grounding |
| `../your-repo` working-tree docs (README, Makefile, `docs/**/*.md`, `docker-compose*.yml`, `Dockerfile*`) | `codescribe_train/rag/sources/worktree_docs_source.py` | Markdown split by heading; other files whole | Each chunk carries `doc_path` + nearest heading — answers "how do I run/build this?" |

The store is **`sqlite-vec` + FTS5** in a single file at `indices/rag.db` (tens of MB after a full index — on the order of a few hundred issues, a hundred-or-so PRs, a few thousand commits, and the issue→PR links between them). The single-file design is deliberate: it's the simplest thing that rsyncs cleanly to a new host via the porter.

Four MCP **tools** are exposed (the connection point with MCP, the next section):

| Tool | What it returns |
|---|---|
| `find_similar_issues(query, k=5, min_confidence=0.8)` | Top-k issues by RRF score, each with its highest-confidence linked fix PR |
| `get_pr_diff(pr_number, max_chars=8000)` | The unified diff for a specific PR (truncated deterministically) |
| `search_commits(query, k=5)` | Top-k commits by RRF score — commits without a linked PR are first-class results |
| `search_docs(query, k=5)` | Top-k working-tree doc chunks by RRF score, each with its `doc_path` + nearest heading — the "how do I run/build this?" path |

**Detail docs:** [`RAG.md`](RAG.md) for the conceptual background + decision tree ("when RAG vs fine-tuning vs tools") and the per-source index design rationale, [`ARCHITECTURE.md`](ARCHITECTURE.md) for how the sidecar composes with the rest of the stack.

---

## 3. MCP — letting the agent actually do things

### What it is

RAG is a one-shot dance: the agent gets retrieved chunks, generates an answer, done. But what if it needs to *run* something — `pytest`, `grep`, `git blame`, a database query? Encoding all of those as RAG indices is wasteful (they're functions of the *current* repo state) and impossible for some of them (`pytest` needs to actually execute).

**MCP (Model Context Protocol)**, open-sourced by Anthropic in 2024, is a small JSON-RPC protocol that standardises how an LLM agent talks to **tool servers**. The agent (the *client*; in our case, the vendored `claw-code` harness) connects to one or many MCP servers, discovers their capabilities at runtime, and emits `tools/call` requests when the model wants to invoke a tool.

Think of MCP as **the LSP of agent tools** — same shape, different domain. LSP standardised editor↔language-server traffic so any editor could mix with any toolchain. MCP standardises agent↔tool-server traffic so any harness can mix with any tool provider.

### Wire format — JSON-RPC over stdio

The server is a subprocess. Requests come in on stdin, responses go to stdout, logs go to stderr. The four methods every server must implement:

| Method | Purpose |
|---|---|
| `initialize` | Handshake: protocol version + capabilities |
| `tools/list` | Return the tool names + JSON-Schema input schemas |
| `tools/call` | Execute a tool with the given args; return content |
| `resources/list` + `resources/read` | Optional: expose read-only typed data |

There is one **subtlety we learned the hard way**: MCP's stdio framing has two incompatible variants in the wild — newline-delimited JSON (used by the official Python SDK 1.27.x) and LSP-style `Content-Length` framing (used by `claw-code`). Wiring a Python MCP server directly into claw produces JSON parse errors and timed-out tool calls. The fix is a small bridge shim ([`codescribe_train/servers/_mcp_framing_bridge.py`](../codescribe_train/servers/_mcp_framing_bridge.py)) that translates between them; the porters bake it into every `.claw/settings.json` they generate. Full diagnosis in [`docs/known-issues.md`](known-issues.md#mcp-stdio-framing-mismatch-claw--python-mcp-sdk--fixed).

### Pros

- **Composable.** A new tool is a new server. The harness doesn't change; the model's tool list grows.
- **Language-agnostic.** Server is any process speaking JSON-RPC. Python, Rust, TypeScript, Go SDKs exist.
- **Local-respecting.** stdio is the default transport — no port to expose, no auth to handle, no cloud round-trip. Fits this project's strictly-local posture cleanly.
- **Sandbox-able.** Because servers are subprocesses, you can wrap the harness in `docker run --network none` and trust the kernel to enforce the boundary (see [`STRICTLY-LOCAL-POSTURE.md §4`](STRICTLY-LOCAL-POSTURE.md)).

### Cons

- **Stdio framing gotcha** (above). Pick a side; bridge if you need to.
- **The model has to know when to call which tool.** That's a property of its training and the prompt; MCP itself doesn't help here. If the model doesn't realise `find_similar_issues` exists or when to use it, the tool is dead weight.
- **Per-call latency.** Spawn cost + RPC round-trip. Acceptable for chat-paced tools, less so for tight loops.
- **Cross-server state is the agent's problem.** No transactions across servers; if your "run tests then read coverage" workflow needs them, you build it on the client side.

### How codescribe-train applies MCP to ../your-repo

Three MCP servers ship today, all stdio:

| Server | Tools | Backing implementation |
|---|---|---|
| `repo-rag` | `find_similar_issues`, `get_pr_diff`, `search_commits`, `search_docs` | `codescribe_train/servers/rag_server/` — reads `indices/rag.db` (the RAG sidecar from §2) |
| `repo-grep` | `grep(pattern, path, language, max_results)` | `codescribe_train/servers/repo_grep/` — `ripgrep` scoped to `../your-repo`'s working tree |
| `repo-docs` | `list_docs(subdir, pattern)`, `read_doc(path, max_chars)` | `codescribe_train/servers/repo_docs/` — direct file access to README, Makefile, `docs/**/*.md`, `docker-compose*.yml`, Dockerfiles, root-level shell scripts; sandboxed to `TARGET_REPO` |

The three are complementary: **rag** for semantic "find me something like X" against the corpus snapshot, **grep** for "find me lines matching this regex" against the live working tree, **docs** for "give me this exact file" — the path you actually want when the user asks the model to run `make up` or `docker compose up` and the model needs to read the Makefile/compose file end-to-end before invoking it through the harness's Bash tool.

The registry that tells `claw-code` where these servers live is `.claw/settings.json` (per-host, generated by the porters). It wraps every server through `_mcp_framing_bridge` so the LSP↔newline mismatch above doesn't bite. See [`CLAW-MCP.md`](CLAW-MCP.md) for the schema, [`MCP-SERVERS.md`](MCP-SERVERS.md) for the protocol background, and `tests/servers/test_mcp_framing_bridge.py` for the regression coverage.

**Detail docs:** [`MCP-SERVERS.md`](MCP-SERVERS.md) (protocol + Python SDK + a canonical server skeleton), [`CLAW-MCP.md`](CLAW-MCP.md) (`.claw/settings.json` MCP registry schema), [`known-issues.md`](known-issues.md) (the framing bug).

---

## 4. Putting it together — runtime composition for ../your-repo

A single chat turn against the local Mac/Linux deployment, with all three pillars active:

```
                    ┌─────────────────────────────────────────────────┐
                    │ 1. You type a question into claw                │
                    │    "Show me a similar past issue to: NPE on     │
                    │     order import"                               │
                    └────────────────┬────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────────────┐
                    │ 2. claw-code's TUI sends OpenAI Chat Completions │
                    │    to llama-server at 127.0.0.1:8080            │
                    └────────────────┬────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────────────┐
                    │ 3. The FINE-TUNED Qwen2.5-Coder-7B writes a     │
                    │    response.  Because it's been QLoRA'd on      │
                    │    sample, its phrasing + variable names + tool │
                    │    selection patterns are sample-shaped.        │
                    │    It emits a tools/call(find_similar_issues).  │
                    └────────────────┬────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────────────┐
                    │ 4. claw routes the call via MCP stdio (LSP frame)│
                    │    → _mcp_framing_bridge                        │
                    │    → rag_server stdin (newline JSON)            │
                    └────────────────┬────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────────────┐
                    │ 5. rag_server embeds the query with bge-large,  │
                    │    fetches top-k chunks from sqlite-vec + FTS5, │
                    │    fuses with RRF, returns issue/PR JSON.       │
                    └────────────────┬────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────────────┐
                    │ 6. The RAG result becomes part of the           │
                    │    conversation; the model writes a final       │
                    │    answer citing real issue numbers + PRs.      │
                    └─────────────────────────────────────────────────┘
```

What each pillar contributed:

- **Fine-tuning** (step 3) made the model's choice of *which* tool to call + *how* to phrase the question sample-shaped. Without it, you'd get a generic-feeling response that called the tool less reliably or asked for the wrong fields.
- **RAG** (step 5) gave the answer **specific, current, sourced facts** — real issue numbers, real PR diffs, with a watermark of when the index was last updated. Without it, the model would confabulate plausible-looking but fake issue numbers.
- **MCP** (step 4) made step 5 reachable from step 3 *as a structured tool* the model knows how to call. Without it, the rag store would just be a sqlite file on disk that nothing in the agent loop knows how to query.

Each pillar fixes a problem the others can't. Stripping any of them degrades a different axis of the experience: drop fine-tuning → generic-style code; drop RAG → confabulated facts; drop MCP → no way for the model to reach the index in the first place.

---

## 5. Further reading

By topic, ordered roughly novice → expert:

**Fine-tuning**
1. [`QLoRA-AND-UNSLOTH.md §1–4`](QLoRA-AND-UNSLOTH.md#1-the-problem-fine-tuning-solves) — LoRA + QLoRA math, why each layer exists
2. [`QLoRA-AND-UNSLOTH.md §5`](QLoRA-AND-UNSLOTH.md#5-why-three-formatters) — why three formatters
3. [`QLoRA-AND-UNSLOTH.md §6`](QLoRA-AND-UNSLOTH.md#6-hyperparameters-for-our-run) — hyperparameter rationale
4. [`FLASH-ATTENTION.md`](FLASH-ATTENTION.md) — the FA2 wheel saga + current deferred status
5. [`HARDWARE-AND-PERFORMANCE.md`](HARDWARE-AND-PERFORMANCE.md) — measured numbers + the seq_len decision

**RAG**
1. [`RAG.md §1–2`](RAG.md#1-the-problem-rag-solves) — the problem + the canonical flow
2. [`RAG.md §3`](RAG.md#3-embedding-models--how-to-pick) — embedder comparison + selection
3. [`RAG.md §6`](RAG.md#6-when-rag-vs-fine-tuning-vs-tools) — the RAG-vs-fine-tuning-vs-tools decision tree

**MCP**
1. [`MCP-SERVERS.md §1`](MCP-SERVERS.md#1-the-headline) — what MCP is + the LSP analogy
2. [`MCP-SERVERS.md §3`](MCP-SERVERS.md#3-the-server-side--what-you-build) — Python server skeleton
3. [`CLAW-MCP.md`](CLAW-MCP.md) — `.claw/settings.json` MCP registry schema (claw's strict-key parser)
4. [`known-issues.md` "MCP stdio framing mismatch"](known-issues.md#mcp-stdio-framing-mismatch-claw--python-mcp-sdk--fixed) — the framing bug + the bridge fix

**Architecture + posture (read first if you're brand-new)**
1. [`ARCHITECTURE.md`](ARCHITECTURE.md) — the four-phase composition with the diagram
2. [`STRICTLY-LOCAL-POSTURE.md`](STRICTLY-LOCAL-POSTURE.md) — why no cloud, no telemetry, no upload
3. [`GLOSSARY.md`](GLOSSARY.md) — jargon dictionary
4. [`PORTING.md`](PORTING.md) — moving the stack to another box you own (Mac / Linux / Windows)
