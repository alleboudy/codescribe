# RAG — Retrieval-Augmented Generation

What RAG is, when it earns its keep over (or alongside) fine-tuning, and how to build a local-only RAG stack that fits this project's strictly-local posture.

> **New to the project?** Start with [`CONCEPTS.md §2`](CONCEPTS.md#2-rag--retrieving-project-state-at-inference-time) for the 30,000-foot view of where RAG fits alongside fine-tuning and MCP, then come back here for the depth.

---

## 1. The problem RAG solves

Fine-tuning (what [`docs/QLoRA-AND-UNSLOTH.md`](QLoRA-AND-UNSLOTH.md) covers) bakes **style and idiom** into the model's weights. After a 3-epoch QLoRA on `../your-repo`, the model knows the project's naming conventions, library choices, async patterns, and SAEnum quirks. What it *doesn't* know:

- Yesterday's commit. (Training data is a snapshot.)
- The exact value of a config field as it exists right now.
- The current state of an issue or PR.
- A specific schema column's data type, where it actually lives in the DB.
- Which test files mention `on_time_delivery_rate` today.
- The output of a freshly run pytest.

Two reasons why baking these into weights is a bad idea:

1. **Stale.** Weights change only when you re-train. The repo changes whenever someone commits.
2. **Wrong tool.** Models don't reliably memorise factual lists. They confabulate plausible-looking values. For "what columns does the `orders` table have?" you want a **lookup**, not a guess.

**RAG separates "fluent style" (weights) from "current facts" (a query layer).** The weights stay fluent and project-shaped via fine-tuning. A retrieval pass at inference time fetches the specific facts the current question needs and pastes them into the prompt. The model reads them as plain context and integrates them into its answer.

---

## 2. The mechanics

A canonical RAG flow at inference time:

```
                 ┌────────────────────┐
   user query ──▶│   1. EMBED query   │
                 └────────┬───────────┘
                          │ vector q
                          ▼
                 ┌────────────────────┐    ┌──────────────────────────┐
                 │   2. RETRIEVE      │◀──▶│  vector store            │
                 │   top-k by cosine  │    │  (FAISS / sqlite-vec /   │
                 │   sim(q, doc_emb)  │    │   Chroma / etc.)         │
                 └────────┬───────────┘    └──────────────────────────┘
                          │ k retrieved chunks
                          ▼
                 ┌────────────────────┐
                 │   3. AUGMENT       │
                 │   prompt = system +│
                 │     [retrieved] +  │
                 │     user query     │
                 └────────┬───────────┘
                          │ augmented prompt
                          ▼
                 ┌────────────────────┐
                 │   4. GENERATE      │
                 │   (your LLM call)  │
                 └────────────────────┘
```

The corresponding "indexing time" flow runs once (or whenever the corpus changes):

```
   corpus  ──▶  CHUNK  ──▶  EMBED chunks  ──▶  STORE (chunk, vector, metadata)
```

Five components to design — each has a default that's hard to beat:

| Component | What it does | Sensible local default |
|---|---|---|
| **Loader** | Walk the corpus, yield raw text + metadata | `git ls-files` (you already have this) |
| **Chunker** | Split each source into ~256–1024-token chunks | Recursive split on paragraph → sentence → token |
| **Embedder** | Map text → fixed-length vector | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) (33 M params, fast, MIT) or `BAAI/bge-m3` (multilingual, ~600 M params) |
| **Vector store** | Store {vector, chunk, metadata}, support k-NN search | `sqlite-vec` for ≤1 M chunks; `FAISS` for ≥1 M with HNSW |
| **Retriever** | Glue: embed query, search store, return chunks | A 50-line wrapper around the above |

---

## 3. Embedding models — how to pick

A **bi-encoder embedder** maps both queries and documents into the same vector space, where cosine similarity correlates with semantic similarity. The model is small (10 M – 1 B params), fast, and runs CPU-only. You don't fine-tune it; off-the-shelf is usually fine.

| Model | Params | Strength | Footprint |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 33 M | Fast, good for English-only code/text | 130 MB |
| `BAAI/bge-base-en-v1.5` | 110 M | Better quality at modest cost | 440 MB |
| `BAAI/bge-large-en-v1.5` | 335 M | Top of the BGE-en family | 1.3 GB |
| `BAAI/bge-m3` | 568 M | Multilingual + dense + sparse + reranker in one | 2.3 GB |
| `sentence-transformers/all-MiniLM-L6-v2` | 22 M | Tiny, fast, classic baseline | 90 MB |
| `nomic-ai/nomic-embed-text-v1.5` | 137 M | Long context (8192), Apache-2 licence | 550 MB |

For a code-only corpus like `../your-repo` (mostly Python + TypeScript + Markdown), **`bge-small-en-v1.5` is the right starting point** — it's small enough to embed thousands of chunks in seconds on CPU, and it handles code-flavoured text well. Upgrade to `bge-base` if recall feels poor.

For a corpus with non-English content (German Slack messages, French commit messages) or where you want one model to do both English and non-English: `bge-m3`. It's bigger but multilingual, and as a bonus it produces both **dense** (cosine similarity) and **sparse** (BM25-like) representations from the same forward pass.

### Local install + use

```bash
.venv/bin/python -m pip install sentence-transformers  # installs torch dependencies if not already
```

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
emb = model.encode(["What columns does the orders table have?"], normalize_embeddings=True)
# emb.shape == (1, 384)
```

The first run downloads ~130 MB to `~/.cache/huggingface/`. After that, fully offline.

---

## 4. Vector stores — how to pick

For our scale (one mid-size repo's worth of chunks — say 5 000–50 000 entries), the choice is:

| Store | Pros | Cons | When |
|---|---|---|---|
| `sqlite-vec` extension | Single file, ACID, queryable with normal SQL, no separate process | Newer, fewer features than FAISS | **Default for ≤1 M chunks** |
| FAISS (in-memory + pickle) | Fast, mature, supports HNSW | No persistence layer, you handle it | Big indices, batch retrieval |
| FAISS + sidecar metadata DB | Same speed + queryable metadata | Two stores to keep in sync | When you need rich metadata filters |
| Chroma | Convenient, embedded mode, good docs | Heavier dep tree | Quickstart prototypes |
| Qdrant / Weaviate / Milvus | Production-grade, filtered search | Network service, overkill here | Multi-user / production scale |

For sample-scale corpora, **`sqlite-vec`** is the right call: it lives in a single `.db` file you can commit (or gitignore), it's queryable with `SELECT … MATCH …`, the vectors and the metadata live in the same DB so you can filter by file path / language / commit-date in a single query.

```python
import sqlite_vec, sqlite3
conn = sqlite3.connect("sample.rag.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)

# Schema:
conn.execute("""
CREATE VIRTUAL TABLE vec_chunks USING vec0(
    embedding FLOAT[384],
    chunk_id INTEGER PRIMARY KEY
)
""")
conn.execute("""
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,
    src_path TEXT,
    chunk TEXT,
    line_start INTEGER,
    line_end INTEGER
)
""")

# k-NN query:
cursor = conn.execute("""
SELECT chunks.src_path, chunks.chunk, distance
FROM vec_chunks JOIN chunks USING (chunk_id)
WHERE embedding MATCH ?
ORDER BY distance
LIMIT 5
""", (query_emb,))
```

That's the whole retriever, more or less.

---

## 5. The augmenter — prompt construction

The retrieved chunks need to be threaded into the prompt sensibly. A defensible pattern for our agent:

```python
SYSTEM = (
    "You are a coding assistant with access to the sample project. "
    "Below is excerpted context retrieved from the project's source. "
    "Use it to inform your answer; do not invent details not present in it."
)

def augment(query: str, retrieved: list[Chunk]) -> str:
    context = "\n\n".join(
        f"--- {c.src_path}:{c.line_start}-{c.line_end} ---\n{c.chunk}"
        for c in retrieved
    )
    return (
        f"<|im_start|>system\n{SYSTEM}\n\n"
        f"# Retrieved context\n\n{context}\n<|im_end|>\n"
        f"<|im_start|>user\n{query}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
```

Things to get right:
- **Citation hints.** Including the file:line header next to each chunk gives the model the option to cite (and helps the human reader trace claims).
- **Token budget.** Don't blindly stuff k=20 chunks in if your model has a 4K context. With Qwen 7B Instruct's 32K, you can afford k=5–10 of ~512-token chunks comfortably.
- **Order matters at inference time.** The model attends more to recently-seen context. Put the most relevant chunks last (just before the query) — i.e. reverse-sort by similarity.
- **No leading/trailing whitespace surprises.** Trim chunks; keep the tag-and-fence structure consistent so the model learns to recognise it.

For Qwen 2.5 Coder Instruct's chat template, `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` does the right thing — feed the messages list to it instead of hand-building the markers.

---

## 6. RAG vs fine-tuning — what to use when

| You want … | Best tool |
|---|---|
| The model writes in your project's style and idioms | **Fine-tune** |
| The model knows your specific function names and patterns | Mostly fine-tune; RAG fills gaps |
| The model answers questions about the *current* state of the repo | **RAG** |
| The model can quote exact lines from a file when asked | **RAG** (or a tool — see [MCP](MCP-SERVERS.md)) |
| The model can answer "what does function `score_delivery_reliability` do?" | RAG (looks it up); fine-tune can describe but might confabulate signature |
| The model can answer "list all migrations since v0.5" | **A tool, not RAG** ([MCP](MCP-SERVERS.md)) |
| The model can generate plausible new code in your style | **Fine-tune** |
| The model can plan multi-file changes touching specific functions | RAG to anchor the plan + fine-tune for style |

Short rule of thumb:

> **Fine-tune for *how* the model speaks. RAG for *what* it knows. Tools for *what it can do*.**

The three layers are complementary. Most production agents combine all three.

---

## 7. Performance and tradeoffs

### Recall vs precision

A k-NN retrieve always returns *some* k chunks. If the right answer isn't in the corpus, you get k irrelevant chunks. The model can be misled by bad context. Defences:

- **Distance threshold.** If the best similarity is < 0.5, return nothing and let the model say "I don't know" rather than read garbage.
- **Reranker.** A tiny cross-encoder model (`BAAI/bge-reranker-base`) re-scores the retrieved chunks against the query with attention between them. Slower per query but much better precision. Good when the corpus has many near-duplicates.
- **Hybrid retrieval.** Run BM25 (sparse, lexical) and dense (semantic) retrievers, take the union or RRF (reciprocal rank fusion). Catches both lexical-exact-match queries and semantic-near-miss queries.

### Latency budget

For an interactive agent, you want < 200 ms retrieval. Order-of-magnitude per-step on CPU:

| Step | Latency at corpus size 10 K chunks |
|---|---|
| Embed query (`bge-small`) | 5–10 ms |
| sqlite-vec k=10 search | 5–20 ms |
| Total | **< 30 ms typical** |

For 1 M chunks → consider FAISS + HNSW; sqlite-vec slows below ~100 ms at that scale.

### Embedding drift

If you bump the embedding model, **all old vectors become incompatible** — they live in a different space. Plan: tag the index with `embedder=<model-id> embedder_revision=<commit>` and refuse to mix. When you upgrade, re-embed the whole corpus.

### Stale indices

If the corpus changes (commit lands), your index is wrong until re-indexed. Two patterns:

1. **Periodic full rebuild** — fine for ≤100k chunks. Run `python -m sample_rag rebuild` weekly or post-commit-hook.
2. **Incremental updates** — track `(src_path, mtime)` per chunk; on rebuild only re-embed paths whose mtime changed.

For a project that changes a few times a day, periodic full rebuild is simpler and good enough.

---

## 8. Strictly-local stack

Everything above runs offline once the embedder is cached. To stay local:

- **No OpenAI/Cohere/Voyage embeddings.** Ever. They send your text to their server.
- **No Pinecone/Weaviate Cloud.** Embedded sqlite-vec or local FAISS only.
- **No external rerankers.** `BAAI/bge-reranker-base` runs locally.
- **First-run model download** comes from HF Hub. Acceptable; same envelope as the base model. After cache, fully offline.

`HF_HUB_OFFLINE=1` after the first run if you want belt-and-braces.

---

## 9. When NOT to use RAG

- **Tiny corpus (< 50 chunks).** Just include all of it in the system prompt. Retrieval overhead isn't worth it.
- **Highly structured queries** ("what columns are in `orders`?"). Use a tool that hits the live system. ⇒ [`docs/MCP-SERVERS.md`](MCP-SERVERS.md).
- **The fine-tuned model already knows it.** If your eval suite says the model nails a class of questions without RAG, don't slow down inference for that class.
- **Adversarial or low-signal corpus.** Embedding things like raw stack traces or autogenerated boilerplate produces noisy vectors that drag down recall.
- **Every sample needs to be in the retrieval-aware format.** Mixing raw fine-tune samples with RAG-augmented samples in the same training set teaches the model to expect retrieval markers when there are none. Either fine-tune the model to expect a known retrieval format consistently, or keep them separate.

---

## 10. Where to put RAG in this project

The pluggable abstractions in `harness/` and `backends/` don't have an obvious slot for RAG yet. Two integration patterns to choose between:

### Pattern A — RAG as an MCP server

The retriever exposes a `search_sample(query: str, k: int = 5)` tool over the [Model Context Protocol](MCP-SERVERS.md). The model decides when to call it. Pros: the model only retrieves when it actually needs to, no token bloat on simple queries. Cons: requires the model to be good at deciding.

### Pattern B — RAG as a prompt-augmenter at session start

The harness, before launching claw-code, retrieves k chunks based on the user's first prompt and prepends them to the system prompt. Pros: simple, no tool-calling required. Cons: retrieves on every turn or only at session start (both have downsides).

For sample specifically, **pattern A is the right call** — the user often asks questions where retrieval would be wasted (e.g., "what's wrong with this snippet I just wrote?"). Letting the model decide saves a lot of bandwidth.

---

## 11. Further reading

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al. 2020, the original RAG paper
- [BGE: One Embedder, Any Task](https://arxiv.org/abs/2402.03216) — the BGE family is the current best free option for English text/code
- [`sqlite-vec` docs](https://github.com/asg017/sqlite-vec) — the simplest-good vector store you can run
- [`sentence-transformers` docs](https://www.sbert.net/) — the canonical Python library for embeddings + rerankers
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — practical research on prompt-position effects (TL;DR: put the important context near the end)
- [HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496) — neat trick when queries and docs differ in style
- [Recursive splitting strategies](https://huggingface.co/learn/cookbook/advanced_rag) — HF's RAG cookbook
