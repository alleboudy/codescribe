# 08 — RAG: Retrieval-Augmented Generation

## What RAG is

**RAG** (Retrieval-Augmented Generation) is the pattern where, at inference time, you:

1. Take the user's prompt.
2. Search a local knowledge index for the most relevant snippets.
3. Inject those snippets into the model's context.
4. Let the model generate, now grounded in retrieved facts.

It's the cheap, fast alternative (and complement) to fine-tuning. Where fine-tuning bakes knowledge into the model's weights, RAG bolts a fact lookup onto the prompt at runtime.

```
User: "How did we handle NPEs on startup last time?"
   ↓
Retriever: search the index for "NPE on startup", get top-5 past bugs + their fixes
   ↓
Prompt assembler: append "## Similar bugs:\nBug 12345: ... CL 67890: ...\n..." to the system prompt
   ↓
Model: generates an answer informed by the retrieved bugs
```

## Why RAG is necessary even with a fine-tune

Fine-tuning ages. The moment you finish training, every new commit, every new bug, every new PR is invisible to the model. RAG can stay current — just keep re-indexing.

Other reasons RAG beats fine-tuning for specific cases:

- **Recent events** that postdate training.
- **High-precision recall**: a specific function name, an exact error message — easier to retrieve than to encode in weights.
- **Provenance**: the retrieved snippet is shown alongside the answer; the user knows *where* the model got the fact from.
- **Update cost**: re-indexing is minutes; re-fine-tuning is hours-to-days.

## When NOT to RAG

- **Simple questions the model already handles well**: retrieval adds noise and burns context window for zero benefit.
- **Highly synthesised reasoning**: if the answer requires inferring patterns *across* many documents, retrieval gives you piecemeal context the model has to stitch — usually worse than a fine-tune that has the pattern baked in.
- **Latency-sensitive use cases**: each retrieval adds ~100 ms; in a fast-typing IDE completion loop this matters.

Always **measure RAG lift** (see [issue #4 §15](https://github.com/alleboudy/codescribe/issues/4)) before turning it on by default. If lift is <5 percentage points on `task_mean`, leave RAG opt-in.

## The components of a RAG system

```
   raw data sources                     embeddings + indices                retrieval
   ──────────────────                   ─────────────────────              ──────────
   ┌──────────────┐
   │ Code (git/p4)│ ──┐
   └──────────────┘   │     ┌────────────────┐         ┌────────────────┐
                      ├───> │  Chunker +     │ ──────> │  Embedder      │
   ┌──────────────┐   │     │  filters +     │         │  (local model) │
   │ Issues       │ ──┤     │  dedup         │         └───────┬────────┘
   └──────────────┘   │     └────────────────┘                 │
                      │                                         ▼
   ┌──────────────┐   │                              ┌──────────────────┐
   │ PRs / Diffs  │ ──┘                              │  Vector store    │
   └──────────────┘                                  │  (sqlite-vec)    │
                                                     └──────────────────┘
                                                              ▲
                                                              │
                                              ┌───────────────┴───────────────┐
                                              │                               │
                                          retrieval                       BM25 sidecar
                                          (cosine search)                 (FTS5 / Tantivy)
                                              │                               │
                                              └─────────────┬─────────────────┘
                                                            ▼
                                                   ┌──────────────────┐
                                                   │  Hybrid merger   │
                                                   │  (RRF / weighted)│
                                                   └────────┬─────────┘
                                                            ▼
                                              top-K results, formatted for the LLM
```

Each component in detail below.

## Embeddings: vectors of meaning

An **embedding** is a fixed-size numeric vector representing the meaning of a piece of text. Two pieces of text with similar meanings produce vectors close together (small cosine distance). "The cat sat on the mat" and "a feline rested on the carpet" → embeddings ~0.85 cosine similarity. "The cat sat on the mat" and "JSON parsing in Rust" → ~0.15.

**Embedding dimensions**: usually 384, 768, or 1024 floats per vector. We use 1024 (BAAI/bge-large-en-v1.5).

**L2 normalisation**: we always normalise our vectors to unit length (`||v|| = 1`). For unit vectors, cosine similarity equals the inner product (`v1·v2`), which is the fast operation in vector databases. Forgetting to normalise gives subtly worse results.

### Embedding models we use (and alternatives)

| Model | Dim | Size | Quality (MTEB code/retrieval) | Speed (GPU batch) |
|---|---|---|---|---|
| **BAAI/bge-large-en-v1.5** | 1024 | 1.3 GB | ~64 (top tier for English) | ~400 chunks/s on RTX 5070 |
| BAAI/bge-base-en-v1.5 | 768 | 440 MB | ~63 | ~800 chunks/s |
| BAAI/bge-small-en-v1.5 | 384 | 130 MB | ~58 | ~1500 chunks/s |
| nomic-embed-text-v1.5 | 768 | 540 MB | ~62 | ~700 chunks/s |
| jinaai/jina-embeddings-v3 | 1024 | 1.1 GB | ~64 | ~500 chunks/s |
| mxbai-embed-large-v1 | 1024 | 1.3 GB | ~64 | ~400 chunks/s |

We default to **`bge-large-en-v1.5`** because:
- Top-tier quality (MTEB ~64).
- Apache 2.0 license.
- 1024-dim is a sweet spot — small enough for fast lookup, big enough to capture nuance.
- Mature; well-tested with sentence-transformers.

For code-heavy corpora, code-specific embedders (`microsoft/unixcoder-base`, `Salesforce/SFR-Embedding-Code`) are tempting — but in practice, modern general-purpose embedders score equally well on code retrieval and have better natural-language understanding for bug descriptions. We stick with `bge-large-en-v1.5`.

### Embedding-time vs query-time

The same model embeds both the documents (at index time) and the query (at retrieval time). **You cannot mix embedders.** If you swap from `bge-large-en-v1.5` to `nomic-embed-text-v1.5`, every vector in the store is invalidated. The store records the model SHA at init time; the embedder asserts the SHA at startup.

## Vector stores

A vector store holds your embeddings and supports fast "find the K nearest neighbours" queries.

### What "nearest neighbour" actually means

For unit vectors, "nearest" by cosine = "highest dot product" = "smallest Euclidean distance" (all equivalent). Naively, finding nearest among N vectors is O(N) — compute the dot product against each. For 100K vectors at 1024 dims that's 100M float multiplications, ~100 ms — slow but tolerable.

Real vector stores use approximate-nearest-neighbour (ANN) algorithms (HNSW, IVF, ScaNN) to bring this down to O(log N) or thereabouts. For 100K vectors, ANN finishes in ~5 ms.

### Choice we made: sqlite-vec

`sqlite-vec` is an extension to SQLite that adds vector-search capabilities via the `vec0` virtual table type. Why we use it:

| Property | sqlite-vec | qdrant | pinecone | chromadb | faiss |
|---|---|---|---|---|---|
| Local-only | ✅ | ✅ (Docker) | ❌ | ✅ | ✅ |
| Single file | ✅ | ❌ (daemon) | ❌ | ❌ | ❌ (in-mem) |
| Built-in BM25 | ✅ (FTS5) | ❌ | ❌ | ❌ | ❌ |
| Backup = `cp file` | ✅ | ❌ | ❌ | ❌ | ❌ |
| Metadata filtering | ✅ | ✅ | ✅ | ✅ | ❌ |
| Scaling beyond 1M docs | ⚠️ | ✅ | ✅ | ✅ | ✅ |

For our use case (≤500K documents on one host) sqlite-vec is the right choice. Everything lives in ONE `.db` file. Backup is `cp`. No daemon. No port management. The same DB also holds the BM25 sidecar (via FTS5), so hybrid retrieval is one process.

The only caveat: sqlite-vec's `vec0` virtual tables don't support JOINs with regular tables, which means we look up metadata in Python after retrieving rowids. Not a big deal in practice; documented in [issue #5 §4](https://github.com/alleboudy/codescribe/issues/5).

### Other options briefly

- **FAISS** (Facebook AI Similarity Search): the original C++ vector library. Fast but in-memory only; no native metadata; you need a sidecar for that. Right choice if you have millions of vectors and they all fit in RAM.
- **Qdrant**: production-grade vector DB; runs as a daemon (Docker). Right choice for multi-user services with rich filtering needs.
- **Pinecone / Weaviate Cloud**: managed services. Disqualified by strictly-local posture.
- **pgvector**: vector extension for Postgres. Right choice if you already have Postgres in the stack.
- **ChromaDB**: friendly Python wrapper. Quick to get started; heavier deps; less efficient than sqlite-vec at the scales we care about.

## BM25 and FTS5: the lexical baseline

Embeddings capture *semantic* similarity but can miss *exact* matches — specific function names, error strings, CL numbers, GUIDs. A pure-vector query for "NullPointerException in ConfigLoader.load" might miss a bug whose summary contains exactly those tokens but is semantically described differently in the description.

**BM25** is the textbook lexical retrieval algorithm. Given a query and a document, BM25 produces a score based on:
- Term frequency in the document (how often the query terms appear).
- Inverse document frequency (rare terms count more than common ones).
- Document length normalisation (so longer docs don't dominate just by having more words).

**FTS5** is SQLite's built-in full-text search module; it implements BM25 as the default ranker. Same database as our vector store; query syntax is a slightly extended boolean grammar.

Why we run BM25 alongside vector search: catches exact-string queries the embedder might miss, especially for proper nouns and specific identifiers. The complementarity is real — embeddings get the gist; BM25 gets the literal.

## Hybrid retrieval: combining embeddings + BM25

You run both vector and BM25 queries on the same query string, then merge the rankings. Several merge strategies:

### Reciprocal Rank Fusion (RRF)

Each ranker (vector, BM25) produces an ordered list. For each candidate document, the fused score is:

```
score(d) = sum over rankers of (1 / (k + rank_in_ranker(d)))
```

where `k = 60` is the standard textbook constant. A doc at rank 1 in the vector ranking and rank 5 in BM25 gets `1/61 + 1/65 ≈ 0.032`. A doc only in the BM25 ranking at rank 1 gets `1/61 ≈ 0.0164`. The combined ranking favours docs that appear in *both* rankings.

We use RRF. Tested; simple; no per-corpus tuning required.

### Weighted score combination

Alternatively, you can normalise both scores to [0, 1] and weight them: `score = w_vec * vec_score + w_bm25 * bm25_score`. Requires tuning weights per corpus. We don't use this.

### Reranking with a cross-encoder

A more elaborate scheme: retrieve a large candidate set (e.g., top-50 from each ranker = 100 candidates), then run them through a cross-encoder (a model that takes the query and the doc together and produces a relevance score). Higher quality; ~10× slower. Worth considering if RAG lift is plateauing.

## Chunking strategies

You can't embed an entire 10K-line source file as one vector — it loses fine-grained meaning AND exceeds the embedder's max token length (usually 512 for `bge-large`).

Common chunking strategies:

| Strategy | When to use |
|---|---|
| **Sliding window** (e.g., 512 tokens with 64-token overlap) | Default for prose; safe baseline |
| **Per-paragraph** | Markdown docs, comments |
| **Per-function / per-class** (parse the AST; embed each symbol separately) | Code retrieval; better recall for "find similar function" |
| **Per-hunk** (split unified diffs at `@@ ... @@`) | What we use for PR diffs in the RAG plan |
| **Semantic chunking** (split where embedding distance jumps) | Experimental; expensive |

For this stack, we use per-hunk for diffs and (title+description+first-N-comments) for bugs. Documented in [issue #5 §7](https://github.com/alleboudy/codescribe/issues/5).

## The pairing problem (the hard part of our RAG)

For a *bug-fix retrieval* RAG, the index of "bug bodies" is only half the value. The other half is "what was the actual fix?" — i.e., the CL/PR that resolved the bug.

Linking bugs to fixes is messy:

- Some bug comments cite a CL number ("Fixed in CL 12345").
- Some CL descriptions cite a bug ID ("Closes bug 4567").
- Some bugs and CLs have no explicit cross-reference, but the CL was submitted shortly after the bug closed by the bug's assignee.
- Some bugs reference multiple CLs (one initial attempt, a follow-up fix, etc.).

The RAG plan in [issue #4 §9](https://github.com/alleboudy/codescribe/issues/4) defines a confidence-scoring scheme that aggregates these signals. Default strict threshold: 0.8 — typically requires explicit cross-reference. Below that, retrieval might surface unrelated CLs.

For GitHub-based stacks the pairing problem mostly disappears — GitHub's GraphQL API exposes `closingIssuesReferences` directly. For Perforce + Bugzilla you need the heuristics.

## Recency weighting

A 5-year-old CL in a since-refactored area can mislead the model. Strategies to deprioritise stale results:

- **Hard cutoff**: drop everything older than N years.
- **Soft decay**: multiply the retrieval score by `exp(-age_years / 3)` so older docs are penalised but not removed.
- **Re-rank with a recency feature**: post-retrieve, sort by a weighted combination of score and recency.

We document this as an open question in [issue #4 §17](https://github.com/alleboudy/codescribe/issues/4). For v1, no recency weighting; measure first, tune second.

## Latency budget

Inside an interactive coding session, retrieval should be **<200 ms p99**, ideally <100 ms. Budget breakdown:

| Step | Time |
|---|---|
| Embed query (bge-large on GPU) | ~30–50 ms |
| Vector search (sqlite-vec, 100K docs) | ~10–30 ms |
| BM25 search (FTS5) | ~5–20 ms |
| RRF merge + metadata lookup | <5 ms |
| Format Markdown output | <5 ms |
| **Total** | **~50–110 ms** |

That leaves headroom for slower paths (cold-start embedder warm-up, larger corpora). If you're hitting >200 ms regularly, either your corpus is much bigger than expected OR the embedder is misconfigured (CPU instead of GPU, or no batching).

## Measuring RAG lift

Before turning RAG on by default, measure it. The lift measurement:

1. Pick a held-out task set (`evals/<repo>_tasks.json`).
2. For each task, run two completions:
   - **Baseline**: prompt only.
   - **With RAG**: prompt + retrieved top-3 bugs/fixes injected into the system prompt.
3. Score both via `expected_token_substrings` / `forbidden_token_substrings` (same scorer as the fine-tune eval).
4. Compute `lift = mean(with_rag_score) - mean(baseline_score)`.

Threshold for default-on: lift ≥ 0.05 (5 percentage points on a 0..1 scale). Below that, leave RAG opt-in via the harness's MCP config. See [issue #4 §15](https://github.com/alleboudy/codescribe/issues/4).

## Further reading

- The original RAG paper (Lewis et al., 2020): https://arxiv.org/abs/2005.11401
- MTEB (Massive Text Embedding Benchmark) leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- BGE model family card: https://huggingface.co/BAAI/bge-large-en-v1.5
- sqlite-vec docs: https://github.com/asg017/sqlite-vec
- SQLite FTS5 docs: https://www.sqlite.org/fts5.html
- Reciprocal Rank Fusion paper (Cormack et al., 2009): https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- BM25 (Robertson et al., 1994): https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf
- "Lost in the Middle" — context-position effects on RAG: https://arxiv.org/abs/2307.03172
