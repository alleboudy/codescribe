# 14 — Agent memory and "dreaming"

The fine-tune ([`02-fine-tuning.md`](02-fine-tuning.md)) and the RAG index ([`08-rag.md`](08-rag.md)) both have a blind spot: they are **statistical and stateless about the operator**. RAG answers "what *looks* like this query" via vectors and BM25. The fine-tune bakes house style into the weights but forgets every correction the moment a session ends. Neither can answer an *exact structural* question — "what *calls* `parse_config`", "what convention governs the `store` module", "what decision is *why* the schema looks like that" — and neither accrues anything from the operator fixing the same mislabel twice.

This doc adds a **symbolic memory** — a typed knowledge graph the assistant reads at query time and writes to over time — and a nightly **consolidation loop** (the "dream") that distills the codebase and the operator's own interactions into durable facts. It is the layer that makes the stack *neuro-symbolic in practice*: the model reaches for a structured query when precision matters and vector search when the question is fuzzy.

> Scope note: this is optional, sits *on top of* the RAG, and touches no weights until the gated retrain step (§7). Everything before that is reversible — delete the store and you are back to the prior system exactly.

## 1. Why symbolic memory, next to a vector index

The two are complementary the same way BM25 and embeddings are (see [`08-rag.md § BM25`](08-rag.md)):

| | Vector RAG (`rag.db`) | Symbolic memory (`memory.db`) |
|---|---|---|
| Question shape | fuzzy / semantic ("something like this") | exact / structural ("what calls X") |
| Storage | chunks + embeddings | typed entities + `(subject, predicate, object)` facts |
| Answer | ranked snippets | a fact with provenance + confidence |
| Failure mode | plausible-but-approximate | empty result (honest "I don't know that") |
| Freshness | re-index | re-extract (cheap; deterministic) |

The classic AI framing is Kahneman's **System 1 / System 2**: fast associative recall (the neural net, RAG) versus slow deliberate rule-following (the symbolic graph). A capable assistant wants both — intuition to *propose*, structure to *check*.

## 2. The store: a typed fact graph in one SQLite file

`memory.db` is a sibling of `rag.db` — plain SQLite, one file, `cp` to back up (same posture as the vector store, [`08-rag.md`](08-rag.md)). The atoms are the classic knowledge-graph triple, updated with three properties the 1980s expert systems lacked:

```sql
entities   (entity_id, kind, name, repo, path, meta, first_seen, last_seen)
           -- kind ∈ file | symbol | module | convention | decision | bug
facts      (fact_id, subject_id, predicate, object_id, object_lit,
            confidence, extractor, valid_from, valid_to, superseded_by, …)
           -- predicate ∈ calls | imports | defined-in | depends-on |
           --             convention-applies-to | decision-affects | supersedes
provenance (prov_id, fact_id, source_kind, source_ref, excerpt, …)
           -- source_ref = 'pkg/util.py:142' | '<commit-sha>' | 'docs/x.md' | 'session:<id>'
```

Every fact carries three things a bare triple does not:

- **Provenance** — the source line / commit / doc / interaction it came from. (This is the thing a Prolog proof gave you *for free*; here you store it by hand.)
- **Confidence** — `0.0–1.0`. Deterministic facts are `1.0`; model-guessed facts are lower. This is the old expert-system "certainty factor", reborn with better statistics.
- **Validity interval** (`valid_from` / `valid_to`) — supersession is **soft**: a conflicting fact sets `valid_to` on the old row and links `superseded_by`; nothing is ever `DELETE`d, so you can time-travel ("what did the graph believe last Tuesday") and audit every change.

## 3. Two streams, deliberately different trust levels

Facts enter the graph two ways, and the split is the whole point:

| | **Code facts** (the backbone) | **Interaction facts** (episodic) |
|---|---|---|
| Source | the source tree | the coding harness's session transcripts |
| Extraction | **deterministic** — AST / imports / defs / call graph | **LLM distillation** — the local served model reads sessions and proposes facts |
| Predicates | `calls`, `imports`, `defined-in`, `depends-on` | `convention-applies-to`, `decision-affects`, `supersedes`, `bug` |
| Confidence | `1.0` — exact, cheap, re-derivable | `< 1.0` — rises with corroboration |
| Character | the trustworthy floor | sparse early, richer with use |

The deterministic backbone means the store is **useful on day one** with zero interactions: the moment you extract the code facts, "what calls X" and "what imports Y" answer correctly with `file:line` provenance. The interaction stream is where the operator's corrections finally accrue — but because it is model-distilled, it is stored at lower confidence and is **always outranked by a deterministic fact in the same slot**.

### The Python call-graph caveat

The deterministic extractor uses the stdlib `ast` module (zero dependency — same no-tree-sitter stance as the chunker in [`08-rag.md`](08-rag.md)). Call edges are matched **by name**: a call to `self.parse()` records an edge to the name `self.parse`, which only chains to a defined symbol if the names line up. So transitive call reasoning is richest in Python and where names resolve cleanly; `imports` / `depends-on` (module-level) chain more reliably. The proof trace is always *exact* (it cites real `fact_id`s and lines) regardless — a compiled call graph is the tracked next improvement, not a silent hole.

## 4. The "dream" loop: offline consolidation

If the assistant can write to its own memory, the obvious next question is: can it use *idle time* to get better? The trainer GPU (see [`11-hardware.md`](11-hardware.md)) is idle ~95% of the time. A nightly job turns that free compute into a sharper graph by morning.

The lineage here is real, not hype — it is the same **reflection / consolidation** move from the agent-memory literature (Generative Agents, Reflexion), and structurally it is a batch RETE pass: "what new facts did today's commits produce, and what higher-level conclusions change?"

```
  1. INGEST      new commits/diffs since the last cursor
                 + new harness session transcripts
        │
  2. EXTRACT     deterministic → exact code facts (confidence 1.0, file:line)
                 LLM (the local served model) → fuzzy facts from sessions
                 every fact carries provenance + confidence + extractor id
        │
  3. REFLECT     dedup identical triples (merge provenance, raise confidence);
                 conflict-resolve (newer + better-provenanced wins → soft supersede);
                 prune facts whose source file vanished
        │
  4. SYNTHESISE  emit grounded training examples — each traces to a real fact
     (optional)  → real source. NO free-form self-talk. (§6)
        │
  5. RETRAIN     [periodic, gated — never nightly, never automatic] (§7)
```

Each run writes a lineage row (cursors + counts) so the whole history is auditable and bisectable. It runs on the trainer under a user-level timer; the freshly-dreamed `memory.db` ships to the serving box the same way `rag.db` does (build on the trainer, `rsync` to the always-on host — see [`11-hardware.md`](11-hardware.md) and [`17-runbooks.md § Two-box index lifecycle`](17-runbooks.md)).

## 5. Served-model integration: two read-only tools

The graph is exposed to the harness as MCP tools, beside `search_code` (see [`07-mcp.md`](07-mcp.md)). Both are read-only and return Markdown with clickable provenance:

- **`query_facts`** — structured graph query. "what calls X" → `{predicate:'calls', object:'X'}`; "what conventions apply to Y"; "what decisions affect Z". This is backward chaining against your graph — a Prolog query in modern clothes. An empty result returns a canonical "no facts — fall back to `search_code`" message, so the model (and the fine-tune) learn to treat it as a signal.
- **`explain_entity`** — a provenance-cited dossier for one entity: its outgoing facts (defined-in, calls, imports) and incoming facts (called-by, conventions that apply, decisions that affect it, known bugs). Pure graph traversal, no generation, fully attributable.

The model now *chooses*, turn by turn, between a structured query when precision matters and vector search when the question is fuzzy. That choice, made by the model itself, is neuro-symbolic behaviour at runtime — observable in the harness transcript.

## 6. Grounded synthesis (optional, and where the danger lives)

The dream can emit synthetic training examples for the periodic retrain — but this is the single most dangerous thing in the stack, so it is worth understanding *why* before turning it on.

**Model collapse** ("the curse of recursion", Shumailov et al., *Nature* 2024): train a model on its own generated output, recursively, and quality degrades — each generation under-samples the tails (rare facts, unusual phrasings) and over-represents the mode, so variance shrinks and the model forgets the rare-but-true. It is the brittleness failure of 1980s expert systems reappearing through a neural door.

The defenses are ordinary engineering discipline — "anchor to ground truth, keep a held-out check, don't ship a regression" — enforced by the pipeline, not by good intentions:

1. **Grounding.** Every synthetic example must trace to a real `fact_id → real source line/commit/doc`. No provenance → not emitted. Free-form self-talk is structurally impossible to produce.
2. **Real stays the majority.** Synthetic is a *capped minority* seasoning (a single hard ratio, e.g. `max_synth_ratio: 0.15`), diversity monitored per run.
3. **A frozen, real-only eval set** the synthetic pipeline can never touch — the only judge.
4. **Beat-or-discard.** Promote a retrained model only if it strictly beats the current one on that frozen eval; else discard and roll back.
5. **Lineage tracked** — which dream cycles fed which weights — so a regression can be bisected to the facts that caused it.

> A hard-won result: the *first* thing to fact-echo synthesis often teaches is *format*, not *capability* — a run that produces "Q: what calls X? A: Y calls X (`file:line`)" examples can **regress** structural recall because it over-fits a terse answer template. Measure it against the frozen eval before believing it helps. [`15-deductive-reasoning-and-imagination.md`](15-deductive-reasoning-and-imagination.md) is the "fundamentally different" synthesis (reasoning chains, not fact echo) that answers this.

## 7. Where retraining fits (and does not)

Stages 1–4 run nightly and touch **no weights**. Stage 5 (retrain) is **periodic, gated, and manual-to-schedule** — it reuses the existing eval-gated QLoRA path (train → frozen-eval gate → export → ship with a `.prev` rollback), and only runs once all of §6's defenses are present. The one non-negotiable: **production changes only for a model that strictly beats the current one on the frozen eval and serves healthy; otherwise discard / auto-rollback.** See [`17-runbooks.md § Eval re-gate`](17-runbooks.md) for how to run and — importantly — how *not* to fool yourself measuring it.

## 8. The through-line

The thing that killed 1980s expert systems was the **knowledge-acquisition bottleneck**: every fact had to be hand-elicited and hand-encoded. An LLM-driven memory system automates exactly that away — the knowledge engineer of 1985 is now a nightly batch job, and the graph it builds is provenance-tracked and confidence-scored rather than crystalline and brittle. Put a reasoner on top of the *same triples* ([`15-deductive-reasoning-and-imagination.md`](15-deductive-reasoning-and-imagination.md)) and the loop closes: the neural half does knowledge acquisition, the symbolic half does precise inference — each covering exactly where the other is weakest.

## Further reading

- **MemGPT / Letta** — treating the context window like OS-managed RAM, paging memory in and out: https://arxiv.org/abs/2310.08560
- **Generative Agents** (Park et al., 2023) — the memory stream + `recency × importance × relevance` retrieval + reflection: https://arxiv.org/abs/2304.03442
- **Reflexion** (Shinn et al., 2023) — agents that reflect on their own experience: https://arxiv.org/abs/2303.11366
- **"The Curse of Recursion"** (Shumailov et al., 2024) — model collapse, the reason grounded + eval-gated synthesis is non-negotiable: https://www.nature.com/articles/s41586-024-07566-y
- **STaR** (Zelikman et al., 2022) — bootstrap by keeping self-generated outputs that pass a check: https://arxiv.org/abs/2203.14465
- **RETE** (Forgy, 1982) — incremental cached evaluation over a changing fact base, the shape of the dream loop: https://www.csl.sri.com/users/mwfong/technical/rete-forgy82.pdf
