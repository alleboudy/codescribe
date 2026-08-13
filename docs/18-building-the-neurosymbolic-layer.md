# 18 — Building the neuro-symbolic layer

[`14-agent-memory-and-dreaming.md`](14-agent-memory-and-dreaming.md) and [`15-deductive-reasoning-and-imagination.md`](15-deductive-reasoning-and-imagination.md) explain *what* the memory + reasoning layer is and *why*. This doc is the **build plan** — the phased, in-repo canonical plan (the same tier as the RAG plan in [`08-rag.md`](08-rag.md) / issue [#4](https://github.com/alleboudy/codescribe/issues/4)): the order to build in, the schema, the entry points, and a Definition of Done per phase.

The organising principle is **value early, risk deferred**: everything that touches weights comes last, behind everything that makes weight-changes safe. Seven phases, in two tracks that share one store.

```
  memory track      P1 substrate ──► P2 dream ──► P3 synthesis ──► P4 retrain (weights)
                        │               │             │                │
  reasoning track    R1 core ──────► R2 tools      R3/R4 imagination ──┘ (feeds P4)
                        └── both read the SAME triples in memory.db ──┘
```

## 0. Prerequisites and keystone decisions

This layer sits **on top of** the existing stack and reuses it wholesale: the code-fact extractor borrows the chunker's file walk ([`08-rag.md`](08-rag.md)); the LLM distillation and the synthesis paraphrase call the *local* served model ([`04-inference.md`](04-inference.md)); the retrain reuses the QLoRA → frozen-eval → export → ship-with-`.prev` path ([`02-fine-tuning.md`](02-fine-tuning.md)). Nothing here is a new external dependency — the store is plain SQLite, the reasoner is stdlib.

Two decisions gate everything downstream; make them explicitly before writing code:

1. **Representation: facts/graph now, a logic engine later.** Build the typed triple store first. A Datalog/Prolog engine can attach to the *same triples* later (R1 is the minimal pure-Python start; Soufflé is the deferred scale path). Do **not** build a full reasoner before the triples flow.
2. **Self-improvement: grounded + eval-gated retraining, or nothing.** Any weights touched are trained on the real corpus plus a *capped minority* of synthetic data where **every** synthetic example traces to real provenance, and a candidate is promoted **only if it strictly beats** the current model on a frozen real-data eval. This is not optional polish — it is the one thing standing between you and model collapse (see [`14 § 6`](14-agent-memory-and-dreaming.md)).

P1–P3 and R1–R2 are **reversible and weight-free**. P4 is the only weight-touching phase and runs only once the substrate *and* the anti-collapse gates exist.

## 1. The schema (`memory.db`)

Commit this as a checked-in `schema.sql`, applied on a fresh file (mirroring the RAG store's placement). Plain SQLite; `PRAGMA journal_mode=WAL` and `foreign_keys=ON` on every connection.

```sql
CREATE TABLE entities (
  entity_id  INTEGER PRIMARY KEY,
  kind       TEXT NOT NULL,        -- file | symbol | module | convention | decision | bug
  name       TEXT NOT NULL,        -- canonical / qualified name
  repo       TEXT, path TEXT, meta TEXT,   -- meta = JSON (lang, signature, docstring, …)
  first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
  UNIQUE(kind, name, repo)
);

CREATE TABLE facts (               -- the triples: (subject) --predicate--> (object)
  fact_id     INTEGER PRIMARY KEY,
  subject_id  INTEGER NOT NULL REFERENCES entities(entity_id),
  predicate   TEXT NOT NULL,       -- calls | imports | defined-in | depends-on |
                                   -- supersedes | convention-applies-to | decision-affects
  object_id   INTEGER REFERENCES entities(entity_id),
  object_lit  TEXT,                -- literal object (e.g. a version string); XOR with object_id
  confidence  REAL NOT NULL,       -- 0..1; deterministic facts = 1.0
  extractor   TEXT NOT NULL,       -- 'det:ast@1' | 'llm:model@genN'
  created_at  TEXT NOT NULL, valid_from TEXT NOT NULL,
  valid_to    TEXT,                -- NULL = current; set on supersede (SOFT delete, never DROP)
  superseded_by INTEGER REFERENCES facts(fact_id),
  UNIQUE(subject_id, predicate, object_id, object_lit, valid_from),
  CHECK ((object_id IS NULL) != (object_lit IS NULL))   -- exactly one object form
);
-- Query-serving indices. The UNIQUE above indexes by subject; the canonical
-- "what calls/affects/applies-to X" queries filter predicate+object, so:
CREATE INDEX idx_facts_pred_obj  ON facts (predicate, object_id, valid_to);
CREATE INDEX idx_facts_subj_pred ON facts (subject_id, predicate, valid_to);
CREATE INDEX idx_facts_valid_to  ON facts (valid_to);

CREATE TABLE provenance (          -- one-to-many evidence per fact; more sources → higher confidence
  prov_id INTEGER PRIMARY KEY, fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
  source_kind TEXT NOT NULL,       -- code_line | commit | doc | interaction
  source_ref  TEXT NOT NULL,       -- 'pkg/util.py:142' | '<sha>' | 'docs/x.md' | 'session:<id>'
  excerpt TEXT, created_at TEXT NOT NULL
);

CREATE TABLE episodes (            -- raw harness-session capture, pre-distillation
  episode_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, ts TEXT NOT NULL,
  kind TEXT NOT NULL,              -- prompt | tool_call | tool_result | error | user_correction
  payload TEXT NOT NULL, distilled INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE dream_runs   (run_id INTEGER PRIMARY KEY, started_at TEXT, finished_at TEXT,
  commit_cursor TEXT, episode_cursor INTEGER, facts_added INTEGER, facts_superseded INTEGER,
  facts_pruned INTEGER, synth_emitted INTEGER, notes TEXT);
CREATE TABLE model_lineage(gen_id INTEGER PRIMARY KEY, gguf_path TEXT, trained_at TEXT,
  dream_runs TEXT, real_examples INTEGER, synth_examples INTEGER, synth_ratio REAL,
  eval_score REAL, prev_gen INTEGER REFERENCES model_lineage(gen_id), promoted INTEGER NOT NULL);
```

Three properties are load-bearing and easy to get wrong:

- **Soft supersession.** Conflicts set `valid_to` + `superseded_by`; nothing is `DELETE`d. "Prune stale" means *archive* (set `valid_to`), never drop — so `as_of` time-travel and audit keep working.
- **The object XOR** is enforced at the DB (`CHECK`) *and* mirrored in the Python insert helper, so a both-null or both-set fact can never be written.
- **Provenance is many-to-one.** Two sources asserting the same triple are two `provenance` rows on one `fact`; consolidation raises that fact's `confidence`.

---

## P1 — Symbolic substrate *(no weights, reversible)*

The deterministic backbone: `memory.db` + code-fact extractors + the `query_facts` tool. Useful on day one, with zero LLM in the write path.

**Build.** A `store` module (context-managed SQLite handle, idempotent `insert_fact`/`insert_provenance`/`get_or_create_entity`). An `extract` module: Python via the stdlib `ast` (imports, defs, and a name-matched `calls` graph — see the caveat in [`14 § 3`](14-agent-memory-and-dreaming.md)); other languages via cheap line regexes for `imports`/`defined-in`/`depends-on`. A `build` CLI that walks the configured repos (**excluding the agent scratch dir** — [`16 § 4`](16-lessons-and-fixes.md)), extracts, and upserts conf-`1.0` facts with `file:line` provenance. A `query` read path + the `query_facts` MCP tool ([`07-mcp.md`](07-mcp.md)).

**Definition of Done**
- [ ] `schema.sql` committed; a fresh `memory.db` has all six tables + the XOR `CHECK` + the three indices.
- [ ] Extractor pass over the source tree produces `imports`/`defined-in`/`depends-on` (+ Python `calls`), all `confidence 1.0`, all with `file:line` provenance.
- [ ] `query_facts` registered on the retrieval server; "what calls X" / "what imports Y" return provenance-cited Markdown; an empty result returns a canonical "no facts → try `search_code`" message (asserted by a test).
- [ ] Parallel extraction has a `BrokenProcessPool` → serial fallback ([`16 § 14`](16-lessons-and-fixes.md)).
- [ ] Fully reversible: delete `memory.db` + de-register the tool → prior system exactly. No weights touched.

## P2 — Dream consolidation *(no weights, reversible)*

The nightly job: capture → extract → reflect. Interaction facts start accreting; `explain_entity` ships.

**Build.** An `episodes` parser (read the harness's on-disk session transcripts — no new hook needed) + idempotent ingest. A `distill` module with an **injectable client** (a stub for tests, a real one that calls the local served model) that proposes `convention-applies-to`/`decision-affects`/`bug` facts at `confidence < 1.0` with `session:` provenance — **rejecting any proposal with an empty subject** ([`16 § 14`](16-lessons-and-fixes.md)). A `consolidate` module: dedup identical triples (merge provenance, bump confidence), conflict-resolve single-valued slots (newer + better-provenanced wins → **soft** supersede; a deterministic fact is **never** retired by an LLM fact), and prune stale facts **scoped per-repo** ([`16 § 13`](16-lessons-and-fixes.md)). A `dream` orchestrator (stages 1–3) writing a `dream_runs` lineage row. A shell wrapper + user-level timer on the trainer.

**Definition of Done**
- [ ] Timer-driven `dream` runs stages 1–3; the first run writes a `dream_runs` row with non-zero `facts_added`.
- [ ] Episode capture live; distillation emits interaction facts with `session:` provenance and `confidence < 1.0`.
- [ ] Consolidation: dedup + provenance-merge raises confidence; conflicts soft-supersede; deterministic facts never retired by LLM facts; stale facts archived, not dropped.
- [ ] `explain_entity` registered; a dossier for a real symbol shows code facts *and* the conventions/decisions/bugs that touch it. No weights touched.

> Watch-outs proven in practice: a nightly re-run must **not** re-insert the whole backbone as new rows — key idempotency on the current triple, not on a fresh wall-clock `valid_from`. And an A→B→A flip within one run must leave the single-valued slot with a current row (reopen the soft-closed fact).

## P3 — Grounded synthesis *(no weights, reversible)*

The dream emits grounded training examples; the frozen eval gets built. Still no retrain — this phase exists to make P4 *safe*.

**Build.** A `synthesize` stage: for each current, provenance-backed fact, emit a Q&A whose answer *is* the fact + its source. Two generators (a deterministic template and an LLM paraphrase), one validator (`is_grounded`) that **rejects** any example whose load-bearing token was dropped or whose paraphrase introduced an out-of-vocabulary identifier — plus a benchmark-contamination guard that drops any example grounded in a frozen-eval file. A frozen, real-only, held-out eval set (asserted synthetic-free at build time). Diversity metrics (unique-bigram ratio, entropy) logged per run.

**Definition of Done**
- [ ] Every emitted example carries `fact_id` provenance to real source; a validator rejects any that can't (drop + count).
- [ ] Frozen real-data eval set built (held-out, synthetic-free) as an extension of the eval harness.
- [ ] Synthetic ratio cap + diversity floor computed and logged per run.
- [ ] The synthetic set is inspectable and discardable; no weights touched.

> Calibrate the diversity floor against *grounded* output, not a textbook constant: mandatory path-citations + dotted symbol names *are* the dominant repeated bigrams, so a naïve 0.6 floor is mathematically unreachable. Measure your real synth, then set the floor just under it, and let an **entropy** floor be the primary degeneracy guard.

## P4 — Eval-gated retrain *(touches weights — gated)*

Periodic QLoRA on real + minority grounded-synthetic, gated promote/rollback. Runs only with the full anti-collapse stack present.

**Build.** Assemble the training mix, enforcing `synth_ratio ≤ cap` (abort or deterministically down-sample above it). Train (reuse the QLoRA path). Score the candidate *and* the current model on the frozen eval in **separate subprocesses** (VRAM isolation on a small card). A `gate_decision`: promote only if the candidate **strictly beats** current (with a hard guard that a task-suite regression can never be masked by a perplexity gain). On promote: export the GGUF, ship to the serving box with a `.prev` written, health-check, auto-rollback on failure, and record a `model_lineage` row.

**Definition of Done**
- [ ] A run whose computed synth ratio exceeds the cap is **aborted before training**, not just logged.
- [ ] The gate promotes only on a strict win, else discards; a deliberately non-improving candidate is shown to auto-discard.
- [ ] Promote ⇒ export + ship + `.prev` + a `model_lineage` row linking the contributing `dream_runs`; rollback drill verified.
- [ ] The whole run is fail-closed: any crash leaves production untouched.

See [`17-runbooks.md § Runbook B`](17-runbooks.md) for how to *measure* the gate without fooling yourself — the eval-scoring traps in [`16 § 5–7`](16-lessons-and-fixes.md) are exactly the ones that make a gate lie.

---

## R1 — Deductive core *(no weights, reversible)*

A pure-Python, read-only, backward-chaining evaluator over the same triples. Rules are Datalog-shaped (see [`15 § 2`](15-deductive-reasoning-and-imagination.md)): `reaches`/`depends` (transitive), `trace_path` (shortest), `impacts` (reverse call-reachability). Every answer is a `Derivation` — the derived triple plus the ordered proof chain of real `fact_id`/`file:line` steps. Cycle-safe (visited set), depth-bounded, memoised per call; confidence = **min-of-chain**.

**Definition of Done**
- [ ] `reaches`/`impacts`/`trace_path` return complete proof traces; cycle-safe + depth-bounded (a cyclic call graph never hangs).
- [ ] Confidence = min-of-chain: a chain touching an LLM fact is visibly `< 1.0`; deterministic chains stay `1.0`.
- [ ] Unit tests over a hand-built graph (linear, diamond/dedup, cycle terminates, depth-bound stops) **plus** a live spot-check over a real `memory.db`.
- [ ] Zero new dependencies; read-only; reversible.

## R2 — Reasoning tools *(no weights, reversible)*

Wire the core to the harness: `trace_path` and `imagine_impact` MCP tools ([`15 § 3`](15-deductive-reasoning-and-imagination.md)), sharing the server's read-only memory connection, returning Markdown with the proof trace inlined, degrading canonically when the graph is absent.

**Definition of Done**
- [ ] Both tools registered; the tool-count test updated to assert the new total and names.
- [ ] Live over a real `memory.db`: `trace_path` returns a cited path; `imagine_impact` returns a correct impact cone; unreachable goals return the canonical fallback.
- [ ] Read-only loopback; the egress audit ([`17 § Runbook E`](17-runbooks.md)) still clean.

## R3 / R4 — Imagination synthesis *(feeds the gated P4)*

> **Status: built and measured.** In the reference deployment these two generators produced 750 grounded examples (0 ungrounded, 4.5× the diversity of fact-echo), and a fine-tune trained on them beats the baseline on both eval suites — the first synthetic source that *helps*. Results: [`19 § 8.3`](19-evaluating-quality.md). The design below is what that build followed.

Add two generators to the P3 synthesis stage: **reasoning-chain** (multi-hop derivations narrated with the proof trace as provenance) and **counterfactual** (a hypothesised change + its deductively-true impact cone). Grounding is *stronger* than fact-echo: an example is admissible only if every proof step resolves to a current provenance-backed fact **and** the derivation type-checks (the symbolic verifier); the model only paraphrases, under the same clamps. Build a **multi-hop structural eval** to judge them ([`15 § 6`](15-deductive-reasoning-and-imagination.md)) — never the substring gate. Record the generator mix in `model_lineage`.

**Definition of Done**
- [ ] Every imagined example is proof-trace-grounded; an adversarial test shows zero unprovable emissions.
- [ ] Imagination synthesises only from facts at/above the confidence floor (fuzzy interaction facts never reach weights).
- [ ] A multi-hop, de-echoed eval exists; a retrain including imagined synth either promotes a measurably-better model or auto-discards; `model_lineage` records the generator mix so a regression bisects to the imagination *kind*.

---

## Cross-cutting build concerns

**Two-box integration.** `memory.db` is built/dreamed on the trainer and `rsync`'d to the serving box, exactly like `rag.db` — build to a temp path, verify (0 pollution, sane counts, FTS integrity where relevant), back up, swap, ship. See [`17-runbooks.md § Runbook C`](17-runbooks.md).

**Testing.** The deterministic backbone makes P1 testable with **no model** (extract → query over a fixture repo). The distill/synthesis paths are testable with an injectable **stub** client — no live model, exactly as the RAG embedder is stubbed. Write the **negative** tests the bug catalog calls for: old term gone after re-upsert, empty-subject fact rejected, cross-repo prune scoped, ungrounded paraphrase rejected.

**Strictly-local.** Every new path is filesystem + local-model only. The deterministic extractors are filesystem + `git`; the tools are read-only loopback MCP; the distill/synthesis LLM calls the *local* served model. Re-run the egress audit ([`17 § Runbook E`](17-runbooks.md)) after wiring any new sync step; default every model-load path offline ([`16 § 15`](16-lessons-and-fixes.md)).

**Build order is a safety property, not a preference.** P1–P3 and R1–R2 are reversible and weight-free; P4 (and the imagined synth feeding it) is the only weight-touching path and runs only once the substrate *and* all the anti-collapse gates exist. You build the safety net before you walk the wire.

## The graph in production: growth relations, freshness, and a daemon

The deterministic substrate kept earning after the initial build. Four grounded
relation families joined the base extractors, each provable and re-derivable:

- **tests** — a fail-closed coverage edge: a test symbol links to a subject only
  when the test module *imports* it AND the test function *references* it. No
  naming guesses. (First-party detection matters: indexing a famous library's
  own repo must not deny its own package name.)
- **co-changed-with** — git-history coupling: files that repeatedly change
  together, with mega-commit blast-radius bounding so a 50-file reformat cannot
  mint C(50,2) fake pairs.
- **may-raise** — bounded exception propagation along real call paths, each fact
  carrying its proof path, confidence decaying per hop.
- **transitively-depends-on** — module-dependency closure, run as a second
  derivation phase because it reads the module facts phase one produces.

Two operational pieces close the loop. A **source-freshness metric** reports how
many commits each source checkout is behind its upstream, from git plumbing
only — it exists because a production graph was once found built on a checkout
hundreds of commits stale, and no internal health metric could see it. And an
**event-driven daemon** turns that metric into maintenance: fetch, compare, and
only when something is actually behind, fast-forward the *clean* repos (a dirty
working tree is never touched) and rebuild. A single-instance lock makes
overlapping passes skip instead of stacking writes. Quiet days cost nothing.

## Open source: groundgraph

The generic core of this layer is now public as **groundgraph** — a
strictly-local, zero-runtime-dependency code-memory graph for coding agents
(MIT): the deterministic extractors, the proof-carrying derivation layer, the
trust tiers with query-time decay, the anti-rot and freshness dashboards, the
watch daemon, serve-time recall with fired-signal instrumentation, an MCP stdio
server, and a zero-dep agent-loop example. Indexing Flask end-to-end takes ~3
seconds for ~12,500 facts, on stdlib Python alone.

https://github.com/alleboudy/groundgraph

The evaluation methodology (docs 19 §8.16-8.17) ships with it, null result
included — `docs/honest-eval.md` in the repo. The pitch is the thesis of this
whole chapter: a smaller graph of facts you can prove beats a bigger graph of
facts you can vibe.
