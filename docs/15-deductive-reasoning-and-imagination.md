# 15 — Deductive reasoning and imagination

[`14-agent-memory-and-dreaming.md`](14-agent-memory-and-dreaming.md) gave the stack a knowledge graph that *writes itself*. This doc attaches the **reasoner that reads it** — a small, dependency-free deductive engine that derives *new* facts (transitive reachability, dependency, and change-impact) from the ones already stored — and then grows an **imagination** layer on top: the assistant simulates hypothetical changes against the codebase and reasons about the consequences, soundly, because a symbolic derivation (not a neural guess) owns every claim.

The two pay one debt each. The reasoner is the "put a logic engine on the same triples" step the memory doc deferred. Imagination is the answer to the memory doc's warning that *fact-echo* synthesis regresses structural recall: what you actually want to teach a model is **cross-file reasoning chains**, and those fall straight out of a deductive engine with a proof trace.

## 1. What the graph can't do yet

`query_facts` ([`14 § 5`](14-agent-memory-and-dreaming.md)) answers **one hop**. The interesting questions are multi-hop:

- "Does `ReportBuilder.render` **transitively** reach `numpy`?" — a *derivation* over `calls` + `imports`, not a stored row.
- "If I change the signature of `load_settings`, **what breaks**?" — the reverse-reachability cone; nothing computes it.
- "Show me **the path** from `Service.handle` to `WEIGHTS`." — a *proof*, not a fact.

These are exactly where the neural half is weakest (precise multi-step deduction) and the symbolic half is strongest. They are also where a Prolog-shaped backward-chaining engine lands naturally.

## 2. The deductive core: backward chaining, no dependencies

The rules are Datalog-shaped — function-free, guaranteed-terminating:

```
reaches(A, B)  :- calls(A, B).
reaches(A, C)  :- calls(A, B), reaches(B, C).
depends(A, B)  :- depends-on(A, B).
depends(A, C)  :- depends-on(A, B), depends(B, C).
impacts(X, Y)  :- reaches(Y, X).        -- changing X impacts everything that reaches it
```

A small evaluator (a few hundred lines of stdlib Python over `memory.db`, read-only) answers a *specific* goal by expanding these against the stored triples — memoised per call, depth-bounded, cycle-safe. It returns a **Derivation**: the derived triple *plus* the ordered chain of real facts that proves it, each step naming a real `fact_id` and its `file:line`.

### Two design choices worth stating

**Backward, not forward.** You could forward-chain and *materialise* the whole transitive closure into the store. Don't: the closure of a call graph is O(N²) edges, supersession of derived rows is ill-defined (they're recomputable, not observed), and every dream run would re-derive them. Backward chaining answers the *asked* goal on demand — no blow-up, always fresh, and the proof trace is a natural byproduct. This is exactly the operator's Prolog instinct (SLD resolution), reborn.

**Confidence = min-of-chain.** A derivation is only as trustworthy as its weakest real link. A chain of deterministic (`confidence 1.0`) facts stays `1.0`; a chain that touches one model-distilled `0.7` fact surfaces as `0.7`. So a derivation over the reliable backbone is visibly more trustworthy than one leaning on a fuzzy interaction fact — and imagination (§4) can refuse to train on anything below a floor.

> **Provenance was free in Prolog; here it's the whole point.** A successful proof *is* its own explanation — the resolution trace tells you exactly why the answer holds. Modern neural systems spend enormous effort buying back the provenance Prolog gave you for nothing. A Derivation hands it back: every hop is a real, clickable source line.

## 3. Two more read-only tools

Beside `query_facts` / `explain_entity`, the harness gets two reasoning tools ([`07-mcp.md`](07-mcp.md)), each returning Markdown with the proof trace inlined:

- **`trace_path {source, target}`** — the shortest call/dependency path from one entity to another, each hop a cited `file:line`. "How does A end up reaching D?" The multi-hop question `query_facts` can't answer in one hop.
- **`imagine_impact {entity, change_kind}`** — the deductively-true **impact cone**: everything that transitively calls `entity` and may need updating, grouped by hop-distance, each with its shortest proof path. "What breaks if I change X?"

Both degrade to a canonical "no derivation — try `query_facts` / `search_code`" when the graph is absent or the goal is unreachable, so they are safe to register before the graph is rich.

## 4. Imagination, precisely (not poetically)

The memory doc had *dreaming* — offline consolidation of *real* experience. Imagination is its generative sibling: **generating and evaluating hypothetical states the system never observed**, kept honest by grounding. The lineage is real:

- **World Models** (Ha & Schmidhuber, 2018) — an agent trains *inside imagined rollouts* of a learned model of its environment. Here the **codebase world-model is `memory.db`**, and an imagined rollout is a deductive walk over it.
- **Dreamer** (Hafner et al.) — "learning behaviours by latent imagination." Ours is symbolic, so the rollout is exact and inspectable, not latent.
- **AlphaGeometry** (2024) — **neural proposes, symbolic verifies**. Imagination inverts the collapse risk the same way: the *derivation* is symbolic (sound by construction); the neural model only *phrases* it in natural language, clamped by the grounding gates below.

Concretely, imagination is two operations over the deductive core:

1. **Multi-hop derivation** (a "reasoning chain"): chase a real chain — `A calls B` → `B imports C` → `C defines D` — and state the conclusion (`A transitively reaches D`) with the whole chain as provenance. *Novel as a statement; true as a derivation.*
2. **Counterfactual simulation** (a "what-if"): pick an entity, imagine a concrete change (rename / signature-change / delete), compute the deductively-true consequence (the impact cone), and narrate it as a hypothetical. *Novel as a scenario; verifiable as a fact about the graph.*

## 5. Reasoning-chain and counterfactual synthesis

This is the "fundamentally different, not fact-echo" training data the memory doc ([`14 § 6`](14-agent-memory-and-dreaming.md)) demanded. The dream's synthesis stage gains two generators beside the plain template one:

**Reasoning-chain** example:

> **Q:** Does `ReportBuilder.render` transitively depend on `numpy`?
> **A:** Yes. `ReportBuilder.render` calls `read_rows` (`builder.py:412`), which imports `numpy` (`builder.py:7`) — so the dependency holds transitively.

**Counterfactual** example:

> **Q:** If the signature of `load_settings` changed, which callers in `reporting/` would need updating?
> **A:** Three: `ReportBuilder.render` (`builder.py:661`), `Service.handle` (`service.py:…`), and `apply_defaults` (`settings.py:…`) — each calls `load_settings` directly or transitively.

The grounding bar is *higher* than fact-echo, and it is met structurally: an example is admissible only if **every** proof step resolves to a current, provenance-backed fact **and** the derivation type-checks against the rules (the symbolic verifier). The model's only job is to paraphrase, under the same vocabulary/word-boundary clamps the plain synthesis uses. Diversity is naturally higher (varied paths, varied hop counts), which relieves the diversity-floor pressure fact-echo hits. And because the confidence floor (§2) applies, imagination over fuzzy interaction facts never becomes training data — only the deterministic backbone imagines into weights.

All of [`14 § 6`](14-agent-memory-and-dreaming.md)'s anti-collapse gates still apply unchanged; the retrain records the *generator mix* (fact-echo vs reasoning-chain vs counterfactual) so a regression bisects to the *kind* of imagination that caused it.

## 6. Judge it on a reasoning eval, not a substring gate

Imagined synthesis is worthless if judged by an echo-prone scorer (see [`16-lessons-and-fixes.md § The eval scored the echoed prompt`](16-lessons-and-fixes.md) — a real trap that inflated a whole gate history). It needs a **multi-hop structural eval**: tasks whose answer requires a *derivation* (transitive reach, impact set, path), scored on the *generated* answer only, with signals in `file.py:NN` form (never bare integers, which substring-match anything). The reasoner itself can author candidate tasks (grounded, proof-checked), which are then adversarially verified against the real source — the same author-then-verify recipe used for the structural suite.

## 7. The punchline

The engine at the end of this pipeline is **Datalog / Prolog / ASP over the same triples** the graph already stores (Datalog first — reachability and impact map cleanly onto recursive rules; keep Prolog/ASP for later constraint work). When it attaches, look at what each half is doing:

- The **LLM does knowledge acquisition** — reading code, commits, and corrections, populating the fact base automatically. *It solves the exact bottleneck that killed expert systems.*
- The **symbolic reasoner does precise inference** — exact multi-hop deduction with a real proof trace. *Exactly where neural is weakest.*

So the symbolic reasoning of the 1980s was never obsolete. It was waiting for someone to solve its data-entry problem — and the neural revolution built the machine that finally populates, and now *imagines over*, the knowledge base for you.

## Further reading

- **World Models** (Ha & Schmidhuber, 2018) — training inside imagined rollouts: https://worldmodels.github.io/
- **Dreamer v3** (Hafner et al., 2023) — mastering diverse domains by latent imagination: https://arxiv.org/abs/2301.04104
- **AlphaGeometry** (Trinh et al., *Nature* 2024) — neural proposes, symbolic verifies: https://www.nature.com/articles/s41586-023-06747-5
- **Soufflé** — a fast Datalog engine that compiles to C++ (the natural "bolt-on" when the graph outgrows the pure-Python evaluator): https://souffle-lang.github.io/
- **The Kautz taxonomy of neuro-symbolic systems** (AAAI 2020 keynote) — the map of ways to wire neural and symbolic together: https://www.cs.rochester.edu/u/kautz/talks/
- **ReAct** (Yao et al., 2022) — interleaving reasoning and tool actions, the pattern MCP tools implement: https://arxiv.org/abs/2210.03629
