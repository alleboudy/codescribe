# 19 — Evaluating quality: LLM vs +RAG vs +neuro-symbolic

Once you have a fine-tune ([`02-fine-tuning.md`](02-fine-tuning.md)), a RAG index ([`08-rag.md`](08-rag.md)), and a symbolic memory + reasoner ([`14`](14-agent-memory-and-dreaming.md)/[`15`](15-deductive-reasoning-and-imagination.md)), the obvious question is: **does each layer actually make the assistant better, and at what?** This doc is the evaluation methodology — the metrics, the baselines, and the ablation that answers it — plus the results from the reference deployment.

## 1. Why one number is the wrong answer

"The assistant scores 0.6" tells you nothing actionable, because the layers help on **different axes**:

- the **fine-tune** encodes house style and broad structure into weights — but ages, and invents specific facts;
- **RAG** adds fuzzy recall of specific snippets — fresh, but shallow about structure;
- the **neuro-symbolic** layer adds exact structural facts with real provenance and multi-hop reasoning — precise where the others guess, silent outside the graph.

So evaluation must be a **capability × configuration matrix**: several metrics, each config scored on all of them. The useful output is not a scalar but "RAG wins freshness, the symbolic layer wins provenance and reasoning, and here's the cost of each."

## 2. The ablation (configurations)

The same served model under five context conditions, plus the untrained floor:

| Config | Fine-tune | RAG context | Symbolic context | Isolates |
|---|---|---|---|---|
| **Base** | ✗ (stock base model) | ✗ | ✗ | the floor — a generic coder LLM |
| **FT** | ✓ | ✗ | ✗ | "LLM only" — the fine-tune alone |
| **FT+RAG** | ✓ | ✓ | ✗ | RAG's lift over the fine-tune |
| **FT+NS** | ✓ | ✗ | ✓ | the symbolic layer's lift |
| **FT+RAG+NS** | ✓ | ✓ | ✓ | the full system; do the layers compose? |

**Fair context injection.** RAG/NS are compared by *injecting retrieved context into the prompt* (retrieve → prepend → complete — the standard RAG-lift protocol), not by driving the full agent loop. Every config sees the same task, the same generation settings, and the same scorer; only the injected block differs — for RAG, the top-k retrieved code chunks; for NS, the current provenance-backed facts about the file/symbols the task concerns (plus, for multi-hop tasks, the reasoner's derivations).

This is deliberately *generous* to the augmented configs: if the symbolic layer surfaces the exact fact, the model should just relay it with provenance. **That is the value proposition, not cheating** — the eval measures whether the *end-to-end system* answers correctly and cites truthfully. What it reveals is *how much context each layer must supply* to get the model there.

## 3. Metrics (the axes)

Every answer is scored on all of these; report per-config means (and per-relation breakdowns).

1. **Structural accuracy** — fraction of a task's expected signals present in the *generated* answer (echo-stripped by token count — see the trap in [`16 § 5`](16-lessons-and-fixes.md)). The headline "does it know the code" number. Break it down per relation (imports / defines / calls / transitive-reach / impact) — the layers win different relations.
2. **Provenance precision** — of the `path:NN` citations in the answer, the fraction that point to a file that *exists and has ≥ NN lines*. The symbolic layer cites real provenance by construction (→ ~1.0); the bare fine-tune invents plausible line numbers; RAG is in between. **This is the metric that most cleanly separates neuro-symbolic from the rest**, and the one a user actually trusts.
3. **Citation rate** — fraction of answers that cite any `path:NN` at all (context for provenance precision — a config can score high precision by rarely citing).
4. **Hallucination / refusal rate** — fraction of answers containing a forbidden signal (a refusal, or a symbol from the *wrong* project). Lower is better.
5. **Latency** — wall-clock per answer, split into retrieval/query and generation. A config that wins +2% accuracy at +2 s/answer may not be worth it interactively. (The symbolic `query_facts` is sub-millisecond; RAG retrieval tens of ms; a deep impact derivation is the slowest symbolic path.)

Deliberately **not** collapsed into one weighted scalar — report the matrix. If a gate needs a scalar, use structural accuracy with a hard guard that provenance precision and hallucination did not regress.

## 4. Baselines

- **Floor** — the **Base** config. Any layer that doesn't beat this is dead weight.
- **Reference point** — the **FT** config. Every augmentation's *lift* is `Δ over FT`, because that's the real decision: "given the fine-tune I already ship, does turning on RAG / the graph help?"
- **Ceiling** — the task answer keys define a perfect score (1.0 structural, 1.0 provenance, 0 hallucination). The gap to ceiling is the headroom.
- **Turn-on threshold** — reusing the RAG-lift rule ([`08 § Measuring RAG lift`](08-rag.md)): turn a layer on by default only if its lift is ≥ 5 points on the axis it targets *and* it doesn't regress another. Below that, leave it opt-in.

## 5. The suites

- **Structural suite** — grounded, adversarially-verified single-hop tasks (imports / defines / calls), pinned to a repo commit, with signals in `file.py:NN` form (never bare integers — the substring-match trap of [`16 § 7`](16-lessons-and-fixes.md)).
- **Multi-hop suite** — tasks whose answer needs a *derivation* ("does A transitively reach B", "what breaks if you change C"). Neither the bare LLM nor RAG can chain these; the reasoner can. Authored by the reasoner over the real graph, then adversarially verified.
- **Freshness suite** (optional) — questions about changes newer than the fine-tune's training cut, where RAG should win and the frozen fine-tune must lose.

## 6. Hypotheses (stated up front, so the results are falsifiable)

- **Base → FT**: a real structural lift (the fine-tune learned the code) and lower refusal.
- **FT → FT+RAG**: modest structural lift; a provenance lift (it saw a real chunk); best on freshness; little help on multi-hop.
- **FT → FT+NS**: the **largest** provenance lift (→ ~1.0) and the **only** config that wins the multi-hop suite; strong single-hop lift; lowest hallucination.
- **FT+RAG+NS**: best or tied-best everywhere, at the highest latency.

If the results *contradict* these, that's the interesting finding — e.g. if the fine-tune already answers structural questions so well that NS adds nothing there, the symbolic layer's value is confined to provenance and reasoning, and that should change what you ship.

## 7. Threats to validity (and their controls)

1. **Echo-inflated scoring** ([`16 § 5`](16-lessons-and-fixes.md)) — inflates every config equally and dilutes deltas. *Control:* the token-slice scorer fix, asserted active at runtime (the editable-install-shadows-`PYTHONPATH` trap, [`16 § 12`](16-lessons-and-fixes.md), makes this non-optional).
2. **The eval's own facts in training/synth data** — contaminates. *Control:* the frozen held-out split + the benchmark-contamination guard exclude eval-cited files from synthesis.
3. **"NS injects the answer" looks like cheating** — addressed in §2: provenance precision + the multi-hop suite are what make it honest (a config can only score provenance by citing *real* lines).
4. **Small, noisy suites** — ±1 task is a meaningful fraction on a ~50-task suite. *Control:* report per-relation n, grow the suites, treat sub-threshold deltas as ties.
5. **VRAM co-residency** — the retrieval embedder and the language model don't co-fit on a small card; an in-process embedder free leaves residue that offloads the model to CPU and crashes generation. *Control:* precompute all retrieval/graph context in a process that **exits** (fully freeing the GPU) before the model process loads. (A real failure — the two-process split is load-bearing, not tidiness; see [`17 § Runbook B`](17-runbooks.md).)

## 8. Results (reference deployment)

A real run of this ablation — the fine-tune vs the stock 7B base, on a ~50-task grounded single-hop structural suite, on an 8 GB laptop card (context capped so the prompt's prefill fits alongside a resident sibling model; the echo-fixed scorer verified active at runtime).

| config | structural `task_mean` | provenance precision | citation rate | hallucination | mean gen ms |
|---|---|---|---|---|---|
| **Base** (stock 7B) | 0.537 | 1.000 | 0.018 | 0.000 | 2451 |
| **FT** (fine-tune only) | 0.590 | 0.867 | 0.232 | 0.000 | 2494 |
| **FT+RAG** | 0.610 | 0.727 | 0.196 | 0.000 | 3034 |
| **FT+NS** | 0.623 | **0.941** | **0.518** | 0.000 | 2467 |
| **FT+RAG+NS** | **0.633** | 0.864 | 0.393 | 0.000 | 2745 |

Lift over the fine-tune (the real "should I turn this on?" number):

| | Δ structural | Δ provenance | Δ citation rate | Δ latency |
|---|---|---|---|---|
| +RAG | +0.020 | **−0.140** | −0.036 | +540 ms |
| +NS | +0.033 | **+0.074** | **+0.286** | −27 ms |
| +RAG+NS | +0.044 | −0.003 | +0.161 | +251 ms |

What it says (four findings, one of them a surprise):

1. **Every layer lifts structural accuracy, and they compose.** Base → FT is worth ~+5 points; each augmentation adds a bit more (NS > RAG), and the full stack is best. Modest, because the base already knows generic structure and the suite is small — but consistent.
2. **Provenance is where the symbolic layer earns its place, decisively.** FT+NS cites a real source in **52%** of answers (vs 23% for the bare fine-tune, 2% for base) *and* at the best precision (0.94). Handed real facts with real `file:line`, the model relays and cites them. This is the axis the others structurally can't reach — the killer feature for a user who needs "…and where is that?"
3. **RAG *hurts* provenance — the hypothesis that flipped.** We predicted RAG would *lift* provenance (it saw a real chunk). It did the opposite (precision 0.73, down from 0.87): retrieved chunks tempt the model into confident-but-misattributed line numbers. RAG helps recall and freshness — its real jobs — but should **not** be trusted for provenance; the symbolic layer should own that.
4. **NS is nearly free; RAG is not.** FT+NS latency ≈ FT (a graph query is sub-millisecond); FT+RAG adds ~540 ms. On quality-per-latency, the symbolic layer is the bargain.

(No hallucination appeared in any config; and this is single-hop only — the multi-hop suite, where the symbolic layer should be the *only* config that can answer at all, is the strongest expected win and is left as the next measurement.)

## 9. What to take from this

The headline is not "config X is best" — it is **which layer to turn on for which job**. Read the matrix by column, not by row: pick the axis you care about (structural precision, trustworthy provenance, freshness, multi-hop reasoning, latency budget) and turn on the cheapest layer that wins it. The neuro-symbolic layer earns its place on the axes the others structurally cannot reach — exact provenance and multi-hop reasoning — while RAG owns freshness and the fine-tune owns fluent house style. The full stack is the union, and this methodology is how you prove each piece pays for itself rather than assuming it does.
