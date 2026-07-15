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
6. **Format-sensitive metrics** — a citation counter that only matches `path:NN` scores a model that cites in prose ("line 57 of `api/tokens.py`") as *not citing*. Worse, it can correlate with the treatment: if the symbolic block's own rendering teaches the countable format while RAG chunk headers teach prose, the metric silently favours one arm. *Control:* broaden extraction to the forms real answers use (including backtick/quote-wrapped paths — the first broadening here still missed those), and never compare numbers across extractor versions.
7. **Probe what each arm actually received.** An injected-context config is only as measured as its injection. Three silent failure modes bit the reference deployment (§8.4): the lexical retrieval arm returning **zero rows on every prompt**, the context block truncating away all but the first chunk, and an all-YES suite half of whose expected signals could be scored by politely restating the question. *Control:* before believing a comparative result, dump a sample of the actual injected blocks and check the suite's answer distribution.

## 8. Results (reference deployment)

A real run of this ablation — the fine-tune vs the stock 7B base, on a ~50-task grounded single-hop structural suite, on an 8 GB laptop card (context capped so the prompt's prefill fits alongside a resident sibling model; the echo-fixed scorer verified active at runtime).

| config | structural `task_mean` | provenance precision | citation rate | hallucination | mean gen ms |
|---|---|---|---|---|---|
| **Base** (stock 7B) | 0.537 | 1.000 | 0.018 | 0.000 | 2451 |
| **FT** (fine-tune only) | 0.590 | 0.867 | 0.232 | 0.000 | 2494 |
| **FT+RAG** | 0.611 | 0.929 | 0.214 | 0.000 | 3238 |
| **FT+NS** | 0.623 | 0.941 | **0.518** | 0.000 | 2467 |
| **FT+RAG+NS** | 0.624 | **0.952** | 0.375 | 0.000 | 3229 |

Lift over the fine-tune (the real "should I turn this on?" number):

| | Δ structural | Δ provenance | Δ citation rate | Δ latency |
|---|---|---|---|---|
| +RAG | +0.021 | +0.062 | −0.018 | +744 ms |
| +NS | +0.033 | +0.074 | **+0.286** | −27 ms |
| +RAG+NS | +0.034 | **+0.085** | +0.143 | +735 ms |

What it says:

1. **Base → FT is a real structural lift (~+5 points); the augmentation deltas are ties.** By this doc's own rules (±1 task ≈ ±0.018 noise floor; 5-point turn-on threshold), the structural deltas over FT (+0.021 RAG, +0.033 NS) are one-to-two tasks on a ~50-task suite — do not read composition into them. A later audit (§8.4) also found most single-hop tasks carry two or three prompt-restatable signals, so the suite's discriminating range is narrower than its size suggests. The single-hop findings that *clear* the bar are the next two.
2. **Every augmentation *improves* provenance, and the combination is best (0.95).** Grounding the model in real context — retrieved chunks *or* graph facts — makes its citations more accurate. NS and RAG reach a similar precision (0.94 / 0.93); combined is best.
3. **The symbolic layer's distinctive win is citation *rate*.** FT+NS cites a source in **52%** of answers (vs 23% FT, 21% RAG, 2% base). Handed exact facts with exact `file:line`, the model volunteers a checkable citation far more often — it answers *and shows its work*. RAG matches precision *when it cites*, but cites less often.
4. **NS is nearly free; RAG is not.** FT+NS latency ≈ FT; FT+RAG adds ~740 ms. On quality-per-latency, the symbolic layer is the bargain.

(No hallucination in any config. This is the single-hop matrix; the multi-hop suite — where the symbolic layer should be the *only* config that can answer at all — is measured in §8.2.)

### 8.1 A caveat that became a case study: a stale index poisoned this result

The first run of this ablation reported FT+RAG provenance at **0.73** — *below* the bare fine-tune — and nearly shipped the conclusion "RAG hurts provenance." That was **wrong**, and root-causing it is the most useful thing in this doc.

- **Symptom:** 27% of FT+RAG citations pointed at a `file:line` that *didn't exist* — not a wrong line in a real file, the *files themselves* were gone.
- **Evidence (model-free, minutes):** **56% of the RAG index's chunk file_paths no longer existed in the repo**, while the eval suite was 96% aligned with current code. RAG was retrieving chunks from *deleted files* and the model faithfully cited their dead paths.
- **Root cause:** the code indexer replaced a file's chunks in place but **never dropped chunks for files deleted from the repo**. The index had been rebuilt *incrementally on top of a stale base* rather than cleanly, so every file removed over months still had chunks.
- **Fix + verification:** a prune step (a full re-index now drops removed files) plus a **staleness detector** in the index-status command (warns when too many chunks point at deleted files). Applying it took staleness 56% → 0% and recovered FT+RAG provenance to **0.929** — the corrected number above.

**The meta-lesson:** *an ablation is only as trustworthy as the quality of the systems it compares.* A stale index made RAG look worse than it is. Control for it — the detector below is that control — and re-measure a surprising result before believing it. (This is [`16 § 4`](16-lessons-and-fixes.md)'s worktree-pollution lesson's cousin: source-index hygiene silently poisons everything downstream.)

### 8.2 The multi-hop result: the symbolic layer's outright win

The multi-hop suite (§5) — 40 tasks, each asking whether one function *transitively* calls another and to name an intermediate on the path with a real `file:line`, authored by the reasoner over the real graph. The two endpoints are in the prompt (so they echo-strip away); the only scorable signals are the **intermediate** and the **source file** — precisely what a config must *chain* to produce. For the symbolic configs the injected context is the reasoner's derivation (the path + every hop's provenance), and all 40 tasks received one. Same fine-tune, same echo-fixed scorer.

| config | structural `task_mean` | provenance precision | citation rate | mean gen ms |
|---|---|---|---|---|
| **FT** (fine-tune only) | 0.319 | 0.063 | 0.175 | 3176 |
| **FT+RAG** | 0.244 | 0.000 | 0.000 | 2376 |
| **FT+NS** | **0.938** | **0.813** | **0.925** | 3179 |
| **FT+RAG+NS** | 0.850 | 0.621 | 0.550 | 4182 |

The headline is the **named-the-intermediate rate** — did the answer contain the connecting function that only chaining reveals:

| config | named the intermediate |
|---|---|
| FT | **0 / 40** |
| FT+RAG | 1 / 40 |
| FT+NS | **39 / 40** |
| FT+RAG+NS | 39 / 40 |

1. **The symbolic layer is the *only* config that answers multi-hop at all.** Bare FT names the connecting intermediate **0 times in 40**; FT+NS **39 times in 40**. This is not a lift but a capability the others structurally lack: the fine-tune's weights hold no reliable transitive-call knowledge, and no prompting recovers a hop never learned. Handed the reasoner's derivation, the model relays it (0.94 structural, 0.93 citation).
2. **RAG fails multi-hop for a deeper reason than "retrieval missed" — and the audit (§8.4) proved it.** The obvious explanation ("the retrieved snippets never contain the connecting hop") turned out to be *false*: probing the actual injected blocks showed the intermediate **was present in 19 of 40** of them, and the true source file in 30 of 40 — yet the RAG-fed model named the intermediate in **one**. Handed the *same* information as an explicit derivation, it names it in 39. The binding constraint is *representation*: a small model cannot perform the cross-chunk join from raw code even when the answer is in its context. And plausible-adjacent chunks actively mislead — a recorded answer: *"The only direct call to `mint_session_token` in the entire project is at line 57 of `api/tokens.py`. No other module calls it."* — a real caller, retrieved, anchored on, over-generalised into a confident wrong "No". That anchoring is how FT+RAG lands *below* bare FT. (The citation-rate 0 was also partly a metric artifact — threat #6: the model cited in backtick-wrapped prose the strict counter didn't match.)
3. **The symbolic layer alone beats it combined with RAG.** Adding retrieved chunks on top of the clean derivation drops provenance (0.81 → 0.62) and citation rate (0.93 → 0.55) and adds latency — the derivation already *is* the answer; the chunks only distract. On multi-hop, turn retrieval off. The honest system-level claim: *single-query retrieve-then-read* cannot chain — not because the chunks lack the hop, but because the model can't assemble it; the symbolic layer is precisely the component that does the assembly and hands over a conclusion.

This is the axis §6 predicted the symbolic layer would own outright, and it does: **multi-hop reasoning is a capability, not a metric delta** — present with the symbolic layer, absent without it. It is the strongest single reason the layer earns its place alongside the fine-tune and RAG.

### 8.3 Training reasoning into the weights: the imagination layer

The results above measure *context injection* (RAG/symbolic facts at inference). The imagination synthesis (R3/R4, [`18-building-the-neurosymbolic-layer.md`](18-building-the-neurosymbolic-layer.md)) asks a different question: can the reasoner's derivations, turned into **grounded training data**, teach the fine-tune *itself* to reason — the "fundamentally different" synthesis the retrain pursuit demanded (reasoning chains, not fact-echo)?

The generator produced **750 examples, 0 ungrounded** from the real graph (contamination-guarded against the eval suites): reasoning-chains ("A calls B calls C, so A transitively reaches C — proof: …") and counterfactuals ("if X changed, these callers break"). Its unique-bigram diversity is **0.458** — versus fact-echo synth's **0.101** and the ~0.42 grounded-synth ceiling. `ft_imag` is a fine-tune retrained on the real corpus + a capped 15% of this synth, the same recipe as the baseline.

| multi-hop config | task_mean | provenance | citation | named intermediate |
|---|---|---|---|---|
| **ft** (baseline) | 0.319 | 0.063 | 0.175 | 0 / 40 |
| **ft_imag** (bare) | **0.556** | **0.768** | **0.775** | **4 / 40** |
| **ft+ns** | 0.938 | 0.813 | 0.925 | 39 / 40 |
| **ft_imag + ns** | 0.938 | **0.974** | 0.500 | 39 / 40 |

| single-hop config | task_mean | provenance | citation |
|---|---|---|---|
| **ft** | 0.590 | 0.867 | 0.232 |
| **ft_imag** | **0.630** | 0.789 | 0.268 |
| **ft+ns** | 0.623 | 0.941 | 0.518 |
| **ft_imag + ns** | **0.693** | **0.972** | 0.393 |

1. **The first synthetic source that *helps*.** Bare `ft_imag` beats the baseline on both suites (multi-hop 0.556 vs 0.319; single-hop 0.630 vs 0.590) and would **promote** through the beat-or-discard gate — where every fact-echo generation regressed. Reasoning-chain synth teaches structure; fact-echo did not.
2. **It bakes in *form + provenance*, not the exact graph.** The largest jumps are multi-hop provenance (0.063 → 0.768) and citation rate (0.175 → 0.775): the model learned to answer in explicit call-chains and cite real `file:line`. But it names the *exact* intermediate only 4/40 from weights alone — it learned to *reason and cite*, not to memorise the graph, so the reasoner is still needed at inference for exactness.
3. **`ft_imag` + the reasoner is the best config on both suites** — trained weights + injected derivations compose to the top numbers; the trained-to-cite model even relays the reasoner's provenance more faithfully (0.974 vs 0.813).

This **closes the loop**: the reasoner's derivations become training data that measurably improves the fine-tune — contamination-guarded (generalization, not leakage) and gated (it cannot collapse). It is the first time any synthetic source moved the model the right way, and the empirical payoff of building the whole reasoning stack.

### 8.4 Auditing your own ablation: the second root-cause pass

The stale-index case study (§8.1) caught one artifact and the result shipped. Weeks later, a deliberate audit — *"is the evaluation correct? probe what each arm actually received"* — found **five more measurement-vs-mechanism gaps stacked under the surviving numbers.** None flipped the headline (the symbolic layer really is the only config that chains); every one of them distorted the size or the *mechanism* of a reported gap. In found order:

1. **The lexical retrieval arm was dead on every prompt.** The FTS safe-query builder joined every token with implicit AND, so a question-shaped query ("does X transitively call Y? Answer yes or no and cite…") demanded chunks containing *every English word* — measured against the production index: **0 rows for 100% of both suites' prompts**. "Hybrid" retrieval had silently been vector-only all along. For multi-hop pairs the AND-collapse is *structural*: A+B share a chunk, B+C share a chunk, A+C never do. *Fix:* retry with OR when AND matches nothing.
2. **The file path was invisible to both retrieval arms.** Not in the FTS index, not in the embedded text — yet file-anchored questions ("what does `analytics/metrics.py` import?") carry the path as their strongest token. *Fix:* index it and lead the embedded text with it ([`08-rag.md`](08-rag.md)).
3. **The injected block silently truncated to ~one chunk.** A flat `block[:1600]` cut let chunk #1 (which can itself be ~1,500 chars) eat the whole budget; chunks 2–3 were dropped and the block usually ended mid-line. *Fix:* chunk-aware packing (equal share + carry-forward, line-boundary cuts).
4. **The citation metric measured format, not grounding** (threat #6) — and the *first* broadened regex still returned zero because the model backtick-wraps paths and `\s+` doesn't match a backtick. Rescoring the recorded answers with the fixed extractor moved several cells (e.g. baseline multi-hop provenance 0.06 → 0.15).
5. **The suite couldn't catch a yes-biased model.** All 40 multi-hop answers were YES, and half of each task's expected signals were the endpoints already present in the prompt — restating the question scored 0.5 with zero reasoning, so part of the FT-vs-FT+RAG gap was *restating style*. *Fix:* a v2 generator that drops endpoint signals and adds **reasoner-proven negative tasks** (same prompt template; kept only when the engine proves unreachability with a complete, frontier-exhausted search) scored by an `answer_polarity` field.

**The meta-lesson, sharpened from §8.1:** a comparative eval is a measurement of the *whole* pipeline — retrieval emptiness, injection truncation, metric format-bias, and suite polarity are all invisible in a score table. Dump what each arm actually received; read your scorer against real answers; check your suite's answer distribution. And when a rebuild for the re-measurement doubles your index, diff its *composition* before blaming the first plausible directory — the audit's own first pollution diagnosis was wrong (the real culprit was nested git checkouts under a novel name, pruned for good by walking with a "contains `.git`" check instead of a name list).

### 8.5 The full lattice: does each layer need the fine-tune?

§2's optional cells (`base_rag`, `base_ns`, `base_rag_ns`) sat unmeasured until the audit forced the question "what is the fine-tune actually buying?" Measuring them costs little — the injected context blocks are *model-independent*, so the base family reuses the ft family's exact injections. `task_mean` (fixed pipeline; the de-echoed v2 suite splits positives/negatives):

| config | single-hop | multi-hop | v2 positives |
|---|---|---|---|
| base | 0.537 | 0.206 | 0.017 |
| base_rag | **0.736** | 0.194 | 0.067 |
| base_ns | 0.641 | 0.856 | **0.975** |
| ft | 0.590 | 0.319 | 0.075 |
| ft_rag | 0.700 | 0.256 | 0.075 |
| ft_ns | 0.609 | **0.938** | 0.950 |

Four findings, each of which changes a default assumption:

1. **The symbolic layer is model-independent.** The *stock* model relays an injected derivation at 0.856–0.975 — on the honest de-echoed suite, *every* NS-carrying config clusters at 0.95–0.975 regardless of model or retrieval, while every non-NS config sits at 0.02–0.08. Total layer separation: the reasoner is the capability, the model is a mouthpiece. It even teaches *citing* by example — the stock model's citation rate jumps 0.05 → 0.93 with derivations in context.
2. **Retrieval's anchoring harm on reasoning is model-independent too** (base_rag < base, mirroring ft_rag < ft). It's a property of retrieve-then-read at small-model scale, not of any particular fine-tune.
3. **Fixed retrieval helps the stock model *more* than the fine-tuned one on lookup** — base_rag posts the best single-hop cell measured (0.736 > ft_rag's 0.700). A model with no baked-in opinions about the codebase reads the retrieved context with less interference.
4. **The generic-corpus fine-tune is the weakest of the three layers on structural axes**: ~+0.05 bare, and consistently *slightly negative* under augmentation once every cell shares identical injection parameters (−0.036 under retrieval, −0.032 under the symbolic layer, −0.046 under both — each ~2 tasks, but the same sign three times). Its real case is fluency, house conventions, and completion styles these suites don't measure — and *targeted* training: the reasoning-chain synthesis retrain lifted bare multi-hop +0.24 where the generic corpus managed +0.11. **What you train on matters more than that you train.**

The strategic consequence: budget effort as symbolic layer ≥ retrieval quality > generic retrains — and let retrains earn their place through targeted capability synthesis, gated by the eval.

**And the re-measurement, which settles the question both ways.** With all five gaps fixed and the index rebuilt clean, the RAG rows were re-run. Single-hop: FT+RAG structural **0.611 → 0.700** — the lift over the bare fine-tune (+0.11) now *clears* the 5-point turn-on bar it previously missed, and citation rate doubles to 0.446 (the model can finally cite `path:NN` from a RAG block, because the blocks finally carry line spans); valid-citations-per-answer (rate × precision — the only citation number comparable across extractor versions) rises **+71%**. Multi-hop: FT+RAG **0.256 — still below bare FT's 0.319.** With retrieval, packing, spans, and the metric all repaired, the remaining deficit is *representational*: the model anchors on plausible-adjacent chunks and cannot join them, which is precisely the capability the symbolic layer supplies. Both headline decisions survive, each in a stronger form: **turn RAG on for single-hop lookup (it now earns it), and off for reasoning (its failure is intrinsic, not an implementation accident).** The combined config also recovered to tie FT+NS on multi-hop structure (0.938) at extra latency — the old "RAG degrades the symbolic layer" was partly broken-RAG noise.

## 9. What to take from this

The headline is not "config X is best" — it is **which layer to turn on for which job**. Read the matrix by column, not by row: pick the axis you care about (structural precision, trustworthy provenance, freshness, multi-hop reasoning, latency budget) and turn on the cheapest layer that wins it. The neuro-symbolic layer earns its place on the axes the others structurally cannot reach — exact provenance and multi-hop reasoning — while RAG owns freshness and the fine-tune owns fluent house style. The full stack is the union, and this methodology is how you prove each piece pays for itself rather than assuming it does.
