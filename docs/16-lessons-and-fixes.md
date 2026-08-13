# 16 — Lessons and fixes

A catalog of real bugs found auditing this class of stack, and the fix pattern for each. Every one is generalizable — none is specific to any one codebase — so treat this as a checklist when you build or review the data, RAG, memory, eval, and harness layers. Each entry states the **trap**, the **fix**, and the **rule** to carry forward. The operational counterpart (how to *find* these with an audit, how to re-gate an eval without fooling yourself) is [`17-runbooks.md`](17-runbooks.md).

---

## Data & retrieval layer

### 1. External-content FTS5 ghosts old terms on re-upsert

**Trap.** The RAG store mirrors each row into an FTS5 table declared `content='issues'` (external-content — the FTS index holds terms, not text, and reads the content table to know which terms a row contributes). A writer that does `UPDATE issues …` first and then `INSERT OR REPLACE INTO issue_fts …` computes FTS5's implicit delete against the *already-updated* content row — so it removes the NEW terms (not yet indexed) and the OLD terms are never cleaned. Every edited/reopened row on the nightly incremental accumulates ghost terms and skews BM25.

**Fix.** For an external-content FTS5 table, **delete the FTS row while the OLD content is still present** — before the content `UPSERT` — then plain-`INSERT` the new mirror:

```python
conn.execute("DELETE FROM issue_fts WHERE rowid = ?", (n,))   # OLD content still in `issues` → right terms removed
conn.execute("INSERT INTO issues(...) ... ON CONFLICT DO UPDATE ...")
conn.execute("INSERT INTO issue_fts(rowid, title, body) VALUES (?, ?, ?)", (n, title, body))
```

**Rule.** External-content FTS5: delete the FTS row *before* mutating the content it mirrors; never trust `INSERT OR REPLACE` to clean the index. Test the *negative* — assert the OLD term no longer `MATCH`es after a re-upsert; an idempotency test that only checks the new term matches will pass while the index rots.

### 2. Raw query text forwarded to FTS5 `MATCH`

**Trap.** Forwarding the user's query straight to `WHERE fts MATCH ?` means code-shaped queries hit FTS5's own grammar: `config.py` → *syntax error near "."*, `fix()` → *near ")"*, `a - b`, `title:x` → reinterpreted as operators. The store swallows the error and the BM25 arm returns nothing, so **hybrid retrieval silently collapses to vector-only** on a large fraction of real code queries.

**Fix.** Sanitise: extract identifier tokens and wrap each as an FTS5 string literal (`"config.py" "timeout"`, implicit-AND), so `MATCH` never sees FTS5 syntax. Return "no BM25 candidates" only when the query has genuinely no indexable token.

**Rule.** Never hand user text to a query grammar unescaped. A silent degrade (one retrieval arm quietly dying) is worse than a loud error — you won't notice the quality loss.

### 2b. …and the sanitiser's implicit-AND kills it a second way

**Trap.** The §2 fix quotes every token and joins with a space — FTS5's implicit AND. Safe against the grammar, but a *question-shaped* query ("does `mint_token` transitively call `hash_secret`? Answer yes or no and cite a file:line.") now demands one chunk containing **every** English token. Probed against a production code index: **zero BM25 rows for 100% of two eval suites' prompts** — hybrid retrieval had silently been vector-only all along, again. For multi-hop endpoint pairs the AND is *structurally* unsatisfiable: A+B share a chunk and B+C share a chunk, but A+C never do.

**Fix.** Keep the AND query first (precise when it hits); when it matches **nothing**, retry the same sanitised phrases joined with `OR` — BM25 ranks best-first, so a chunk matching any strong token (an identifier, a path) surfaces and the weak English tokens just don't help.

**Rule.** After you make a query *safe*, check it can still *match*: run your real query corpus against the index and alarm on a 0-row rate. Both §2 failures were silent for weeks because vector-only results still look plausible.

### 3. `split_salt: ~` becomes the literal salt `"None"`

**Trap.** `salt = str(config.get("split_salt", repo_name))`. A *present-but-null* YAML key (`split_salt: ~`) makes `.get` return `None` — the default only fires on a **missing** key — so `str(None)` salts every train/val/test split with the literal string `"None"`, silently diverging from the intended per-repo salt and setting up a train↔eval split mismatch.

**Fix.** `salt = str(config.get("split_salt") or repo_name)`.

**Rule.** For a nullable-with-fallback config key, use `cfg.get(k) or default`, not `cfg.get(k, default)`. The two differ exactly when the key is present and null.

### 4. Source walkers index nested agent-worktree checkouts

**Trap.** Both the RAG code chunker and the memory extractor exclude `.git`, `node_modules`, `vendor`, … but not a coding agent's scratch directory. Modern agent harnesses create **full git-worktree checkouts** of the repo under a hidden dir (e.g. `.claude/worktrees/<branch>/`). A walker that `rglob`s the tree descends into it and re-indexes an entire duplicate copy. Measured on a real index: **~49% of the graph was worktree-duplicate**, and provenance cited the transient worktree path instead of the canonical one.

**Fix.** Add the agent scratch dir (`.claude`, and any `*/worktrees/*`) to the exclusion set in *both* walkers.

**Rule.** A repo that self-hosts agent worktrees silently doubles every index that walks the tree. Exclude agent scratch dirs, and validate provenance paths against a *real* build, not a toy fixture — a toy fixture never has a nested worktree.

### 4a. Nested checkouts, generation three: prune by construction, not by name

**Trap.** The §4 fix excluded the scratch dir *by name*. Months later worktrees reappeared under a *different* root-level name no list knew — and a clean index rebuild came out **double**: half its chunks were duplicate worktree content. A third walker-pollution generation, same mechanism, new name. Bonus finding: the project's three tree walkers (memory extractor, code source, docs source) each carried a *private* exclude list, and they had drifted — one got enriched in a perf fix, the others didn't.

**Fix.** Stop playing the name game. One shared exclude constant for every walker, plus an `os.walk`-based shared walker that prunes, top-down, **any subdirectory containing `.git`** — a directory for clones, a *file* for worktrees. A nested checkout is duplicate repo content by construction, whatever it's called. (Top-down pruning also never descends into `node_modules` at all — the old `rglob` walked everything and filtered afterwards.)

**Rule.** Duplicate "skip these dirs" knowledge WILL drift; centralise it. Prune nested checkouts structurally (`.git` marker), not by name. And when an index size jumps on rebuild, diff its *composition* (top-dir counts old vs new) before shipping a diagnosis — the first diagnosis here blamed a package-manager store that turned out to contribute eight files.

### 4b. An incremental re-index never prunes chunks for *deleted* files

**Trap.** The code indexer replaces a file's chunks in place (delete-by-`file_path`, re-insert) on every run — but it has no step that removes chunks for files that were **deleted from the repo**. So an incremental re-index *accumulates* stale chunks: every file removed since the base index still has entries pointing at its now-dead path. Rebuild incrementally on top of an old base and the rot compounds. Measured on a real index that had been rebuilt this way: **56% of chunk file_paths no longer existed in the repo.** Retrieval then serves chunks from deleted files, and the model faithfully cites their dead `file:line` — silently degrading provenance (this is what made RAG look like it *hurt* provenance in [`19 § 8.1`](19-evaluating-quality.md) until it was root-caused).

**Fix (remedy).** After a *full* code re-index (the code source is re-walked in full every run, so you have the current file set in hand), prune any chunk whose `file_path` is not in the current set — deleting its vec0 + external-content FTS5 mirrors in the [§1](#1-external-content-fts5-ghosts-old-terms-on-re-upsert) delete-first order. Now every re-index self-heals. Applying it to the polluted index took staleness 56% → 0% and recovered the RAG provenance metric from 0.73 to 0.93.

**Detection (correction mechanism).** Don't wait to *notice* degraded answers — instrument it. Add a **staleness check** to the index-status command: for each distinct indexed `file_path`, does the file still exist under the repo root? Report `stale / total` and **warn past a threshold** (≈5% is ordinary churn; the pathology above was 56%). That single line ("*N% of the code index points at deleted files — re-index or rebuild*") is the signal that tells an operator a rebuild is due *before* it poisons retrieval — and, downstream, before it poisons an eval that compares against RAG.

**Rule.** A "replace-in-place" index silently rots when files are deleted unless something prunes removed entries; make every full re-index self-prune, **and** surface a staleness metric so you can *detect* the need for a rebuild instead of discovering it in a degraded metric. Prefer a clean rebuild over an incremental one on top of a stale base.

---

## Eval layer

### 5. The eval scored the echoed prompt

**Trap.** The task scorer wrapped each prompt in the chat template (`<|im_start|>user\n…`) and stripped the echo with `decoded[len(prompt):] if decoded.startswith(prompt)`. But the decode used `skip_special_tokens=True`, which drops the very control tokens — so the decoded string began `"user\n…"`, the `startswith` never matched, and **the whole prompt echo was scored** by substring matching. When a large fraction of a suite's `expected_signals` also appear in the prompts (measured: ~40%), every model gets the same auto-hit — absolute scores inflate and real deltas dilute.

**Fix.** Strip the echo by **token count, not text**: `generate()` returns the input ids as a prefix, so slice `output_ids[0][inputs["input_ids"].shape[1]:]` and decode only the new tokens. Then keep `expected_signals` out of the prompt text.

**Rule.** When scoring generated text, slice off the prompt by token length; never string-match a prompt you decoded with `skip_special_tokens=True`.

### 6. An eval-scoring bug inflates absolute numbers — but may not change decisions

**Trap (and its inverse).** Having found #5, it is tempting to declare "the entire gate history is invalid." Re-measuring on real hardware showed the opposite: the echo inflated *every* generation by ~4–7 points, but the **relative ordering held** — the winner still won, and the "synthetic-data generation regresses structural recall" conclusion survived the correction (it even sharpened). So #5 is a real fix, but the decisions drawn from the old gate stood.

**Rule.** An eval-scoring bug shifts absolute numbers; whether it changes *decisions* is a separate, empirical question. Re-measure before claiming a whole history is worthless — and re-measure *correctly* (see #12).

### 7. Bare-integer expected signals substring-match anything

**Trap.** A structural task expecting a line number as the bare token `"22"` scores a hit against any response containing a "2". And line-number signals rot: the target file moves, `:22` becomes `:30`, and a now-correct answer is marked wrong.

**Fix.** Require line signals in `file.py:NN` form, never bare integers, and pin the eval suite to a source commit (store the sha in the suite; warn on mismatch at eval time).

**Rule.** An eval signal must be specific enough that only a correct answer contains it, and pinned enough that the moving target doesn't invalidate it.

### 7a. A comparative eval measures the whole pipeline — audit what each arm received

**Trap.** A "RAG hurts reasoning" result survived one root-cause pass (the stale index, §4b) and shipped with a plausible mechanism ("the retrieved chunks never contain the connecting hop"). A later audit *probed the actual injected context* and falsified it: the connecting answer **was present in ~half the injected blocks** — the model just couldn't assemble it from raw snippets (it relayed the same content 39/40 when handed as an explicit derivation). Under the surviving numbers sat five silent gaps: a dead lexical arm (§2b), an invisible file path, a flat block cut that dropped all but one chunk, a citation metric that couldn't see backtick-wrapped prose citations, and an all-YES suite half of whose signals could be scored by restating the question. None flipped the headline; all distorted sizes and mechanisms. Full anatomy: [`19 § 8.4`](19-evaluating-quality.md).

**Fix.** Before believing a comparative result: dump a sample of each arm's actual injected blocks; run the scorer against a handful of real answers by eye; check the suite's answer-polarity distribution; and treat sub-noise-floor deltas as ties per your own methodology.

**Rule.** The score table shows you the *ranking*; only probing the pipeline shows you the *mechanism*. Publish the mechanism only after you've watched the data flow.

---

## Harness & subprocess layer

### 8. Subprocess option injection via a user-controlled pattern

**Trap.** A search tool builds `["rg", "--json", pattern, path]` from a model-supplied `pattern`, guarded only by `re.compile(pattern)`. But `re.compile("--pre=<cmd>")` succeeds, and ripgrep's `--pre` runs a command per file — so the pattern reaches argv *as a flag* and is a local-code-execution chain under an agent that also writes repo files.

**Fix.** Put a literal `--` before the operands so nothing after it is parsed as a flag (`rg --json [--type t] -- pattern path`), and reject a `pattern` beginning with `-`.

**Rule.** Any user/model string that becomes a subprocess *positional* must be preceded by `--`. A regex-validity check is not an injection guard.

### 9. `str(path).startswith(root)` is not a sandbox

**Trap.** Gating file access with `str(resolved).startswith(str(REPO_DIR))` lets a sibling that shares the name prefix through: with root `/…/app`, the path `/…/app-secrets/creds` passes the check.

**Fix.** `resolved.is_relative_to(REPO_DIR)` (py3.9+), or compare against `str(root) + os.sep`.

**Rule.** Never sandbox a resolved path with a string prefix; use `Path.is_relative_to`.

### 10. Config rendered into a file the consumer never reads

**Trap.** The launcher rendered the permission gate + MCP-server config into one filename, but the vendored agent binary discovers config from a *different* set of paths — so the assistant launched with **no permission gate and no MCP servers**, silently. (Confirmed by grepping the binary's source for the filename: zero hits.)

**Fix.** Verify the field names *and* the discovery paths against the consumer's own source, not against a sibling tool's schema, and write to a path the consumer actually reads.

**Rule.** When you generate config for a program you didn't write, its *discovery path* and *schema* are both part of the contract — verify both against that program, not against what a similar program expects.

### 11. A strict-schema consumer rejects unknown bookkeeping keys

**Trap.** Having fixed #10 by writing to the right file, a clobber-guard stamped a `_generatedBy` marker into every config file. The agent binary strict-validates its config and **rejects the whole file on any unknown key** — silently disabling the very gate the fix installed (verified against the built binary: "unknown key … config files loaded 0/N").

**Fix.** Keep bookkeeping markers out of the files the strict consumer parses; put the fingerprint only in a file the consumer ignores, and write the parsed files clean.

**Rule.** Never inject your own keys into a file a strict-schema program parses — verify against the actual binary (`doctor` / config-dump), because a schema read from source is necessary but not sufficient.

### 12. An editable install shadows `PYTHONPATH`

**Trap.** Re-running an eval with a fixed scorer via `PYTHONPATH=/tmp/worktree python -m pkg.eval` returned scores *identical* to the old ones — because a modern editable install (uv / recent setuptools) registers an **import-hook finder that beats `PYTHONPATH`**, so `import pkg.eval` resolved to the *original* code, not the worktree. A single-file overlay also fails — the fixed file imports a module the old tree lacks (`ModuleNotFoundError`).

**Fix.** Assert *which* module ran (`inspect.getsource(fn)` / `module.__file__`) before trusting the result. To actually run patched code under an editable install, check out the whole branch in the install's target dir (a consistent tree), not a partial overlay.

**Rule.** When a result depends on which version of a module ran, verify it at runtime — under editable installs, `PYTHONPATH` does *not* win.

---

## Memory & robustness layer

### 13. A cross-namespace existence check must be scoped

**Trap.** The memory graph's "prune facts whose source file vanished" compared a fact's path against a **flat union of paths across all indexed repos**. A file deleted in repo A but still present at the same relative path in repo B was shielded from pruning forever.

**Fix.** Key the existence check per-repo (by the fact's own subject repo); keep a fact whose repo isn't in the scan (never over-prune).

**Rule.** When multiple sources share a namespace, any existence/dedup check must be scoped to the source, never a union across all of them.

### 14. Validate at the extraction boundary, and give the parallel path a serial fallback

**Trap (two small ones).** (a) The LLM fact-distiller accepted an item with no `subject_name`, defaulting to `""` — minting a junk `('convention', '')` entity every subjectless orphan collapsed into. (b) The parallel extractor ran `ProcessPoolExecutor(...).map(...)` with no `BrokenProcessPool` handling, so a single OOM-killed worker (common on a RAM-tight box) aborted the whole nightly job.

**Fix.** (a) Reject a proposed fact whose subject is empty/whitespace/non-string at parse time. (b) Catch `BrokenProcessPool` and fall back to the identical serial path.

**Rule.** Validate model-proposed data at the boundary where it enters your store, and treat a process pool as an optimisation with a mandatory serial fallback — never a correctness requirement.

---

## Strictly-local layer

### 15. Offline-by-default on every model-load path

**Trap.** HuggingFace / Transformers default to *online* (etag revalidation, cache-miss re-download). Any load path that doesn't set the offline flags — training, each gate-eval subprocess, the RAG embedder cold-start — can silently reach out mid-run, violating the strictly-local posture ([`01-overview.md`](01-overview.md)).

**Fix.** Default `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` (and pass `local_files_only=True` where the loader supports it) on *every* load path; the one sanctioned online path is the operator's explicit first-time bootstrap (`HF_HUB_OFFLINE=0`).

**Rule.** For a strictly-local stack, offline is the fail-closed default, opt-*out* for bootstrap — not opt-in. Cover every load path, and add an egress test that drives a *real* load, not a stubbed one.

## Agentic-training & watcher layer

### 16. Probe enum-ish columns before filtering on them — and make zero-yield self-explaining

**Trap (two halves).** (a) The trajectory miner filtered graph entities with `kind IN ('function','method','class')` — names guessed from convention. The store's real kinds are `symbol` / `module` / `file`, so the query matched **nothing** and generation produced zero rows. (b) Every skip happened *before* the emit path's counters, so the run reported all-zeros with no explanation — a burned debug cycle just to learn *where* it died.

**Fix.** (a) `SELECT DISTINCT kind` first; cite the probe in a comment next to the filter. (b) Count every pre-emit skip (`targets`, `no_window`, `no_grep`, `mined_commits`, …) so a zero-yield run carries its own diagnosis.

**Rule.** Never filter on enum-ish column values you haven't probed, and instrument generators so their *failure* output is as informative as their success output.

### 17. Replay edits against the parent state, not today's tree

**Trap.** Commit-replay training rows showed the model a window of the *current* file, then an edit whose old/new strings came from a *historical* commit. Any file that moved on since could lack the old string entirely — the row grounds a lie, teaching the model that edits apply to text that isn't in what it just "read". A second, quieter trap in the same generator: raw recursive grep swept nested-checkout junk (`.worktrees/…` duplicates) into results, wasting the truncation budget on noise — the same duplicate-content class that once silently doubled a retrieval index.

**Fix.** The miner already fetched the parent-state file to verify the hunk applies uniquely — carry that text through and window *it*. Grep through the shared exclusion list every walker uses.

**Rule.** Grounded training data must be *internally* consistent — every turn of a synthetic trajectory must be true relative to the same snapshot — and any tree walk anywhere (index, grep, miner) goes through one shared exclusion list.

### 18. A hung ssh freezes a watcher silently — bound every scripted session

**Trap.** A monitor loop polled a remote run with `$(ssh … grep …)` guarded only by `ConnectTimeout`. One session established and then stalled; the command substitution never returned, the loop froze mid-tick without ever firing its terminal event, and its half-open sessions starved every later ssh to the box — the watcher *became* the outage.

**Fix.** `-o ServerAliveInterval=5 -o ServerAliveCountMax=3` on every scripted ssh (a stalled session dies in ~15 s); prefer bounded single-shot fetches (grep a marker) over streaming whole logs; stop the previous watcher before arming the next.

**Rule.** A watcher must be strictly more reliable than the thing it watches: every remote call it makes needs a hard upper bound, and silence must be distinguishable from "still running".

### 19. Unmasked SFT on tool-RESULT formats teaches the model to emit results, not call tools

**Trap.** Training an agent on multi-turn tool trajectories, we put the harness's
verbose tool-RESULT payloads (read output, edit-success output) into the training
text — on the loss, unmasked. Offline the model scored beautifully. Live, under the
real prompt, it emitted an edit-RESULT payload (`<tool_response>{"path":…,"oldString":…,"newString":…}`)
*instead of calling the edit tool* — it had learned the result format so well it
generated the result rather than requesting it. The ordinary bare-JSON repair
can't recover a payload with no tool *name*, so nothing executed.

**Fix (two layers).** Immediate: a shim that salvages an edit-result-shaped payload
back into the edit CALL it meant — safe, because the real edit tool then validates
the old-string against the file. Real: mask the SFT loss to assistant-authored
tokens (calls + prose), or keep tool-result payloads out of the loss entirely.

**Rule.** In agent SFT, the tool RESULTS are the *environment's* tokens, not the
model's — never train the model to produce them. Mask assistant-only, or the model
role-confuses call↔result at inference.

### 20. An offline eval that doesn't reproduce the live prompt shape measures a phantom

**Trap.** The tool-use eval scored the model on short, training-shaped prompts and
reported 0.93/0.97. The live harness sends a ~7k-token system prompt, ~50 offered
tools, and `max_tokens=64000`. Under *that*, the same model ran away to context
truncation (25k tokens in one turn) and role-confused call↔result — neither
failure exists in the offline suite. The offline numbers were real and useless.

**Fix.** Model the harness's full live prompt in the eval (and in training): its
token length, its tool count, its `max_tokens`. Where you can't, bound the
pathologies in the harness — clamp `max_tokens`, restrict the offered toolset.

**Rule.** An eval's prompt distribution is part of the metric. If it doesn't match
what the model meets in production, a high score is a measurement of the wrong
thing — verify the top offline result on ONE real end-to-end run before trusting it.

---

## The meta-lesson

Half of these (#5, #10, #11, #12) were only caught by **running on the real target** — a stubbed test, a toy fixture, or a schema read from source would have missed every one. The other half (#1, #2, #8, #9, #13) are the kind a careful reviewer catches by asking "what's the *negative* case?" — the old term that should be gone, the sibling that shares a prefix, the null key, the union that should be scoped. Build both habits: **verify against reality**, and **test the negative**.

### 21. Greedy + identical prompt + identical server ≠ identical output

A server-side prefix cache evaluates cached and fresh prompts down slightly
different numeric paths; greedy argmax flips on knife-edge tokens. In a
multi-case eval this couples case N's outcome to the cases before it — and
shared-history reruns will happily "reproduce" the coupled result. Pin
`cache_prompt: false` (or your server's equivalent) in every eval harness, and
treat any per-case flip whose own prompt didn't change as a harness bug until
proven otherwise.

### 22. A repair layer that touches paths needs the harness's resolver

Our edit-failure hint (feed the file's closest real region back after a failed
exact-match edit) silently never fired in one harness: it tested the model's
raw relative path against the process cwd. The live agent always sent absolute
paths, so the gap was invisible for weeks — and the failures it would have
rescued were misattributed to the model. Context-free repair helpers should
take a `resolve=` hook; verification is the repair showing up in the run
trace, not the code reading correctly.

### 23. Index and eval trees rot from the inside

Three pollution sources each cost us a debugging round: a `.venv` full of
site-packages inside the repo tree (30 minutes of embedding someone else's
library code), a `.worktrees` directory duplicating every file (every top-k
list contained doubles), and `__tests__` slipping a `tests?` regex. Excludes
must be applied at BOTH layers — the tree copy and the indexer — and asserted
on the root-relative path, not the absolute one (a tmp dir named `test_*`
once excluded everything).

### 24. A loop variable shadowed our tokenizer and starved a training run

`tok = grep_token_for(...)` inside a generator loop rebound the function-scope
tokenizer; every later render failed and was silently counted as a routine
drop. Two training runs got far fewer rows of their most important class than
designed — discovered only when a rebalance made the class count land at
exactly zero. Name loop temporaries like temporaries, and make "zero rows of
the class this run exists for" a loud failure, not a counter.

### 25. Right file, wrong line still loses

Retrieval that lands the correct FILE can still lose the case: the line number
you inject steers the model's first read window, and a chunk-start line put
the target just outside it. Sharpen the injected line to the best-matching
line inside the winning chunk. The diff between "right file at line 1" and
"right file at line 24" was a solved case.

### 26. A chat template may re-render EARLIER turns

A newer model family's template strips or keeps reasoning blocks in *prior*
assistant turns depending on what follows them — so rendering a k-message
prefix is not a prefix of the k+1 render, and any incremental-render trick
(ours computed loss-mask spans that way) silently drops every row. The fix
that holds across a family: locate assistant turns by the template's own
structural markers in ONE full render, and fail loud unless the block count
matches the message count. Probe prefix-monotonicity before trusting
incremental rendering on any new template.

### 27. "Fits in 4-bit" is a load-time question, not a parameter-count estimate

An 8B model at 4-bit quantization refused to load on an 8 GB card that a 7B
had trained on comfortably — modules dispatched to CPU/disk, hard error —
with a resident service holding one more GiB than the estimate assumed. The
smoke gate answers fits-or-not in minutes and costs nothing; the download it
would have saved cost 23 GB. Related: a training framework's first load of a
new base may fetch a quantized companion repo — offline pins break there, so
give the run an explicit inbound-only escape hatch.

### 28. A "polish" epoch can scrub what the knowledge stage built

Sequential fine-tuning (knowledge stage on the corpus, then a low-lr
execution stage resuming the same adapter) is not conservative by default:
ours ended BELOW plain mixing on the knowledge score AND lost the flagship
execution capability, because the gentle second stage neither preserved
stage one nor delivered enough task dose to express the skill. Gate a
curriculum like any other candidate — ours was, and the gate said no.

### 29. A fixed synth cap turns "more synth" into "displaced synth"

Doubling a synthetic-data source changed nothing about how much of it trained: a
"majority-real" cap trimmed the extra rows before training, and within the fixed
synthetic slice the enriched source merely displaced *other* synthetic rows — so
the target metric stayed flat and the behavior the displaced rows carried
regressed. Before enriching any capped input, confirm the cap even lets the
extra through; otherwise enrichment is silently displacement.

### 30. A correct scaffold can move nothing — measure against failing cases

Two harness scaffolds targeted a clear failure taxonomy, were individually
correct (one fired on exactly the right cases), and moved the suite score by
zero: the failures never reached the step the scaffolds guarded. Pattern-matching
a taxonomy is not evidence; measuring against the failing cases is. Keep a
zero-gain-but-safe scaffold if you like, but don't ship it as an improvement.

### 31. A null result without a fired-signal is unreadable

Two augmentation arms went into an expensive eval; one recorded whether the
treatment executed, the other recorded nothing. Had its input relations been
missing, its 0 could not be told from "never ran." Instrument every arm
symmetrically (injected flag, call count) and census the treatment's inputs
before the run — or the null proves nothing.

### 32. A multi-repo graph grounds tasks into the wrong repo

Retrieval over a graph built from several repos, with no repo scoping,
injected paths from sibling repos — well-formatted, plausible, and nonexistent
in the task's workspace — on over half the cases. Scope grounding by repo, and
existence-check every injected ref against the workspace before injection.
"Fired" instrumentation does not prove "aimed."

### 33. Greedy decoding is not deterministic across serving configs

The same model, prompts, and temperature-0 sampling flipped three cases
between a CPU-served run and a GPU-served run: different floating-point
accumulation orders move near-tie argmax. Within one server, greedy is
reproducible; across serving configs it is not. Pin the serving config when
comparing against historical anchors.

### 34. A target-file-only judge hides wrong-file edits

The harness diffed and restored only each case's declared target file, so a
"landed" edit that hit a *different* file showed as "no diff" — and the stray
edit silently survived into later arms as workspace contamination. Mtime
forensics found them post-hoc. Either snapshot/diff/restore the whole
workspace, or at minimum detect and flag any write outside the declared
target.
