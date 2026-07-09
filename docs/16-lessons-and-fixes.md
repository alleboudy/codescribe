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

### 3. `split_salt: ~` becomes the literal salt `"None"`

**Trap.** `salt = str(config.get("split_salt", repo_name))`. A *present-but-null* YAML key (`split_salt: ~`) makes `.get` return `None` — the default only fires on a **missing** key — so `str(None)` salts every train/val/test split with the literal string `"None"`, silently diverging from the intended per-repo salt and setting up a train↔eval split mismatch.

**Fix.** `salt = str(config.get("split_salt") or repo_name)`.

**Rule.** For a nullable-with-fallback config key, use `cfg.get(k) or default`, not `cfg.get(k, default)`. The two differ exactly when the key is present and null.

### 4. Source walkers index nested agent-worktree checkouts

**Trap.** Both the RAG code chunker and the memory extractor exclude `.git`, `node_modules`, `vendor`, … but not a coding agent's scratch directory. Modern agent harnesses create **full git-worktree checkouts** of the repo under a hidden dir (e.g. `.claude/worktrees/<branch>/`). A walker that `rglob`s the tree descends into it and re-indexes an entire duplicate copy. Measured on a real index: **~49% of the graph was worktree-duplicate**, and provenance cited the transient worktree path instead of the canonical one.

**Fix.** Add the agent scratch dir (`.claude`, and any `*/worktrees/*`) to the exclusion set in *both* walkers.

**Rule.** A repo that self-hosts agent worktrees silently doubles every index that walks the tree. Exclude agent scratch dirs, and validate provenance paths against a *real* build, not a toy fixture — a toy fixture never has a nested worktree.

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

---

## The meta-lesson

Half of these (#5, #10, #11, #12) were only caught by **running on the real target** — a stubbed test, a toy fixture, or a schema read from source would have missed every one. The other half (#1, #2, #8, #9, #13) are the kind a careful reviewer catches by asking "what's the *negative* case?" — the old term that should be gone, the sibling that shares a prefix, the null key, the union that should be scoped. Build both habits: **verify against reality**, and **test the negative**.
