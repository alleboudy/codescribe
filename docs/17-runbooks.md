# 17 — Operator runbooks

Operational procedures for building, auditing, and maintaining the stack once it exists. These are the "how do I actually *do* it" companions to the concept docs; the bug catalog they help you avoid is [`16-lessons-and-fixes.md`](16-lessons-and-fixes.md). Two machines are assumed throughout, by role (see [`11-hardware.md`](11-hardware.md)):

- **the trainer** — the GPU box that fine-tunes, dreams, and builds indices (idle most of the time).
- **the serving box** — the always-on host that serves the GGUF and hosts the indices the harness reads.

They share a LAN. Nothing here reaches the cloud.

---

## Runbook A — Auditing the stack with parallel read-only agents

When you want a broad correctness/security sweep of a codebase this size, fan out **read-only** agents by subsystem, then verify centrally. The discipline that makes it trustworthy: agents *find*, you *verify*, and nothing is filed or fixed on an agent's word alone.

1. **Partition by subsystem**, not by file — e.g. `{data + splitter}`, `{RAG store + servers}`, `{train + eval + deploy}`, `{memory + reasoning}`. One agent per partition, each with a tight scope and the house invariants (strictly-local, configs-are-data, loopback-by-default) in its brief.
2. **Read-only means read-only.** No writes, no state-changing commands, no network. Cap findings per agent (say 12) so they rank rather than dump.
3. **Every finding cites `path:line` and quotes the code.** A finding without evidence is a hypothesis.
4. **Verify each high-severity finding yourself** against the code before it becomes an issue — re-read the cited lines, and for anything you can reproduce cheaply, run a probe (a 20-line script that triggers the bug). Half the real bugs in [`16`](16-lessons-and-fixes.md) only proved out when reproduced.
5. **File, then fix on a branch.** Group tightly-related findings into one issue with a checklist; fix each with a regression test that fails before and passes after; reference the issue in the commit. Keep the full suite green.
6. **Report honestly, in three lists:** what you *ran* (with counts/output), what you only *statically traced*, and what remains *device- or human-only*. Never blend them — "verified" and "looks right" are different claims.

> Concurrency note: on a shared or rate-limited environment, cap concurrent agents low (2–3) and inline the rest. A parallel fan-out that dies mid-flight wastes more than the parallelism saves.

---

## Runbook B — Re-gating an eval on a single-GPU box without fooling yourself

You changed the scorer, the data, or the synthesis recipe and want to re-measure historical model generations. On an 8 GB card this is fiddly, and there are three ways to get a *false* result. Do it in this order:

1. **Free the whole card.** A 7B-4bit eval needs ~6.7 GB; a sibling small model or a serving process will crowd it out. Stop the other GPU user first; if the load offloads to CPU, a device-guard should fail loud (a CPU-resident layer hitting a CUDA-only kernel dies confusingly otherwise).
2. **Make sure the fixed code actually runs.** An editable install shadows `PYTHONPATH` ([`16 §12`](16-lessons-and-fixes.md)). Assert it: `inspect.getsource(scorer)` must contain your change, and `module.__file__` must point where you expect. If it doesn't, check out the whole fixed branch in the install's target dir — a single-file overlay breaks on the first new import.
3. **Skip the path that OOMs.** If a perplexity forward doesn't fit the card, gate on the task suite alone (which is what the synthetic data actually targets) and say so — a documented `skip_perplexity` is operational truth, not a shortcut to hide.
4. **Restore state on exit.** If you switched branches on the build machine to run the eval, use a trap so a failure can't leave the tree dirty: `trap 'git checkout "$ORIG"' EXIT`.
5. **Interpret honestly.** A scoring fix that moves absolute numbers may leave the *ordering* — and therefore every promote/discard decision — unchanged ([`16 §6`](16-lessons-and-fixes.md)). Re-measure before rewriting history.

---

## Runbook C — Two-box index lifecycle (build → verify → ship)

Both the vector index (`rag.db`) and the memory graph (`memory.db`) are **built on the trainer and served on the serving box**. The pattern is identical:

1. **Build on the trainer.** For code facts / chunks this is CPU-bound (AST/regex/chunking); embedding is the GPU part. Build to a **temp path**, never in place, so the live index is untouched until you've verified.
2. **Verify before swapping.** Check the counts are *sane* (non-zero code chunks, issues/PRs present if it's the RAG), that the pollution you were fixing is gone (e.g. `SELECT COUNT(*) … WHERE path LIKE '%<agent-scratch>/%'` returns 0, [`16 §4`](16-lessons-and-fixes.md)), and that FTS still answers (`… MATCH 'def'` returns hits, and the FTS integrity-check command doesn't raise).
3. **Back up, then swap.** Copy the live index aside (`cp file file.backup-<date>`) before overwriting — every swap must be reversible.
4. **Ship directly.** `rsync` the built index from the trainer straight to the serving box over the LAN — no third hop. On a fast local link this moves a ~½ GB index in well under a minute; don't assume it's slow and route it the long way.
5. **No restart needed if the reader is per-session.** If the harness spawns the retrieval server per session (rather than a long-lived daemon), the next session opens the fresh index automatically — nothing to restart. If it *is* a daemon, restart it (a read-only re-open is low-risk).

Leave the serving model itself alone — shipping an index must never touch the served weights.

**Know when a rebuild is due — don't wait to notice bad answers.** The failure mode of [`16 § 4b`](16-lessons-and-fixes.md) — an index full of chunks for files deleted from the repo — is invisible until it silently degrades retrieval (and any eval that compares against RAG). Instrument it: the index-status command should report a **staleness metric** — of the distinct indexed file_paths, how many no longer exist under the repo root — and **warn past a threshold** (~5% is ordinary churn; 50%+ means an incremental rebuild landed on a stale base). Run `status` after every ship and on a schedule; when it warns, do Runbook D or a clean rebuild. This is the detection half of the correction; the prune (self-healing full re-index) is the other half.

---

## Runbook D — Purging index pollution without a full rebuild

When an index has accumulated bad rows (e.g. the nested-worktree duplication of [`16 §4`](16-lessons-and-fixes.md)) but you can't or don't want to rebuild from scratch, you can **surgically delete** the offending rows — provided you respect the FTS5 ordering rule.

1. **Back up first** (`cp`), always.
2. **Load the vector extension** — a plain `sqlite3.connect` doesn't, so any statement touching a `vec0` table fails with `no such module: vec0`. Load `sqlite-vec` the way the store does before deleting from the vector mirror.
3. **Delete in the right order** for external-content FTS5 ([`16 §1`](16-lessons-and-fixes.md)): the FTS rows **first**, while the content rows are still present (so the correct terms are removed), then the vector rows, then the content rows:
   ```sql
   DELETE FROM code_chunk_fts     WHERE rowid IN (SELECT chunk_id FROM code_chunks WHERE file_path LIKE '%<scratch>/%');
   DELETE FROM code_chunk_vectors WHERE rowid IN (SELECT chunk_id FROM code_chunks WHERE file_path LIKE '%<scratch>/%');
   DELETE FROM code_chunks                         WHERE file_path LIKE '%<scratch>/%';
   ```
4. **Verify integrity**: run the FTS5 integrity-check command (`INSERT INTO fts(fts) VALUES('integrity-check')` — raises if corrupt) and a sanity `MATCH`.
5. **Prefer a proper rebuild when you can.** The surgical purge fixes the pollution but leaves the rest of the index at its last-built freshness. If your code source "re-indexes in full every run," a normal incremental index run with the fixed walker rebuilds the code rows clean *and* current — do that when the prerequisites (embedder present, tracker access) are available, and fall back to the surgical purge when they aren't.

---

## Runbook E — Verifying the strictly-local posture and a config consumer

Two things you must verify against *reality*, not against what you assume:

**The egress audit** (the posture from [`01-overview.md`](01-overview.md)). Run a full session — index build, a served-model query, a harness turn — under `strace -f -e trace=connect` and confirm **zero non-loopback `connect()` calls** (LAN links to your own serving box excepted, and named explicitly). Re-run it after wiring any new tool or sync step. Note that a stubbed embedder hides the real model-load egress — the audit must drive a *real* load ([`16 §15`](16-lessons-and-fixes.md)).

**The config consumer** (the trap from [`16 §10`](16-lessons-and-fixes.md) and [`§11`](16-lessons-and-fixes.md)). When you render config for an agent harness or any program you didn't write:
- Confirm the **discovery path** — grep the consumer's own source for the filename; if it's not there, the consumer never reads your file.
- Confirm the **schema** — field names and whether it strict-rejects unknown keys.
- Confirm **end-to-end** against the built binary: most harnesses have a `doctor` / config-dump subcommand that reports which config files loaded and how many. "Loaded 0/N" with an "unknown key" error means your file was rejected wholesale — a config read from source is necessary but not sufficient; only the binary's own report proves it.

---

## The one rule under all of these

Prefer **evidence over assertion** at every step: a reproduced bug over a suspected one, a verified swap over an assumed one, the binary's own report over a schema you read, three honest lists over one confident summary. Everything in [`16`](16-lessons-and-fixes.md) that bit hard bit *because* a plausible-looking assumption went unchecked.
