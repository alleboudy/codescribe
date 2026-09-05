# 23. Memory Operations & Lifecycle

**The operations companion to the Blueprint (22): turning a built memory
system into a *kept* one — one-command node setup, the four schedules,
consolidation nights, backups that protect the right bytes, and restores
that don't corrupt what they restore.**

Chapter 22 specifies what to build. This chapter specifies how it lives:
what runs when, what a fresh machine needs to become a node, what you back
up (and — more important — what you deliberately don't), and the handful of
operational traps that were each paid for in a real deployment. Like 22, it
is written to be handed to a coding agent and implemented as small, testable
steps.

---

## 0. The lifecycle at a glance

| When | Job | What it does | Cost when idle |
|---|---|---|---|
| every 15 min | **update tick** | check-then-act freshness pass (22 §6): fetch sources, rebuild only on real change | seconds |
| nightly | **semantic refresh** | incremental embedding index over new commits/tickets/docs | minutes |
| nightly, later | **consolidation ("dream") pass** | ingest new episodes → distil → reflect (dedup/merge/supersede/prune) | minutes |
| weekly | **backup** | snapshot + upload + retention | a minute |

Two rules make this table trustworthy:

1. **Every job is a no-op when nothing changed**, and says so in its log.
   A scheduler that does real work on every tick is either rebuilding the
   world (wasteful) or hiding a broken change-detector (worse).
2. **Every job appends to its own log file.** When something looks stale,
   the first move is always `tail <that log>` — never guesswork.

---

## 1. One-command node setup

A memory node has too many parts to assemble by hand twice: source
checkouts, a venv, an embedding model, first builds, agent registration,
and four schedules. Write ONE idempotent script — `setup-memory-node.sh` —
and let re-running it be the upgrade path.

The contract:

```
setup-memory-node.sh [--clone] [--no-build] [--no-schedule] [--dry-run]

1. prereqs     fail fast, name every missing tool in one message
2. sources     verify the source checkouts exist as siblings
               (--clone fetches them; the default reports and stops —
               never surprise-clone gigabytes)
3. venv        one lockfile-driven sync command
4. embedder    fetch the embedding model ONCE to a local dir; skip when
               present (this is the only permitted network fetch — the
               pipelines themselves run with offline flags, 22 §9)
5. first build the full graph pass, then the semantic bootstrap ONLY
               when the index file is absent
6. registration wire the MCP server for the operator's agent harness
7. schedules   per-OS: launchd plists on macOS, systemd user timers on
               Linux, schtasks on Windows — all four jobs from §0
```

Idempotence details that matter:

- Each phase **checks before it acts** and logs `ok:` or `doing:` — the
  second run of the script must print a column of `ok:` lines.
- `--dry-run` prints the plan and touches nothing. Agents should run it
  first and show you the output.
- The scheduler phase **re-installs** (bootout + bootstrap, or
  `systemctl --user daemon-reload && enable --now`) rather than skipping
  when present — schedules are cheap to rewrite and drift otherwise.
- End with a verification line the operator can copy-paste: the
  registration check and the log paths.

**Paid-for caveat — the unscheduled consolidator.** In one deployment the
consolidation loop was fully implemented, tested, and documented — and ran
on no machine for ten days, because a fleet re-arrangement moved every
other job and nobody re-homed the dream pass. The setup script is the fix:
if the script owns ALL four schedules, a node either has all of them or is
visibly not set up. Never wire schedules by hand, and audit with one
command (`launchctl list | grep <prefix>` / `systemctl --user list-timers`).

---

## 2. Consolidation nights: operating the dream pass

Chapter 14 covers why a memory dreams; 22 §5-§6 cover what the graph does
with the results. Operationally, a consolidation pass is three stages with
an audit trail:

```
INGEST    parse new episodes (session transcripts, ticket threads) into
          the episodes table; refresh the deterministic backbone
EXTRACT   distil undistilled episodes through a local LLM into candidate
          facts — ALWAYS below confidence 1.0 (22 §2: only deterministic
          extractors mint 1.0)
REFLECT   consolidate: dedup, merge, supersede contradicted facts, prune
          facts whose source files vanished
```

Operational requirements, each earned the hard way:

- **An audit row per run.** Write `(started, cursors, counts, duration)`
  to a `dream_runs` table. When someone asks "why does the graph believe
  X since Tuesday?", the audit row is the difference between an answer
  and a shrug. It also makes the pass **bisectable** — you can identify
  and delete exactly one night's output.
- **Reversibility.** The pass must be weight-free: deleting the facts and
  episodes it created restores the prior graph exactly. Never let a
  consolidation pass rewrite deterministic facts in place.
- **The remote distiller and the tunnel.** The distilling LLM often lives
  on another box (the one with the GPU). Do not hardcode that topology.
  Give the job an env hook (`DREAM_TUNNEL_HOST=<ssh-host>`): when set, the
  job opens an ssh tunnel for the run and closes it after — use a control
  socket (`ssh -f -N -M -S <sock> -L <port>:127.0.0.1:<port> host`, then
  `ssh -S <sock> -O exit host`) so the teardown is exact, never a pkill
  pattern-match.
- **Graceful degradation.** GPU box off → the tunnel step logs one line
  and the pass runs without distillation (INGEST and REFLECT still earn
  their keep). The next night retries. A consolidation pass that hard-fails
  when a peer is down turns "a box was off" into "the memory stopped".
- **Caps stay in config.** Synthetic-output ratio caps, confidence floors,
  diversity floors (22 §10): keep them in the dream config file, not in
  code, so an experiment is a config diff with an audit row — not a fork.

---

## 3. Backups: protect the irreplaceable, not the rebuildable

First, split the bytes honestly:

| Store | Rebuildable? | Back up? |
|---|---|---|
| deterministic code facts | yes — re-run the extractors | no (rebuild) |
| semantic index (embeddings) | yes — re-embed (slow but exact) | optional, rarely |
| **bitemporal fact history** (`valid_from`/`valid_to` chains) | **no** — a rebuild mints fresh current-truth with new timestamps | **yes** |
| **episodes** (ingested sessions/threads) | **no** — the originals may be gone | **yes** |
| **consolidation outputs** (distilled facts, audit rows) | **no** — they cost LLM nights | **yes** |

All three irreplaceables live in the graph db. So the minimal correct
backup is: **snapshot the graph db; treat the semantic index as optional.**

The snapshot recipe:

```bash
# 1. One consistent, compacted copy — folds the WAL in, needs no sidecars:
sqlite3 graph.db "VACUUM INTO 'snap/graph.db'"

# 2. Compress (text-heavy sqlite compresses 5-8x):
zstd -T0 -12 --rm snap/graph.db

# 3. A manifest makes the snapshot self-describing:
#    row counts, sha256 per asset, and the source-checkout positions
#    (git HEADs etc.) the graph was built from.
```

**Where snapshots go — and where they must not.** Never commit a snapshot
to the repository: a few hundred MB per week in git history bloats every
clone forever, and no retention policy can un-bloat it. Use your forge's
**release/artifact mechanism** (release assets on GitHub/GitLab/Gitea):
assets live on the repo, carry versions and tags, download with one
command, and keep clones small. Apply retention in the same script — keep
the newest N releases (delete tag + release together) and the newest few
local copies for fast restores.

Schedule: weekly is right for most teams — the update tick regenerates
everything rebuildable anyway, so the backup cadence only bounds how much
*history* a disaster can eat.

---

## 4. Restores: the discipline is in the sidecars

A restore script earns its keep on the worst day, so it must be boring:

1. **Fetch and verify.** Download by tag (or use the newest local copy),
   then check every sha256 against the manifest before touching anything.
2. **Move aside, never overwrite.** The live db becomes
   `graph.db.pre-restore-<timestamp>` — rollback is one `mv`.
3. **Delete the stale WAL sidecars.** This is the trap: sqlite pairs
   `graph.db` with `graph.db-wal`/`-shm`. Restore the main file while the
   OLD sidecars remain, and sqlite will happily replay the old WAL onto
   the new file — silently corrupting the restore with pages from the
   database you were trying to replace. `rm -f graph.db-wal graph.db-shm`
   after the swap, always. (`VACUUM INTO` snapshots carry no sidecars, so
   the restored file starts clean.)
4. **Prove it.** Run the health dashboard (22 §6) immediately; print the
   row counts next to the manifest's counts.

**Drill it once per quarter** — restore into a scratch directory, open the
db, run three real queries. An untested restore path is a rumor. And record
honestly which parts of the drill RAN versus which were only traced; the
distinction is the difference between a runbook and a wish.

---

## 5. Topology: one node or a fleet?

The Blueprint runs anywhere; the operational question is where the
builder, the store, and the consumer live. Three arrangements, in the
order teams usually try them:

1. **Fleet** (builder on a server, consumers pull replicas): survives any
   single laptop, but adds replica lag, a sync job, and a cross-node
   dependency for every consumer. Right when several people consume the
   same memory.
2. **Single node — build where you consume**: when the memory has exactly
   one consumer (one operator's agent sessions), hosting the builder on
   the consumer's machine deletes the replica lag and the sync machinery
   outright. The cost: the builder now walks the operator's *working
   trees*, so a dirty checkout skips its pull (loudly — 22 §6) and the
   graph tracks the tree as it is. That is often a feature — the operator's
   agent should know the operator's actual tree — but write it down.
3. **Hybrid**: single-node memory + a remote GPU peer for consolidation
   distillation only, reached through the tunnel hook (§2), degrading
   gracefully when the peer sleeps.

Whichever you pick: the setup script (§1) IS the topology document. A new
arrangement means editing one script, re-running it on each node, and
letting idempotence tear down what moved.

---

## 6. The operator's one-page cheat sheet

```
is it fresh?     tail the update-tick log; look for "nothing to do" ticks
did it dream?    tail the dream log; check the newest dream_runs row
is it backed up? list the release tags; check the newest manifest's counts
restore          restore-script --latest   (drill: into a scratch dir)
new source repo  add to the canonical source list; run one forced pass
node from zero   setup-memory-node.sh --clone
everything odd   one forced full pass, then the health dashboard
```

If a step here needs more than one command on your deployment, the
deployment — not the cheat sheet — is what needs fixing.
