# 22. The Memory-System Blueprint

**A from-scratch, agent-implementable specification for a deterministic
code-memory graph over enterprise sources — Perforce, Bugzilla, git, the
build/test toolchains, requirements & specification portals, and OneDrive
documents —
with a semantic sidecar, a fine-tuned embedding model for your
organization's jargon, and the validation harness that keeps all of it
honest.**

This chapter is written to be handed to a coding agent (Visual Studio /
VS Code Copilot in agent mode, or any equivalent) and implemented milestone
by milestone. Every module has a contract, every caveat here was paid for in
a real deployment, and every milestone ends in tests an agent can run. A
reference implementation of the core exists as open source
([groundgraph](https://github.com/alleboudy/groundgraph), MIT, zero runtime
dependencies); this blueprint extends it to enterprise sources and the
tuned-embedder tier.

---

## 0. How to use this document with an implementing agent

1. Feed the agent **one milestone at a time** (§12), never the whole
   document as a single task. Each milestone names its modules, its tests,
   and its acceptance gate.
2. Enforce **RED first**: the agent writes the milestone's failing tests
   before the implementation, and shows you the failure. A test that never
   failed proves nothing about the code it guards.
3. Enforce the **fired-signal discipline** (§10) from milestone one: every
   lever the system grows must record whether it actually executed. You
   will thank yourself at evaluation time.
4. Keep the agent inside the **module boundaries** (§1). Each module has
   one responsibility and a typed contract; an agent that edits three
   modules for one feature is drifting.

---

## 1. Thesis and architecture

**A smaller graph of facts you can prove beats a bigger graph of facts you
can vibe.** Most agent-memory systems ask a model to summarize code and
conversations into "memories"; the graph inherits hallucination,
cross-project contamination, and precision rot that grows with scale. This
system takes the opposite bet: the pipeline is **100% deterministic** —
parsers, version-control plumbing, and mechanical derivations with proof
paths. Generation is for *answering* questions, never for *remembering*.

The neural layer exists, but in exactly one role: **the embedder proposes,
the graph verifies.** Semantic recall finds candidate sites the lexical
layer cannot name; the symbolic layer confirms they exist, scopes them to
the right repo and workspace, and expands them into verified, provenance-
carrying context.

```
sources                     substrate                        consumers
───────                     ─────────                        ─────────
git ────────┐
Perforce ───┤   extractors      ┌──────────────┐   derivations
Bugzilla ───┤──(deterministic)──▶ entities     │──(proof-carrying)──┐
toolchain ──┤                   │ facts        │◀───────────────────┘
specs ──────┤                   │ provenance   │
OneDrive ───┘                   │ embeddings   │
tuned embedder ──(proposes)────▶└──────────────┘──(anti-rot dashboards)
(local /v1/embeddings)                 │
                                       ▼
                    query API · explain dossiers · pre-injection assist
                    agentic tools · MCP server (Copilot / any MCP client)
```

One SQLite file. Three trust tiers. Everything else is an adapter.

| tier | extractor prefix | examples | decay |
|---|---|---|---|
| ground truth | `det:` | AST edges, Perforce co-change, Bugzilla fix links, test outcomes, doc rules | never — re-verified every build |
| grounded derivation | `der:` | inverse edges, transitive closures, exception flow — each with a proof path | never — re-derivable |
| risky / inferred | anything else | model-distilled facts, if you ever add them | halves every 60 days unless re-corroborated |

---

## 2. The store

One SQLite file, WAL mode, foreign keys on. Three core tables plus the
semantic sidecar's vectors:

```sql
CREATE TABLE entities (
  entity_id    INTEGER PRIMARY KEY,
  kind         TEXT NOT NULL,   -- file | symbol | module | repo | bug | changelist
                                -- | doc | lesson-topic | test-run
  name         TEXT NOT NULL,   -- canonical / qualified name
  repo         TEXT,            -- source label (NULL for cross-source entities)
  path         TEXT,            -- file path for file/symbol-bound entities
  meta         TEXT,            -- optional JSON
  first_seen   TEXT NOT NULL,   -- ISO-8601
  last_seen    TEXT NOT NULL,
  UNIQUE(kind, name, repo)
);

CREATE TABLE facts (
  fact_id       INTEGER PRIMARY KEY,
  subject_id    INTEGER NOT NULL REFERENCES entities(entity_id),
  predicate     TEXT NOT NULL,
  object_id     INTEGER REFERENCES entities(entity_id),
  object_lit    TEXT,
  confidence    REAL NOT NULL,
  extractor     TEXT NOT NULL,   -- 'det:ast@1' | 'det:p4-cochange@1' | 'der:inverse@1' ...
  created_at    TEXT NOT NULL,
  valid_from    TEXT NOT NULL,
  valid_to      TEXT,            -- NULL = current; set on supersede (SOFT, never DELETE)
  superseded_by INTEGER REFERENCES facts(fact_id),
  UNIQUE(subject_id, predicate, object_id, object_lit, valid_from),
  CHECK ((object_id IS NULL) != (object_lit IS NULL))   -- object XOR
);

CREATE INDEX idx_facts_pred_obj  ON facts (predicate, object_id, valid_to);
CREATE INDEX idx_facts_subj_pred ON facts (subject_id, predicate, valid_to);
CREATE INDEX idx_facts_valid_to  ON facts (valid_to);

CREATE TABLE provenance (
  prov_id      INTEGER PRIMARY KEY,
  fact_id      INTEGER NOT NULL REFERENCES facts(fact_id),
  source_kind  TEXT NOT NULL,   -- code_line | commit | changelist | bug | doc | test_run | derived
  source_ref   TEXT NOT NULL,   -- 'src/app.py:142' | 'p4:812345' | 'bz:44121' | webUrl | ...
  excerpt      TEXT,
  created_at   TEXT NOT NULL
);

CREATE TABLE embeddings (        -- the semantic sidecar (§8); entity-keyed
  entity_id   INTEGER PRIMARY KEY REFERENCES entities(entity_id),
  model       TEXT NOT NULL,
  dim         INTEGER NOT NULL,
  vec         BLOB NOT NULL,     -- little-endian float32, L2-normalized
  src_hash    TEXT NOT NULL,     -- sha256 of the embedded text (idempotency)
  created_at  TEXT NOT NULL
);
```

**Store invariants — every one of these was violated once and cost real
debugging time:**

- **Supersession is SOFT.** `valid_to` + `superseded_by`, never DELETE. The
  old row survives for `as_of` time-travel and audit. Pruning (source file
  vanished) archives the same way, with no winner.
- **NULL-safe everything.** SQLite's UNIQUE treats NULLs as distinct, so
  entity lookups and fact dedup must SELECT with `IS ?` before inserting,
  or you mint duplicate entities for every NULL-repo row.
- **Path backfill, never clobber.** An entity created without a path gains
  one when a later fact supplies it; a populated path is never overwritten.
- **Flip-flop reopen.** A value that goes A→B→A within one run re-asserts a
  row that was soft-closed earlier in the same run (dedup includes
  `valid_from`); reopen it, or the slot ends with zero current rows.
- **Idempotent builds.** Re-running over unchanged sources writes zero new
  facts (dedup on the current triple). This is what makes an event-driven
  scheduler free.
- **The deterministic tier outranks.** In a single-valued slot, a `det:`
  fact is never retired by a lower-tier candidate. Corroboration raises
  confidence but caps at 0.99 — the 1.0 floor is reserved for ground truth.
- Writes run inside transactions; a batch context manager wraps bulk
  ingest. One DB file; never split it.

---

## 3. Source adapters — the contract

Every source plugs in through the same narrow contract:

```
adapter(source_config) -> Iterable[ExtractedFact | ProposedFact]
```

- `ExtractedFact` — deterministic code facts with file:line provenance,
  written at confidence 1.0.
- `ProposedFact` — everything else (relations, lessons, links), carrying
  `origin` (becomes the provenance `source_ref`), `confidence`, `extractor`,
  and an evidence `excerpt`. Consolidation (§2 invariants) does dedup,
  corroboration, and conflict resolution.

Adapter rules (uniform across all five sources):

1. **Fail-closed.** A malformed input the adapter does not recognize yields
   an error or zero facts — never a guessed fact. (A file line before any
   commit header in a `git log` parse raises; an unparsable Bugzilla
   comment yields nothing.)
2. **Incremental by cursor.** Every adapter records a sync cursor
   (commit SHA, changelist number, `last_change_time`, delta token) and
   resumes from it. Full re-ingest must be idempotent anyway.
3. **Provenance or it didn't happen.** Every fact names its exact source:
   `path:line`, `p4:<CL>`, `bz:<bug-id>`, the document's URL, the test-run
   identifier.
4. **Bounded blast radius.** Any source event touching more than N items
   (a 50-file changelist, a mass-edited wiki page) is skipped whole for
   pair-mining relations — mechanical churn fabricates coupling that isn't
   there (pair count grows as C(n,2)).
5. **Walker hygiene.** Tree walkers share ONE exclude list and prune any
   nested checkout (a subdirectory containing `.git`, or a Perforce client
   root inside another) *by construction, not by name* — a nested checkout
   is duplicate content whatever it is called. This once silently doubled
   an index.

### 3.1 git

- **Code facts**: Python via the stdlib AST (defined-in with line numbers,
  calls, imports, inherits, raises, decorated-by); other languages via
  line regexes for imports + declarations (C/C++/C#/Swift/Kotlin/TS/...).
  Do NOT regex-guess cross-language call graphs — that violates the
  ground-truth mandate. Syntax errors return zero facts for that file.
- **Co-change**: parse `git log --name-only --pretty=format:%H -n <cap>`.
  Pure parse first (testable on a string), git second. Blast-radius bound
  (default: skip commits touching >12 files), support threshold (default:
  pair must co-occur ≥3 times), code-extensions-only filter, symmetric
  emission (store BOTH directions — the graph is traversed
  subject-anchored, and a canonical-order-only edge makes half the lookups
  miss). Confidence rises with support, capped below 1.0.
- **Freshness**: `git rev-list --count HEAD..@{u}` per repo; `None` (not 0)
  when there is no upstream — an honest unknown beats a guessed zero.

### 3.2 Perforce

Same shapes, different plumbing:

- **Co-change**: `p4 changes -s submitted -m <cap> //depot/path/...` for
  the changelist list, then `p4 describe -s <CL>` (or `p4 files @=<CL>`)
  for the file set. Apply the SAME blast-radius bound per changelist —
  branch/integrate changelists routinely touch thousands of files and MUST
  be skipped for pair mining (check for `integrate`/`branch` actions in
  the describe output and skip those CLs entirely).
- **Path mapping**: facts store *client-relative* paths (the workspace the
  agent actually opens), not depot syntax. Resolve through the client view
  (`p4 where`) once per sync and cache. A fact whose path the agent cannot
  open is worse than no fact (§7).
- **Fix links**: `p4 fixes -j <job>` / `p4 fixes -c <CL>` yields
  job↔changelist links for shops using p4 jobs — these become
  `bug fixed-by changelist` facts at `det:` tier with `p4:<CL>` provenance.
- **Cursor**: the highest ingested changelist number (monotonic).
  **Freshness**: compare the client's have-list head
  (`p4 changes -m1 @<client>`) against depot head (`p4 changes -m1`);
  report the gap in changelists.
- **Caveats**: `p4 describe` on a huge CL is slow — always `-s` (no diffs);
  shelved changelists are not submitted state (skip); renamed/moved files
  appear as delete+add pairs (do not mint co-change between a file and its
  own new name — detect `move/add`+`move/delete` pairs and rewrite to one
  identity).

### 3.3 Bugzilla

Bugs are first-class entities (`kind=bug`, `name=bz:<id>`), and the richest
source of *symptom vocabulary* you have — which is exactly what the
embedding fine-tune (§8) needs.

- **Sync**: the REST API (`/rest/bug?last_change_time>=<cursor>` for
  incremental; `include_fields` to keep payloads small). Respect rate
  limits; page.
- **Facts** (all `det:` tier — these are records, not inferences):
  - `bug affects component` (Bugzilla's own component field),
  - `bug fixed-by changelist|commit` — from `p4 fixes` where jobs are
    wired, else from a STRICT comment regex (`(?:fixed in|submitted as)\s+
    (?:CL\s*)?(\d{4,})` and commit-SHA patterns). Strict means strict: a
    miss yields no fact, never a guess.
  - `bug touches file` — derived (`der:`) by joining fixed-by to the
    changelist's file set. This is the join that turns bug text into
    code-anchored training pairs.
  - `bug duplicate-of bug` — from Bugzilla's dupe field; follow chains to
    the terminal bug, cycle-safe, and store only the terminal link.
- **Lessons**: resolution comments mined with deterministic markers ONLY
  (`Root cause:`, `Fix:`, `Workaround:` at line start). Strip quoted reply
  blocks (`> ` prefixes) BEFORE mining — quoted text repeats markers and
  once minted the same lesson five times from one thread.
- **Caveats**: security-restricted bugs will 401/403 — skip them *silently
  is wrong*; count them and surface the count in the sync report (an
  access-shaped hole in the graph should be visible). Comment 0 is the
  description. `cf_*` custom fields vary per instance — treat unknown
  fields as absent, never crash.

### 3.4 Build/test toolchains — the generic toolchain adapter

Your build/test/analysis toolchain — a CI system, a compiler plus a test
runner, a static analyzer — emits structured results, and the implementing
agent has your toolchain's own output formats in front of it. So this section
specifies the ADAPTER CONTRACT rather than guessing formats; the same contract
fits any toolchain that emits structured results.

- **Discover the structured outputs first.** Enumerate what the toolchain writes
  per run — test-result files, static-analysis reports, build logs — and
  their formats (JSON/XML/log). Write ONE parser per format, pure (parses
  a string/bytes, no I/O), so every parser is testable on captured
  fixtures. Capture real fixture files into the test suite on day one.
- **Facts to emit** (all `det:toolchain@1`, run-scoped provenance
  `toolrun:<run-id>`):
  - `test-run passed|failed test-symbol` — outcomes CORROBORATE the static
    `tests` relation (§5): a test that exists and passes is stronger
    evidence than a test that merely exists. On failure, the message +
    file:line ride in the excerpt.
  - `diagnostic` facts for analyzer/compiler findings with file:line.
    Bound them (a legacy target can emit tens of thousands): cap per run,
    keep the newest run per target, and soft-supersede the previous run's
    diagnostics — they are *state*, not history; the supersession model
    handles this for free.
- **Caveats**: version-detect the report schema and fail closed on unknown
  versions (toolchains change formats between releases — a silent
  best-effort parse mints wrong facts); parse and discard artifacts, never
  store bundles/logs in the graph; distinct run configurations (target,
  platform, debug/release) are distinct run entities, or a passing release
  run will supersede a failing debug run's evidence.

### 3.5 Requirements & specification portals

Requirements and specs are the vocabulary bridge between "what the product
must do" and "where the code does it" — and a requirements-management portal's
typed work-item links make that bridge deterministic. Most enterprise
specification tools expose the same shape over a REST API: work items with
typed links to revisions, tests, and defects.

- **Sync**: the portal's REST API, per project: work items with
  `include_fields`-style narrowing, paged, incremental via each item's
  `updated` timestamp as the cursor. Auth by personal access token in the
  OS credential store (§9). Where the portal renders live spec documents
  that CONTAIN work items, ingest the work items through the API, not the
  rendered document HTML.
- **Entities**: `kind=spec-item`, `name=<SPEC-ID>`, the title in `meta`.
- **Facts** (all `det:spec@1`, provenance = the work-item URL):
  - typed link facts from the portal's own typed link roles: `spec-item
    implemented-by changelist|commit` (from revision links / linked
    revisions), `spec-item verified-by test-case`, `spec-item links-to
    bug` (Bugzilla cross-links), parent/child structure as `part-of`.
  - `spec-item touches file` — **derived** (`der:`) by joining
    implemented-by to the changelist's file set (§3.2), exactly like the
    Bugzilla join. This turns requirement language into code-anchored
    context and training pairs.
  - spec section rules mined as lessons ONLY where the text carries the
    deterministic markers (§3.6-style); requirement prose is otherwise
    indexed for the semantic tier, not force-fit into lesson facts.
- **Caveats**: link ROLES are per-instance configuration — enumerate the
  instance's actual role IDs at sync start and map them explicitly; an
  unmapped role is skipped and COUNTED, never guessed. Work items are
  revisioned — use the revision as `valid_from` and soft-supersede on
  update (state, not history). Suspect/stale link flags (most portals expose
  one) mean
  the link needs human re-validation — carry the flag into `meta`, and
  exclude suspect links from `det:` (drop to a lower-confidence tier).
  Access-restricted spaces 403 like Bugzilla's private bugs: count and
  surface, never silently skip.

### 3.6 OneDrive documents

Engineering knowledge that never made it into the repo: design docs,
postmortems, runbooks.

- **Sync**: Microsoft Graph — `GET /drives/{id}/root/delta` with a stored
  delta token (incremental, exactly what the cursor contract wants). Only
  folders the operator allow-lists; record each document's `webUrl` as
  provenance.
- **Extraction**: `.md`/`.txt` directly; `.docx` WITHOUT a dependency — a
  docx is a zip of XML (stdlib `zipfile` + `xml.etree`): map paragraph
  styles `Heading1..3` to section headings, then run the SAME
  lesson-mining used for repo markdown (bolded rules, `Rule:`/`Why:`
  markers, `file.ext:NN` citations). PDFs are out of scope for the
  zero-dependency core; route them through a local converter if you must.
- **Heading formats**: support markdown `#` headings AND
  reStructuredText-style underlines AND docx styles through ONE
  section-splitter with three detectors — sources will surprise you.
- **Caveats — read these twice**:
  - **Auth**: device-code flow; tokens in the OS credential store, NEVER in
    the graph, NEVER in config files. The graph must be shareable without
    leaking a secret.
  - **Permission boundary = index boundary.** Index only what the syncing
    identity can read, and record the `webUrl` so consumers hit the
    document's own ACL when they follow provenance. Do not copy restricted
    document *content* into excerpts beyond the mined rule sentence.
  - **Versioning**: use `lastModifiedDateTime` as `valid_from`; a re-edited
    document soft-supersedes its previous facts (state, not history).
  - Export formats lie: "docx" from some tools is actually HTML — sniff
    the zip magic bytes (`PK`), fail closed otherwise.

---

## 4. Jargon: the alias layer

Every organization has vocabulary drift: British/American spellings,
project code-names vs product names, `HUD` vs `overlay`, an internal
abbreviation per subsystem. Two mechanisms, layered:

1. **A deterministic alias map** (config, reviewed by humans):
   `{"colour": "color", "hud": "overlay", "<codename>": "<component>"}`.
   Applied word-wise to BOTH sides of every lexical match — task terms AND
   indexed text. A one-sided fold silently blanks the lever (a real field
   miss: tasks folded to "color" while the notes said "colour", and the
   lesson-matching stage returned nothing for weeks).
   Seed it from your docs' glossary; grow it from Bugzilla dup clusters
   (two bugs marked duplicates with disjoint vocabulary are an alias
   signal — propose, human-review, then add).
2. **The tuned embedder** (§8) — the statistical generalization of the
   alias map. The map stays even after tuning: deterministic, reviewable,
   and free.

Also in this layer: keep snake_case/camelCase identifiers WHOLE as match
terms in addition to their split words — naive word-splitting turns
`url_for` into `url` + a stopword and the symbol becomes unmatchable.

---

## 5. Relations catalogue

**Extracted (`det:`)** — `defined-in` (file:line), `calls`, `imports`,
`depends-on`, `inherits`, `raises`, `decorated-by`; `tests` (fail-closed:
the test must IMPORT the subject AND REFERENCE it inside the test body —
never a naming guess; deny stdlib/test-stack/famous-library imports, but
let a repo's OWN package override the deny list — indexing a famous
library's own repo must not deny `import <itself>`); `co-changed-with`
(git §3.1 / Perforce §3.2); `fixed-by` / `touches` / `duplicate-of`
(Bugzilla §3.3); test outcomes and diagnostics (§3.4); `lesson` /
`because` / `cites` (docs §3.6).

**Derived (`der:`)** — each cycle-safe, depth-bounded, fan-out-capped, and
**logging its truncations** (a silent cap reads as "covered everything"):

- inverse edges: `called-by`, `imported-by`, `defines`, `raised-by` —
  materialized because "who uses this?" is the #1 agent query;
- transitive `isa` from `inherits` (depth ≥2 only; 1-hop already exists);
- module-level `depends-on` aggregated from cross-file symbol edges
  (support count in the proof);
- bounded call `reaches` with the path as proof;
- `may-raise`: `A calls B calls C, C raises E ⇒ A may-raise E`, confidence
  decaying per hop, full call path in the excerpt;
- transitive module depends — **phase 2**: it reads the module facts phase
  1 produces, so it runs after phase 1 consolidates. Ordering is data
  dependency, not convention.

---

## 6. Anti-rot: the graph watches itself

A graph that grows in count while rotting in precision is a regression.

- **Query-time decay**: `det:`/`der:` never decay; anything else halves per
  60-day half-life at QUERY time. Nothing is destroyed; a stale inferred
  fact sinks below the recall floor until re-corroborated.
- **Health dashboard** (run after every build; warn thresholds in
  parentheses): live/superseded counts; tier ratio (risky >40% of live =
  precision risk); exact duplicates (>5% = consolidation lagging); orphan
  facts (empty-named subjects — junk nodes); decay-dormant count;
  **contradiction candidates restricted to single-valued predicates
  ONLY** — multi-valued relations (`calls`, `imports`, `raises`...)
  legitimately hold many objects per subject, and a naive detector once
  raised 12,000+ false alarms before this restriction.
- **Source freshness**: the graph is only as current as its checkouts. Per
  source: git commits-behind-upstream; Perforce changelists between client
  have-head and depot head; OneDrive delta-cursor age. From plumbing only,
  with honest `unknown` when the tool cannot answer — never a guessed 0.
  This metric exists because a production graph was once found built on a
  checkout **hundreds of commits behind** and no internal metric could see
  it.
- **The watch daemon** — the freshness metric turned into maintenance.
  Check-then-act: fetch/compare; when nothing is behind, do nothing (a
  near-free no-op); when something moved, fast-forward the CLEAN repos
  only — **a dirty working tree is never touched** (no stash, no reset;
  the operator's uncommitted work is sacred and the skip is logged
  loudly) — then rebuild and re-run the dashboard. A **single-instance
  lock** (fcntl / a Windows named mutex) makes an overlapping pass skip
  cleanly instead of stacking writes: a rebuild can outlast the scheduler
  interval. Schedule with cron / launchd / **Windows Task Scheduler**
  (`schtasks /create /sc minute /mo 15 ...`) — and verify the scheduler
  daemon itself is actually running; an installed schedule on a dead
  scheduler is theatre (a real deployment ran "scheduled" for weeks on a
  box whose cron daemon had never started).
- **Stale pruning**: facts whose source file vanished *from their own
  repo* soft-archive on every build. Repo-aware: a file deleted in repo A
  but present at the same relative path in repo B must not shield A's
  stale fact.

---

## 7. Serve-time: how agents consume it

**Query API** (read-only connection, always): `query_facts(subject,
predicate, object, repo, min_confidence, as_of, limit)` and
`explain_entity(name, repo)` — a depth-1 dossier (outgoing + incoming
facts) resolved by entity id, with a repo preference so same-named symbols
in different repos never swap dossiers.

**Pre-injection assist**: ground the task's terms (alias-folded, §4)
against defined symbols; expand the top candidates through their
neighbourhoods (called-by / calls / tests / raises / file-level co-change /
fixed-by bugs) and matching lessons; prepend ONE compact block. Rules that
were each paid for:

- **A wrong file is worse than none.** Weak matches degrade to a loud
  no-op (task returned unchanged). A misgrounded injection actively steers
  the agent into editing the wrong file — observed repeatedly.
- **Repo scoping**: a multi-repo graph must not ground a task against a
  sibling repo's paths. (Field measurement, before the fix: over half the
  evaluation tasks received mostly nonexistent-in-workspace paths.)
- **Workspace existence-check**: drop any ref whose path does not exist in
  the agent's working directory — BEFORE injection, not post-hoc. For
  Perforce, this is also your client-view safety net.
- **Nothing raises into the agent loop.** Missing DB, partial schema,
  malformed model-supplied arguments (`"limit": "many"` happens) — every
  path degrades to a no-op or an explicit error string. One uncaught error
  aborts an entire evaluation batch.
- **Co-change is stored on FILE entities.** Query it by the candidate's
  file path, not its symbol name — a symbol name never resolves to a
  file-kind entity, and the wrong-kind query is a *silent* no-op that
  shipped once and returned nothing on every task while looking healthy.

**Agentic tools**: the same two queries exposed as OpenAI-style function
schemas, executed in-process, read-only. Log per-task how often the model
actually calls them — in a measured deployment the model consulted memory
on fewer than a third of tasks unprompted, and you cannot see that without
the counter.

**MCP server** — the integration path for Visual Studio / VS Code Copilot
and any MCP client. Newline-delimited JSON-RPC 2.0 over stdio; implement
`initialize` (echo the client's protocolVersion), `ping`, `tools/list`,
`tools/call`; notifications get no response; unknown methods get `-32601`;
a malformed line gets a parse error and the loop continues. Tools:
`query_facts`, `explain_entity`, `assist` (returns the grounded prompt PLUS
its fired-signal report). Keep stdout pure JSON-RPC; logs to stderr. VS
Code: add to `.vscode/mcp.json`; Visual Studio: the MCP servers
configuration UI. The server opens the store read-only per call and can
never write.

**Instrumentation** (§10 depends on this): every assist returns
`{injected, grounded_symbols, lexical_hits, semantic_hits}` so any
evaluation can attribute every grounding to its proposer.

---

## 8. The semantic sidecar and the tuned embedder

### 8.1 Sidecar mechanics

- Embed a source window (≈30 lines) around every defined symbol, plus every
  lesson text, via any **local** OpenAI-compatible `/v1/embeddings`
  endpoint (llama.cpp server with an embedding model, Ollama, LM Studio).
  Vectors are L2-normalized float32 blobs in the `embeddings` table,
  **keyed to graph entities** — a semantic hit IS a graph node, so it
  expands through the same verified `explain_entity` neighbourhoods.
- Idempotent re-index via `src_hash`; a model/dim change is detected per
  row and mismatched rows are skipped honestly (embeddings are not
  comparable across models — re-embed all on a model swap).
- Query: cosine over the store (brute force is fine to ~100k entities;
  `numpy` if present, pure Python otherwise). Proposals pass the SAME repo
  scope + workspace existence-check as lexical grounding, then merge after
  lexical hits.
- **Endpoint down ⇒ lexical-only, logged, never a crash.** Index time is
  an explicit operator action and MAY fail hard; query time never.

Why hybrid, in one measured sentence: dense retrieval lands on *consumers*
and prose-adjacent code the lexical layer cannot name (it was the only
mechanism that ever automatically solved symptom-phrased evaluation cases),
while the graph supplies definitions, relations, tests, and proofs the
embedder cannot — each is the other's missing half.

### 8.2 Fine-tuning the embedder on YOUR jargon

Off-the-shelf embedders don't know your code-names, subsystem
abbreviations, or that "the widget flashes" means the render throttle. The
fix is a contrastive fine-tune on pairs mined **deterministically from the
graph you already built** — no annotation project:

**Positive pair mining (all from `det:` facts):**

| pair | source | why it teaches jargon |
|---|---|---|
| bug title+description ↔ source windows of files the fixing CL touched | Bugzilla × Perforce `fixed-by`/`touches` joins | maps SYMPTOM vocabulary to code — the highest-value pairs you own |
| requirement title+text ↔ source windows of implementing-CL files | spec-portal × Perforce `implemented-by`/`touches` joins | maps SPEC vocabulary to code — the compliance-grade pairs |
| commit/CL message ↔ changed-file windows | git / `p4 describe` | maps intent language to code |
| doc section ↔ its `cites` file:line windows | lesson extraction | maps design/runbook language to code |
| test name + docstring ↔ subject-symbol window | the `tests` relation | maps behavior names to implementations |

**Hard negatives**: same-module different-symbol windows (near-misses teach
the boundary); co-change partners of the positive file are *ambiguous* —
exclude them from negatives entirely rather than teaching the model that
coupled files are unrelated.

**Recipe** (single consumer GPU, hours not days): start from a small open
embedding model; train with a multiple-negatives ranking loss (in-batch
negatives + the mined hard negatives), LoRA or full fine-tune, low epochs
(1–2), modest LR — embedders overfit fast on small corpora. Serve the
result behind the same `/v1/embeddings` interface so the sidecar never
knows which model it is talking to.

**Evaluation — this is where fine-tunes lie to you:**

- **Split by FILE PATH, never by chunk** — chunks of one file across
  train/eval is leakage that inflates every metric.
- **Split bug pairs by TIME** — train on older bugs, evaluate on newer;
  bug language drifts and future-bug leakage flatters recall.
- Metric: recall@k / MRR on held-out symptom→file pairs; report the
  off-the-shelf baseline in the SAME table; **promote only if the tuned
  model beats baseline on held-out** — a gate, not a vibe. Keep the frozen
  eval set under version control with a checksum.
- End-to-end check: re-run your assist evaluation (§10) with the tuned
  embedder as one arm. Retrieval-metric gains that do not move
  grounded-task performance are real but bounded information — report
  both.

---

## 9. Security and privacy posture

- **Strictly local by default**: the graph, the vectors, and the embedder
  run on hardware you control; the only network egress is to the
  enterprise APIs you configured (Perforce, Bugzilla, Microsoft Graph) and
  a loopback embeddings endpoint.
- **No secrets in the graph, ever.** Tokens live in the OS credential
  store; config files carry endpoint URLs only. The store must be
  shareable with a teammate without a leak-scan panic — enforce it with a
  pre-share scan for your organization's secret patterns anyway.
- Provenance may carry URLs into access-controlled systems (Bugzilla,
  OneDrive); that is correct — the consumer hits the source's own ACL.
  Restricted-content *text* stays out of excerpts beyond the mined rule
  sentence.
- Count and surface what you could NOT index (403'd bugs, excluded
  folders): an access-shaped hole in the graph should be visible, not
  silent.

---

## 10. Validation: the harness that keeps it honest

### 10.1 Unit tests every module must ship (the paid-for list)

- **Store**: NULL-safe entity dedup; object XOR enforcement; soft
  supersession; flip-flop reopen; idempotent rebuild writes zero.
- **Closures/derivations**: 2-cycle terminates with no self-ancestry;
  depth bounds hold on long chains; per-source fan-out caps LOG when hit;
  phase-2 ordering (transitive module depends on an empty phase-1 output
  returns cleanly).
- **Co-change**: blast-radius bound (the 20-file commit mints nothing);
  support threshold; malformed log fails closed; symmetric emission;
  non-code pairs dropped; Perforce integrate-CL skip; move/add+delete
  identity rewrite.
- **Tests relation**: no-import ⇒ no fact; import-without-reference ⇒ no
  fact; stdlib/test-stack denied; first-party override works.
- **Lessons**: quoted-reply stripping; marker mining; the three heading
  detectors (md / rst-underline / docx styles); citation regex ignores
  periods inside `file.py:9`.
- **Assist**: loud no-op on nothing-grounds / missing DB / partial schema
  (build a db whose `facts` table lacks a column the query layer selects —
  the "grounds shallow, crashes deep" case); repo scoping excludes
  sibling-repo paths; workspace check drops nonexistent refs BEFORE
  injection; alias fold applies to BOTH sides; snake_case identifiers
  survive whole; malformed tool args (`"limit": "many"`) coerce instead of
  raising.
- **Semantic**: index idempotency via src_hash; dim/model mismatch rows
  skipped; endpoint-down degrades to lexical-only; the complementarity
  test — a task sharing ZERO vocabulary with symbol names grounds via the
  semantic proposer alone, and the fired-signal attributes it.
- **Watch**: the pure decision function (unknown/fresh/dirty/behind ⇒
  skip/skip/skip-loudly/pull); lock mutual exclusion (second instance
  skips, exit 0); no-op pass writes nothing.
- **MCP**: notifications get no response; unknown method ⇒ -32601;
  malformed line ⇒ parse error and the loop continues; a real subprocess
  round-trip over stdio (initialize → tools/list → tools/call).
- **Agent loop**: a canned in-thread HTTP server exercising the full
  tool-call round-trip, plus the clean endpoint-error path.

### 10.2 Baselines to record on day one

Index a well-known public repo (e.g. Flask) and commit the numbers: facts
by tier, build wall-clock, health dashboard all-zeros (duplicates, orphans,
dormant), and the same for YOUR main repo. Every later change diffs against
these. An index whose size jumps on rebuild gets its *composition* diffed
(top-directory counts, old vs new) before anyone ships a diagnosis.

### 10.3 The experiment protocol (for any "does memory help?" question)

1. **Arms**: bare floor; treatment(s); an **oracle ceiling** fed perfect
   grounding. The oracle prices your maximum possible win — without it you
   cannot tell "retrieval is weak" from "the model can't use retrieval".
2. **Fired-signal on EVERY arm, symmetric**: per-task, did the treatment
   execute, and how much (injected flag, grounded-symbol count, tool-call
   count). A null you cannot attribute — "didn't help" vs "never ran" — is
   a wasted run.
3. **Input census before the run**: count the facts each lever consumes,
   per predicate; warn loudly on absent relations. A lever starved of
   input reads exactly like an ineffective lever.
4. **Fired is not aimed**: post-run, deterministically re-derive what each
   arm injected and verify every ref against the task's workspace. In one
   audit, over half the tasks had received well-formatted paths that did
   not exist there — instrumentation had proven firing, not aim.
5. **Pin the serving configuration.** Greedy decoding is NOT deterministic
   across serving configs — CPU vs GPU floating-point accumulation order
   flipped multiple cases between otherwise-identical runs. Compare
   against anchors only within a pinned config.
6. **Judge the whole workspace, not the target file.** A target-only diff
   hides wrong-file edits ("landed, no diff") and lets strays contaminate
   later arms. Snapshot/diff/restore everything writable, or at minimum
   detect and flag any write outside the declared target. Disable any
   server-side prompt-prefix cache during evaluation (order-coupling: a
   case's outcome must not depend on which cases ran before it).
7. **Refresh and re-validate fixtures** before every run: evaluation
   workspaces drift from the suites authored against them; validate
   anchors (expected markers unique and present) on the fresh trees and
   report skips by name.
8. Publish the null if that is what you measured. A well-instrumented null
   tells you where the next unit of effort goes; ours said "generation,
   not retrieval," and it was right — and when the hybrid later earned a
   real solve, the same instrumentation is what made the win credible.

---

## 11. Caveat compendium (the short list to tattoo somewhere)

1. Deterministic-first: an LLM distiller produced hundreds of plausible,
   grounded-looking, WRONG facts; a parser beat it at zero cost.
2. Grounding gates are necessary, not sufficient — they cannot catch
   wrong-project contamination. Scope your sources.
3. Multi-valued predicates are not contradictions.
4. Nested checkouts double indexes; prune by construction.
5. Freshness is a first-class metric; schedulers must be verified alive.
6. A dirty working tree is sacred.
7. Locks, or overlapping rebuilds stack on one DB file.
8. Repo-scope and existence-check every injected ref; wrong file beats no
   file at steering agents into the weeds.
9. Fold aliases on both sides; keep identifiers whole.
10. Store co-change on files, query it on files.
11. Every failure path in serve-time code degrades, never raises.
12. Fired-signal + input census before every expensive run; verify aim
    after.
13. Split evals by file path and by time; freeze and checksum the eval
    set; beat the baseline or don't promote.
14. Blast-radius-bound every bulk event (commits, CLs, mass edits).
15. Count what you could not index; silence is not coverage.

---

## 12. Milestones for the implementing agent

Each milestone: build → its §10.1 tests green → acceptance check → commit.

| # | deliverable | acceptance |
|---|---|---|
| M1 | store + schema + consolidation | store invariants tests green; idempotent double-build writes zero |
| M2 | git code facts + co-change + lessons(md) | index a public repo; baseline numbers recorded (§10.2) |
| M3 | derivation layer (2 phases) | closure/bound/log tests green; health dashboard zeros |
| M4 | health + freshness + watch daemon | lock test green; no-op pass proven free; scheduler verified alive |
| M5 | assist (lexical) + guards + fired-signal | no-op discipline tests green, incl. partial-schema |
| M6 | MCP server + agentic tools | stdio subprocess round-trip green; registered in the IDE and answering |
| M7 | Perforce adapter | integrate-CL skip + path mapping tests; co-change parity with git shapes |
| M8 | Bugzilla adapter | fixed-by strict-regex tests; restricted-bug counting; dup-chain terminal |
| M9 | Toolchain adapter (build/test results) | fixture-driven parsers; outcomes + bounded/superseded diagnostics; unknown schema fails closed |
| M10 | Specifications-portal adapter | link-role mapping enumerated; implemented-by/touches joins; suspect links demoted; revision supersession |
| M11 | OneDrive adapter | delta-cursor sync; docx-via-zip extraction; auth in OS store |
| M12 | semantic sidecar (off-the-shelf embedder) | complementarity test green; endpoint-down degradation |
| M13 | embedder fine-tune + gated promotion | pairs mined from the graph; held-out recall@k beats baseline; frozen eval checksummed |
| M14 | evaluation harness (§10.3) | one full instrumented run, all arms, published — win or null |

---

*The reference core (M1–M6, M12) is implemented and tested in
[groundgraph](https://github.com/alleboudy/groundgraph); the enterprise
adapters and the tuned-embedder tier are specified here for your
infrastructure. The evaluation history behind every caveat lives in
chapters 16 and 19.*
