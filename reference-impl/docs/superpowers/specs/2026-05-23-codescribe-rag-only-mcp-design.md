# Design — RAG-only MCP server for Copilot Chat (issue #6)

- **Date:** 2026-05-23
- **Status:** Approved (design); pending implementation plan
- **Implements:** GitHub issue #6 (RAG-only path), drawing on #2 Phase 0, #4 Phases 6–11, and #5 skeletons §1–§8/§11
- **Lives in:** `codescribe/reference-impl/` (subdir of the planning repo)
- **Python package:** `codescribe_rag`

## 1. Goal

Build the retrieval half of the codescribe stack and wire it into **VS Code GitHub Copilot Chat** — skipping the fine-tune, `llama-server`, and CLI orchestrator entirely (issue #6 §5). The end state:

1. A populated `indices/rag.db` (sqlite-vec + FTS5) holding Perforce changelists + Bugzilla bugs, paired with confidence scores.
2. A local MCP server (`codescribe_rag.servers.rag_server`) exposing `find_similar_bugs(query, k, min_confidence)` and `get_fix_diff(cl_number, max_chars)` over stdio.
3. A documented VS Code Copilot Chat configuration that spawns the server and surfaces both tools.
4. A nightly incremental indexer (cron) to keep the index fresh.

**Source systems are fixed:** code lives in **Perforce** (changelists + unified diffs), bugs live in **Bugzilla** (REST API). Both are read-only, on the corporate LAN.

### In scope
Phase 0 scaffolding (package tree, AGENTS.md, copilot-instructions, CI config, pyproject) + Phases 6–11 (sources, extract/pair, store/embed, indexer, retriever, MCP server) + the Copilot wiring doc.

### Out of scope (YAGNI)
Fine-tune / `train` / `serve` / `llama-server` / top-level `cli` orchestrator; Phase-12 RAG-lift eval (no fine-tune baseline to measure against — issue #6 §5 replaces it with manual A/B); recency weighting (#4 §12, v1 defers); head-revision code indexing (#4 §18); LangChain / LlamaIndex; Bugzilla attachment patches (#4-referenced scope creep).

## 2. Decisions beyond the issues

The issues leave a few choices to the implementer. These are the ones made here:

### 2.1 Dual embedder behind a `Protocol`
The issues assume a single `sentence-transformers` embedder loading `BAAI/bge-large-en-v1.5` (1.3 GB, GPU-friendly). But the chosen **full acceptance test suite** cannot be deterministic or CI-runnable if every store / retrieve / pipeline / MCP test needs that model.

Resolution — two implementations of one interface:

```python
class SupportsEmbedding(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...      # (N, 1024), L2-normalised, float32
    def embed_one(self, text: str) -> np.ndarray: ...         # (1024,)
```

- **`Embedder`** — production. `sentence-transformers` + bge-large, lazy-load, device auto-probe (Pascal→fp16 note), pre-warm. Verbatim per #5 §6.
- **`HashingEmbedder`** — numpy-only, deterministic, 1024-dim, L2-normalised. Hashes token n-grams into the vector. No model download, no GPU. Used by the test suite and by anyone wiring the system before downloading the real model.

`embed.backend` in `rag.yaml` selects (`"sentence-transformers"` | `"hashing"`). `RagTools.load` (MCP server) reads the config to pick. Genuinely *semantic* tests (e.g. "near-identical strings → cosine > 0.9") require real embeddings and are gated with `@pytest.mark.skipif(model not present)`. Contract tests (shape, normalisation, empty input, lazy-load) run against `HashingEmbedder` and always execute.

### 2.2 Latency test gating
`test_retrieve_latency_under_200ms` (100K-row fixture, p99 over 100 queries) is kept but marked `@pytest.mark.slow`. Default CI runs the fast suite; `pytest -m slow` runs the latency check. Fixture vectors are generated with numpy — no model needed.

### 2.3 Version pins & tooling
`mcp>=1.0,<2.0` (SDK API has drifted between minors), `numpy<2.0`, `sqlite-vec>=0.1.6`, `sentence-transformers>=3.0`, `httpx>=0.27`, `pydantic>=2.6`, `tenacity`, `pyyaml`, `python-dateutil`, `pathspec`. **`uv`** manages dependencies; **Python 3.12**; type hints throughout; `from __future__ import annotations` in every module.

## 3. Architecture & dataflow

```
Perforce server (p4 -G marshal, subprocess) ┐
                                            ├─→ sources (Phase 6, read-only, rate-limited, resumable)
Bugzilla server (REST, httpx, X-API-KEY)    ┘        │ raw P4Change + Bug
                                                     ▼
                                   extract/pair (Phase 7, pure-functional)
                                   regex link extraction + confidence scoring
                                                     │ scored FixLink (bug_id, cl, confidence)
                                                     ▼
                                   embed + chunk (Phase 8) → store (Phase 8)
                                   bge-large / hashing       one indices/rag.db
                                                             sqlite-vec + FTS5 + metadata
                                                     │
                                   indexer pipeline (Phase 9): bootstrap + incremental, SIGTERM-safe
                                                     ▼
                                   hybrid retriever (Phase 10): vec top-K + BM25 top-K + RRF merge
                                                     │
                                   MCP server (Phase 11, stdio JSON-RPC)
                                   find_similar_bugs · get_fix_diff
                                                     ▼
                                   VS Code Copilot Chat  ──→ GitHub cloud ──→ model provider
                                   (the cloud boundary — see §7 privacy)
```

Lifecycle: Copilot Chat spawns `python -m codescribe_rag.servers.rag_server` as a stdio child → `initialize` + `tools/list` handshake → user prompts → planner calls `find_similar_bugs` → server reads `rag.db`, returns Markdown → result injected into the next model turn.

## 4. Repository layout (`reference-impl/`)

```
reference-impl/
  pyproject.toml                       # uv project; rag + mcp extras groups
  README.md                            # quickstart + pointer to docs/SETUP-COPILOT.md
  .github/copilot-instructions.md      # #2 §6 verbatim, <project>→codescribe_rag
  AGENTS.md                            # repo-root rules (#2 §6 cross-cutting, trimmed to RAG-only)
  configs/
    rag.yaml                           # RagConfig instance (see §5)
    mcp.json                           # canonical MCP server config (single source of truth)
  codescribe_rag/
    __init__.py
    rag/
      __init__.py
      __main__.py                      # CLI: index [--bootstrap ...] | query | status
      config.py                        # pydantic RagConfig + sub-configs + from_yaml
      AGENTS.md                        # #4 §11 verbatim
      sources/
        __init__.py
        __main__.py                    # CLI: p4-probe | bz-probe
        perforce.py                    # #5 §1 — P4Source, P4Change, FileChange, P4AuthExpired
        bugzilla.py                    # #5 §2 — BugzillaSource, Bug, BugComment
        _ratelimit.py                  # #5 §3 — TokenBucket
        AGENTS.md                      # #4 §8 verbatim
      extract/
        __init__.py
        links.py                       # #4 §9 — CL_PATTERNS, BUG_PATTERNS, extract_*_refs
        pairing.py                     # #4 §9 — FixLink, score_pair, pair_bugs_to_cls
        AGENTS.md                      # #4 §9 verbatim
      store/
        __init__.py
        schema.sql                     # #4 §10 verbatim
        writer.py                      # #5 §4 — Store (open, upserts, batch, state, get_diff_text)
        retrieve.py                    # #5 §5 — Retriever (find_similar_bugs, get_fix_diff)
        types.py                       # RetrievedBugFix, CLDiff, FileDiff
        AGENTS.md                      # #4 §10 verbatim
      embed/
        __init__.py
        embedder.py                    # #5 §6 — Embedder + HashingEmbedder + SupportsEmbedding
        chunker.py                     # #5 §7 — chunk_diff, CLChunk
      pipelines/
        __init__.py
        bootstrap.py                   # #5 §10 — run_bootstrap, BootstrapStats
        incremental.py                 # run_incremental, IncrementalStats
    servers/
      __init__.py
      rag_server/
        __init__.py
        __main__.py                    # #5 §8 — MCP stdio entry, logger isolation, SIGTERM
        tools.py                       # #5 §8 — RagTools, schemas, formatters
        AGENTS.md                      # #4 §13.5 verbatim
  tests/
    conftest.py                        # shared fixtures: hashing embedder, populated tmp store, p4 stub path
    rag/
      sources/    {test_no_writes, test_no_credentials_in_source, test_perforce_parser,
                   test_bugzilla_parser, test_rate_limit, test_egress_allowlist,
                   test_no_p4python, test_p4_subprocess_safety, fixtures/p4_stub.py}
      extract/    {test_links_cl_patterns, test_links_bug_patterns, test_links_min_id_threshold,
                   test_pairing_signals, test_pairing_temporal, test_pairing_threshold,
                   test_pairing_no_double_count}
      store/      {test_open_creates_schema, test_upsert_bug_roundtrip, test_upsert_cl_chunks,
                   test_fts_bm25_works, test_vec_query_works, test_diff_compressed,
                   test_no_unsafe_serialisation, test_retrieve_hybrid_ranks,
                   test_retrieve_confidence_filter, test_retrieve_no_unlinked_bug,
                   test_retrieve_latency_under_200ms (slow), test_get_fix_diff_round_trip,
                   test_query_normalisation}
      embed/      {test_normalisation (skipif), test_dim_is_1024, test_l2_normalised,
                   test_lazy_load, test_empty_input_returns_empty_array,
                   test_chunker_respects_max_chars, test_chunker_never_splits_inside_hunk_header,
                   test_chunker_utf8_boundary_preserved, test_chunker_per_file_isolation}
      pipelines/  {test_bootstrap_idempotent, test_incremental_resumes, test_late_arriving_cl,
                   test_sigterm_safe, test_batch_size, test_no_stdout_in_library}
    servers/rag_server/ {test_initialize_handshake, test_tools_list, test_call_find_similar_bugs,
                   test_call_get_fix_diff, test_invalid_params, test_no_stdout, test_no_egress,
                   test_no_db_writes, test_sigterm_clean_shutdown, test_logger_no_stream_handler,
                   test_tools_list_schemas_match}
  docs/
    SETUP-COPILOT.md                   # step-by-step VS Code Copilot Chat wiring (§6)
    superpowers/specs/                 # this design doc
```

## 5. Config schema (`configs/rag.yaml` → `RagConfig`)

Pydantic v2 models in `rag/config.py`; `RagConfig.from_yaml(path)` rejects unknown keys (`model_config = ConfigDict(extra="forbid")`).

```yaml
perforce:
  p4port: "ssl:perforce.corp.example.com:1666"
  p4user: "svc-rag"
  ticket_path: "~/.p4tickets"
  depot_path: "//depot/main/..."
  binary: "p4"
  subprocess_timeout_s: 60.0
bugzilla:
  base_url: "https://bugzilla.corp.example.com"
  allowlist_hostname: "bugzilla.corp.example.com"   # asserted at client construction
  api_key_path: "~/.config/codescribe_rag/bugzilla.key"  # mode-600 enforced
  page_size: 100
  request_timeout_s: 30.0
  connect_timeout_s: 5.0
store:
  db_path: "indices/rag.db"
embed:
  backend: "sentence-transformers"        # | "hashing"
  model_path: "~/.hf-models/bge-large-en-v1.5"
  device: "auto"
  batch_size: 32
  max_length: 512
  dim: 1024
pairing:
  weights:
    bug_comment_cites_cl: 0.5
    cl_desc_cites_bug: 0.5
    temporal_proximity: 0.3
    assignee_author_match: 0.2
    bug_status_fixed: 0.1
  threshold: 0.8
  temporal_signal_days: 7                 # CL within N days of bug resolution → temporal signal
  candidate_window_days: 60               # only score bug×CL pairs within this window
  min_bug_id: 1000
  min_cl_id: 1000
retrieve:
  k: 5
  k_vec: 20
  k_bm25: 20
  confidence_threshold: 0.8
  rrf_constant: 60
rate_limit_rps: 5.0
```

## 6. The Copilot wiring deliverable (`docs/SETUP-COPILOT.md`)

The artifact the user explicitly asked for. Contents, in order:

1. **Privacy gate first.** Restate issue #6 §3: every retrieved bug title / diff hunk / author travels to GitHub's cloud and onward to the routed model provider. Disqualified for NDA codebases. One-paragraph sign-off prompt.
2. **Verify tier supports MCP.** `code --list-extensions --show-versions | grep -i copilot`; settings-key drift (`github.copilot.chat.mcpServers` → `chat.mcp.servers` → Settings UI "mcp"); fallback harnesses (Cursor `.cursor/mcp.json`, Continue `~/.continue/config.json`, Claude Code) all taking the same `mcpServers` shape.
3. **Bootstrap the index.** `uv run python -m codescribe_rag.rag index --bootstrap` (after Day-0: `p4 login`, Bugzilla key, model pre-stage).
4. **Verify the server boots.** `npx @modelcontextprotocol/inspector uv run python -m codescribe_rag.servers.rag_server` → click `find_similar_bugs`.
5. **The exact `settings.json` block** (abs `RAG_DB_PATH`, `RAG_LOG_PATH`), reload window, confirm tools in the picker.
6. **Nightly cron** for `... rag index` (incremental).
7. **Day-to-day** + tailing `logs/rag-server-*.log` to confirm dispatch.

```json
{
  "github.copilot.chat.mcpServers": {
    "codescribe-rag": {
      "command": "uv",
      "args": ["run", "python", "-m", "codescribe_rag.servers.rag_server"],
      "env": {
        "RAG_DB_PATH": "/abs/path/to/reference-impl/indices/rag.db",
        "RAG_LOG_PATH": "/abs/path/to/reference-impl/logs/rag-server.log"
      }
    }
  }
}
```

`configs/mcp.json` holds the canonical version (the "single source of truth" per #4 §13.7 / #7 doc); SETUP-COPILOT.md shows how to copy it into VS Code settings.

## 7. Security & privacy

- **Read-only everywhere.** No `p4 submit/edit/add/delete/reopen`; no Bugzilla POST/PUT/PATCH/DELETE. Static tests grep the source and fail CI on any hit.
- **No `shell=True`** in any `subprocess` call (AST/regex static test).
- **Egress allowlist:** Bugzilla `httpx.Client` constructed once against one host; hostname asserted at construction; `test_egress_allowlist` monkey-patches `httpx.Client.send`. MCP server asserts *zero* egress (`test_no_egress` patches `socket.connect`).
- **Credentials:** P4 ticket from `$P4TICKETS`/`~/.p4tickets`; Bugzilla key from `~/.config/codescribe_rag/bugzilla.key` with mode-600 enforced at load. Never logged; a `redact()` helper strips key-shaped strings before logging. `test_no_credentials_in_source` greps for inline key-shaped literals.
- **MCP stdout discipline:** `_configure_logging()` runs first in `__main__.py`, removes all root handlers, installs a single `FileHandler`. `test_no_stdout` + `test_logger_no_stream_handler` enforce it.
- **Cloud caveat** is documentation, surfaced loudly in SETUP-COPILOT.md — this path is intentionally not strictly-local (the harness is GitHub's cloud).

## 8. Error handling

- **Perforce:** `subprocess.run(check=True)`; `P4AuthExpired` raised on "session has expired"/"P4-AUTH" stderr so the operator knows to re-`p4 login`; diff > ~1 MB sets `diff_truncated`.
- **Bugzilla:** `tenacity` 3 attempts, exponential backoff, retry on 5xx + connect/read-timeout only; **no** retry on 401/403/404; `raise_for_status()` surfaces the rest.
- **Store:** foreign keys ON, WAL, batched transactions; `vec0` uses DELETE+INSERT (no UPSERT); gzip on diffs.
- **MCP tools:** arg validation (defence-in-depth atop SDK schema validation) → `ValueError` → SDK maps to `InvalidParams`; all exceptions logged to file via `logger.exception`, never to stdout.

## 9. Testing strategy

- **pytest**, markers: `slow` (latency). Default CI = fast suite, no model download, no network, no real `p4`.
- **Perforce:** `tests/rag/sources/fixtures/p4_stub.py` (a tiny Python `p4` emulator, #5 §11) invoked via `P4Source(binary=...)`.
- **Bugzilla:** `httpx.MockTransport` swapped onto the client (#5 §11).
- **Embeddings:** `HashingEmbedder` for determinism; real-model semantic tests `skipif` absent.
- **Store/retrieve:** real `sqlite-vec` against `tmp_path` DBs (the extension ships per-OS wheels; works on darwin arm64).
- **MCP:** in-process transport via `create_connected_server_and_client_session` (#5 §11) for handshake/list/call; subprocess only for the SIGTERM test.
- **Static-analysis tests** (no-writes, no-print, no-shell-True, no-p4python, no-unsafe-serialisation, no-credentials) implemented as source greps / AST walks over the package dirs.

## 10. Build phasing

Mirrors #4's phases; each lands with its tests green before the next:

| Order | Phase | Modules | Key tests |
|---|---|---|---|
| 1 | Phase 0 | pyproject, AGENTS.md ×5, copilot-instructions, CI, config.py, package skeleton, README | config parser; package imports |
| 2 | Phase 6 | sources/{perforce,bugzilla,_ratelimit}, sources/__main__ | parsers, rate limit, no-writes, egress, no-p4python, subprocess-safety |
| 3 | Phase 7 | extract/{links,pairing} | link patterns, pairing signals/temporal/threshold/no-double-count |
| 4 | Phase 8 | store/{schema,writer,types}, embed/{embedder,chunker} | open/upsert/fts/vec/gzip, embedder contract, chunker boundaries |
| 5 | Phase 10 | store/retrieve | RRF ranking, confidence filter, normalisation, latency (slow) |
| 6 | Phase 9 | pipelines/{bootstrap,incremental}, rag/__main__ | idempotent, resume, late-arriving CL, sigterm, batch size, no-stdout |
| 7 | Phase 11 | servers/rag_server/{__main__,tools} | handshake, list, call, invalid params, no-stdout/egress/db-writes, sigterm, logger |
| 8 | Wiring | docs/SETUP-COPILOT.md, configs/mcp.json | manual (operator) |

Phase 10 precedes Phase 9 because the indexer's final assertions are easiest to validate once retrieval works; the store stub (`retrieve.py` raising `NotImplementedError` after Phase 8) is filled in at step 5.

## 11. Open questions / risks

- **`sqlite-vec` wheel on darwin arm64** — expected to work (vendored binaries); confirmed at Phase 8.
- **MCP SDK surface** (`Server`, `stdio_server`, `InitializationOptions`, `NotificationOptions`) is pinned to `>=1.0,<2.0`; if the installed minor differs from the #5 §8 skeleton, adapt imports in Phase 11 (its own concern, documented in PR/notes).
- **bge-large not downloaded** — production retrieval quality needs it; the system runs (lower quality) on `HashingEmbedder` until then. SETUP-COPILOT.md documents the `hf download` step.
- **Real Perforce/Bugzilla not available in this environment** — all automated tests use stubs/fixtures; operator-verified probes (`p4-probe`, `bz-probe`) are documented for the user to run against real servers.
