# rag — package rules

Stitches sources → extract → embed → store. Cron-friendly. Idempotent. Resumable.

## Hard rules

- All pipelines are idempotent. Re-running over the same data must produce the same store state (upsert semantics).
- State checkpoints are atomic — `set_state` writes happen in the same transaction as the data upserts they correspond to.
- Logs are structured JSON to `logs/rag-indexer-<unix_ts>.log`. One line per phase event.
- Bootstrap and incremental share the same inner loop; the only difference is the initial state.
- The pipeline can be SIGTERM'd safely; on restart, picks up from `state.*`.

## Anti-patterns

- Do NOT print to stdout from the pipeline; logger only. Stdout is reserved for CLI summaries.
- Do NOT hold all pulled bugs/CLs in memory. Stream them; upsert in batches of ≤256.
- Do NOT skip the gzip step for the diff_text column "to save CPU cycles". It saves DB size by 10×.
