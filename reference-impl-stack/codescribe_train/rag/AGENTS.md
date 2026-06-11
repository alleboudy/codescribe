# rag — package rules (concrete: sample sources)

Stitches Git + GitHub sources → extract → embed → store. Cron-friendly. Idempotent. Resumable.

## Hard rules

- All pipelines are idempotent — re-running produces the same store state (upsert semantics).
- State checkpoints (last_seen_commit_sha, last_seen_issue_updated_at) are atomic with the data upserts they correspond to.
- Logs are structured JSON to `logs/rag-indexer-<unix_ts>.log`. One line per phase event.
- Bootstrap and incremental share the same inner loop.
- Pipeline tolerates SIGTERM; restart picks up from state.*.

## Sample-specific defaults

- Source repo path: `../your-repo` (relative to the repo root of this project).
- GitHub repo: `example-org/sample`.
- Pairing strict default threshold: 0.8.
- Trusted pairing signal: GitHub's `closingIssuesReferences` (GraphQL) → confidence 1.0.

## Anti-patterns

- Do NOT print to stdout from the pipeline; logger only.
- Do NOT hold all pulled records in memory; stream + upsert in batches of ≤256.
- Do NOT pull the whole GitHub history if state.last_seen_issue_updated_at is set — that's what incremental mode is for.
- Do NOT cache GitHub responses for longer than one pipeline run; the API authoritative.
- Do NOT skip the gzip on diff_text.
- Do NOT touch any cloud service: embedding is local, vector store is local, only allowed egress is `api.github.com`.
