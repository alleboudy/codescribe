# rag.sources — package rules

Two read-only clients: Perforce (p4 CLI subprocess) and Bugzilla (REST via httpx).

## Hard rules

- **Read-only.** `p4 submit`, `p4 edit`, `p4 delete`, any Bugzilla POST/PUT/PATCH/DELETE — all forbidden. Static check `tests/rag/sources/test_no_writes.py` greps for these and fails CI if found.
- **No credentials in code.** P4 ticket from `$P4TICKETS` (default `~/.p4tickets`). Bugzilla key from `~/.config/codescribe_rag/bugzilla.key` (mode 600). Reject startup if a key is found inline in source.
- **Allowlisted egress.** `httpx.Client(base_url=...)` is constructed exactly once per Bugzilla; assert at startup that `urlparse(base_url).hostname` matches the configured allowlist (single hostname per env).
- **Rate-limit.** Both clients use a token bucket (default 5 req/s, configurable). Bugzilla in particular will lock you out at 100+ req/s.
- **Resumable.** Each client exposes `iter_*_since(checkpoint) -> Iterator[...]`. State (`last_seen_cl`, `last_seen_bug_modtime`) lives in the store's `state` table.
- **No `p4python`.** Use `subprocess` only. p4python's C extension complicates installation and offers nothing we need here.

## Anti-patterns

- Do NOT log credentials, even partially. Mask in error messages.
- Do NOT use `p4 sync` from the indexer — we only need metadata + diffs, not file content checkout.
- Do NOT swallow `p4`'s non-zero exit codes. Surface them with the full stderr.
- Do NOT use Bugzilla XML-RPC; it's deprecated. REST only.
- Do NOT cache Bugzilla bugs in-process beyond one batch — let the indexer's checkpointing handle resumability.
