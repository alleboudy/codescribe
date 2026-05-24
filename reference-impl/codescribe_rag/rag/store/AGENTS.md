# rag.store — package rules

Single-SQLite-file store; sqlite-vec for vectors, FTS5 for BM25, plain tables for metadata.

## Hard rules

- One DB file: `indices/rag.db`. NEVER split into multiple DBs.
- All writes go through `codescribe_rag/rag/store/writer.py`'s upsert API. Never bypass with raw SQL elsewhere.
- All reads in the retrieval path go through `codescribe_rag/rag/store/retrieve.py`. Bypassing fragments query semantics.
- `sqlite-vec` extension is loaded explicitly via `conn.enable_load_extension(True); sqlite_vec.load(conn)` — never relies on it being globally available.
- Diff text is GZIP-compressed in `changes.diff_text`. Same for `raw_marshal`. Decompression happens at read time in `retrieve.py`.
- Foreign-key constraints ON (`PRAGMA foreign_keys = ON`). Schema migrations append-only; never destructive without a versioned migration script.
- WAL journal mode (`PRAGMA journal_mode = WAL`) for safe concurrent read while indexer writes.

## Anti-patterns

- Do NOT use Python's unsafe binary serialisation modules for any stored field. JSON + gzip only. The `marshal` module is acceptable only for the immutable `p4 -G` raw payload round-trip.
- Do NOT skip the gzip on diffs "to save dev time". Large diffs blow the DB to 10× without it.
- Do NOT mix `sqlite-vec`'s `vec0` virtual tables with `JOIN` to regular tables in a single query — `vec0` only supports SELECT by rowid; join in Python after retrieval.
- Do NOT batch UPSERTs without `BEGIN`/`COMMIT` — single-row writes through WAL are 10× slower.
