-- Metadata ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bugs (
    bug_id INTEGER PRIMARY KEY,
    summary TEXT NOT NULL,
    description TEXT,
    component TEXT,
    severity TEXT,
    status TEXT,
    resolution TEXT,
    creation_time TEXT,
    last_change_time TEXT,
    assigned_to TEXT,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS changes (
    cl_number INTEGER PRIMARY KEY,
    author TEXT,
    submitted_at TEXT,
    description TEXT,
    file_count INTEGER,
    diff_text BLOB,                  -- gzip-compressed unified diff
    raw_marshal BLOB                 -- (reserved) gzip p4 -G round-trip; unused in v1
);

CREATE TABLE IF NOT EXISTS fix_links (
    bug_id INTEGER,
    cl_number INTEGER,
    confidence REAL,
    signals TEXT,                    -- JSON of signal_name -> weight
    extracted_at TEXT,
    PRIMARY KEY (bug_id, cl_number)
);
CREATE INDEX IF NOT EXISTS idx_fix_links_bug ON fix_links (bug_id);
CREATE INDEX IF NOT EXISTS idx_fix_links_cl  ON fix_links (cl_number);
CREATE INDEX IF NOT EXISTS idx_fix_links_conf ON fix_links (confidence DESC);

-- Vector indices (sqlite-vec; the writer loads the extension before this runs) -
CREATE VIRTUAL TABLE IF NOT EXISTS bug_vectors USING vec0(
    embedding FLOAT[1024]
);
-- bug_vectors.rowid is set to bugs.bug_id by the writer.

CREATE VIRTUAL TABLE IF NOT EXISTS cl_chunk_vectors USING vec0(
    embedding FLOAT[1024]
);

CREATE TABLE IF NOT EXISTS cl_chunks (
    chunk_id INTEGER PRIMARY KEY,
    cl_number INTEGER NOT NULL,
    file_path TEXT,
    hunk_index INTEGER,
    chunk_text TEXT,
    FOREIGN KEY (cl_number) REFERENCES changes (cl_number)
);
CREATE INDEX IF NOT EXISTS idx_cl_chunks_cl ON cl_chunks (cl_number);

-- FTS5 (BM25 default ranker). Standalone tables with an UNINDEXED id column, so
-- the writer can INSERT/DELETE by id and the retriever can SELECT the id back.
CREATE VIRTUAL TABLE IF NOT EXISTS bug_fts USING fts5(
    bug_id UNINDEXED, summary, description, tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS cl_fts USING fts5(
    cl_number UNINDEXED, chunk_id UNINDEXED, chunk_text, tokenize='porter unicode61'
);

-- Indexer state -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT
);
-- e.g. state['last_seen_cl'] = '123456', state['last_seen_bug_modtime'] = '2026-05-20T18:00:00Z'
