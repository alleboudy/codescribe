CREATE TABLE issues (
    issue_number INTEGER PRIMARY KEY,         -- example-org/sample#N
    title TEXT NOT NULL,
    body TEXT,                                -- comment 0 equivalent
    state TEXT,                               -- open / closed
    state_reason TEXT,                        -- completed / not_planned / reopened
    labels TEXT,                              -- JSON array
    assignees TEXT,                           -- JSON array
    created_at TEXT,
    updated_at TEXT,
    closed_at TEXT,
    raw_json TEXT
);

CREATE TABLE pulls (
    pr_number INTEGER PRIMARY KEY,            -- example-org/sample#N
    title TEXT NOT NULL,
    body TEXT,
    state TEXT,                               -- open / closed / merged
    head_sha TEXT,                            -- commit SHA of PR head at merge time
    base_branch TEXT,
    merged_at TEXT,
    author TEXT,
    file_count INTEGER,
    diff_text BLOB,                           -- gzip-compressed unified diff
    raw_json TEXT
);

CREATE TABLE commits (
    sha TEXT PRIMARY KEY,
    author TEXT,
    author_email TEXT,
    authored_at TEXT,
    message TEXT,
    file_count INTEGER,
    diff_text BLOB,                           -- gzip-compressed
    pr_number INTEGER                         -- soft reference to pulls.pr_number;
                                              -- NOT a foreign key: the indexer
                                              -- streams commits and pulls
                                              -- concurrently, so commits can be
                                              -- written before the PR they
                                              -- reference (or the PR may never
                                              -- exist if it was deleted/squashed
                                              -- before the API window). The
                                              -- relationship is informational at
                                              -- retrieval time, not an integrity
                                              -- constraint at index time.
);

CREATE TABLE issue_pr_links (
    issue_number INTEGER,
    pr_number INTEGER,
    confidence REAL,
    source TEXT,                              -- "closingIssuesReferences" | "body_closes" | "temporal_author"
    extracted_at TEXT,
    PRIMARY KEY (issue_number, pr_number)
);
CREATE INDEX idx_issue_pr_links_issue ON issue_pr_links (issue_number);
CREATE INDEX idx_issue_pr_links_pr    ON issue_pr_links (pr_number);
CREATE INDEX idx_issue_pr_links_conf  ON issue_pr_links (confidence DESC);

-- Vector indices (sqlite-vec; 1024 for bge-large-en-v1.5)
CREATE VIRTUAL TABLE issue_vectors USING vec0(embedding FLOAT[1024]);
CREATE VIRTUAL TABLE pr_vectors    USING vec0(embedding FLOAT[1024]);
CREATE VIRTUAL TABLE commit_vectors USING vec0(embedding FLOAT[1024]);

CREATE TABLE pr_chunks (
    chunk_id INTEGER PRIMARY KEY,
    pr_number INTEGER NOT NULL,
    file_path TEXT,
    hunk_index INTEGER,
    chunk_text TEXT,
    FOREIGN KEY (pr_number) REFERENCES pulls (pr_number)
);
CREATE VIRTUAL TABLE pr_chunk_vectors USING vec0(embedding FLOAT[1024]);

-- Working-tree documentation chunks (README, Makefile, docs/**/*.md,
-- docker-compose*.yml, Dockerfile*). One row per heading-delimited
-- chunk; vec0 + FTS5 mirrors keyed by chunk_id, exactly like pr_chunks.
-- Docs are re-indexed in full each run (tiny corpus) so there's no
-- per-doc watermark; upsert_doc deletes a path's old chunks first.
CREATE TABLE doc_chunks (
    chunk_id INTEGER PRIMARY KEY,
    doc_path TEXT NOT NULL,        -- relative to the repo root, e.g. "docs/DEVELOPMENT.md"
    heading TEXT,                  -- nearest preceding heading, or "" for preamble / non-markdown
    chunk_index INTEGER,           -- ordinal within the doc (0-based)
    chunk_text TEXT
);
CREATE INDEX idx_doc_chunks_path ON doc_chunks (doc_path);
CREATE VIRTUAL TABLE doc_chunk_vectors USING vec0(embedding FLOAT[1024]);
CREATE VIRTUAL TABLE doc_chunk_fts USING fts5(heading, chunk_text, content='doc_chunks', content_rowid='chunk_id', tokenize='porter unicode61');

-- FTS5 (BM25 default)
CREATE VIRTUAL TABLE issue_fts  USING fts5(title, body, content='issues',  content_rowid='issue_number', tokenize='porter unicode61');
CREATE VIRTUAL TABLE pr_fts     USING fts5(title, body, content='pulls',   content_rowid='pr_number',    tokenize='porter unicode61');
CREATE VIRTUAL TABLE commit_fts USING fts5(message,    content='commits', content_rowid='rowid',         tokenize='porter unicode61');

CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT);
-- e.g., state['last_seen_commit_sha'] = '...', state['last_seen_issue_updated_at'] = '<ISO-8601 UTC>'
