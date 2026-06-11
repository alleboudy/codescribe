# `codescribe_train/data/`

Generic git repo → train/val/test JSONL pipeline. Phase 1 of `codescribe-train`.

This package walks any git repository, applies a per-repo YAML config to filter the file list, deduplicates, splits deterministically by file path, and emits text-only training samples through three orthogonal formatters (document, FIM, diff-instruction).

> Full background on **why** these decisions: [`docs/QLoRA-AND-UNSLOTH.md § 5 — why three formatters`](../../docs/QLoRA-AND-UNSLOTH.md#5-why-three-formatters).

---

## Public surface

```bash
python -m codescribe_train.data build \
    --repo /path/to/git/repo \
    --config configs/<repo-name>.yaml \
    --out datasets/<repo-name>/
```

Outputs:
- `datasets/<repo-name>/train.jsonl`
- `datasets/<repo-name>/val.jsonl`
- `datasets/<repo-name>/test.jsonl`
- `datasets/<repo-name>/manifest.json` — provenance: source repo path, `repo_name`, config path, split weights, salt, seed, per-split totals.

Each line in the JSONLs is `{"text": "<sample>", "meta": {"format": "document"|"fim"|"diff_instr", ...}}`. `text` is the only field the trainer reads; `meta` is for human inspection.

---

## File layout

```
data/
├── __init__.py
├── __main__.py        # → python -m codescribe_train.data <subcommand>
├── cli.py             # argparse + orchestration: walk → filter → dedup → split → format → write
├── walker.py          # `git ls-files -z` enumeration → FileRecord stream
├── filters.py         # YAML config → FilterConfig → predicate over FileRecord
├── dedup.py           # SHA-256 exact-content dedup
├── splitter.py        # SplitWeights + deterministic-by-relpath bucket assignment
└── formatters/
    ├── __init__.py
    ├── base.py        # Sample dataclass
    ├── document.py    # whole-file → one Sample with <|file_sep|> markers
    ├── fim.py         # random-line-span mask → PSM-format Sample
    └── diff_instr.py  # git log → ChatML user/assistant pair from commit + diff
```

---

## Pipeline order

```
git_ls_files(repo)  →  walker  →  filter  →  dedup  →  splitter
                                                          │
                                  ┌───────────────────────┴───────────────────────┐
                                  │                       │                       │
                              [train recs]            [val recs]              [test recs]
                                  │                       │                       │
                                  ▼                       ▼                       ▼
                          format_document            format_document         format_document
                          format_fim                 format_fim              format_fim
                          format_diff_instr          (skipped — train only)  (skipped — train only)
                                  │                       │                       │
                                  ▼                       ▼                       ▼
                            train.jsonl              val.jsonl              test.jsonl
```

Notes:
- `format_diff_instr` runs **only on train** because it pulls from `git log` (repo-global), not per-FileRecord. We don't want commit-level signal leaking into val/test.
- The splitter operates on `FileRecord`s (metadata only) before any content reading. This keeps memory bounded and ensures the split is reproducible.

---

## Conventions

- **Splits are by file path, never by sample.** The splitter hashes `salt + ":" + relpath` → bucket. All samples derived from the same file land in the same split. Don't undo this.
- **Salts are append-only.** Once a dataset is built and used for training, **don't change the salt**. Same applies to filter rules — changing them after training mixes train/val/test boundaries silently.
- **Configs are YAML, not Python.** Per-repo behaviour belongs in `configs/`, not in code. Add a new repo by copying `configs/sample.yaml` and editing.
- **Formatters are pure functions of input.** No state between samples (other than the seeded RNG). Reproducibility is non-negotiable.
- **Every sample has `meta.format`.** Downstream consumers can filter by format if needed. The CLI only reads `meta.src_path` and `meta.format` for the diagnostics in `manifest.json`.

---

## Adding a new formatter

A formatter is a function `(records, **kwargs) -> Iterator[Sample]`. To add one:

1. Create `formatters/<your_format>.py` with a top-level function.
2. Re-export from `formatters/__init__.py`.
3. Add a config schema entry in `cli.py:_samples_for_split` and the relevant YAML files.
4. Add tests in `tests/data/test_formatters.py`.

Required tests:
- Format produces the expected special tokens / structure.
- Output is deterministic given a seeded RNG.
- Edge cases: empty file, single-line file, file > seq_len.

---

## Adding a new repo to train against

Copy `configs/sample.yaml` to `configs/<your-repo>.yaml`, edit:

```yaml
repo_name: <your-repo>
seed: <some-int>
split_salt: <your-repo>-v1   # bump v1 → v2 if you ever regenerate
filter:
  include:
    extensions: [py, ts, ...]   # languages to train on
  exclude:
    paths:
      - "**/tests/**"
      - "**/.venv/**"
      # ... add anything project-specific
  size:
    min_bytes: 16
    max_bytes: 524288
formatters:
  document: { enabled: true }
  fim: { enabled: true, samples_per_file: 2 }
  diff_instr: { enabled: false }   # opt-in; needs a clean commit history
```

Run:

```bash
python -m codescribe_train.data build \
    --repo /path/to/your/repo \
    --config configs/<your-repo>.yaml \
    --out datasets/<your-repo>/
```

Verify: open `train.jsonl`, eyeball 5–10 samples. Check `manifest.json` for plausible totals. Spot-check that no test/migration/lock files made it through (`.meta.src_path`).

---

## Test layout

`tests/data/`:
- `test_walker.py` — git ls-files enumeration, symlink/gitlink skipping, unicode names
- `test_filters.py` — extension allowlists, gitignore-style globs, size bounds
- `test_dedup.py` — exact-dup detection, missing-file handling
- `test_splitter.py` — determinism, ratio approximation, file-stability across re-runs
- `test_formatters.py` — per-formatter invariants, FIM round-trip property, diff_instr ChatML structure
- `test_cli.py` — end-to-end on a synthetic repo; asserts split disjointness, no-leak rules

All 45 tests pass under `pytest tests/data/`. Don't break this without a very good reason.

---

## See also

- [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) — how this phase fits in the overall pipeline
- [`docs/QLoRA-AND-UNSLOTH.md`](../../docs/QLoRA-AND-UNSLOTH.md) — what training does with these JSONLs
- [`docs/CONCEPTS.md`](../../docs/CONCEPTS.md) — design rationale for the pipeline
