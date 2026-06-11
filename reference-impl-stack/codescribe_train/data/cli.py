"""CLI entry point: ``python -m codescribe_train.data build``.

Wires the pipeline:

    walk -> filter -> dedup -> split (by file)
                                 \
                                  +-> format_document
                                  +-> format_fim
                                  +-> format_diff_instructions  (train only)
                                                  \
                                                   -> JSONL per split + manifest.json

Configuration is YAML; defaults are conservative. The pipeline never reaches
the network — local filesystem and ``git`` subprocess only.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from codescribe_train.data.dedup import dedup_records
from codescribe_train.data.filters import FilterConfig, filter_records
from codescribe_train.data.formatters import (
    Sample,
    format_diff_instructions,
    format_document,
    format_fim,
)
from codescribe_train.data.splitter import SplitWeights, split_records
from codescribe_train.data.walker import FileRecord, assert_git_root, walk_repo

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def _write_jsonl(samples: Iterator[Sample], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"text": s.text, "meta": s.metadata}, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def _samples_for_split(
    *,
    split_name: str,
    records: list[FileRecord],
    repo: Path,
    repo_name: str,
    rng: random.Random,
    fmt_doc: dict[str, Any],
    fmt_fim_cfg: dict[str, Any],
    fmt_diff: dict[str, Any],
) -> Iterator[Sample]:
    """Yield samples for one split, lazily — no full buffering."""
    if fmt_doc.get("enabled", True):
        yield from format_document(records, repo_name=repo_name)
    if fmt_fim_cfg.get("enabled", True):
        yield from format_fim(
            records,
            rng=rng,
            samples_per_file=int(fmt_fim_cfg.get("samples_per_file", 2)),
            min_middle_lines=int(fmt_fim_cfg.get("min_middle_lines", 1)),
            max_middle_lines=int(fmt_fim_cfg.get("max_middle_lines", 8)),
        )
    # diff_instr is global to the repo, not per-split. Emit only on train.
    if split_name == "train" and fmt_diff.get("enabled", False):
        yield from format_diff_instructions(
            repo,
            max_diff_lines=int(fmt_diff.get("max_diff_lines", 400)),
            min_subject_chars=int(fmt_diff.get("min_subject_chars", 12)),
            max_subject_chars=int(fmt_diff.get("max_subject_chars", 200)),
            skip_merge_commits=bool(fmt_diff.get("skip_merge_commits", True)),
            max_commits=fmt_diff.get("max_commits"),
        )


def build(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    repo = assert_git_root(Path(args.repo))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 0))
    repo_name = str(config.get("repo_name") or repo.name)

    # Walk -> filter -> dedup
    filter_config = FilterConfig.from_dict(config.get("filter"))
    records: list[FileRecord] = list(dedup_records(filter_records(walk_repo(repo), filter_config)))
    logger.info("survived filter+dedup: %d files", len(records))

    # Splits
    weights_dict = config.get("split", {}) or {}
    weights = SplitWeights(
        train=float(weights_dict.get("train", 0.90)),
        val=float(weights_dict.get("val", 0.05)),
        test=float(weights_dict.get("test", 0.05)),
    )
    salt = str(config.get("split_salt", repo_name))
    buckets = split_records(records, weights=weights, salt=salt)

    # Formatter selection
    fmt_cfg = config.get("formatters", {}) or {}
    fmt_doc = fmt_cfg.get("document", {}) or {}
    fmt_fim_cfg = fmt_cfg.get("fim", {}) or {}
    fmt_diff = fmt_cfg.get("diff_instr", {}) or {}

    manifest: dict[str, Any] = {
        "repo": str(repo),
        "repo_name": repo_name,
        "config_path": str(args.config),
        "split_weights": {
            "train": weights.train,
            "val": weights.val,
            "test": weights.test,
        },
        "split_salt": salt,
        "seed": seed,
        "totals": {},
    }

    rng_master = random.Random(seed)
    for split_name in ("train", "val", "test"):
        split_recs = buckets[split_name]
        n = _write_jsonl(
            _samples_for_split(
                split_name=split_name,
                records=split_recs,
                repo=repo,
                repo_name=repo_name,
                rng=random.Random(rng_master.randint(0, 1 << 32)),
                fmt_doc=fmt_doc,
                fmt_fim_cfg=fmt_fim_cfg,
                fmt_diff=fmt_diff,
            ),
            out / f"{split_name}.jsonl",
        )
        manifest["totals"][split_name] = n
        manifest["totals"][f"{split_name}_files"] = len(split_recs)
        logger.info("%s: %d samples from %d files", split_name, n, len(split_recs))

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("wrote manifest to %s", out / "manifest.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="codescribe_train.data",
        description="Build train/val/test datasets from a git repo (strictly local).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build_p = sub.add_parser("build", help="Build train/val/test splits from a repo.")
    build_p.add_argument("--repo", required=True, help="Path to the git repo root.")
    build_p.add_argument("--config", required=True, help="Path to the YAML config.")
    build_p.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        return build(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
