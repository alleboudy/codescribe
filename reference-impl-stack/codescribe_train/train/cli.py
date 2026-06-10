"""CLI: ``python -m codescribe_train.train <subcommand> ...``.

Subcommands:

* ``smoke``  — 10-step smoke training (Phase 2 DoD verification gate)
* ``run``    — full QLoRA training run
* ``eval``   — score a saved adapter on the held-out test set + sample tasks
* ``export`` — merge LoRA + optionally convert to GGUF
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _cmd_smoke(args: argparse.Namespace) -> int:
    from codescribe_train.train.sft import smoke_train

    result = smoke_train(
        config_path=Path(args.config),
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.out),
    )
    logger.info(
        "smoke OK: %d steps, train_loss=%s, adapter_dir=%s",
        result.steps,
        result.train_loss,
        result.adapter_dir,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from codescribe_train.train.sft import train

    result = train(
        config_path=Path(args.config),
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.out),
    )
    logger.info(
        "train complete: %d steps, train_loss=%s, eval_loss=%s, adapter_dir=%s",
        result.steps,
        result.train_loss,
        result.eval_loss,
        result.adapter_dir,
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from codescribe_train.train.eval import evaluate

    # --dataset is optional: without it (e.g. the fleet orchestrator's eval call)
    # we score the task suite only and skip held-out perplexity.
    result = evaluate(
        adapter_dir=Path(args.adapter),
        dataset_dir=Path(args.dataset) if args.dataset else None,
        tasks_path=Path(args.tasks),
        skip_perplexity=bool(args.skip_perplexity) or not args.dataset,
    )
    out = {
        "perplexity": result.perplexity,
        "task_mean": result.task_mean,
        "tasks": [
            {
                "id": t.task_id,
                "category": t.category,
                "score": t.score,
                "expected_hits": t.expected_hits,
                "expected_total": t.expected_total,
                "forbidden_hits": t.forbidden_hits,
                "forbidden_total": t.forbidden_total,
            }
            for t in result.tasks
        ],
    }
    if args.report:
        Path(args.report).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", args.report)
    print(json.dumps(out, indent=2))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from codescribe_train.train.export_gguf import export

    paths = export(
        adapter_dir=Path(args.adapter),
        base_model_id=args.base_model,
        merged_dir=Path(args.merged),
        gguf_path=Path(args.gguf),
        llama_cpp_dir=Path(args.llama_cpp) if args.llama_cpp else None,
    )
    logger.info(
        "export complete: merged=%s, gguf=%s",
        paths.merged_dir,
        paths.gguf_path or "(skipped — llama.cpp not provided)",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="codescribe_train.train",
        description="QLoRA SFT + eval + GGUF export. Strictly local.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_smoke = sub.add_parser("smoke", help="10-step smoke training (Phase 2 DoD)")
    p_smoke.add_argument("--config", required=True)
    p_smoke.add_argument("--dataset", required=True)
    p_smoke.add_argument("--out", required=True)

    p_run = sub.add_parser("run", help="full QLoRA training")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--out", required=True)

    p_eval = sub.add_parser("eval", help="score a saved adapter")
    p_eval.add_argument("--adapter", required=True)
    p_eval.add_argument(
        "--dataset",
        default=None,
        help="data dir with test.jsonl for perplexity; omit to score the task suite only",
    )
    p_eval.add_argument("--tasks", default="evals/sample_tasks.json")
    p_eval.add_argument("--skip-perplexity", action="store_true")
    p_eval.add_argument("--report", help="optional JSON output path")

    p_export = sub.add_parser("export", help="merge LoRA + GGUF quantize")
    p_export.add_argument("--adapter", required=True)
    p_export.add_argument("--base-model", required=True)
    p_export.add_argument("--merged", required=True)
    p_export.add_argument("--gguf", required=True)
    p_export.add_argument(
        "--llama-cpp",
        default=None,
        help="path to vendored llama.cpp (Phase 4); if omitted, only merge runs",
    )

    args = parser.parse_args(argv)
    handlers = {
        "smoke": _cmd_smoke,
        "run": _cmd_run,
        "eval": _cmd_eval,
        "export": _cmd_export,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
