#!/usr/bin/env python3
"""Mock trainer — a stand-in for the real `train` CLI (codescribe issue #2 Phase 2).

It honours the SAME command contract the fleet orchestrator drives:

    fake_train.py smoke  --config C --dataset D --out O
    fake_train.py run    --config C --dataset D --out O
    fake_train.py eval   --adapter O [--tasks T] --report R
    fake_train.py export --adapter O --base-model M --merged MM --gguf G --llama-cpp L

...but instead of fine-tuning it sleeps briefly and writes *deterministic* artefacts
derived from the config's hyperparameters, so the whole Path A pipeline (schedule →
train → score → pull → summarise) runs end-to-end on one laptop with no GPU.

`task_mean` peaks at lr=2e-4, r=16 — the QLoRA reference point — so the summary
table's ordering is meaningful in the demo. Replace this script with the real
trainer by pointing `train.module` at it in fleet.yaml.

This is a CLI/script (not library code), so it prints to stdout by design.
Stdlib-only: it does not require PyYAML (falls back to a tiny parser).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

REFERENCE_LR = 2.0e-4
REFERENCE_R = 16


def _load_hparams(config_path: str) -> tuple[float, int, int]:
    """Return (lr, r, seed) from a rendered train config. PyYAML-optional."""
    text = Path(config_path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(text)
        return (
            float(cfg["optim"]["lr"]),
            int(cfg["lora"]["r"]),
            int(cfg.get("seed", 0)),
        )
    except ImportError:
        return _parse_hparams_fallback(text)


def _parse_hparams_fallback(text: str) -> tuple[float, int, int]:
    """Minimal indent-aware reader for our 2-level rendered config."""
    lr: float | None = None
    r: int | None = None
    seed = 0
    section: str | None = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1]
            continue
        if indent == 0 and stripped.startswith("seed:"):
            seed = int(stripped.split(":", 1)[1].strip())
            section = None
            continue
        if section == "optim" and stripped.startswith("lr:"):
            lr = float(stripped.split(":", 1)[1].strip())
        elif section == "lora" and stripped.startswith("r:"):
            r = int(stripped.split(":", 1)[1].strip())
    if lr is None or r is None:
        raise ValueError(f"could not parse lr/r from config (lr={lr}, r={r})")
    return lr, r, seed


def _scores(lr: float, r: int, seed: int) -> tuple[float, float]:
    """Deterministic (task_mean, train_loss) from hyperparameters."""
    lr_pen = abs(math.log10(lr) - math.log10(REFERENCE_LR))
    r_pen = abs(math.log2(r) - math.log2(REFERENCE_R))
    jitter = ((seed % 7) - 3) * 0.003
    task_mean = max(0.0, 0.62 - 0.18 * lr_pen - 0.05 * r_pen + jitter)
    train_loss = 0.30 + 0.50 * lr_pen + 0.10 * r_pen - jitter
    return round(task_mean, 4), round(train_loss, 4)


def _sleep() -> None:
    # Default 0 so the test suite is instant; the demo can set FAKE_TRAIN_SLEEP=0.3
    # for a more realistic feel.
    time.sleep(float(os.environ.get("FAKE_TRAIN_SLEEP", "0")))


def _train(config: str, out: str, *, steps: int, log_every: int) -> int:
    lr, r, seed = _load_hparams(config)
    task_mean, final_loss = _scores(lr, r, seed)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(0, steps, log_every):
        # loss decays toward final_loss across the (fake) run
        loss = final_loss + (steps - step) / steps * 0.4
        print(f"step {step} train_loss={loss:.4f}")
    print(f"step {steps} train_loss={final_loss:.4f}")
    _sleep()

    (out_dir / "adapter_model.safetensors").write_bytes(b"FAKE-LORA-ADAPTER\x00" * 4)
    (out_dir / "adapter_config.json").write_text(
        json.dumps({"r": r, "lr": lr, "seed": seed, "base_model_name_or_path": "fake"}),
        encoding="utf-8",
    )
    (out_dir / "run-meta.json").write_text(
        json.dumps(
            {
                "lr": lr,
                "r": r,
                "seed": seed,
                "train_loss": final_loss,
                "task_mean_oracle": task_mean,
                "wall_clock_h": round(4.5 + (seed % 5) * 0.2, 2),
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "DONE").write_text("ok\n", encoding="utf-8")
    return 0


def _eval(adapter: str, report: str, tasks: str | None) -> int:
    adapter_dir = Path(adapter)
    meta_path = adapter_dir / "run-meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lr, r, seed = float(meta["lr"]), int(meta["r"]), int(meta["seed"])
    else:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        lr, r, seed = float(cfg["lr"]), int(cfg["r"]), int(cfg.get("seed", 0))

    task_mean, _ = _scores(lr, r, seed)
    n_tasks = 0
    if tasks and Path(tasks).is_file():
        try:
            n_tasks = len(json.loads(Path(tasks).read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            n_tasks = 0

    report_path = Path(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {"task_mean": task_mean, "n_tasks": n_tasks, "adapter": str(adapter_dir)},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"task_mean={task_mean}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fake_train")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, n_steps in (("smoke", 10), ("run", 200)):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.add_argument("--dataset", required=False)  # mock ignores dataset contents
        sp.add_argument("--out", required=True)
        sp.set_defaults(_steps=n_steps)

    ev = sub.add_parser("eval")
    ev.add_argument("--adapter", required=True)
    ev.add_argument("--report", required=True)
    ev.add_argument("--tasks", required=False)

    ex = sub.add_parser("export")
    ex.add_argument("--adapter", required=True)
    ex.add_argument("--base-model", required=False)
    ex.add_argument("--merged", required=False)
    ex.add_argument("--gguf", required=False)
    ex.add_argument("--llama-cpp", required=False)

    args = p.parse_args(argv)
    if args.cmd in ("smoke", "run"):
        return _train(args.config, args.out, steps=args._steps, log_every=10)
    if args.cmd == "eval":
        return _eval(args.adapter, args.report, args.tasks)
    if args.cmd == "export":
        print(f"[fake] would merge {args.adapter} -> {args.gguf or 'model.gguf'}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
