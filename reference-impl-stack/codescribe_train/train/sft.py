"""QLoRA SFT loop for Qwen 2.5 Coder 7B Instruct on sample data.

Targets a consumer Blackwell-class laptop GPU, ~8 GB VRAM / sm_120, as the
reference example hardware. Uses Unsloth's
4-bit base loading + Triton kernels (mandatory at this VRAM size), `peft`
LoRA, and `trl.SFTTrainer`. Strictly local — `report_to="none"` is non-
negotiable.

Two entry points:

* :func:`train` — full training run as configured in the YAML.
* :func:`smoke_train` — 10 steps, no eval, no save. Used as the Phase 2
  Definition-of-Done verification gate.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result of a training run, surface-level enough for tests."""

    adapter_dir: Path
    train_loss: float | None
    eval_loss: float | None
    steps: int


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not loaded:
        raise ValueError(f"empty config: {path}")
    return loaded


def _enforce_local_only_env() -> None:
    """Disable remote logging / hub-push surfaces. Defence in depth on top of
    `report_to="none"` — these env vars catch any incidental SDK that we
    don't pass `report_to` to."""
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "0")  # opt-in
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "0")
    # Always disable telemetry.
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _build_dataset(dataset_dir: Path):
    """Load the train + eval splits from a phase-1 output directory.

    We project to **text-only** here. The JSONL also carries a ``meta``
    object with format-specific keys (document has ``size_bytes``, FIM has
    ``middle_lines/start_line``, diff_instr has ``sha/subject/diff_lines``),
    which is heterogeneous across rows. Hugging Face ``datasets`` infers a
    schema from the first batch and then fails to cast subsequent batches
    with different metadata shapes — projecting up front avoids the issue
    and meta is anyway only useful for human inspection of the JSONL.
    """
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"missing {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"missing {val_path}")
    return _load_text_only(train_path), _load_text_only(val_path)


def _load_text_only(path: Path):
    from datasets import Dataset

    texts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            texts.append(row["text"])
    return Dataset.from_dict({"text": texts})


def _build_model(config: dict[str, Any]):
    """Load Qwen 2.5 Coder 7B with 4-bit NF4 quantization and attach a LoRA."""
    from unsloth import FastLanguageModel

    base = str(config["base_model"])
    seq_len = int(config["training"]["seq_len"])
    lora = config["lora"]

    logger.info("loading base model %s (4-bit NF4, seq_len=%d)", base, seq_len)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=seq_len,
        load_in_4bit=bool(config["quantization"]["load_in_4bit"]),
        dtype=None,  # bf16 on Blackwell, fp16 elsewhere
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    logger.info(
        "attaching LoRA (r=%d, alpha=%d, dropout=%s)", lora["r"], lora["alpha"], lora["dropout"]
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias=str(lora["bias"]),
        use_gradient_checkpointing="unsloth",  # Unsloth's optimised variant
        random_state=int(config["training"]["seed"]),
    )
    return model, tokenizer


def _build_trainer(
    *, model, tokenizer, train_ds, val_ds, config: dict[str, Any], smoke: bool, output_dir: Path
):
    """Construct the trl.SFTTrainer with all hyperparams from the config."""
    from trl import SFTConfig, SFTTrainer

    training = config["training"]
    intervals = config["intervals"]
    smoke_cfg = config.get("smoke", {})

    if smoke:
        max_steps = int(smoke_cfg.get("max_steps", 10))
        eval_steps = int(smoke_cfg.get("eval_steps", 0))
        save_steps = int(smoke_cfg.get("save_steps", 0))
        logging_steps = int(smoke_cfg.get("logging_steps", 1))
        epochs = int(smoke_cfg.get("epochs", 1))
        eval_strategy = "no"
        save_strategy = "no"
    else:
        max_steps = -1
        eval_steps = int(intervals["eval_steps"])
        save_steps = int(intervals["save_steps"])
        logging_steps = int(intervals["logging_steps"])
        epochs = int(training["epochs"])
        eval_strategy = "steps"
        save_strategy = "steps"

    args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_length=int(training["seq_len"]),
        packing=False,  # samples are already wrapped — never pack
        per_device_train_batch_size=int(training["batch_size"]),
        per_device_eval_batch_size=int(training["batch_size"]),
        gradient_accumulation_steps=int(training["grad_accum"]),
        warmup_ratio=float(training["warmup_ratio"]),
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        bf16=bool(training.get("bf16", True)),
        fp16=bool(training.get("fp16", False)),
        optim=str(training["optimizer"]),
        lr_scheduler_type=str(training["lr_schedule"]),
        max_grad_norm=float(training["max_grad_norm"]),
        seed=int(training["seed"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        logging_steps=logging_steps,
        report_to="none",  # NEVER change. Strictly-local posture.
        dataloader_num_workers=int(config["dataloader"]["num_workers"]),
        dataloader_pin_memory=bool(config["dataloader"]["pin_memory"]),
        remove_unused_columns=False,
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
    )


def _do_train(
    *, config_path: Path, dataset_dir: Path, output_dir: Path, smoke: bool
) -> TrainResult:
    """Shared inner-loop for train/smoke."""
    _enforce_local_only_env()
    config = _load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = _build_dataset(dataset_dir)
    logger.info("train: %d samples; val: %d samples", len(train_ds), len(val_ds))
    model, tokenizer = _build_model(config)

    trainer = _build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        config=config,
        smoke=smoke,
        output_dir=output_dir,
    )

    metrics = trainer.train()
    train_loss = float(metrics.training_loss) if metrics.training_loss is not None else None
    eval_loss = None
    if not smoke:
        eval_metrics = trainer.evaluate()
        eval_loss = float(eval_metrics.get("eval_loss")) if eval_metrics else None
        # Path B (distributed): trl/HF Trainer pick up DDP from torchrun's env
        # vars automatically, so the same `run` is the distributed entry point.
        # Only rank 0 writes the adapter + tokenizer, to avoid concurrent writes
        # to the shared output dir. (Trainer.save_model is already main-only; the
        # tokenizer save is not, so we guard both.) See docs/TRAINING-PATHS.md.
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

    steps = int(trainer.state.global_step)
    return TrainResult(
        adapter_dir=output_dir,
        train_loss=train_loss,
        eval_loss=eval_loss,
        steps=steps,
    )


def train(*, config_path: Path, dataset_dir: Path, output_dir: Path) -> TrainResult:
    """Full training run."""
    return _do_train(
        config_path=config_path, dataset_dir=dataset_dir, output_dir=output_dir, smoke=False
    )


def smoke_train(*, config_path: Path, dataset_dir: Path, output_dir: Path) -> TrainResult:
    """10-step smoke training for Phase 2 verification — no eval, no save."""
    return _do_train(
        config_path=config_path, dataset_dir=dataset_dir, output_dir=output_dir, smoke=True
    )
