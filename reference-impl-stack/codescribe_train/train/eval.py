"""Qualitative + quantitative evaluation of a fine-tuned adapter.

Two scores:

* **Held-out perplexity** on ``test.jsonl`` from the data pipeline. Necessary
  but not sufficient.
* **Sample task suite** in ``evals/sample_tasks.json``. Each task has
  ``expected_signals`` (substrings that should appear, case-insensitive) and
  ``forbidden_signals`` (must NOT appear). The score is the fraction of
  expected signals matched minus the fraction of forbidden ones.

HumanEval before/after is documented in the plan but **not** implemented yet —
running it locally requires an internet-fetched dataset, which contradicts the
strictly-local posture during a hot training session. Offline cache + manual
sanity prompts for now; full HumanEval is a deferred task.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TaskScore:
    task_id: str
    category: str
    response: str
    expected_hits: int
    expected_total: int
    forbidden_hits: int
    forbidden_total: int

    @property
    def score(self) -> float:
        positive = self.expected_hits / self.expected_total if self.expected_total else 1.0
        penalty = self.forbidden_hits / self.forbidden_total if self.forbidden_total else 0.0
        return positive - penalty


@dataclass
class EvalResult:
    perplexity: float | None = None
    tasks: list[TaskScore] = field(default_factory=list)

    @property
    def task_mean(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(t.score for t in self.tasks) / len(self.tasks)


def _load_adapter(adapter_dir: Path, *, max_seq_length: int = 1024):
    """Load a saved LoRA adapter on top of its base. Returns (model, tokenizer).

    ``max_seq_length`` defaults to 1024 to match the training-time setting on
    a typical 8 GB-VRAM GPU. Higher values trigger bitsandbytes' "some modules
    dispatched on CPU" error during 4-bit load on such hardware.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, *, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic for eval
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip the echoed prompt.
    return decoded[len(prompt) :] if decoded.startswith(prompt) else decoded


def score_tasks(adapter_dir: Path, tasks_path: Path) -> list[TaskScore]:
    """Run every task through the adapter and score by signal matching."""
    tasks_payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = tasks_payload["tasks"]
    model, tokenizer = _load_adapter(adapter_dir)

    results: list[TaskScore] = []
    for task in tasks:
        prompt = task["prompt"]
        wrapped = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        response = _generate(model, tokenizer, wrapped)
        lowered = response.lower()
        expected = [s.lower() for s in task.get("expected_signals", [])]
        forbidden = [s.lower() for s in task.get("forbidden_signals", [])]
        expected_hits = sum(1 for s in expected if s in lowered)
        forbidden_hits = sum(1 for s in forbidden if s in lowered)
        results.append(
            TaskScore(
                task_id=str(task["id"]),
                category=str(task.get("category", "")),
                response=response,
                expected_hits=expected_hits,
                expected_total=len(expected),
                forbidden_hits=forbidden_hits,
                forbidden_total=len(forbidden),
            )
        )
        logger.info(
            "task %s: score %.2f (%d/%d expected, %d/%d forbidden)",
            task["id"],
            results[-1].score,
            expected_hits,
            len(expected),
            forbidden_hits,
            len(forbidden),
        )
    return results


def compute_perplexity(adapter_dir: Path, dataset_dir: Path, *, max_examples: int = 200) -> float:
    """Approximate held-out perplexity on test.jsonl. Capped at max_examples."""
    import torch
    from datasets import load_dataset

    test_path = dataset_dir / "test.jsonl"
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    ds = load_dataset("json", data_files=str(test_path), split="train")
    if max_examples and len(ds) > max_examples:
        ds = ds.select(range(max_examples))

    model, tokenizer = _load_adapter(adapter_dir)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for row in ds:
            inputs = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            n = inputs["input_ids"].shape[1]
            total_loss += float(outputs.loss) * n
            total_tokens += n
    if not total_tokens:
        return float("nan")
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def evaluate(
    *,
    adapter_dir: Path,
    dataset_dir: Path | None = None,
    tasks_path: Path,
    skip_perplexity: bool = False,
) -> EvalResult:
    """End-to-end eval: perplexity + task suite.

    ``dataset_dir`` is only needed for held-out perplexity. Omit it (or pass
    ``skip_perplexity=True``) to score the task suite alone — this is what the
    fleet sweep orchestrator does when ranking adapters by ``task_mean``.
    """
    perplexity = None
    if not skip_perplexity:
        if dataset_dir is None:
            raise ValueError(
                "perplexity needs dataset_dir (test.jsonl); pass skip_perplexity=True to skip"
            )
        perplexity = compute_perplexity(adapter_dir, dataset_dir)
    tasks = score_tasks(adapter_dir, tasks_path)
    return EvalResult(perplexity=perplexity, tasks=tasks)
