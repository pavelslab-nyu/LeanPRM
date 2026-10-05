#!/usr/bin/env python3
"""Training-format external evaluation with deterministic benchmark conversion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from external_eval import (
    hard2verify_metrics,
    hard2verify_rows,
    processbench_metrics,
    processbench_rows,
    write_results,
)
from verifier_task import OVERALL_RE, STEP_RE, SYSTEM_PROMPT


def format_aligned_messages(problem: str, proof_steps: list[str]) -> list[dict[str, str]]:
    body = "\n".join(
        f"<step{index}>{step}</step{index}>" for index, step in enumerate(proof_steps, 1)
    )
    user = (
        "Problem:\n"
        f"{problem}\n\n"
        "Solution:\n"
        f"{body}\n\n"
        f"Label all {len(proof_steps)} steps. Then give the overall proof verdict.\n"
        "Use exactly this format:\n"
        "Step 1: correct or incorrect\n"
        "Step 2: correct or incorrect\n"
        "...\n"
        "Overall: correct or incorrect"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_training_labels(text: str, step_count: int) -> tuple[list[int] | None, str]:
    think_end = text.rfind("</think>")
    if think_end < 0:
        return None, "missing_thinking_end"
    reasoning = text[:think_end].removeprefix("<think>").strip()
    if not reasoning:
        return None, "missing_thinking"
    answer = text[think_end + len("</think>") :]
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    predictions: dict[int, str] = {}
    overall = None
    for line in lines:
        if match := STEP_RE.fullmatch(line):
            index = int(match.group(1))
            if index in predictions:
                return None, "duplicate_label"
            predictions[index] = match.group(2).lower()
        elif match := OVERALL_RE.fullmatch(line):
            if overall is not None:
                return None, "duplicate_verdict"
            overall = match.group(1).lower()
        else:
            return None, "malformed_output"
    expected_indices = set(range(1, step_count + 1))
    if set(predictions) != expected_indices:
        return None, "missing_or_extra_labels"
    if overall not in {"correct", "incorrect"}:
        return None, "missing_verdict"
    labels = [1 if predictions[index] == "correct" else 0 for index in range(1, step_count + 1)]
    implied = "correct" if all(labels) else "incorrect"
    if overall != implied:
        return None, "inconsistent_verdict"
    if 0 in labels and any(labels[index] for index in range(labels.index(0) + 1, len(labels))):
        return labels, "valid_non_monotonic_suffix"
    return labels, "valid"


def first_error(labels: list[int]) -> int:
    return next((index for index, label in enumerate(labels) if label == 0), -1)


def convert_prediction(dataset: str, labels: list[int] | None) -> list[int] | int | None:
    if labels is None:
        return None
    if dataset == "processbench":
        return first_error(labels)
    return labels


def prmbench_rows(dataset_name: str = "hitsmy/PRMBench_Preview") -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = []
    for item in load_dataset(dataset_name, split="train"):
        steps = list(item["modified_process"])
        error_steps = {int(index) for index in item.get("error_steps") or []}
        gold = [0 if index in error_steps else 1 for index in range(1, len(steps) + 1)]
        rows.append(
            {
                "dataset": "prmbench",
                "source": item.get("classification") or "unknown",
                "id": item["idx"],
                "problem": item.get("modified_question") or item["question"],
                "proof_steps": steps,
                "step_count": len(steps),
                "gold": gold,
                "gold_first_error": first_error(gold),
            }
        )
    return rows


def prmbench_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = sum(row["prediction"] is not None for row in rows)
    gold_labels = []
    predictions = []
    first_error_hits = 0
    for row in rows:
        gold = row["gold"]
        pred = row["prediction"]
        scored = pred if pred is not None else [1 - label for label in gold]
        gold_labels.extend(gold)
        predictions.extend(scored)
        if pred is not None:
            first_error_hits += first_error(pred) == row["gold_first_error"]

    def accuracy(label: int) -> float:
        indices = [index for index, value in enumerate(gold_labels) if value == label]
        return sum(predictions[index] == label for index in indices) / len(indices) if indices else 0.0

    return {
        "examples": len(rows),
        "parse_rate": valid / len(rows) if rows else 0.0,
        "correct_step_accuracy": accuracy(1),
        "incorrect_step_accuracy": accuracy(0),
        "total_step_accuracy": sum(a == b for a, b in zip(gold_labels, predictions, strict=True)) / len(gold_labels),
        "first_error_accuracy": first_error_hits / len(rows) if rows else 0.0,
        "binary_score_conversion": {"correct": 1.0, "incorrect": 0.0},
    }


def score_generated(rows: list[dict[str, Any]], generations: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != len(generations):
        raise ValueError("Rows and generations differ in length")
    scored = []
    for row, generation in zip(rows, generations, strict=True):
        labels, parser = parse_training_labels(generation, row["step_count"])
        scored.append(
            {
                **row,
                "messages": format_aligned_messages(row["problem"], row["proof_steps"]),
                "generation": generation,
                "step_prediction": labels,
                "prediction": convert_prediction(row["dataset"], labels),
                "parser": parser,
            }
        )
    h2v = [row for row in scored if row["dataset"] == "hard2verify"]
    process = [row for row in scored if row["dataset"] == "processbench"]
    prm = [row for row in scored if row["dataset"] == "prmbench"]
    summary = {
        "protocol": "format_aligned",
        "failure_categories": dict(Counter(row["parser"] for row in scored)),
        "label_coverage": sum(row["step_prediction"] is not None for row in scored) / len(scored) if scored else 0.0,
        "hard2verify": hard2verify_metrics(h2v) if h2v else None,
        "processbench": processbench_metrics(process) if process else None,
        "prmbench": prmbench_metrics(prm) if prm else None,
    }
    return scored, summary


def generate(model: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [
        tokenizer.apply_chat_template(
            format_aligned_messages(row["problem"], row["proof_steps"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for row in rows
    ]
    llm = LLM(
        model=model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed),
        use_tqdm=True,
    )
    return [output.outputs[0].text for output in outputs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hard2verify-repo", type=Path, required=True)
    parser.add_argument("--processbench-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prmbench-dataset", default="hitsmy/PRMBench_Preview")
    parser.add_argument("--max-tokens", type=int, default=6656)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = hard2verify_rows(args.hard2verify_repo) + processbench_rows(args.processbench_repo) + prmbench_rows(args.prmbench_dataset)
    if args.limit_per_dataset is not None:
        rows = [
            row
            for dataset in ("hard2verify", "processbench", "prmbench")
            for row in [item for item in rows if item["dataset"] == dataset][: args.limit_per_dataset]
        ]
    generations = generate(args.model, rows, args)
    scored, summary = score_generated(rows, generations)
    summary.update(
        {
            "name": args.name,
            "model": args.model,
            "decoding": {"temperature": args.temperature, "seed": args.seed, "max_tokens": args.max_tokens},
            "prmbench_adapter": "modified_question + modified_process; binary labels converted to 1.0/0.0 scores",
        }
    )
    write_results(args.output_dir / args.name, scored, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
