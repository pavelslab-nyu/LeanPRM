"""Held-out evaluation for the base and trained verifier."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from verifier_task import _parse_truth, _score_one
from solver_task import score_one as solver_score_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["verifier", "solver"], default="verifier")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = list(load_dataset("parquet", data_files=str(args.data), split="train"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for row in rows
    ]
    model = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=16384,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=True,
    )
    outputs = model.generate(
        prompts,
        SamplingParams(temperature=args.temperature, max_tokens=6656, seed=args.seed),
    )

    totals: dict[str, list[float]] = defaultdict(list)
    details = []
    for row, output in zip(rows, outputs, strict=True):
        text = output.outputs[0].text
        if args.task == "solver":
            score = solver_score_one(text, str(row["reward_model"]["ground_truth"]))
        else:
            score = _score_one(text, _parse_truth(row["reward_model"]["ground_truth"]))
        for key, value in score.items():
            if isinstance(value, (int, float)):
                totals[key].append(float(value))
        details.append({"extra_info": row["extra_info"], "output": text, "metrics": score})

    metrics = {key: sum(values) / len(values) for key, values in totals.items()}
    metrics["examples"] = len(rows)
    metrics["model"] = args.model
    metrics["temperature"] = args.temperature
    metrics["seed"] = args.seed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.name}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.output_dir / f"{args.name}.jsonl").open("w") as handle:
        for row in details:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
