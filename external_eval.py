#!/usr/bin/env python3
"""Evaluate native model outputs on Hard2Verify and ProcessBench.

The benchmark prompts and model generations are left unchanged. The parser
accepts the equivalent label formats these verifier models naturally emit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

LABELS = {
    "yes": 1,
    "y": 1,
    "correct": 1,
    "c": 1,
    "true": 1,
    "no": 0,
    "n": 0,
    "incorrect": 0,
    "i": 0,
    "false": 0,
}
LABEL_PATTERN = r"incorrect|correct|false|true|yes|no|[ynci]"


def answer_text(text: str) -> str:
    """Return only the answer after thinking, never labels mentioned in CoT."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        return ""
    return text.strip()


def label_list(text: str) -> list[int]:
    tokens = re.findall(rf"\b({LABEL_PATTERN})\b", text.lower())
    return [LABELS[token.replace(" ", "")] for token in tokens]


def parse_step_labels(text: str, step_count: int) -> tuple[list[int] | None, str]:
    """Parse Y/N, C/I, correct/incorrect, or native numbered step lines."""
    answer = answer_text(text)
    if not answer:
        return None, "missing_final_answer"

    verdicts = re.findall(
        r"(?im)^\s*(?:verdict|labels?)\s*:\s*(.+?)\s*$", answer
    )
    for verdict in reversed(verdicts):
        labels = label_list(verdict)
        if len(labels) == step_count:
            return labels, "verdict"

    step_matches = re.findall(
        rf"(?i)\bstep\s*\[?(\d+)\]?\s*(?::|=|-|\bis\b)\s*({LABEL_PATTERN})\b",
        answer,
    )
    indexed = {
        int(index): LABELS[token.lower().replace(" ", "")]
        for index, token in step_matches
    }
    if set(indexed) == set(range(step_count)):
        return [indexed[i] for i in range(step_count)], "indexed_steps_zero_based"
    if set(indexed) == set(range(1, step_count + 1)):
        return [indexed[i] for i in range(1, step_count + 1)], "indexed_steps_one_based"

    all_correct = r"\ball\s+(?:of\s+the\s+)?steps\s+(?:are\s+)?correct\b"
    if re.search(all_correct, answer, re.IGNORECASE) and not re.search(
        rf"\bnot\s+{all_correct}", answer, re.IGNORECASE
    ):
        return [1] * step_count, "all_steps_correct"

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if lines:
        labels = label_list(lines[-1])
        if len(labels) == step_count:
            return labels, "final_line"
    return None, "unparsed"


def parse_first_error(text: str, step_count: int) -> tuple[int | None, str]:
    """Parse ProcessBench's boxed index or the verifier's native output."""
    answer = answer_text(text)
    if not answer:
        return None, "missing_final_answer"

    boxed = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", answer)
    if boxed:
        value = int(boxed[-1])
        return (value, "boxed") if -1 <= value < step_count else (None, "out_of_range")

    verdicts = re.findall(r"(?im)^\s*verdict\s*:\s*(-?\d+)\s*[.!]?\s*$", answer)
    if verdicts:
        value = int(verdicts[-1])
        return (value, "verdict") if -1 <= value < step_count else (None, "out_of_range")

    labels, parser = parse_step_labels(text, step_count)
    if labels is not None:
        return next((i for i, label in enumerate(labels) if label == 0), -1), parser

    overall = re.findall(
        rf"(?im)^\s*overall\s*:\s*({LABEL_PATTERN})\b", answer
    )
    if overall and LABELS[overall[-1].lower().replace(" ", "")] == 1:
        return -1, "overall"

    paragraph = re.findall(
        r"(?i)(?:earliest|first)\s+(?:error|incorrect\s+(?:step|paragraph)).*?\b(?:step|paragraph)\s*(\d+)\b",
        answer,
    )
    if paragraph:
        value = int(paragraph[-1])
        return (value, "error_statement") if value < step_count else (None, "out_of_range")

    answer_index = re.findall(r"(?i)\b(?:answer|index)\s*(?:is|:)\s*(-?\d+)\b", answer)
    if answer_index:
        value = int(answer_index[-1])
        return (value, "answer_index") if -1 <= value < step_count else (None, "out_of_range")

    if re.search(r"(?i)\b(?:no\s+(?:errors?|incorrect\s+steps?)|all\s+steps\s+(?:are\s+)?correct)\b", answer):
        return -1, "no_error_statement"

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if lines and re.fullmatch(r"-?\d+", lines[-1]):
        value = int(lines[-1])
        return (value, "final_integer") if -1 <= value < step_count else (None, "out_of_range")
    return None, "unparsed"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hard2verify_rows(repo: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    utils = load_module(repo / "utils.py", "hard2verify_utils")
    dataset = load_dataset("Salesforce/Hard2Verify", split="test").map(utils.decrypt_sample)
    rows = []
    for item in dataset:
        tagged = "\n".join(
            f"<step>[{i}] {step}</step>"
            for i, step in enumerate(item["model_response_by_step"])
        )
        user = utils.step_level_user_prompt.format(problem=item["question"], steps=tagged)
        rows.append(
            {
                "dataset": "hard2verify",
                "source": "hard2verify",
                "id": item["unique_id"],
                "problem": item["question"],
                "proof_steps": list(item["model_response_by_step"]),
                "messages": [
                    {"role": "system", "content": utils.step_level_system_prompt},
                    {"role": "user", "content": user},
                ],
                "step_count": len(item["model_response_by_step"]),
                "gold": [int(label) for label in item["human_labels"]],
            }
        )
    return rows


def processbench_rows(repo: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    template = (repo / "code/templates/critique_template.txt").read_text().strip()
    rows = []
    for source in ("gsm8k", "math", "olympiadbench", "omnimath"):
        for item in load_dataset("Qwen/ProcessBench", split=source):
            tagged = "\n\n".join(
                f"<paragraph_{i}>\n{step}\n</paragraph_{i}>"
                for i, step in enumerate(item["steps"])
            )
            rows.append(
                {
                    "dataset": "processbench",
                    "source": source,
                    "id": f"{source}:{item['id']}",
                    "problem": item["problem"],
                    "proof_steps": list(item["steps"]),
                    "messages": [{"role": "user", "content": template.format(problem=item["problem"], tagged_response=tagged)}],
                    "step_count": len(item["steps"]),
                    "gold": int(item["label"]),
                }
            )
    return rows


def generate(model: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
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
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
        use_tqdm=True,
    )
    results = []
    for row, output in zip(rows, outputs, strict=True):
        generation = output.outputs[0].text
        if row["dataset"] == "hard2verify":
            prediction, parser = parse_step_labels(generation, row["step_count"])
        else:
            prediction, parser = parse_first_error(generation, row["step_count"])
        results.append({**row, "generation": generation, "prediction": prediction, "parser": parser})
    return results


def recall(gold: list[int], pred: list[int], label: int) -> float:
    indices = [i for i, value in enumerate(gold) if value == label]
    return sum(pred[i] == label for i in indices) / len(indices) if indices else 0.0


def harmonic(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if a + b else 0.0


def hard2verify_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    gold_steps: list[int] = []
    pred_steps: list[int] = []
    gold_proofs: list[int] = []
    pred_proofs: list[int] = []
    valid = first_error_hits = 0
    for row in rows:
        gold = row["gold"]
        pred = row["prediction"]
        gold_proof = int(all(gold))
        if pred is None:
            scored = [1 - label for label in gold]
            pred_proof = 1 - gold_proof
        else:
            valid += 1
            scored = pred
            pred_proof = int(all(pred))
            gold_first = next((i for i, label in enumerate(gold) if label == 0), -1)
            pred_first = next((i for i, label in enumerate(pred) if label == 0), -1)
            first_error_hits += gold_first == pred_first
        gold_steps.extend(gold)
        pred_steps.extend(scored)
        gold_proofs.append(gold_proof)
        pred_proofs.append(pred_proof)
    step_tpr = recall(gold_steps, pred_steps, 1)
    step_tnr = recall(gold_steps, pred_steps, 0)
    proof_tpr = recall(gold_proofs, pred_proofs, 1)
    proof_tnr = recall(gold_proofs, pred_proofs, 0)
    return {
        "examples": len(rows),
        "parse_rate": valid / len(rows),
        "step_tpr": step_tpr,
        "step_tnr": step_tnr,
        "step_balanced_accuracy": (step_tpr + step_tnr) / 2,
        "proof_tpr": proof_tpr,
        "proof_tnr": proof_tnr,
        "proof_balanced_accuracy": (proof_tpr + proof_tnr) / 2,
        "first_error_accuracy": first_error_hits / len(rows),
    }


def processbench_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"examples": len(rows)}
    valid = sum(row["prediction"] is not None for row in rows)
    result["parse_rate"] = valid / len(rows)
    source_f1 = []
    for source in ("gsm8k", "math", "olympiadbench", "omnimath"):
        subset = [row for row in rows if row["source"] == source]
        incorrect = [row for row in subset if row["gold"] != -1]
        correct = [row for row in subset if row["gold"] == -1]
        error_acc = (
            sum(row["prediction"] == row["gold"] for row in incorrect) / len(incorrect)
            if incorrect
            else 0.0
        )
        correct_acc = (
            sum(row["prediction"] == -1 for row in correct) / len(correct)
            if correct
            else 0.0
        )
        f1 = harmonic(error_acc, correct_acc)
        result[source] = {
            "error_accuracy": error_acc,
            "correct_accuracy": correct_acc,
            "f1": f1,
        }
        source_f1.append(f1)
    result["macro_f1"] = sum(source_f1) / len(source_f1)
    return result


def write_results(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / "results.jsonl.tmp"
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path / "results.jsonl")
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hard2verify-repo", type=Path, required=True)
    parser.add_argument("--processbench-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=6656)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hard2verify = hard2verify_rows(args.hard2verify_repo)
    processbench = processbench_rows(args.processbench_repo)
    if args.limit_per_dataset is not None:
        hard2verify = hard2verify[: args.limit_per_dataset]
        processbench = processbench[: args.limit_per_dataset]
    rows = generate(args.model, hard2verify + processbench, args)
    h2v = [row for row in rows if row["dataset"] == "hard2verify"]
    pb = [row for row in rows if row["dataset"] == "processbench"]
    summary = {
        "name": args.name,
        "model": args.model,
        "decoding": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "generation": "native model output; no constrained decoding or rewriting",
            "accepted_labels": ["yes/no", "Y/N", "correct/incorrect", "C/I", "true/false"],
        },
        "hard2verify": hard2verify_metrics(h2v),
        "processbench": processbench_metrics(pb),
    }
    write_results(args.output_dir / args.name, rows, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
