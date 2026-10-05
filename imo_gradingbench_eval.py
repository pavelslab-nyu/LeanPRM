#!/usr/bin/env python3
"""IMO-GradingBench evaluation.

Each row is (problem, candidate response, human grade). We binarise the grade as
Correct vs not-Correct and ask each model for an overall verdict, giving each
model the prompt format it handles best:

  ours   - the verifier's training layout: paragraphs of the response become
           <stepN> elements, the model writes a reasoning section and then a
           label per step plus `Overall:`; we read the Overall line.
  native - a plain instruction that asks for a single word, used for the base
           model, which cannot follow the training layout.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from verifier_task import OVERALL_RE, STEP_RE, SYSTEM_PROMPT

MAX_STEPS = 30
MAX_RESPONSE_CHARS = 24000
NATIVE_RE = re.compile(r"\b(correct|incorrect|yes|no)\b", re.IGNORECASE)


def split_steps(response: str, max_steps: int = MAX_STEPS) -> list[str]:
    """Deterministic paragraph split, merging evenly if there are too many."""
    parts = [p.strip() for p in response.split("\n\n") if p.strip()]
    if not parts:
        parts = [response.strip()]
    if len(parts) <= max_steps:
        return parts
    per = -(-len(parts) // max_steps)
    return ["\n\n".join(parts[i : i + per]) for i in range(0, len(parts), per)]


def ours_messages(problem: str, steps: list[str]) -> list[dict[str, str]]:
    body = "\n".join(f"<step{i}>{s}</step{i}>" for i, s in enumerate(steps, 1))
    user = (
        "Problem:\n"
        f"{problem}\n\n"
        "Solution:\n"
        f"{body}\n\n"
        f"Label all {len(steps)} steps. Then give the overall proof verdict.\n"
        "Use exactly this format:\n"
        "Step 1: correct or incorrect\n"
        "Step 2: correct or incorrect\n"
        "...\n"
        "Overall: correct or incorrect"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def native_messages(problem: str, response: str) -> list[dict[str, str]]:
    user = (
        "You are grading a proposed solution to a competition mathematics problem.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Proposed solution:\n{response}\n\n"
        "Is the proposed solution completely correct? Reply with exactly one word, "
        "either Correct or Incorrect."
    )
    return [{"role": "user", "content": user}]


def parse_ours(text: str, step_count: int) -> tuple[int | None, str]:
    end = text.rfind("</think>")
    if end < 0:
        return None, "missing_thinking_end"
    answer = text[end + len("</think>") :]
    overall = None
    for line in (l.strip() for l in answer.strip().splitlines()):
        if not line:
            continue
        if match := OVERALL_RE.fullmatch(line):
            overall = match.group(1).lower()
        elif not STEP_RE.fullmatch(line):
            continue
    if overall is None:
        return None, "missing_verdict"
    return int(overall == "correct"), "valid"


def parse_native(text: str) -> tuple[int | None, str]:
    tail = text[text.rfind("</think>") + len("</think>") :] if "</think>" in text else text
    words = [m.group(1).lower() for m in NATIVE_RE.finditer(tail)]
    if not words:
        return None, "missing_verdict"
    negated = re.search(r"\bnot\s+correct\b", tail, re.IGNORECASE) is not None
    if negated or "incorrect" in words or "no" in words:
        return 0, "valid"
    return 1, "valid"


def load_rows(limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = []
    for item in load_dataset("Hwilner/imo-gradingbench", split="train"):
        response = item["Response"] or ""
        if not response.strip() or len(response) > MAX_RESPONSE_CHARS:
            continue
        rows.append(
            {
                "id": item["Grading ID"],
                "problem_id": item["Problem ID"],
                "problem": item["Problem"],
                "response": response,
                "steps": split_steps(response),
                "reward": item["Reward"],
                "gold": int(item["Reward"] == "Correct"),
            }
        )
    return rows[:limit] if limit else rows


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r["prediction"] is not None]
    pos = [r for r in rows if r["gold"] == 1]
    neg = [r for r in rows if r["gold"] == 0]
    hit = lambda group: sum(r["prediction"] == r["gold"] for r in group)
    recall_pos = hit(pos) / len(pos) if pos else 0.0
    recall_neg = hit(neg) / len(neg) if neg else 0.0
    return {
        "examples": len(rows),
        "parse_rate": len(valid) / len(rows) if rows else 0.0,
        "accuracy": hit(rows) / len(rows) if rows else 0.0,
        "correct_recall": recall_pos,
        "incorrect_recall": recall_neg,
        "balanced_accuracy": (recall_pos + recall_neg) / 2,
        "predicted_correct_rate": sum(r["prediction"] == 1 for r in valid) / len(valid) if valid else 0.0,
        "failure_categories": dict(Counter(r["parser"] for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--protocol", choices=["ours", "native"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=6656)
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_rows(args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    build = (
        (lambda r: ours_messages(r["problem"], r["steps"]))
        if args.protocol == "ours"
        else (lambda r: native_messages(r["problem"], r["response"]))
    )
    prompts = [
        tokenizer.apply_chat_template(
            build(row), tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for row in rows
    ]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=args.temperature, seed=args.seed, max_tokens=args.max_tokens),
    )
    for row, output in zip(rows, outputs, strict=True):
        text = output.outputs[0].text
        if args.protocol == "ours":
            prediction, status = parse_ours(text, len(row["steps"]))
        else:
            prediction, status = parse_native(text)
        row["generation"] = text
        row["prediction"] = prediction
        row["parser"] = status

    summary = {
        "name": args.name,
        "model": args.model,
        "protocol": args.protocol,
        "temperature": args.temperature,
        "step_cap": MAX_STEPS,
        "max_response_chars": MAX_RESPONSE_CHARS,
        "imo_gradingbench": metrics(rows),
    }
    out = args.output_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary["imo_gradingbench"], indent=2))


if __name__ == "__main__":
    main()
