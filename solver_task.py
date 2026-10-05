#!/usr/bin/env python3
"""RL-on-solving baseline: GRPO on NuminaMath problems with a final-answer reward.

Same recipe as the verifier runs (batch 64, group 8, lr 1e-6, KL 0.001, 6,656-token
responses); only the task differs: the model solves the problem and is rewarded for
a \\boxed{} answer equivalent to the reference. The resulting solver is then
evaluated as a verifier with the paper's suite (prompted, not trained, to verify).

`prepare` writes train/train_weighted/validation/test parquet files in the VERL row
schema plus a manifest that `training_contract.py --write-contract` can freeze.
`compute_score_batch` is the custom reward function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATA_SOURCE = "numina_solver"
SYSTEM_PROMPT = (
    "You are an expert competition mathematician. Solve the problem. Reason carefully in your "
    "thinking section, then give only the final answer after </think> as \\boxed{...}."
)
HF_DATASET = "AI-MO/NuminaMath-1.5"
SOURCES = ("olympiads", "amc_aime", "aops_forum", "cn_contest", "number_theory", "inequalities", "olympiads_ref")
MAX_ANSWER_CHARS = 40
MAX_PROBLEM_CHARS = 2500
BAD_ANSWERS = {"", "proof", "notfound", "none", "n/a"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _problem_key(problem: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", problem.strip().lower()).encode()).hexdigest()


def collect(limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    seen: set[str] = set()
    rows = []
    for item in load_dataset(HF_DATASET, split="train"):
        answer = (item.get("answer") or "").strip()
        problem = (item.get("problem") or "").strip()
        if (
            item.get("question_type") != "math-word-problem"
            or item.get("problem_is_valid") != "Yes"
            or item.get("solution_is_valid") != "Yes"
            or str(item.get("synthetic")) == "True"
            or item.get("source") not in SOURCES
            or answer.lower() in BAD_ANSWERS
            or len(answer) > MAX_ANSWER_CHARS
            or len(problem) > MAX_PROBLEM_CHARS
            or "\\boxed" in problem
        ):
            continue
        key = _problem_key(problem)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "id": f"numina15_{key[:16]}",
                "problem": problem,
                "answer": answer,
                "source": item.get("source"),
                "problem_type": item.get("problem_type"),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def to_row(item: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["problem"]},
        ],
        "ability": "math_solving",
        "reward_model": {"style": "rule", "ground_truth": item["answer"]},
        "extra_info": {"id": item["id"], "source": item["source"], "problem_type": item["problem_type"], "split": split},
    }


def prepare(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw_path = Path(args.raw_output)
    if raw_path.exists():
        pool = json.loads(raw_path.read_text())
    else:
        pool = collect()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(pool, ensure_ascii=False, indent=0) + "\n")
    # keep answers that a boxed-string / math-verify check can score: no equations, functions or prose
    pool = [r for r in pool if "=" not in r["answer"] and " for " not in r["answer"] and "\\text" not in r["answer"]]
    rng = random.Random(args.seed)
    order = sorted(pool, key=lambda r: r["id"])
    rng.shuffle(order)
    if len(order) < args.train_rows + args.validation_rows + args.test_rows:
        raise SystemExit(f"pool too small: {len(order)}")
    validation = order[: args.validation_rows]
    test = order[args.validation_rows : args.validation_rows + args.test_rows]
    train = order[args.validation_rows + args.test_rows : args.validation_rows + args.test_rows + args.train_rows]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "train_weighted": train, "validation": validation, "test": test}
    hashes = {}
    for name, items in splits.items():
        table = pa.Table.from_pylist([to_row(it, name.replace("_weighted", "")) for it in items])
        path = out / f"{name}.parquet"
        pq.write_table(table, path)
        hashes[f"{name}.parquet"] = _sha256(path)
    manifest = {
        "task": "rl_on_solving",
        "seed": args.seed,
        "duplicate_policy": "error",
        "balance_classes": False,
        "exclude_obvious_leaks": False,
        "validation_per_class": args.validation_rows,
        "test_per_class": args.test_rows,
        "usable_candidates": len(pool),
        "short_proofs_removed": 0,
        "obvious_leaks_removed": 0,
        "group_duplicates_removed": 0,
        "theorem_group_overlap": 0,
        "unique_theorem_ids": len(pool),
        "train_counts": {"problems": len(train)},
        "validation_counts": {"problems": len(validation)},
        "test_counts": {"problems": len(test)},
        "train_source_counts": dict(Counter(r["source"] for r in train)),
        "weighted_train": {"rows": len(train), "unique_rows": len(train)},
        "inputs": [str(raw_path)],
        "input_sha256": {str(raw_path): _sha256(raw_path)},
        "parquet_sha256": hashes,
        "hf_dataset": HF_DATASET,
        "filters": {"sources": list(SOURCES), "max_answer_chars": MAX_ANSWER_CHARS, "max_problem_chars": MAX_PROBLEM_CHARS, "answer_excludes": ["=", " for ", "\\text"]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("parquet_sha256", "input_sha256", "filters")}, indent=2))


# ----------------------------------------------------------------- reward -----------------

_BOXED_RE = re.compile(r"\\boxed\s*\{")


def last_boxed(text: str) -> str | None:
    starts = [m.end() for m in _BOXED_RE.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None


def _normalize(answer: str) -> str:
    s = answer.strip().strip("$").strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("\\,", "").replace("\\!", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s


def answers_match(prediction: str, truth: str) -> bool:
    if _normalize(prediction) == _normalize(truth):
        return True
    try:
        from math_verify import parse, verify  # installed on the training pod

        gold = parse(f"\\boxed{{{truth}}}")
        pred = parse(f"\\boxed{{{prediction}}}")
        return bool(verify(gold, pred))
    except Exception:
        return False


def score_one(solution: str, truth: str) -> dict[str, float]:
    think_end = solution.rfind("</think>")
    thinking_valid = float(think_end >= 0 and solution[:think_end].strip() != "")
    answer_region = solution[think_end + len("</think>") :] if think_end >= 0 else solution
    prediction = last_boxed(answer_region)
    boxed = float(prediction is not None)
    correct = float(prediction is not None and answers_match(prediction, truth))
    return {
        "score": correct,
        "accuracy": correct,
        "format_valid": boxed,
        "thinking_valid": thinking_valid,
    }


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    del data_sources, extra_infos
    return [score_one(solution, str(truth)) for solution, truth in zip(solution_strs, ground_truths, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--raw-output", required=True, help="filtered NuminaMath pool (json); reused if present")
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--train-rows", type=int, default=19200)
    prep.add_argument("--validation-rows", type=int, default=200)
    prep.add_argument("--test-rows", type=int, default=200)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
