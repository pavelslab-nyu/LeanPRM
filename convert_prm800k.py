#!/usr/bin/env python3
"""Convert PRM800K labeler records into the verifier dataset format.

Each record is one labelled solution to a MATH problem. Row layout matches the
Lean dataset: steps[0] is the problem statement, steps[1:] are solution steps.
Human ratings give labels: the first step rated -1 is the error step (label
reasoning_error); every step before it is correct; every step after it is
labelled incorrect as well (prefix validity, same convention as the Lean data).
Solutions with no -1 rating and finish_reason "solution" are correct proofs.
Quality-control and screening records, give-ups, bad problem statements and
records whose rated prefix disagrees with the pre-generated steps are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def rating_of(step: dict) -> int | None:
    completions = step.get("completions") or []
    if step.get("chosen_completion") is not None and 0 <= step["chosen_completion"] < len(completions):
        chosen = completions[step["chosen_completion"]]
    elif completions:
        chosen = completions[0]
    else:
        return None
    return chosen.get("rating")


def convert(paths: list[Path]) -> tuple[dict, dict]:
    out = {"correct_proofs": [], "incorrect_proofs": []}
    stats = Counter()
    seen_solutions: set[str] = set()
    for path in paths:
        phase = path.stem
        with path.open() as handle:
            for line_no, line in enumerate(handle):
                if not line.strip():
                    continue
                rec = json.loads(line)
                stats["records"] += 1
                if rec.get("is_quality_control_question") or rec.get("is_initial_screening_question"):
                    stats["skip_qc"] += 1
                    continue
                label = rec["label"]
                finish = label.get("finish_reason")
                question = rec["question"]
                problem = (question.get("problem") or "").strip()
                pre_steps = [s.strip() for s in (question.get("pre_generated_steps") or [])]
                if not problem or not pre_steps:
                    stats["skip_empty"] += 1
                    continue
                if finish not in ("found_error", "solution"):
                    stats[f"skip_{finish}"] += 1
                    continue
                rated = []
                mismatch = False
                for i, step in enumerate(label.get("steps") or []):
                    r = rating_of(step)
                    text = None
                    comps = step.get("completions") or []
                    if comps:
                        idx = step.get("chosen_completion")
                        text = (comps[idx] if idx is not None and 0 <= idx < len(comps) else comps[0]).get("text")
                    if step.get("human_completion"):
                        mismatch = True  # labeler wrote their own step; solution diverges from the sampled one
                        break
                    if text is not None and i < len(pre_steps) and text.strip() != pre_steps[i]:
                        mismatch = True
                        break
                    rated.append(r)
                if mismatch:
                    stats["skip_diverged"] += 1
                    continue
                first_error = next((i for i, r in enumerate(rated) if r == -1), None)
                if finish == "found_error" and first_error is None:
                    stats["skip_no_minus_one"] += 1
                    continue
                if finish == "solution" and first_error is not None:
                    stats["skip_solution_with_error"] += 1
                    continue
                if finish == "solution" and len(rated) < len(pre_steps):
                    stats["skip_partial_solution"] += 1
                    continue
                if any(r is None for r in rated[: (first_error + 1) if first_error is not None else len(rated)]):
                    stats["skip_unrated"] += 1
                    continue
                key = hashlib.sha256((problem + "\n".join(pre_steps)).encode()).hexdigest()
                if key in seen_solutions:
                    stats["skip_duplicate_solution"] += 1
                    continue
                seen_solutions.add(key)
                steps = [{"step": "0", "lean_code": "", "natural_language_description": problem, "label": "correct"}]
                for i, text in enumerate(pre_steps, 1):
                    if first_error is None or i - 1 < first_error:
                        lab = "correct"
                    elif i - 1 == first_error:
                        lab = "reasoning_error"
                    else:
                        lab = "incorrect"
                    steps.append({"step": str(i), "lean_code": "", "natural_language_description": text, "label": lab})
                row = {
                    "id": f"prm800k_{phase}_{line_no}",
                    "lean_statement": problem,
                    "proof_index": 0,
                    "steps": steps,
                    "category": question.get("subject") or "math",
                }
                if first_error is None:
                    out["correct_proofs"].append(row)
                    stats["correct"] += 1
                else:
                    row["error_step"] = first_error + 1
                    row["error_class"] = "human_labelled_error"
                    out["incorrect_proofs"].append(row)
                    stats["incorrect"] += 1
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, stats = convert(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False))
    stats["unique_problems"] = len({r["lean_statement"] for b in data.values() for r in b})
    stats["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    Path(str(args.output) + ".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
