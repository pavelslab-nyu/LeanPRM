"""Prepare the Lean verifier data and score verifier rollouts for verl."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SYSTEM_PROMPT = """You are a mathematical proof verifier. Reason carefully about every numbered natural-language step. Once a step is incorrect, every later step is incorrect because it depends on an invalid proof state. Use your thinking section for the analysis. After </think>, return only the requested labels with no explanation."""

STEP_RE = re.compile(r"^Step\s+(\d+):\s*(correct|incorrect)\s*$", re.IGNORECASE)
OVERALL_RE = re.compile(r"^Overall:\s*(correct|incorrect)\s*$", re.IGNORECASE)
STEP_REWARD_WEIGHT = 0.7
VERDICT_REWARD_WEIGHT = 0.3
OBVIOUS_LEAK_RE = re.compile(
    r"\b(unjustified|invalid|false|does not follow|not justified|cannot be concluded|not actually)\b",
    re.IGNORECASE,
)
SOURCE_PREFIXES = (
    "lean_workbook_hf",
    "lean_workbook",
    "pythagoras",
    "proofnet",
    "numina",
    "corpus",
    "herald",
    "fqb",
)


def _label(step: dict) -> str:
    return "correct" if step.get("label") == "correct" else "incorrect"


def _proof_prompt(proof: dict) -> tuple[str, list[str]]:
    steps = proof["steps"]
    labels = [_label(step) for step in steps[1:]]
    problem = steps[0]["natural_language_description"].strip()
    body = "\n".join(
        f"<step{i}>{step['natural_language_description'].strip()}</step{i}>"
        for i, step in enumerate(steps[1:], 1)
    )
    return (
        "Problem:\n"
        f"{problem}\n\n"
        "Solution:\n"
        f"{body}\n\n"
        f"Label all {len(labels)} steps. Then give the overall proof verdict.\n"
        "Use exactly this format:\n"
        "Step 1: correct or incorrect\n"
        "Step 2: correct or incorrect\n"
        "...\n"
        "Overall: correct or incorrect",
        labels,
    )


def _stable_key(seed: int, proof: dict) -> str:
    value = f"{seed}:{proof['id']}:{proof.get('proof_index', 0)}"
    return hashlib.sha256(value.encode()).hexdigest()


def _source_name(proof: dict, path: Path) -> str:
    for key in ("source", "dataset", "source_dataset"):
        value = proof.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    theorem_id = str(proof.get("id", "")).lower()
    for prefix in SOURCE_PREFIXES:
        if theorem_id.startswith(prefix + "_"):
            return prefix
    return path.stem


def _theorem_group(proof: dict) -> str:
    statement = str(proof.get("lean_statement", ""))
    statement = re.sub(r"\b(theorem|lemma)\s+[^\s(:]+", r"\1 __name__", statement, count=1)
    statement = re.sub(r"\s*:=\s*by\s*$", "", statement.strip())
    canonical = re.sub(r"\s+", " ", statement)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _obvious_leak(proof: dict) -> bool:
    if proof.get("_overall") != "incorrect":
        return False
    error_step = proof.get("error_step")
    steps = proof.get("steps", [])
    if not isinstance(error_step, int) or not 0 <= error_step < len(steps):
        return False
    text = str(steps[error_step].get("natural_language_description", ""))
    return bool(OBVIOUS_LEAK_RE.search(text))


def _load_proofs(
    paths: list[Path], duplicate_policy: str, exclude_obvious_leaks: bool
) -> tuple[list[dict], dict[str, int]]:
    candidates = []
    skipped_short_proofs = 0
    obvious_leaks_removed = 0
    for path in paths:
        data = json.loads(path.read_text())
        for bucket, overall in (("correct_proofs", "correct"), ("incorrect_proofs", "incorrect")):
            for proof in data[bucket]:
                proof = dict(proof)
                proof["_overall"] = overall
                if exclude_obvious_leaks and _obvious_leak(proof):
                    obvious_leaks_removed += 1
                    continue
                if len(proof.get("steps", [])) < 2:
                    skipped_short_proofs += 1
                    continue
                proof["_source"] = _source_name(proof, path)
                proof["_theorem_group"] = _theorem_group(proof)
                candidates.append(proof)

    by_id = {}
    for proof in candidates:
        theorem_id = proof["id"]
        if theorem_id in by_id and duplicate_policy == "error":
            raise ValueError(f"duplicate theorem id across inputs: {theorem_id}")
        if theorem_id not in by_id or (
            duplicate_policy == "prefer-incorrect"
            and proof["_overall"] == "incorrect"
            and by_id[theorem_id]["_overall"] == "correct"
        ):
            by_id[theorem_id] = proof

    stats = {
        "usable_candidates": len(candidates),
        "short_proofs_removed": skipped_short_proofs,
        "obvious_leaks_removed": obvious_leaks_removed,
        "duplicate_candidates_removed": len(candidates) - len(by_id),
    }
    return list(by_id.values()), stats


def _split_by_theorem_group(
    rows: list[dict], seed: int, validation_per_class: int, test_per_class: int
) -> tuple[list[dict], list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["extra_info"]["theorem_group"]].append(row)
    remaining = dict(grouped)

    def take(split_name: str, per_class: int) -> list[dict]:
        target = {"correct": per_class, "incorrect": per_class}
        counts = Counter()
        selected_groups = []
        ordered = sorted(
            remaining,
            key=lambda group: hashlib.sha256(f"split:{seed}:{split_name}:{group}".encode()).hexdigest(),
        )
        for group in ordered:
            group_counts = Counter(row["extra_info"]["overall"] for row in remaining[group])
            if any(group_counts[label] and counts[label] + group_counts[label] > target[label] for label in target):
                continue
            if not any(group_counts[label] and counts[label] < target[label] for label in target):
                continue
            selected_groups.append(group)
            counts.update(group_counts)
            if counts == Counter(target):
                break
        if counts != Counter(target):
            raise ValueError(f"could not form {split_name} theorem-group split: {dict(counts)}")
        selected = [row for group in selected_groups for row in remaining.pop(group)]
        return selected

    validation_rows = take("validation", validation_per_class)
    test_rows = take("test", test_per_class)
    train_rows = [row for group in remaining.values() for row in group]
    return train_rows, validation_rows, test_rows


def _weighted_training_view(rows: list[dict], seed: int) -> tuple[list[dict], dict]:
    """Make a deterministic source-by-verdict balanced epoch without dropping any row."""
    cells = defaultdict(list)
    for row in rows:
        info = row["extra_info"]
        cells[(info["source"], info["overall"])].append(row)

    rng = random.Random(seed)
    weighted = []
    repeats = Counter()
    sources = sorted({source for source, _ in cells})
    for source in sources:
        correct = list(cells[(source, "correct")])
        incorrect = list(cells[(source, "incorrect")])
        if not correct or not incorrect:
            weighted.extend(correct + incorrect)
            continue
        target = max(len(correct), len(incorrect))
        for label, cell in (("correct", correct), ("incorrect", incorrect)):
            ordered = sorted(cell, key=lambda row: row["extra_info"]["id"])
            rng.shuffle(ordered)
            weighted.extend(ordered)
            for index in range(target - len(ordered)):
                weighted.append(ordered[index % len(ordered)])
                repeats[(source, label)] += 1
    rng.shuffle(weighted)
    counts = Counter((row["extra_info"]["source"], row["extra_info"]["overall"]) for row in weighted)
    unique_rows = {
        (row["extra_info"]["id"], row["extra_info"].get("proof_index", 0)) for row in weighted
    }
    return weighted, {
        "method": "oversample smaller verdict cell to parity within each source",
        "rows": len(weighted),
        "unique_rows": len(unique_rows),
        "source_verdict_counts": {f"{source}:{label}": count for (source, label), count in sorted(counts.items())},
        "added_repeats": {f"{source}:{label}": count for (source, label), count in sorted(repeats.items())},
    }


def _write_filtered_json(inputs: list[Path], output: Path) -> dict[str, int]:
    result = {"correct_proofs": [], "incorrect_proofs": []}
    removed = 0
    for path in inputs:
        data = json.loads(path.read_text())
        result["correct_proofs"].extend(data.get("correct_proofs", []))
        for original in data.get("incorrect_proofs", []):
            proof = dict(original)
            proof["_overall"] = "incorrect"
            if _obvious_leak(proof):
                removed += 1
            else:
                result["incorrect_proofs"].append(original)
    result["filter_manifest"] = {
        "rule": "strict admission-at-error-step phrase filter",
        "removed_incorrect_proofs": removed,
        "remaining_correct_proofs": len(result["correct_proofs"]),
        "remaining_incorrect_proofs": len(result["incorrect_proofs"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return result["filter_manifest"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trivial_baseline(rows: list[dict], step_prediction: str, verdict_prediction: str) -> dict[str, float]:
    truths = [_parse_truth(row["reward_model"]["ground_truth"]) for row in rows]
    scores = []
    for truth in truths:
        labels = "\n".join(
            [
                *(f"Step {i}: {step_prediction}" for i in range(1, len(truth["steps"]) + 1)),
                f"Overall: {verdict_prediction}",
            ]
        )
        solution = f"baseline reasoning\n</think>\n{labels}"
        scores.append(_score_one(solution, truth))
    return {
        key: sum(float(score[key]) for score in scores) / len(scores)
        for key in (
            "score",
            "step_score",
            "step_accuracy",
            "exact_step_vector_accuracy",
            "first_error_accuracy",
            "verdict_accuracy",
            "exact_proof_accuracy",
            "label_consistency",
        )
    }


def prepare(
    inputs: list[Path],
    output_dir: Path,
    seed: int,
    validation_per_class: int,
    test_per_class: int,
    duplicate_policy: str,
    balance_classes: bool,
    exclude_obvious_leaks: bool,
    write_weighted_train: bool,
    filtered_json: Path | None,
    pin_split: Path | None = None,
    dedupe_by_group: bool = False,
) -> None:
    from datasets import Dataset

    proofs, load_stats = _load_proofs(inputs, duplicate_policy, exclude_obvious_leaks)
    by_label = {
        label: sorted((proof for proof in proofs if proof["_overall"] == label), key=lambda p: _stable_key(seed, p))
        for label in ("correct", "incorrect")
    }
    if balance_classes:
        per_class = min(map(len, by_label.values()))
        selected = by_label["correct"][:per_class] + by_label["incorrect"][:per_class]
    else:
        selected = by_label["correct"] + by_label["incorrect"]

    rows = []
    for proof in selected:
        prompt, labels = _proof_prompt(proof)
        truth = {"steps": labels, "overall": proof["_overall"]}
        rows.append(
            {
                "data_source": "lean_nl_verifier",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "proof_verification",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(truth, separators=(",", ":")),
                },
                "extra_info": {
                    "id": proof["id"],
                    "proof_index": proof.get("proof_index", 0),
                    "category": proof.get("category"),
                    "source": proof["_source"],
                    "theorem_group": proof["_theorem_group"],
                    "overall": proof["_overall"],
                    "step_count": len(labels),
                },
            }
        )

    pinned = json.loads(pin_split.read_text()) if pin_split else None
    protected_groups = (
        set(pinned.get("validation", {}).get("groups", [])) | set(pinned["test"]["groups"]) if pinned else set()
    )
    group_duplicates_removed = 0
    if dedupe_by_group:
        # One data point per theorem: among rows that share a theorem group and
        # are not part of a pinned held-out split, keep the one with the
        # smallest stable key. Pinned validation/test rows are left untouched.
        best: dict[str, tuple[str, int]] = {}
        for index, row in enumerate(rows):
            group = row["extra_info"]["theorem_group"]
            if group in protected_groups:
                continue
            key = _stable_key(seed, {"id": row["extra_info"]["id"], "proof_index": row["extra_info"]["proof_index"]})
            if group not in best or key < best[group][0]:
                best[group] = (key, index)
        keep = {index for _, index in best.values()}
        deduped = [
            row
            for index, row in enumerate(rows)
            if row["extra_info"]["theorem_group"] in protected_groups or index in keep
        ]
        group_duplicates_removed = len(rows) - len(deduped)
        rows = deduped

    if pinned:
        test_groups = set(pinned["test"]["groups"])
        test_rows = [row for row in rows if row["extra_info"]["theorem_group"] in test_groups]
        if sorted(row["extra_info"]["id"] for row in test_rows) != sorted(pinned["test"]["ids"]):
            raise ValueError("pinned test split does not reproduce the recorded rows")
        if "validation" in pinned:
            validation_groups = set(pinned["validation"]["groups"])
            validation_rows = [row for row in rows if row["extra_info"]["theorem_group"] in validation_groups]
            if sorted(row["extra_info"]["id"] for row in validation_rows) != sorted(pinned["validation"]["ids"]):
                raise ValueError("pinned validation split does not reproduce the recorded rows")
            train_rows = [
                row for row in rows if row["extra_info"]["theorem_group"] not in validation_groups | test_groups
            ]
        else:
            # Test pinned, validation drawn fresh (seeded, theorem-disjoint) from the remaining rows.
            remaining = [row for row in rows if row["extra_info"]["theorem_group"] not in test_groups]
            train_rows, validation_rows, _ = _split_by_theorem_group(remaining, seed, validation_per_class, 0)
    else:
        held_out = validation_per_class + test_per_class
        for label in ("correct", "incorrect"):
            if sum(row["extra_info"]["overall"] == label for row in rows) <= held_out:
                raise ValueError(f"not enough {label} examples for {held_out} held-out rows")
        train_rows, validation_rows, test_rows = _split_by_theorem_group(
            rows, seed, validation_per_class, test_per_class
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_rows).to_parquet(output_dir / "train.parquet")
    weighted_manifest = None
    if write_weighted_train:
        weighted_rows, weighted_manifest = _weighted_training_view(train_rows, seed)
        Dataset.from_list(weighted_rows).to_parquet(output_dir / "train_weighted.parquet")
    Dataset.from_list(validation_rows).to_parquet(output_dir / "validation.parquet")
    Dataset.from_list(test_rows).to_parquet(output_dir / "test.parquet")

    filtered_manifest = _write_filtered_json(inputs, filtered_json) if filtered_json else None
    split_groups = {
        name: {row["extra_info"]["theorem_group"] for row in split}
        for name, split in (("train", train_rows), ("validation", validation_rows), ("test", test_rows))
    }
    assert not split_groups["train"] & split_groups["validation"]
    assert not split_groups["train"] & split_groups["test"]
    assert not split_groups["validation"] & split_groups["test"]

    manifest = {
        "seed": seed,
        "inputs": [str(path) for path in inputs],
        "duplicate_policy": duplicate_policy,
        "balance_classes": balance_classes,
        "exclude_obvious_leaks": exclude_obvious_leaks,
        "validation_per_class": validation_per_class,
        "test_per_class": test_per_class,
        **load_stats,
        "candidate_counts": dict(Counter(proof["_overall"] for proof in proofs)),
        "selected_counts": dict(Counter(proof["_overall"] for proof in selected)),
        "train_counts": dict(Counter(row["extra_info"]["overall"] for row in train_rows)),
        "validation_counts": dict(Counter(row["extra_info"]["overall"] for row in validation_rows)),
        "test_counts": dict(Counter(row["extra_info"]["overall"] for row in test_rows)),
        "train_source_verdict_counts": {
            f"{source}:{label}": count
            for (source, label), count in sorted(
                Counter((row["extra_info"]["source"], row["extra_info"]["overall"]) for row in train_rows).items()
            )
        },
        "weighted_train": weighted_manifest,
        "filtered_json": filtered_manifest,
        "theorem_group_counts": {name: len(groups) for name, groups in split_groups.items()},
        "theorem_group_overlap": 0,
        "pin_split": str(pin_split) if pin_split else None,
        "group_duplicates_removed": group_duplicates_removed,
        "input_sha256": {str(path): _sha256(path) for path in inputs},
        "parquet_sha256": {
            path.name: _sha256(path)
            for path in sorted(output_dir.glob("*.parquet"))
        },
        "unique_theorem_ids": len(
            {row["extra_info"]["id"] for row in train_rows + validation_rows + test_rows}
        ),
        "trivial_validation_baselines": {
            "all_correct": _trivial_baseline(validation_rows, "correct", "correct"),
            "all_incorrect": _trivial_baseline(validation_rows, "incorrect", "incorrect"),
            "correct_steps_incorrect_verdict": _trivial_baseline(validation_rows, "correct", "incorrect"),
            "incorrect_steps_correct_verdict": _trivial_baseline(validation_rows, "incorrect", "correct"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def _parse_truth(value) -> dict:
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, str):
        value = json.loads(value)
    return value


def _score_one(solution: str, truth: dict) -> dict[str, float | str]:
    think_end = solution.rfind("</think>")
    reasoning = solution[:think_end].removeprefix("<think>").strip() if think_end >= 0 else ""
    answer = solution[think_end + len("</think>") :] if think_end >= 0 else ""
    thinking_valid = bool(reasoning)
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    predictions = {}
    overall = None
    valid = thinking_valid
    for line in lines:
        if match := STEP_RE.fullmatch(line):
            index = int(match.group(1))
            if index in predictions:
                valid = False
            predictions[index] = match.group(2).lower()
        elif match := OVERALL_RE.fullmatch(line):
            if overall is not None:
                valid = False
            overall = match.group(1).lower()
        else:
            valid = False

    labels = truth["steps"]
    valid &= set(predictions) == set(range(1, len(labels) + 1))
    valid &= overall in {"correct", "incorrect"}
    if not valid:
        return {
            "score": 0.0,
            "task_score": 0.0,
            "ungated_task_score": 0.0,
            "step_score": 0.0,
            "step_accuracy": 0.0,
            "exact_step_vector_accuracy": 0.0,
            "first_error_accuracy": 0.0,
            "correct_step_recall": 0.0,
            "incorrect_step_recall": 0.0,
            "verdict_accuracy": 0.0,
            "exact_proof_accuracy": 0.0,
            "format_valid": 0.0,
            "thinking_valid": float(thinking_valid),
            "label_consistency": 0.0,
        }

    predicted_labels = [predictions[i] for i in range(1, len(labels) + 1)]
    expected_overall = "correct" if all(label == "correct" for label in predicted_labels) else "incorrect"
    label_consistency = float(overall == expected_overall)

    step_accuracy = sum(predictions[i] == labels[i - 1] for i in range(1, len(labels) + 1)) / len(labels)
    exact_step_vector_accuracy = float(step_accuracy == 1.0)
    true_first_error = next((i for i, label in enumerate(labels, 1) if label == "incorrect"), None)
    predicted_first_error = next((i for i, label in enumerate(predicted_labels, 1) if label == "incorrect"), None)
    first_error_accuracy = float(predicted_first_error == true_first_error)
    correct_steps = [i for i, label in enumerate(labels, 1) if label == "correct"]
    incorrect_steps = [i for i, label in enumerate(labels, 1) if label == "incorrect"]
    correct_recall = (
        sum(predictions[i] == "correct" for i in correct_steps) / len(correct_steps) if correct_steps else 1.0
    )
    incorrect_recall = (
        sum(predictions[i] == "incorrect" for i in incorrect_steps) / len(incorrect_steps)
        if incorrect_steps
        else 1.0
    )
    if truth["overall"] == "incorrect":
        step_score = correct_recall * incorrect_recall
    else:
        step_score = float(all(predictions[i] == "correct" for i in range(1, len(labels) + 1)))
    verdict_accuracy = float(overall == truth["overall"])
    exact_proof_accuracy = exact_step_vector_accuracy * verdict_accuracy
    ungated_task_score = STEP_REWARD_WEIGHT * step_score + VERDICT_REWARD_WEIGHT * verdict_accuracy
    task_score = label_consistency * ungated_task_score
    return {
        "score": task_score,
        "task_score": task_score,
        "ungated_task_score": ungated_task_score,
        "step_score": step_score,
        "step_accuracy": step_accuracy,
        "exact_step_vector_accuracy": exact_step_vector_accuracy,
        "first_error_accuracy": first_error_accuracy,
        "correct_step_recall": correct_recall,
        "incorrect_step_recall": incorrect_recall,
        "verdict_accuracy": verdict_accuracy,
        "exact_proof_accuracy": exact_proof_accuracy,
        "format_valid": 1.0,
        "thinking_valid": 1.0,
        "label_consistency": label_consistency,
    }


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    del data_sources, extra_infos
    return [_score_one(solution, _parse_truth(truth)) for solution, truth in zip(solution_strs, ground_truths, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--duplicate-policy", choices=("error", "first", "prefer-incorrect"), default="error")
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument("--exclude-obvious-leaks", action="store_true")
    parser.add_argument("--write-weighted-train", action="store_true")
    parser.add_argument("--filtered-json", type=Path)
    parser.add_argument("--pin-split", type=Path, help="JSON with recorded validation/test theorem groups and ids to reuse verbatim")
    parser.add_argument("--dedupe-by-group", action="store_true", help="keep one row per theorem group outside the pinned splits")
    args = parser.parse_args()
    prepare(
        args.input,
        args.output_dir,
        args.seed,
        args.validation_per_class,
        args.test_per_class,
        args.duplicate_policy,
        args.balance_classes,
        args.exclude_obvious_leaks,
        args.write_weighted_train,
        args.filtered_json,
        args.pin_split,
        args.dedupe_by_group,
    )


if __name__ == "__main__":
    main()
