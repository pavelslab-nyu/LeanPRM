#!/usr/bin/env python3
"""Select a balanced set of Lean-workbook statements from HuggingFace."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = "Goedel-LM/Lean-workbook-proofs"
DEFAULT_SPLIT = "train"
DEFAULT_TOTAL = 4_000
DEFAULT_SEED = 20260523
DEFAULT_OUTPUT = SCRIPT_DIR / "final_valid_lean_statements_hf_balanced_4000.json"

CATEGORY_SHARES = (
    ("inequality", 0.20),
    ("equality_identity", 0.20),
    ("divisibility_modular", 0.25),
    ("existence_construction", 0.20),
    ("iff_logic", 0.15),
)

STATEMENT_KEYS = (
    "lean_statement",
    "formal_statement",
    "statement",
    "theorem_statement",
)

DIVISIBILITY_MODULAR_PATTERNS = (
    "∣",
    "[ZMOD",
    "ZMOD",
    "%",
    "Nat.gcd",
    "Nat.lcm",
    "Int.gcd",
    "Int.lcm",
    "gcd",
    "lcm",
    "Coprime",
    ".Prime",
    "Prime",
)

IFF_LOGIC_PATTERNS = ("↔", "¬", "∧", "∨", "→", "False", "True")
INEQUALITY_RE = re.compile(r"(?<![<>=:])[<>≤≥](?![<>=])")
EQUALITY_RE = re.compile(r"(?<![<>=:])=(?![<>=])")
FORBIDDEN_STATEMENT_RE = re.compile(
    r"(?<![A-Za-z_])(sorry|admit|by_contra!|exact\?|apply\?|rw\?|simp\?|aesop\?)"
    r"(?![A-Za-z_])"
)


def target_counts(total: int) -> dict[str, int]:
    floors: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    assigned = 0
    for category, share in CATEGORY_SHARES:
        raw = total * share
        count = math.floor(raw)
        floors[category] = count
        assigned += count
        fractions.append((raw - count, category))

    for _, category in sorted(fractions, reverse=True)[: total - assigned]:
        floors[category] += 1
    return floors


def normalized_statement(text: str) -> str | None:
    text = text.strip()
    start = text.find("theorem ")
    if start < 0:
        return None
    theorem_text = text[start:]
    if ":= by" in theorem_text:
        theorem_text = theorem_text.split(":= by", 1)[0]
    elif ":=" in theorem_text:
        theorem_text = theorem_text.split(":=", 1)[0]
    statement = re.sub(r"\s+", " ", theorem_text).strip()
    if not statement.startswith("theorem ") or " : " not in statement:
        return None
    if FORBIDDEN_STATEMENT_RE.search(statement):
        return None
    return statement


def statement_from_row(row: dict[str, Any]) -> str | None:
    for key in STATEMENT_KEYS:
        value = row.get(key)
        if isinstance(value, str):
            statement = normalized_statement(value)
            if statement is not None:
                return statement

    full_proof = row.get("full_proof")
    if isinstance(full_proof, str):
        return normalized_statement(full_proof)
    return None


def output_id(row: dict[str, Any], fallback_index: int) -> str:
    raw = str(row.get("problem_id") or row.get("id") or fallback_index)
    raw = raw.strip()
    if raw.startswith("lean_workbook_hf_"):
        return raw
    raw = re.sub(r"^lean_workbook(?:_plus)?_", "", raw)
    raw = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_")
    return f"lean_workbook_hf_{raw or fallback_index}"


def rename_theorem(statement: str, theorem_id: str) -> str:
    return re.sub(r"^theorem\s+\S+", f"theorem {theorem_id}", statement, count=1)


def conclusion_text(statement: str) -> str:
    depth = 0
    for idx, char in enumerate(statement):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            return statement[idx + 1 :].strip()
    return statement


def classify_statement(statement: str) -> str:
    conclusion = conclusion_text(statement)
    if "∃" in conclusion or re.search(r"\bExists\b", conclusion):
        return "existence_construction"
    if any(token in conclusion for token in IFF_LOGIC_PATTERNS):
        return "iff_logic"
    if any(token in conclusion for token in DIVISIBILITY_MODULAR_PATTERNS):
        return "divisibility_modular"
    if INEQUALITY_RE.search(conclusion):
        return "inequality"
    if EQUALITY_RE.search(conclusion):
        return "equality_identity"
    if any(token in statement for token in DIVISIBILITY_MODULAR_PATTERNS):
        return "divisibility_modular"
    return "equality_identity"


def collect_candidates(dataset_name: str, split: str) -> dict[str, list[dict[str, str]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: run `uv add datasets` first.") from exc

    rows = load_dataset(dataset_name, split=split)
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_statements: set[str] = set()
    seen_ids: set[str] = set()

    for idx, row in enumerate(rows):
        statement = statement_from_row(dict(row))
        if statement is None:
            continue
        theorem_id = output_id(dict(row), idx)
        if theorem_id in seen_ids:
            theorem_id = f"{theorem_id}_{idx}"
        statement = rename_theorem(statement, theorem_id)
        if statement in seen_statements:
            continue
        category = classify_statement(statement)
        seen_statements.add(statement)
        seen_ids.add(theorem_id)
        buckets[category].append(
            {
                "id": theorem_id,
                "lean_statement": statement,
                "category": category,
                "source_dataset": dataset_name,
                "source_problem_id": str(row.get("problem_id") or row.get("id") or idx),
            }
        )

    return buckets


def select_balanced(
    buckets: dict[str, list[dict[str, str]]],
    targets: dict[str, int],
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    shortages: dict[str, tuple[int, int]] = {}

    for category, target in targets.items():
        candidates = list(buckets.get(category, []))
        rng.shuffle(candidates)
        if len(candidates) < target:
            shortages[category] = (len(candidates), target)
        selected.extend(candidates[:target])

    if shortages:
        detail = ", ".join(
            f"{category}: available {available}, need {target}"
            for category, (available, target) in shortages.items()
        )
        raise SystemExit(f"Not enough HuggingFace statements for target balance: {detail}")

    rng.shuffle(selected)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--target-total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = target_counts(args.target_total)
    buckets = collect_candidates(args.dataset, args.split)
    selected = select_balanced(buckets, targets, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(selected)} statements to {args.output}", flush=True)
    print("Targets:", json.dumps(targets, sort_keys=True), flush=True)
    print(
        "Available:",
        json.dumps({category: len(rows) for category, rows in sorted(buckets.items())}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
