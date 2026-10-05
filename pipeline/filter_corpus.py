"""Filter new generation runs before preparing a new training dataset."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from verifier_task import _obvious_leak, _theorem_group

MATHEMATICAL_CLASSES = frozenset(
    [
        "unjustified_assumption",
        "invalid_inequality_bound",
        "missing_case_or_side_condition",
        "bad_witness_or_construction",
        "algebraic_identity_error",
        "modular_or_divisibility_error",
    ]
)


def filter_proof(proof, bucket):
    steps = proof.get("steps", [])
    if len(steps) < 2:
        return "empty_or_short"
    if bucket == "incorrect_proofs":
        if proof.get("error_class") not in MATHEMATICAL_CLASSES:
            return "formalization_or_unclassified_error"
        if _obvious_leak({**proof, "_overall": "incorrect"}):
            return "explicit_error_admission"
    else:
        if re.search(
            r"\b(sorry|admit)\b", "\n".join(s.get("lean_code", "") for s in steps)
        ):
            return "positive_placeholder"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--exclude-groups", type=Path, help="JSON list of held-out theorem-group hashes"
    )
    a = p.parse_args()
    if a.output.exists():
        p.error("Output already exists")
    seen = set(json.loads(a.exclude_groups.read_text())) if a.exclude_groups else set()
    ids = set()
    out = {"correct_proofs": [], "incorrect_proofs": []}
    rejected = Counter()
    for path in a.input:
        data = json.loads(path.read_text())
        for bucket in out:
            for proof in data[bucket]:
                why = filter_proof(proof, bucket)
                group = _theorem_group(proof)
                if why:
                    rejected[why] += 1
                    continue
                if group in seen or proof["id"] in ids:
                    rejected["duplicate_or_heldout_theorem"] += 1
                    continue
                seen.add(group)
                ids.add(proof["id"])
                out[bucket].append(proof)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False) + "\n")
    report = {
        "kept": {k: len(v) for k, v in out.items()},
        "rejected": dict(rejected),
        "policy": "For new data. Historical paper data is exported separately without relabeling.",
    }
    a.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
