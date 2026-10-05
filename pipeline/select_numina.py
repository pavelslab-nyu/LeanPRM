"""Select hard NuminaMath statements from the difficulty classifier output."""

import argparse
import json
import re
from pathlib import Path


def clean_statement(formal: str) -> str | None:
    """Extract a clean, self-contained single-theorem signature ending in ':= by'.

    Numina formal_statements include 'import Mathlib', 'open ...' lines, and a
    comment block. Many also bundle auxiliary `lemma ... := by sorry` helpers and
    custom `def`/`structure` declarations before the real goal theorem. Those are
    NOT self-contained (they won't compile under just `import Mathlib`, and the
    model only reproduces the final goal), so we reject them. Returns None if the
    statement is not a clean single self-contained theorem.
    """
    s = formal.strip()
    # Drop comment blocks.
    s = re.sub(r"/-.*?-/", "", s, flags=re.DOTALL)
    # Drop import/open/variable/set_option preamble lines.
    body_lines = [
        ln
        for ln in s.splitlines()
        if not re.match(r"\s*(import|open|set_option|variable|namespace|section)\b", ln)
    ]
    s = "\n".join(body_lines).strip()

    # Reject non-self-contained statements: custom definitions, auxiliary sorrys,
    # or more than one top-level declaration.
    if "sorry" in s:
        return None
    if re.search(r"(?m)^\s*(def|structure|inductive|abbrev|class|instance)\b", s):
        return None
    decls = re.findall(r"(?m)^\s*(theorem|lemma|example)\b", s)
    if len(decls) != 1:
        return None

    # Take from the single declaration keyword to ':= by'.
    m = re.search(r"(?m)^\s*(theorem|lemma|example)\b", s)
    if not m:
        return None
    s = s[m.start() :].strip()
    if s.endswith(":="):
        s = s + " by"
    else:
        m2 = re.search(r":=\s*by", s)
        if m2:
            s = s[: m2.end()]
        elif s.endswith(":= by") is False:
            return None
    s = s.strip()
    if not s.endswith(":= by"):
        return None
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=2000)
    a = p.parse_args()
    from datasets import load_dataset

    hard = {
        r["uuid"]
        for line in a.labels.read_text().splitlines()
        if line.strip()
        for r in [json.loads(line)]
        if r["difficulty"] == 3
    }
    rows = []
    for row in load_dataset("AI-MO/NuminaMath-LEAN", split="train"):
        if row["uuid"] not in hard:
            continue
        statement = clean_statement(row["formal_statement"])
        if statement is None:
            continue
        rows.append(
            {
                "id": "numina_" + row["uuid"],
                "lean_statement": statement,
                "category": row.get("problem_type", ""),
                "difficulty": 3,
            }
        )
        if len(rows) >= a.limit:
            break
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(rows)} statements")


if __name__ == "__main__":
    main()
