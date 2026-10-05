#!/usr/bin/env python3
"""Minimal end-to-end builder for ``final_dataset.json``.

Single entrypoint: generates Claude proof attempts, verifies each with Lean
(``lean-interact`` + Mathlib), applies lightweight error bucketing,
checkpoints merged results to JSONL, and writes ``final_dataset.json``.

The Anthropic API key is taken from ``ANTHROPIC_API_KEY`` after loading a
``.env`` file next to this script when present."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


# -----------------------------------------------------------------------------
# Paths and tunables (edit here instead of proliferating CLI flags).
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
INPUT_PATH = SCRIPT_DIR / "final_valid_lean_statements.json"
RUNS_DIR = SCRIPT_DIR / "runs"
RUN_TIMESTAMP_FORMAT = "%m-%d-%y_%H-%M-%S"
CACHE_ROOT = Path(os.environ.get("LEAN_VERIFIER_CACHE_ROOT", SCRIPT_DIR / ".cache"))
LEAN_PROJECT_DIR = CACHE_ROOT / "lean_interact_mathlib"
LEAN_INTERACT_CACHE_DIR = CACHE_ROOT / "lean_interact_cache"
TMP_DIR = CACHE_ROOT / "tmp"

DEFAULT_PROVIDER = "anthropic"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
DEFAULT_AZURE_OPENAI_REVIEW_DEPLOYMENT = os.environ.get("AZURE_OPENAI_REVIEW_DEPLOYMENT", "")
DEFAULT_AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_TOGETHER_MODEL = "moonshotai/Kimi-K2.6"
DEFAULT_MAX_TOKENS = 64_000
DEFAULT_THINKING_BUDGET = 32_000
# Adaptive-thinking effort for review and independent proof generation.
# Generation stays at high because these are best-shot solver trajectories.
THINKING_EFFORT = "high"
GENERATION_THINKING_EFFORT = "high"
# Models that take adaptive thinking + `effort`. On these, the older
# {"type": "enabled", "budget_tokens": N} form is removed and returns a 400 --
# which is why Sonnet 5 / Opus 5 could not run through this pipeline at all.
ADAPTIVE_THINKING_RE = re.compile(
    r"^claude-(opus-(4-[678]|5)|sonnet-5|fable-5|mythos-5)"
)
LEAN_VERSION_DEFAULT = "v4.25.0-rc2"
PROOFS_PER_STATEMENT = 1
MAX_RETRIES = 5
RETRY_SLEEP_S = 5.0
API_TIMEOUT_S = 600.0
MAX_CORRECT_ENTRIES = 4000
MAX_INCORRECT_ENTRIES = 4000
TWO_STEP_TARGET_KEPT = 100
TWO_STEP_MAX_ATTEMPTS = 300
RECOVER_DEAD_TACTICS = True

# Full simp tracing can turn a hard proof check into minutes of CPU and
# gigabytes of diagnostics. Lean's ordinary errors and warnings are sufficient
# for locating the first failing or dead tactic.
TRACE_LINES: tuple[str, ...] = ()

DEFAULT_AUDIT_OUTPUT = SCRIPT_DIR / "audited_final_dataset.json"
DEFAULT_REWRITE_OUTPUT = SCRIPT_DIR / "rewritten_final_dataset.json"

DEAD_TACTIC_PATTERNS = (
    "no goals to be solved",
    "this tactic is never executed",
    "tactic does nothing",
)

FORBIDDEN_NL_TERMS = (
    "lean",
    "tactic",
    "compiler",
    "syntax",
    "solver",
    "automation",
    "automated",
    "verifier",
    "engine",
    "linarith",
    "nlinarith",
    "field_simp",
    "norm_num",
    "aesop",
    "omega",
)
FORBIDDEN_NL_RE = re.compile(
    r"(?<![A-Za-z_])("
    + "|".join(re.escape(term) for term in FORBIDDEN_NL_TERMS)
    + r")(?![A-Za-z_])",
    re.IGNORECASE,
)

FORBIDDEN_LEAN_RE = re.compile(
    r"(?<![A-Za-z_])(sorry|admit|native_decide)(?![A-Za-z_])"
)

SYSTEM_PROMPT = """You are an expert Lean 4 and mathematics proof writer.
Generate Lean proofs for dataset construction.  Each proof must be a sequence
of small tactic steps, and every step must have a matching mathematical natural
language explanation that does not mention Lean syntax, tactic names, formal
verification, or code.

The Lean code and the natural-language proof must be faithful to each other:
the Lean proof should implement the same mathematical argument described in the
natural language, not skip over several natural-language claims with one opaque
automation call.

Do not ban arithmetic discharge tactics outright, but do not use `linarith`,
`nlinarith`, or `omega` as opaque one-shot proofs of the main argument.  If one
of these tactics is used, the preceding steps must expose the concrete
mathematical facts, identities, cases, or inequalities it relies on.  Never use
`sorry`, `admit`, or any placeholder proof term.

Two hard requirements for the natural language:

1. Each NL description must be 1:1 with its corresponding Lean step --
   describing exactly what that step does mathematically, not a higher-level
   summary and not a vague gloss.
2. Concatenating all NL descriptions in order must read as a complete,
   standalone mathematical proof that a student could have written and a
   mathematician could follow without ever seeing the Lean code.  It must not
   read like comments about code or a list of tactic explanations."""

FEW_SHOT_EXAMPLE = """Few-shot guide for the expected level of detail and 1:1 alignment:

{
  "id": "lean_workbook_19000_1",
  "lean_statement": "theorem lean_workbook_19000 (x : ℝ) (hx : x ≥ 3) : (x - 3) * (x^3 + 3 * x^2 + 9 * x - 27) ≥ 0 := by",
  "proof_index": 0,
  "steps": [
    {
      "step": 0,
      "lean_code": "theorem lean_workbook_19000 (x : ℝ) (hx : x ≥ 3) : (x - 3) * (x^3 + 3 * x^2 + 9 * x - 27) ≥ 0 := by",
      "natural_language_description": "We must prove: for any real x with x ≥ 3, the quantity (x − 3)(x^3 + 3x^2 + 9x − 27) is nonnegative.",
      "label": "correct"
    },
    {
      "step": 1,
      "lean_code": "  have h1 : x - 3 ≥ 0 := by linarith",
      "natural_language_description": "From x ≥ 3, we immediately obtain x − 3 ≥ 0.",
      "label": "correct"
    },
    {
      "step": 2,
      "lean_code": "  have h2 : x + 3 ≥ 0 := by linarith",
      "natural_language_description": "Since x ≥ 3, we also have x + 3 ≥ 3 + 3 = 6 ≥ 0.",
      "label": "correct"
    },
    {
      "step": 3,
      "lean_code": "  have h3 : x ^ 2 ≥ 0 := sq_nonneg x",
      "natural_language_description": "For any real x, its square is nonnegative: x^2 ≥ 0.",
      "label": "correct"
    },
    {
      "step": 4,
      "lean_code": "  have h4 : (x - 3) * x ^ 2 * (x + 3) ≥ 0 := mul_nonneg (mul_nonneg h1 h3) h2",
      "natural_language_description": "The product (x − 3)·x^2·(x + 3) is nonnegative because it is a product of three nonnegative factors (x − 3 ≥ 0, x^2 ≥ 0, x + 3 ≥ 0).",
      "label": "correct"
    },
    {
      "step": 5,
      "lean_code": "  have h5 : (x - 3) ^ 2 ≥ 0 := sq_nonneg (x - 3)",
      "natural_language_description": "Also, (x − 3)^2 ≥ 0, since a square is always nonnegative.",
      "label": "correct"
    },
    {
      "step": 6,
      "lean_code": "  nlinarith [h4, h5, sq_nonneg x, sq_nonneg (x - 3)]",
      "natural_language_description": "Note the algebraic decomposition: x^3 + 3x^2 + 9x − 27 = x^2(x + 3) + 9(x − 3). Hence (x − 3)(x^3 + 3x^2 + 9x − 27) = (x − 3)x^2(x + 3) + 9(x − 3)^2. By steps 4 and 5, each term on the right is ≥ 0, so their sum is ≥ 0. Therefore (x − 3)(x^3 + 3x^2 + 9x − 27) ≥ 0.",
      "label": "correct"
    },
    {
      "step": 7,
      "lean_code": "--     [linarith] ✅️ Invoking oracle\\n--       [linarith] found a contradiction: [(1, 9), (7, 1), (9, 1)]",
      "natural_language_description": "Equivalently, if one were to assume for contradiction that (x − 3)(x^3 + 3x^2 + 9x − 27) < 0 while keeping h4 ≥ 0 and h5 ≥ 0, then combining the identity above with these nonnegativities would contradict that assumption.",
      "label": "correct"
    },
    {
      "step": 8,
      "lean_code": "--     [linarith] ✅️ Building final expression\\n--       [linarith] \\n--             9 * (0 - (x - 3) ^ 2) + (0 - (x - 3) * x ^ 2 * (x + 3)) + ((x - 3) * (x ^ 3 + 3 * x ^ 2 + 9 * x - 27) - 0)\\n--           should be both 0 and negative",
      "natural_language_description": "Indeed, from (x − 3)(x^3 + 3x^2 + 9x − 27) = (x − 3)x^2(x + 3) + 9(x − 3)^2 with both summands ≥ 0, the left-hand side cannot be negative. This confirms the desired inequality.",
      "label": "correct"
    }
  ]
}

Use this as the standard for detail: each NL line states the concrete mathematical
claim behind the corresponding Lean line, and opaque algebra is expanded into the
actual identity or contradiction being used.  Do not write generic summaries like
"algebra finishes the proof", "standard processing applied", or "the final
transformation is resolved".

The final comment-only steps in this example show how trace-derived comments can
be explained when they are already available.  Do not fabricate trace comments in
new proofs.  If no trace comments are provided in the input, output only the
theorem line and executable proof lines."""

USER_PROMPT = """Generate one independent proof attempt for this Lean theorem:

```lean
{lean_statement}
```

Requirements:
- Use Lean 4 with Mathlib.
- Keep the theorem name and statement exactly as given.
- Do not use `linarith`, `nlinarith`, or `omega` as opaque one-shot proofs.
  They are allowed only as final arithmetic discharge after the preceding steps
  have made the algebraic situation, case split, divisibility fact, modular
  computation, or inequality combination explicit enough that the NL remains
  1:1 and mathematically informative.
- Do not use `sorry`, `admit`, `by_contra!` as a placeholder, or any unfinished
  proof placeholder.
- Do not use one opaque tactic to hide the main argument.  If a proof needs
  algebra, inequalities, case splits, contradiction, divisibility facts, or
  modular arithmetic, introduce those facts as explicit intermediate `have`,
  `calc`, `rw`, or case-analysis steps before any final arithmetic discharge.
- The proof must be step-by-step: step 0 is the theorem line ending in `:= by`,
  and each later step is one Lean tactic or one tightly scoped Lean proof line.
- Keep the proof concise. Aim for at most 40 steps and do not split routine
  rewriting or arithmetic into dozens of tiny steps.
- Each step must include a natural-language mathematical explanation matching
  that exact Lean step 1:1 (see rules and example below).
- Natural language must explain the mathematics only.  Do not mention Lean,
  tactics, automation, syntax, or the compiler.
- The natural language must not describe a stronger or more detailed argument
  than the Lean line actually proves.  If the NL says "we write k = 4q + 1",
  the Lean line must actually establish that statement in the corresponding
  case; if the NL cites an algebraic identity, the Lean line must establish that
  identity or use it directly.
- It is acceptable if the proof attempt is mathematically wrong, but avoid
  syntax errors and malformed Lean where possible.
- Return only JSON.  No Markdown fences.

Natural-language rules:
- 1:1 alignment with the Lean step — describe exactly what that step does.
- All steps concatenated in order must read as a standalone mathematical
  proof a student could have written (no gaps, no references to code).
- Do not write comments about the formal proof.  Write the mathematical proof
  itself.  If the Lean code were stripped out, the NL should still read
  naturally from beginning to end.
- Prefer concrete inequalities, computations, named lemmas — not generic filler.
- For opaque arithmetic steps, spell out the exact identity, contradiction,
  inequality combination, or modular case being used.  If you cannot state that
  exact mathematical content, split the proof into smaller explicit steps.
- Avoid vacuous or degenerate proofs unless the hypotheses are genuinely
  contradictory; when using contradiction, explicitly derive and explain the
  contradiction before concluding.

{few_shot}

JSON schema:
{{
  "steps": [
    {{
      "step": 0,
      "lean_code": "{escaped_theorem} := by",
      "natural_language_description": "We prove ... "
    }},
    {{
      "step": 1,
      "lean_code": "  ...",
      "natural_language_description": "..."
    }}
  ]
}}
"""

INCORRECT_PROMPT_ADDITION = """\nIntentional incorrect-proof mode:
- Generate a proof attempt that is syntactically well-formed Lean, but contains
  exactly one genuine mathematical reasoning error after a plausible correct
  prefix.
- The first wrong step should be a Lean line that tries to prove a false or
  unjustified mathematical claim and therefore fails verification as a proof
  error, not as malformed syntax.
- The natural language for that step must faithfully describe the same wrong
  mathematical claim.  Do not make the NL correct while the Lean line is merely
  too weak; the negative label should correspond to a real mathematical mistake.
- Prefer plausible errors such as an invalid inequality direction, an
  unjustified divisibility/modular conclusion, a missing hypothesis in a case,
  an algebraic identity with an incorrect sign or coefficient, or applying a
  theorem outside its conditions.
- Especially prefer non-inequality reasoning errors when the theorem permits
  them: a wrong modular residue, a bad divisibility witness, an invalid parity
  case, an unjustified existential witness, or a false algebraic identity.
- Keep all steps before the first wrong step correct, explicit, and aligned.
- Do not use `sorry`, `admit`, placeholders, malformed Lean, unknown constants,
  or deliberately broken syntax.
- Avoid examples where the natural language is a valid mathematical proof but
  the Lean line fails only because the automation or formalization is missing
  an intermediate fact.
- After the wrong step, you may include a few continuation steps that depend on
  the wrong claim; their NL should remain faithful to the erroneous path.
"""

INCORRECT_ERROR_STYLES = (
    (
        "theorem_misapplication",
        "Apply a true theorem while missing one of its hypotheses or using it outside its conditions.",
    ),
    (
        "wrong_rewrite_direction",
        "Rewrite an implication, equality, or inequality in an invalid direction.",
    ),
    (
        "missing_case_or_side_condition",
        "Ignore a necessary case, nonzero condition, positivity condition, or range restriction.",
    ),
    (
        "modular_or_divisibility_error",
        "Use a wrong residue, parity split, divisibility witness, or modular conclusion.",
    ),
    (
        "bad_witness_or_construction",
        "Choose an existential witness or constructed object that almost works but violates a required property.",
    ),
    (
        "algebraic_identity_error",
        "Use a false expansion, factorization, sign, coefficient, or algebraic identity.",
    ),
    (
        "automation_misuse",
        "Use arithmetic automation only after stating an explicit but false arithmetic or nonlinear premise.",
    ),
)

AUDIT_SYSTEM_PROMPT = """You are a strict mathematical data-quality auditor for a Lean-backed PRM dataset.
You judge natural-language proof traces as mathematics, not as Lean code.
Return only JSON."""

FAILURE_REVIEW_PROMPT = """A generated proof attempt failed Lean verification.

Your job is to decide whether the failed step is a genuine mathematical reasoning
error in the natural-language proof, or merely a formalization/syntax/tactic
failure where the mathematical prose may still be correct.

Keep as a reasoning error only if the natural-language step at the first failure
asserts a false or unjustified mathematical claim.  Do not keep failures caused
only by missing formal details, insufficient automation, syntax/type errors,
version drift, or a Lean line that is too weak while the natural-language proof
is mathematically valid.

Lean statement:
```lean
{lean_statement}
```

Available Lean context (imports, namespaces, and helper definitions; do not
repeat this in step 0):
```lean
{lean_context}
```

Candidate first failing step: {candidate_error_step}

Compiler errors:
{compiler_errors}

Steps JSON:
{steps_json}

Return exactly this JSON shape:
{{
  "verdict": "reasoning_error" | "not_reasoning_error" | "syntax_error",
  "first_error_step": <integer step index or null>,
  "reason": "short explanation"
}}
"""

AUDIT_PROOF_PROMPT = """Audit this labeled incorrect PRM proof.

The dataset currently labels step {error_step} and all following steps as
reasoning_error.  Decide whether this is a genuine mathematical reasoning error
in the natural-language proof.

Keep only if the natural-language proof itself makes a false or unjustified
mathematical claim at the first error step.  Remove if the issue is only Lean
syntax, missing formal detail, failed automation, theorem translation quality,
bad attribution, or a Lean-code mistake that the natural-language prose does not
faithfully share.

If the sample is good but the first error step is wrong, return keep with the
correct first_error_step.

Lean statement:
```lean
{lean_statement}
```

Requested error style: {requested_error_style}
Category: {category}

Steps JSON:
{steps_json}

Return exactly this JSON shape:
{{
  "verdict": "keep" | "remove",
  "first_error_step": <integer step index or null>,
  "reason": "short explanation",
  "confidence": "high" | "medium" | "low"
}}
"""

REWRITE_SYSTEM_PROMPT = """You rewrite natural-language mathematical proof steps for a PRM dataset.
Return only JSON.  Never mention Lean, tactics, syntax, compilers, automation,
or verification.  Preserve the mathematical content and labels."""

REWRITE_PROOF_PROMPT = """Rewrite only the natural-language descriptions for this proof.

Hard constraints:
- Preserve the exact number of steps and every step index.
- Do not change Lean code, labels, error_step, IDs, or metadata.
- Each rewritten description must match the corresponding Lean line's
  mathematical content 1:1.
- The concatenated descriptions must read as a standalone mathematical proof.
- Do not mention Lean, tactics, code, syntax, compiler, verifier, automation, or
  tool names.
- For labels marked reasoning_error, do not silently fix the mathematics.  State
  the same erroneous mathematical claim confidently as part of the attempted
  proof; do not meta-comment that it is wrong.

Proof JSON:
{proof_json}

Return exactly this JSON shape:
{{
  "steps": [
    {{"step": 0, "natural_language_description": "..."}}
  ]
}}
"""

# Fixed taxonomy of mathematical reasoning-error classes for the two-step pipeline.
# The second LLM call picks exactly one of these for a "reasoning_error" verdict.
# Aligned where possible with the main pipeline's requested_error_style values so
# the two datasets can be sliced together; "other" is the catch-all.
ERROR_CLASSES = [
    "algebraic_identity_error",       # false or invalid algebraic manipulation/identity
    "invalid_inequality_bound",       # wrong direction or unjustified bound in an inequality
    "bad_witness_or_construction",    # wrong witness for an existential / invalid construction
    "theorem_misapplication",         # applies a lemma/theorem whose hypotheses do not hold
    "modular_or_divisibility_error",  # incorrect modular arithmetic or divisibility reasoning
    "missing_case_or_side_condition", # omits a required case or side condition
    "wrong_rewrite_direction",        # invalid or backwards rewrite / substitution
    "unjustified_assumption",         # asserts something not established by prior steps
    "automation_misuse",              # claims automation discharges a goal it cannot
    "other",                          # genuine reasoning error not captured above
]
ERROR_CLASS_OPTIONS_TEXT = "\n".join(f"- {c}" for c in ERROR_CLASSES)


TWO_STEP_REVIEW_PROMPT = """Classify this immutable Stage-1 Lean/NL proof attempt after Lean verification and, only when appropriate, return minimal surface-form Lean syntax patches.

Stage 1 is the true blind attempt. Its natural-language descriptions and
mathematical plan are frozen. You MUST NOT rewrite, improve, complete, or correct
them. Compiler feedback is available only for classification and private syntax
repair; it must never leak into the natural-language trajectory.

This is the second and final LLM call for this datapoint. Decide one verdict:

- "correct": Lean verification already SUCCEEDED and the natural-language proof is
  faithful to the Lean proof.
- "reasoning_error": Lean verification FAILED because the mathematical proof
  attempt is false, unjustified, or incomplete at the first failing step. This
  includes a missing branch, missing direction, unclosed goal, or an argument
  that stops before proving the theorem. These are valuable natural solver
  failures and MUST be retained as reasoning errors, not discarded. If the
  attempt simply stops with an unclosed goal, use its final provided step as
  `first_error_step`, not step 0.
- "repair": Lean verification FAILED for a purely MECHANICAL/code reason while the
  underlying mathematics is VALID, such as punctuation, parentheses, namespace,
  a renamed identifier with exactly the same proposition, or an explicit type or
  coercion annotation expressing the same term. Return only the changed Lean lines.
- "discard": the statement is ill-posed, the output is malformed, or the NL and
  Lean silently describe different mathematics. Do NOT discard merely because
  the proof is incomplete or would need substantial new mathematics to finish.

REPAIR RULES (only when verdict is "repair"):
- Fix ONLY surface-form Lean syntax or elaboration. Do NOT change the mathematics.
- Do NOT alter step 0 in any way.
- Do NOT weaken the theorem, and do NOT use `sorry`, `admit`, `native_decide`, or any placeholder/cheat tactic.
- Do NOT add, remove, merge, split, or reorder steps.
- Do NOT change a proposition, expression, witness, assumption, case split,
  intermediate claim, tactic kind, tactic strategy, or tactic arguments that carry
  mathematical content.
- Do NOT swap tactics or add hints, facts, automation, or "small nudges."
- If it cannot compile without adding or changing substantive mathematics, use
  "reasoning_error" when the attempt is incomplete or mathematically invalid.

For any verdict other than "repair", return no syntax patches.

ERROR CLASSIFICATION (only when verdict is "reasoning_error"):
Reason about the single first failing mathematical step and choose the ONE label
from this fixed list that best describes the underlying mistake:
{error_class_options}
Pick the most specific label that fits; use "other" only if none clearly apply.
For any verdict other than "reasoning_error", set "error_class" to null.

Lean statement:
```lean
{lean_statement}
```

Lean verification status: {verification_status}
Candidate first failing step: {candidate_error_step}

Compiler errors:
{compiler_errors}

Generated steps JSON:
{steps_json}

Return exactly this JSON shape. `syntax_patches` must be empty unless verdict is
"repair". Never return natural-language text:
{{
  "verdict": "correct" | "reasoning_error" | "repair" | "discard",
  "first_error_step": <integer step index or null>,
  "error_class": "<one label from the list above, or null if not a reasoning_error>",
  "discard_reason": "short reason, or empty string if kept",
  "syntax_patches": [
    {{"step": <changed step index>, "lean_code": "<replacement Lean line only>"}}
  ]
}}
"""

SYNTAX_PATTERNS = (
    "unknown identifier",
    "unexpected token",
    "expected token",
    "invalid field",
    "invalid projection",
    "unknown constant",
    "unknown namespace",
    "unknown tactic",
    "invalid constructor",
    "ambiguous",
    "elaboration function",
    "declaration uses",
    "invalid binder",
    "invalid pattern",
    "invalid dotted identifier",
)

INCOMPLETE_PROOF_RE = re.compile(
    r"\b(incomplete|unfinished|unclosed)\b|"
    r"\bnot addressed\b|"
    r"\bnever (?:completes|derives|proves|constructs|finishes)\b|"
    r"\b(?:missing|omits?) (?:the |a )?"
    r"(?:argument|case|direction|proof|derivation|mathematical step)\b|"
    r"\bleaves?\b.{0,80}\b(?:goal|branch|direction)s?\b",
    re.IGNORECASE,
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def make_run_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime(RUN_TIMESTAMP_FORMAT)
    run_dir = RUNS_DIR / stamp
    suffix = 2
    while run_dir.exists():
        run_dir = RUNS_DIR / f"{stamp}-{suffix}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


def resolve_resume_run(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    if not path.exists():
        bare_run = RUNS_DIR / value
        if bare_run.exists():
            path = bare_run
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"--resume-run does not exist or is not a directory: {value}")
    return path


def resolve_input_path(value: str | None) -> Path:
    if value is None:
        return INPUT_PATH
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    if not path.exists() or not path.is_file():
        raise SystemExit(f"--input does not exist or is not a file: {value}")
    return path


def resolve_script_path(value: str | Path | None, default: Path | None = None) -> Path:
    if value is None:
        if default is None:
            raise SystemExit("Missing required path.")
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path


def ensure_lake_on_path() -> None:
    if shutil.which("lake"):
        return
    elan_bin = Path.home() / ".elan" / "bin"
    if (elan_bin / "lake").exists():
        os.environ["PATH"] = f"{elan_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    if not shutil.which("lake"):
        raise SystemExit(
            "Lean executable 'lake' was not found. Install Lean via elan, "
            "or add lake to PATH before running this script."
        )


def prepare_runtime_storage() -> None:
    for path in (CACHE_ROOT, LEAN_PROJECT_DIR, LEAN_INTERACT_CACHE_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(TMP_DIR)
    os.environ["TMP"] = str(TMP_DIR)
    os.environ["TEMP"] = str(TMP_DIR)


def run_lake(args: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(["lake", *args], cwd=cwd, env=merged_env, check=True)


def disable_proofwidgets_cache_fetch(project_dir: Path) -> None:
    requests = project_dir / ".lake/packages/mathlib/Cache/Requests.lean"
    if not requests.exists():
        return
    text = requests.read_text(encoding="utf-8")
    old = "  getProofWidgets <| ← getProofWidgetsCacheDir"
    new = "  pure () -- ProofWidgets JS is not needed for theorem verification."
    if old in text:
        requests.write_text(text.replace(old, new), encoding="utf-8")


def ensure_lean_project() -> Path:
    prepare_runtime_storage()
    ensure_lake_on_path()
    lakefile = LEAN_PROJECT_DIR / "lakefile.lean"
    if not lakefile.exists():
        (LEAN_PROJECT_DIR / "lean-toolchain").write_text(
            "leanprover/lean4:" + LEAN_VERSION_DEFAULT + "\n",
            encoding="utf-8",
        )
        lakefile.write_text(
            'import Lake\nopen Lake DSL\n\n'
            'package dummy\n\n'
            'require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "766e19e781a0992344fd4f24d2662858b6b87250"\n\n'
            '@[default_target]\nlean_lib Dummy where\n',
            encoding="utf-8",
        )
        (LEAN_PROJECT_DIR / "Dummy.lean").write_text(
            "import Mathlib\n",
            encoding="utf-8",
        )
    if not (LEAN_PROJECT_DIR / "lake-manifest.json").exists():
        run_lake(
            ["update"],
            LEAN_PROJECT_DIR,
            env={"MATHLIB_NO_CACHE_ON_UPDATE": "1", "GITHUB_ACTIONS": "true"},
        )
    disable_proofwidgets_cache_fetch(LEAN_PROJECT_DIR)
    return LEAN_PROJECT_DIR


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def parse_model_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        # Preserve complete step objects from a truncated/malformed JSON array.
        # The resulting unfinished Lean proof is still a valid best-of-N sample:
        # Lean will reject it and provide the formal verdict. Retrying would pay
        # for a second trajectory while silently discarding the first one.
        steps_key = text.find('"steps"')
        array_start = text.find("[", steps_key)
        if steps_key != -1 and array_start != -1:
            decoder = json.JSONDecoder()
            cursor = array_start + 1
            steps: list[dict[str, Any]] = []
            while cursor < len(text):
                while cursor < len(text) and text[cursor] in " \t\r\n,":
                    cursor += 1
                if cursor >= len(text) or text[cursor] == "]":
                    break
                try:
                    value, cursor = decoder.raw_decode(text, cursor)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    break
                steps.append(value)
            if len(steps) >= 2:
                return {"steps": steps, "partial_json": True}
        raise original_error


def open_anthropic_client(api_timeout: float) -> Any:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is missing. Set it in the environment "
            f"or in {ENV_PATH}."
        )
    try:
        import anthropic  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install anthropic") from exc
    return anthropic.Anthropic(timeout=api_timeout)


def openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is missing. Set it in the environment "
            f"or in {ENV_PATH}."
        )
    return key


def together_api_key_from_env() -> str:
    key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHERAI_API_KEY")
    if not key:
        raise SystemExit(
            "TOGETHER_API_KEY is missing. Set it in the environment "
            f"or in {ENV_PATH}."
        )
    return key


def azure_openai_config() -> tuple[str, str, str]:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    version = os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_OPENAI_API_VERSION)
    if not endpoint or not key:
        raise SystemExit(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required for Azure OpenAI."
        )
    return endpoint, key, version


def record_usage(provider: str, model: str, phase: str, usage: Any) -> None:
    """Append per-call token usage to $USAGE_LOG, when set.

    Billing is per API call, but the unit we care about is the kept sample, and
    each kept sample costs several calls (generate + review, times the retry and
    discard rate). Cost modelling needs the per-call numbers; this records them
    on demand and is inert otherwise.
    """
    path = os.environ.get("USAGE_LOG")
    if not path or usage is None:
        return
    if not isinstance(usage, dict):
        usage = {
            k: getattr(usage, k, None)
            for k in ("input_tokens", "output_tokens",
                      "cache_read_input_tokens", "cache_creation_input_tokens")
        }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(
            {"provider": provider, "model": model, "phase": phase, **usage},
            default=str) + "\n")


def call_anthropic_json(
    client: Any,
    *,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int,
    thinking_budget: int,
    usage_phase: str = "unknown",
) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking_budget > 0:
        if ADAPTIVE_THINKING_RE.match(model):
            # opus-4 only supports adaptive thinking; effort controls depth.
            # 'high' makes hard problems think for many minutes and blow past the
            # request timeout (killing throughput), so use a faster effort.
            kw["thinking"] = {"type": "adaptive"}
            kw["output_config"] = {"effort": THINKING_EFFORT}
        else:
            kw["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    for _attempt in range(3):
        message = client.messages.create(**kw)
        record_usage("anthropic", model, usage_phase, getattr(message, "usage", None))
        raw_text = ""
        for block in message.content:
            if getattr(block, "type", None) == "text":
                raw_text += block.text
        if raw_text.strip():
            break
        # Empty response: adaptive thinking consumed the whole budget. On models
        # that still accept it, fall back to an explicit budget; on the newer ones
        # budget_tokens is rejected, so step the effort down instead.
        if ADAPTIVE_THINKING_RE.match(model):
            kw["output_config"] = {"effort": "medium"}
        else:
            kw["thinking"] = {"type": "enabled", "budget_tokens": max_tokens // 2}
            kw.pop("output_config", None)
    return parse_model_json(raw_text)


def call_openai_chat_json(
    *,
    api_key: str,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int,
    api_timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "lean-verifier/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=api_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {body}") from exc
    choices = data.get("choices") or []
    raw_text = choices[0].get("message", {}).get("content", "") if choices else ""
    if not raw_text:
        raise ValueError(f"OpenAI response had no message content: {data}")
    return parse_model_json(raw_text)


def call_together_chat_json(
    *,
    api_key: str,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int,
    api_timeout: float,
    usage_phase: str = "unknown",
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    # GLM-5.2's default thinking mode can consume the entire output budget and
    # return no JSON. Best-of-N needs diverse, inexpensive independent samples,
    # so use the same non-thinking configuration as generate_best_of_n.py.
    if "glm-5.2" in model.lower():
        payload["temperature"] = 1.0
        payload["chat_template_kwargs"] = {"thinking": False}
    req = urllib.request.Request(
        "https://api.together.xyz/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Together sits behind Cloudflare, which answers 403 (code 1010) to
            # requests with no Accept / User-Agent. urllib sends neither by default.
            "Accept": "application/json",
            "User-Agent": "lean-verifier/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=api_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Together HTTP {exc.code}: {body}") from exc

    record_usage("together", model, usage_phase, data.get("usage"))
    choices = data.get("choices") or []
    raw_text = choices[0].get("message", {}).get("content", "") if choices else ""
    if not raw_text:
        raise ValueError(f"Together response had no message content: {data}")
    return parse_model_json(raw_text)


def call_azure_openai_chat_json(
    *,
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
    system: str,
    prompt: str,
    max_tokens: int,
    api_timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": deployment,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_output_tokens": max_tokens,
    }
    url = f"{endpoint}/openai/responses?api-version={api_version}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "lean-verifier/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=api_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Azure OpenAI HTTP {exc.code}: {body}") from exc
    raw_text = str(data.get("output_text", ""))
    if not raw_text:
        chunks: list[str] = []
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                if isinstance(content, dict):
                    chunks.append(str(content.get("text", "")))
        raw_text = "".join(chunks)
    if not raw_text:
        raise ValueError(f"Azure OpenAI response had no message content: {data}")
    return parse_model_json(raw_text)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def normalize_lean_code(src: str) -> str:
    return re.sub(r"(?<=\w)\^(?=\w|\d|\()", " ^ ", src)


def normalize_statement(statement: dict[str, Any]) -> dict[str, Any]:
    updated = dict(statement)
    updated["lean_statement"] = normalize_lean_code(str(updated["lean_statement"]))
    if updated.get("lean_context"):
        updated["lean_context"] = normalize_lean_code(str(updated["lean_context"]))
    return updated


def normalize_step_numbers(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        if isinstance(step, dict):
            lean_code = normalize_lean_code(str(step.get("lean_code", ""))).rstrip()
            description = str(step.get("natural_language_description", "")).strip()
        else:
            # Keep malformed later steps as rejected formal actions instead of
            # paying to resample the whole trajectory.
            lean_code = ""
            description = str(step).strip() or "Malformed generated proof step."
        normalized.append(
            {
                "step": i,
                "lean_code": lean_code,
                "natural_language_description": description,
            }
        )
    return normalized


def build_full_proof(
    steps: list[dict[str, Any]],
    lean_context: str = "",
) -> tuple[str, list[tuple[int, int]]]:
    context = lean_context.strip()
    lines = [*(context.splitlines() if context else ["import Mathlib"]), "", *TRACE_LINES, ""]
    ranges: list[tuple[int, int]] = []
    for step in steps:
        start_line = len(lines) + 1
        chunk = str(step["lean_code"]).splitlines() or [""]
        lines.extend(chunk)
        end_line = len(lines)
        ranges.append((start_line, end_line))
    return "\n".join(lines) + "\n", ranges


def message_text(message: dict[str, Any]) -> str:
    return str(message.get("data", "")).lower()


def is_dead_tactic_message(message: dict[str, Any]) -> bool:
    text = message_text(message)
    return any(p in text for p in DEAD_TACTIC_PATTERNS)


def dead_tactic_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if is_dead_tactic_message(m)]


def is_dead_tactic_only_failure(messages: Iterable[dict[str, Any]]) -> bool:
    messages = list(messages)
    if not dead_tactic_messages(messages):
        return False
    non_dead = [
        m
        for m in messages
        if m.get("severity") == "error" and not is_dead_tactic_message(m)
    ]
    return not non_dead


def first_message_line(messages: Iterable[dict[str, Any]]) -> int | None:
    lines: list[int] = []
    for message in messages:
        pos = message.get("start_pos") or {}
        line = pos.get("line")
        if isinstance(line, int):
            lines.append(line)
    return min(lines) if lines else None


def step_index_for_line(line: int, ranges: list[tuple[int, int]]) -> int | None:
    for idx, (start, end) in enumerate(ranges):
        if start <= line <= end:
            return idx
    return None


def trim_steps_before_line(
    steps: list[dict[str, Any]],
    ranges: list[tuple[int, int]],
    line: int,
) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for step, (start, end) in zip(steps, ranges):
        if end < line:
            trimmed.append(dict(step))
            continue
        if start >= line:
            break
        kept_count = line - start
        kept_lines = str(step["lean_code"]).splitlines()[:kept_count]
        kept_code = "\n".join(kept_lines).rstrip()
        if kept_code:
            kept_step = dict(step)
            kept_step["lean_code"] = kept_code
            trimmed.append(kept_step)
        break

    return [{**step, "step": i} for i, step in enumerate(trimmed)]


def command_response_to_dict(result: Any) -> dict[str, Any]:
    messages = []
    for msg in getattr(result, "messages", []) or []:
        start = getattr(msg, "start_pos", None)
        end = getattr(msg, "end_pos", None)
        messages.append(
            {
                "severity": getattr(msg, "severity", ""),
                "data": getattr(msg, "data", ""),
                "start_pos": {
                    "line": getattr(start, "line", None),
                    "column": getattr(start, "column", None),
                }
                if start
                else None,
                "end_pos": {
                    "line": getattr(end, "line", None),
                    "column": getattr(end, "column", None),
                }
                if end
                else None,
            }
        )
    return {
        "has_errors": bool(result.has_errors()) if hasattr(result, "has_errors") else True,
        "messages": messages,
    }


LEAN_CHECK_TIMEOUT_S = 120.0
LEAN_MEMORY_HARD_LIMIT_MB = 8_192
LEAN_REPL_RECYCLE_EVERY_CHECKS = 20
LEAN_MAX_TOTAL_MEMORY_FRACTION = 0.70
LEAN_MAX_PROCESS_MEMORY_FRACTION = 0.50


def run_lean_check(
    server: Any,
    command_cls: Any,
    steps: list[dict[str, Any]],
    lean_context: str = "",
) -> tuple[str, list[tuple[int, int]], dict[str, Any]]:
    proof_src, ranges = build_full_proof(steps, lean_context)
    # We only consume Lean's ordinary diagnostic messages and source positions.
    # Requesting the full tactic info tree makes large proofs dramatically slower
    # and more memory hungry without changing the verdict or error-step mapping.
    result = server.run(command_cls(cmd=proof_src), timeout=LEAN_CHECK_TIMEOUT_S)
    return proof_src, ranges, command_response_to_dict(result)


def recover_dead_tactics(
    server: Any,
    command_cls: Any,
    steps: list[dict[str, Any]],
    lean_result: dict[str, Any],
    ranges: list[tuple[int, int]],
    lean_context: str = "",
) -> dict[str, Any] | None:
    messages = lean_result.get("messages", [])
    dead_messages = dead_tactic_messages(messages)
    if not is_dead_tactic_only_failure(messages):
        return None

    line = first_message_line(dead_messages)
    if line is None:
        return None

    trimmed_steps = trim_steps_before_line(steps, ranges, line)
    if not trimmed_steps or trimmed_steps == steps:
        return None

    proof_src, trimmed_ranges, trimmed_result = run_lean_check(
        server, command_cls, trimmed_steps, lean_context
    )
    return {
        "trigger_line": line,
        "trigger_step": step_index_for_line(line, ranges),
        "removed_step_count": len(steps) - len(trimmed_steps),
        "trimmed_steps": trimmed_steps,
        "trimmed_full_proof": proof_src,
        "trimmed_ranges": trimmed_ranges,
        "trimmed_lean_result": trimmed_result,
    }


def classify_messages(messages: Iterable[dict[str, Any]]) -> str:
    joined = "\n".join(str(m.get("data", "")) for m in messages).lower()
    if not joined:
        return "ambiguous_error"
    if any(p in joined for p in SYNTAX_PATTERNS):
        return "syntax_error"
    return "ambiguous_error"


def error_step_from_messages(
    messages: Iterable[dict[str, Any]],
    ranges: list[tuple[int, int]],
) -> int:
    line_nums: list[int] = []
    for msg in messages:
        pos = msg.get("start_pos") or {}
        line = pos.get("line")
        if isinstance(line, int):
            line_nums.append(line)
    if line_nums:
        first_line = min(line_nums)
        for idx, (start, end) in enumerate(ranges):
            if start <= first_line <= end:
                return idx
    return max(len(ranges) - 1, 0)


def compiler_errors_block(messages: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in messages:
        pos = msg.get("start_pos") or {}
        loc = ""
        if isinstance(pos.get("line"), int):
            loc = f"line {pos.get('line')}, column {pos.get('column')}: "
        lines.append(f"- {loc}{msg.get('data', '')}")
    return "\n".join(lines) if lines else "- No compiler error text available."


def bounded_step(value: Any, step_count: int) -> int | None:
    if not isinstance(value, int):
        return None
    if 0 <= value < step_count:
        return value
    return None


def review_failed_attempt_with_model(
    client: Any,
    *,
    model: str,
    max_tokens: int,
    thinking_budget: int,
    lean_statement: str,
    steps: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    candidate_error_step: int,
) -> dict[str, Any]:
    prompt = render_template(
        FAILURE_REVIEW_PROMPT,
        lean_statement=lean_statement,
        candidate_error_step=str(candidate_error_step),
        compiler_errors=compiler_errors_block(errors),
        steps_json=json.dumps(steps, indent=2, ensure_ascii=False),
    )
    result = call_anthropic_json(
        client,
        model=model,
        system=AUDIT_SYSTEM_PROMPT,
        prompt=prompt,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
    )
    verdict = str(result.get("verdict", "not_reasoning_error"))
    first = bounded_step(result.get("first_error_step"), len(steps))
    if verdict != "reasoning_error":
        first = None
    elif first is None:
        first = candidate_error_step
    return {
        "verdict": verdict,
        "first_error_step": first,
        "reason": str(result.get("reason", "")),
    }


def proof_key(proof: dict[str, Any]) -> tuple[str, int]:
    return str(proof.get("id", "")), int(proof.get("proof_index", 0))


def proof_section_key(section: str, proof: dict[str, Any]) -> tuple[str, str, int]:
    pid, pidx = proof_key(proof)
    return section, pid, pidx


def completed_log_by_key(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    done: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if "id" not in row or "proof_index" not in row:
            continue
        done[(str(row["id"]), int(row["proof_index"]))] = row
    return done


def completed_rewrite_log_by_key(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    done: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if "id" not in row or "proof_index" not in row or "section" not in row:
            continue
        done[(str(row["section"]), str(row["id"]), int(row["proof_index"]))] = row
    return done


def audit_proof_with_model(
    client: Any,
    *,
    model: str,
    max_tokens: int,
    thinking_budget: int,
    proof: dict[str, Any],
) -> dict[str, Any]:
    prompt = render_template(
        AUDIT_PROOF_PROMPT,
        lean_statement=str(proof.get("lean_statement", "")),
        error_step=str(proof.get("error_step")),
        requested_error_style=str(proof.get("requested_error_style", "")),
        category=str(proof.get("category", "")),
        steps_json=json.dumps(proof.get("steps", []), indent=2, ensure_ascii=False),
    )
    result = call_anthropic_json(
        client,
        model=model,
        system=AUDIT_SYSTEM_PROMPT,
        prompt=prompt,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
    )
    verdict = str(result.get("verdict", "remove"))
    if verdict not in {"keep", "remove"}:
        verdict = "remove"
    first = bounded_step(result.get("first_error_step"), len(proof.get("steps", [])))
    if verdict == "keep" and first is None:
        first = bounded_step(proof.get("error_step"), len(proof.get("steps", [])))
    return {
        "id": proof.get("id"),
        "proof_index": proof.get("proof_index"),
        "verdict": verdict,
        "first_error_step": first,
        "reason": str(result.get("reason", "")),
        "confidence": str(result.get("confidence", "")),
    }


def apply_audit_to_dataset(
    dataset: dict[str, Any],
    audit_rows: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    correct = dataset.get("correct_proofs", [])
    incorrect: list[dict[str, Any]] = []
    for proof in dataset.get("incorrect_proofs", []):
        row = audit_rows.get(proof_key(proof))
        if not row or row.get("verdict") != "keep":
            continue
        updated = dict(proof)
        if isinstance(row.get("first_error_step"), int):
            updated["error_step"] = row["first_error_step"]
            updated["steps"] = steps_with_labels(
                [
                    {
                        "step": step.get("step", i),
                        "lean_code": step.get("lean_code", ""),
                        "natural_language_description": step.get(
                            "natural_language_description", ""
                        ),
                    }
                    for i, step in enumerate(proof.get("steps", []))
                ],
                error_step=row["first_error_step"],
            )
        updated["audit_reason"] = row.get("reason", "")
        updated["audit_confidence"] = row.get("confidence", "")
        incorrect.append(updated)
    return {"correct_proofs": correct, "incorrect_proofs": incorrect}


def validate_rewritten_steps(
    original_steps: list[dict[str, Any]],
    rewritten: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    steps = rewritten.get("steps")
    if not isinstance(steps, list) or len(steps) != len(original_steps):
        return ["rewrite did not return the same number of steps"]
    for i, step in enumerate(steps):
        if step.get("step") != i:
            errors.append(f"step {i} has wrong index")
        nl = str(step.get("natural_language_description", "")).strip()
        if not nl:
            errors.append(f"step {i} has empty natural_language_description")
        if FORBIDDEN_NL_RE.search(nl):
            errors.append(f"step {i} mentions formal tooling")
    return errors


def rewrite_proof_with_model(
    client: Any,
    *,
    model: str,
    max_tokens: int,
    thinking_budget: int,
    proof: dict[str, Any],
) -> dict[str, Any]:
    prompt = render_template(
        REWRITE_PROOF_PROMPT,
        proof_json=json.dumps(proof, indent=2, ensure_ascii=False),
    )
    rewritten = call_anthropic_json(
        client,
        model=model,
        system=REWRITE_SYSTEM_PROMPT,
        prompt=prompt,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
    )
    errors = validate_rewritten_steps(proof.get("steps", []), rewritten)
    if errors:
        raise ValueError("; ".join(errors))
    updated = dict(proof)
    by_index = {step["step"]: step for step in rewritten["steps"]}
    updated_steps = []
    for step in proof.get("steps", []):
        new_step = dict(step)
        new_step["natural_language_description"] = str(
            by_index[step["step"]]["natural_language_description"]
        ).strip()
        updated_steps.append(new_step)
    updated["steps"] = updated_steps
    return updated


def validate_attempt_shape(
    parsed: dict[str, Any],
    statement: dict[str, Any],
) -> tuple[bool, str, list[dict[str, Any]]]:
    steps = parsed.get("steps")
    if not isinstance(steps, list) or not steps:
        return False, "missing non-empty steps array", []
    steps = normalize_step_numbers(steps)
    for step in steps:
        nl = step["natural_language_description"]
        if not nl.strip():
            return (
                False,
                f"empty natural_language_description at step {step['step']}",
                steps,
            )
        if FORBIDDEN_NL_RE.search(nl):
            return False, "natural language mentions discouraged formal-tooling wording", steps
    expected_prefix = statement["lean_statement"].strip()
    first = re.sub(r"\s+", " ", steps[0]["lean_code"].strip())
    expected = re.sub(r"\s+", " ", expected_prefix)
    if expected not in first or ":= by" not in first:
        # Reproducing a long declaration byte-for-byte is formatting, not proof
        # generation.  Normalize a model-supplied declaration to the official
        # statement; if the model omitted step 0 entirely, prepend it.  The proof
        # body is unchanged and is still checked against this exact theorem by Lean.
        if re.search(r"\b(theorem|lemma|example)\b", first) or ":= by" in first:
            steps[0]["lean_code"] = expected_prefix
        else:
            steps.insert(
                0,
                {
                    "step": 0,
                    "lean_code": expected_prefix,
                    "natural_language_description": "We prove the stated proposition.",
                },
            )
            steps = [{**step, "step": i} for i, step in enumerate(steps)]
    return True, "", steps


def steps_with_labels(
    steps: list[dict[str, Any]],
    error_step: int | None = None,
) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        label = "correct"
        if error_step is not None and i >= error_step:
            label = "reasoning_error"
        labeled.append(
            {
                "step": i,
                "lean_code": step["lean_code"],
                "natural_language_description": step["natural_language_description"],
                "label": label,
            }
        )
    return labeled


def pools_from_records(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["id"], int(row["proof_index"]))
        latest[key] = row

    correct_src: list[dict[str, Any]] = []
    incorrect_src: list[dict[str, Any]] = []
    for key in sorted(latest.keys()):
        row = latest[key]
        outcome = row.get("outcome")
        if outcome == "correct":
            correct_src.append(row)
        elif outcome == "reasoning_error":
            incorrect_src.append(row)
    return correct_src, incorrect_src


def assemble_final_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct_src, incorrect_src = pools_from_records(rows)
    correct_out: list[dict[str, Any]] = []
    for row in correct_src[:MAX_CORRECT_ENTRIES]:
        entry: dict[str, Any] = {
            "id": row["id"],
            "lean_statement": row["lean_statement"],
            "proof_index": row["proof_index"],
            "steps": steps_with_labels(row["steps"]),
        }
        if row.get("category"):
            entry["category"] = row["category"]
        correct_out.append(entry)

    incorrect_out: list[dict[str, Any]] = []
    for row in incorrect_src[:MAX_INCORRECT_ENTRIES]:
        err = int(row["error_step"])
        entry = {
            "id": row["id"],
            "lean_statement": row["lean_statement"],
            "proof_index": row["proof_index"],
            "error_step": err,
            "steps": steps_with_labels(row["steps"], error_step=err),
        }
        if row.get("requested_error_style"):
            entry["requested_error_style"] = row["requested_error_style"]
        if row.get("error_class"):
            entry["error_class"] = row["error_class"]
        if row.get("category"):
            entry["category"] = row["category"]
        incorrect_out.append(entry)

    return {"correct_proofs": correct_out, "incorrect_proofs": incorrect_out}


def write_final_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    dataset = assemble_final_dataset(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def pools_full(rows: list[dict[str, Any]]) -> bool:
    good, bad = pools_from_records(rows)
    return len(good) >= MAX_CORRECT_ENTRIES and len(bad) >= MAX_INCORRECT_ENTRIES


def slot_done_outcomes() -> frozenset[str]:
    return frozenset(
        {
            "correct",
            "reasoning_error",
            "syntax_excluded",
            "dead_tactic_discarded",
            "ambiguous_excluded",
            "discarded_after_review",
            "forbidden_tactic_rejected",
            "verification_resource_exceeded",
            "generation_failed",
        }
    )


def completed_verification_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    done = slot_done_outcomes()
    keys = set()
    for row in rows:
        if row.get("outcome") not in done:
            continue
        keys.add((row["id"], int(row["proof_index"])))
    return keys


def requested_error_style(statement: dict[str, Any], proof_index: int) -> tuple[str, str]:
    key = f"{statement['id']}:{proof_index}"
    bucket = sum((i + 1) * ord(ch) for i, ch in enumerate(key))
    return INCORRECT_ERROR_STYLES[bucket % len(INCORRECT_ERROR_STYLES)]


def generation_prompt(
    statement: dict[str, Any],
    *,
    incorrect: bool = False,
    error_style: tuple[str, str] | None = None,
) -> str:
    escaped = statement["lean_statement"].replace("\\", "\\\\").replace('"', '\\"')
    prompt = render_template(
        USER_PROMPT,
        lean_statement=statement["lean_statement"],
        escaped_theorem=escaped,
        few_shot=FEW_SHOT_EXAMPLE,
    )
    context = str(statement.get("lean_context", "")).strip()
    if context:
        prompt += (
            "\nThe following Lean context is already available. Use its definitions, "
            "but do not repeat it in step 0:\n```lean\n"
            + context
            + "\n```\n"
        )
    sample_index = statement.get("_sample_index")
    if sample_index is not None:
        prompt += (
            f"\nThis is independent trajectory {int(sample_index) + 1}. "
            "Use a distinct valid proof strategy when the theorem naturally allows one.\n"
        )
    if incorrect:
        prompt += INCORRECT_PROMPT_ADDITION
        if error_style is not None:
            name, instruction = error_style
            prompt += (
                f"\nRequested reasoning-error style: {name}\n"
                f"- {instruction}\n"
                "- Use this style when it naturally fits the theorem; otherwise use the closest plausible student-like reasoning error.\n"
            )
    return prompt


def generate_one_proof_anthropic(
    client: Any,
    *,
    model: str,
    max_tokens: int,
    thinking_budget: int,
    incorrect: bool,
    statement: dict[str, Any],
    error_style: tuple[str, str] | None,
) -> tuple[list[dict[str, Any]], str]:
    prompt = generation_prompt(statement, incorrect=incorrect, error_style=error_style)
    kw: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking_budget > 0:
        if ADAPTIVE_THINKING_RE.match(model):
            # Adaptive models have no separate thinking-token cap: max_tokens
            # covers both hidden reasoning and visible output. Honor the CLI's
            # requested reasoning allowance here, then use a separate
            # no-thinking continuation below if the model needs output room.
            kw["max_tokens"] = min(max_tokens, thinking_budget)
            kw["thinking"] = {"type": "adaptive"}
            kw["output_config"] = {"effort": GENERATION_THINKING_EFFORT}
        else:
            kw["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    message = client.messages.create(**kw)
    record_usage("anthropic", model, "generate", getattr(message, "usage", None))
    raw_text = "".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    )

    # Sonnet 5 can spend the entire high-effort output allowance thinking and
    # stop before emitting any JSON. Preserve that reasoning and ask it only to
    # serialize the finished proof. This is a continuation of the same blind
    # attempt: it receives no Lean result or other correctness feedback.
    if not raw_text.strip() and getattr(message, "stop_reason", None) in {
        "max_tokens",
        "model_context_window_exceeded",
    }:
        continuation = client.messages.create(
            model=model,
            max_tokens=min(max_tokens, 32_000),
            system=SYSTEM_PROMPT,
            thinking={"type": "disabled"},
            output_config={"effort": "high"},
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": message.content},
                {
                    "role": "user",
                    "content": (
                        "Return the completed proof now as only the requested JSON "
                        "object. Do not restart or continue the reasoning."
                    ),
                },
            ],
        )
        record_usage(
            "anthropic", model, "generate_finalize", getattr(continuation, "usage", None)
        )
        raw_text = "".join(
            block.text
            for block in continuation.content
            if getattr(block, "type", None) == "text"
        )
    parsed = parse_model_json(raw_text)
    ok, reason, steps = validate_attempt_shape(parsed, statement)
    if not ok:
        raise ValueError(reason)
    return steps, raw_text


def generate_one_proof_gemini(
    *,
    api_key: str,
    model: str,
    max_tokens: int,
    api_timeout: float,
    incorrect: bool,
    statement: dict[str, Any],
    error_style: tuple[str, str] | None,
) -> tuple[list[dict[str, Any]], str]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": generation_prompt(
                            statement, incorrect=incorrect, error_style=error_style
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=api_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {exc.code}: {body}") from exc

    candidates = data.get("candidates") or []
    parts = (
        candidates[0].get("content", {}).get("parts", [])
        if candidates
        else []
    )
    raw_text = "".join(str(part.get("text", "")) for part in parts)
    if not raw_text:
        raise ValueError(f"Gemini response had no text: {data}")

    parsed = parse_model_json(raw_text)
    ok, reason, steps = validate_attempt_shape(parsed, statement)
    if not ok:
        raise ValueError(reason)
    return steps, raw_text


def generate_one_proof_together(
    *,
    api_key: str,
    model: str,
    max_tokens: int,
    api_timeout: float,
    incorrect: bool,
    statement: dict[str, Any],
    error_style: tuple[str, str] | None,
) -> tuple[list[dict[str, Any]], str]:
    prompt = generation_prompt(statement, incorrect=incorrect, error_style=error_style)
    parsed = call_together_chat_json(
        api_key=api_key,
        model=model,
        system=SYSTEM_PROMPT,
        prompt=prompt,
        max_tokens=max_tokens,
        api_timeout=api_timeout,
        usage_phase="generate",
    )
    ok, reason, steps = validate_attempt_shape(parsed, statement)
    if not ok:
        raise ValueError(reason)
    return steps, json.dumps(parsed, ensure_ascii=False)


def generate_one_proof_openai(
    *,
    api_key: str,
    model: str,
    max_tokens: int,
    api_timeout: float,
    statement: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    prompt = generation_prompt(statement, incorrect=False, error_style=None)
    parsed = call_openai_chat_json(
        api_key=api_key,
        model=model,
        system=SYSTEM_PROMPT,
        prompt=prompt,
        max_tokens=max_tokens,
        api_timeout=api_timeout,
    )
    ok, reason, steps = validate_attempt_shape(parsed, statement)
    if not ok:
        raise ValueError(reason)
    return steps, json.dumps(parsed, ensure_ascii=False)


def generate_one_proof_azure_openai(
    *,
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
    max_tokens: int,
    api_timeout: float,
    statement: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    prompt = generation_prompt(statement, incorrect=False, error_style=None)
    parsed = call_azure_openai_chat_json(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment=deployment,
        system=SYSTEM_PROMPT,
        prompt=prompt,
        max_tokens=max_tokens,
        api_timeout=api_timeout,
    )
    ok, reason, steps = validate_attempt_shape(parsed, statement)
    if not ok:
        raise ValueError(reason)
    return steps, json.dumps(parsed, ensure_ascii=False)


def two_step_review_rewrite(
    *,
    client: Any | None = None,
    together_key: str = "",
    azure_endpoint: str = "",
    azure_key: str = "",
    azure_api_version: str = DEFAULT_AZURE_OPENAI_API_VERSION,
    provider: str,
    model: str,
    max_tokens: int,
    api_timeout: float,
    thinking_budget: int,
    lean_statement: str,
    lean_context: str,
    steps: list[dict[str, Any]],
    lean_result: dict[str, Any],
    ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    errors = [m for m in lean_result.get("messages", []) if m.get("severity") == "error"]
    candidate = error_step_from_messages(errors, ranges) if errors else None
    prompt = render_template(
        TWO_STEP_REVIEW_PROMPT,
        lean_statement=lean_statement,
        lean_context=lean_context or "(none beyond Mathlib)",
        verification_status="failed" if lean_result.get("has_errors") else "passed",
        candidate_error_step=str(candidate),
        compiler_errors=compiler_errors_block(errors),
        steps_json=json.dumps(steps, indent=2, ensure_ascii=False),
        num_steps=str(len(steps)),
        error_class_options=ERROR_CLASS_OPTIONS_TEXT,
    )
    if provider == "together":
        result = call_together_chat_json(
            api_key=together_key,
            model=model,
            system=AUDIT_SYSTEM_PROMPT,
            prompt=prompt,
            max_tokens=max_tokens,
            api_timeout=api_timeout,
            usage_phase="review",
        )
    elif provider == "azure-openai":
        result = call_azure_openai_chat_json(
            endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=azure_api_version,
            deployment=model,
            system=AUDIT_SYSTEM_PROMPT,
            prompt=prompt,
            max_tokens=max_tokens,
            api_timeout=api_timeout,
        )
    else:
        if client is None:
            raise ValueError("Anthropic review requires a client")
        result = call_anthropic_json(
            client,
            model=model,
            system=AUDIT_SYSTEM_PROMPT,
            prompt=prompt,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            usage_phase="review",
        )
    verdict = str(result.get("verdict", "discard"))
    if verdict not in {"correct", "reasoning_error", "repair", "discard"}:
        verdict = "discard"
    discard_reason = str(result.get("discard_reason", ""))
    # Older reviewer behavior threw away incomplete best-shot proofs. Under the
    # dataset's formal-prefix semantics, an incomplete mathematical trajectory is
    # a natural incorrect example at its first rejected (or final unclosed) step.
    if (
        verdict == "discard"
        and lean_result.get("has_errors")
        and INCOMPLETE_PROOF_RE.search(discard_reason)
    ):
        verdict = "reasoning_error"
        result["error_class"] = result.get("error_class") or "missing_case_or_side_condition"
        result["first_error_step"] = (
            candidate if isinstance(candidate, int) and candidate > 0 else len(steps) - 1
        )
        result["discard_reason"] = ""
    first = bounded_step(result.get("first_error_step"), len(steps))
    if verdict == "reasoning_error" and first is None and candidate is not None:
        first = candidate
    if verdict != "reasoning_error":
        first = None
    # Error classification: only meaningful for a reasoning_error verdict. Coerce
    # anything off-list to "other" so the field is always a valid label or null.
    if verdict == "reasoning_error":
        error_class = result.get("error_class")
        if error_class not in ERROR_CLASSES:
            error_class = "other"
    else:
        error_class = None
    repair_steps: list[dict[str, Any]] = []
    patch_errors: list[str] = []
    if verdict == "repair":
        patches = result.get("syntax_patches")
        if not isinstance(patches, list) or not patches:
            patch_errors.append("repair returned no syntax patches")
        repair_steps = [dict(step) for step in steps]
        seen: set[int] = set()
        for patch in patches if isinstance(patches, list) else []:
            index = patch.get("step") if isinstance(patch, dict) else None
            lean_code = str(patch.get("lean_code", "")).rstrip() if isinstance(patch, dict) else ""
            if not isinstance(index, int) or not 0 < index < len(steps):
                patch_errors.append(f"invalid syntax patch step: {index}")
                continue
            if index in seen:
                patch_errors.append(f"duplicate syntax patch step: {index}")
                continue
            if not lean_code or FORBIDDEN_LEAN_RE.search(lean_code):
                patch_errors.append(f"invalid syntax patch code at step: {index}")
                continue
            seen.add(index)
            repair_steps[index]["lean_code"] = lean_code
        if patch_errors:
            verdict = "discard"
            repair_steps = []
            result["discard_reason"] = "; ".join(patch_errors)
    return {
        "verdict": verdict,
        "first_error_step": first,
        "error_class": error_class,
        "discard_reason": str(result.get("discard_reason", "")),
        "rewrite_fallback": "; ".join(patch_errors),
        "steps": [],
        "repair_steps": repair_steps,
        "candidate_error_step": candidate,
    }


def apply_review_rewrite_steps(
    steps: list[dict[str, Any]],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    # Stage 2 never edits Stage-1 natural language. For syntax-repaired samples,
    # `steps` already contains the privately repaired Lean with the original NL.
    return [dict(step) for step in steps]


def verify_steps(
    server: Any,
    command_cls: Any,
    steps: list[dict[str, Any]],
    lean_context: str = "",
) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[tuple[int, int]]]:
    proof_src, ranges, lean_result = run_lean_check(
        server, command_cls, steps, lean_context
    )
    active_steps = steps

    if RECOVER_DEAD_TACTICS:
        dead_recovery = recover_dead_tactics(
            server, command_cls, steps, lean_result, ranges, lean_context
        )
        if dead_recovery is not None:
            trimmed_result = dead_recovery["trimmed_lean_result"]
            dead_ok = (
                not trimmed_result["has_errors"]
                and not dead_tactic_messages(trimmed_result.get("messages", []))
            )
            if dead_ok:
                active_steps = dead_recovery["trimmed_steps"]
                proof_src = dead_recovery["trimmed_full_proof"]
                ranges = dead_recovery["trimmed_ranges"]
                lean_result = trimmed_result

    return proof_src, active_steps, lean_result, ranges


def audit_dataset_command(args: argparse.Namespace) -> None:
    load_env_file(ENV_PATH)
    input_path = resolve_script_path(args.input, SCRIPT_DIR / "opus_46_final_dataset.json")
    output_path = resolve_script_path(args.output, DEFAULT_AUDIT_OUTPUT)
    log_path = resolve_script_path(
        args.log,
        input_path.with_name(f"{input_path.stem}_audit_results.jsonl"),
    )
    model = args.model or DEFAULT_ANTHROPIC_MODEL
    dataset = load_json(input_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    done = completed_log_by_key(log_path)
    client = open_anthropic_client(args.api_timeout)

    audited_now = 0
    for proof in dataset.get("incorrect_proofs", []):
        key = proof_key(proof)
        if key in done:
            continue
        if args.limit is not None and audited_now >= args.limit:
            break
        print(f"auditing {key[0]} proof_index={key[1]}...", flush=True)
        row = audit_proof_with_model(
            client,
            model=model,
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            proof=proof,
        )
        row["audited_at"] = now_iso()
        append_jsonl(log_path, row)
        done[key] = row
        audited_now += 1
        write_json(output_path, apply_audit_to_dataset(dataset, done))

    output = apply_audit_to_dataset(dataset, completed_log_by_key(log_path))
    write_json(output_path, output)
    print(
        f"Wrote {output_path}: "
        f"{len(output.get('correct_proofs', []))} correct, "
        f"{len(output.get('incorrect_proofs', []))} audited incorrect",
        flush=True,
    )


def rewrite_nl_command(args: argparse.Namespace) -> None:
    load_env_file(ENV_PATH)
    input_path = resolve_script_path(args.input, DEFAULT_AUDIT_OUTPUT)
    output_path = resolve_script_path(args.output, DEFAULT_REWRITE_OUTPUT)
    log_path = resolve_script_path(
        args.log,
        input_path.with_name(f"{input_path.stem}_rewrite_results.jsonl"),
    )
    model = args.model or DEFAULT_ANTHROPIC_MODEL
    dataset = load_json(input_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    done = completed_rewrite_log_by_key(log_path)
    client = open_anthropic_client(args.api_timeout)

    entries: list[tuple[str, dict[str, Any]]] = []
    entries.extend(("correct_proofs", proof) for proof in dataset.get("correct_proofs", []))
    entries.extend(("incorrect_proofs", proof) for proof in dataset.get("incorrect_proofs", []))

    rewritten_now = 0
    for section, proof in entries:
        key = proof_section_key(section, proof)
        if key in done:
            continue
        if args.limit is not None and rewritten_now >= args.limit:
            break
        print(f"rewriting {key[1]} proof_index={key[2]}...", flush=True)
        try:
            updated = rewrite_proof_with_model(
                client,
                model=model,
                max_tokens=args.max_tokens,
                thinking_budget=args.thinking_budget,
                proof=proof,
            )
            row = {
                "id": proof.get("id"),
                "proof_index": proof.get("proof_index"),
                "section": section,
                "status": "ok",
                "rewritten_at": now_iso(),
                "proof": updated,
            }
        except Exception as exc:
            row = {
                "id": proof.get("id"),
                "proof_index": proof.get("proof_index"),
                "section": section,
                "status": "failed",
                "rewritten_at": now_iso(),
                "error": str(exc),
            }
        append_jsonl(log_path, row)
        done[key] = row
        rewritten_now += 1
        write_json(output_path, apply_rewrites_to_dataset(dataset, done))

    output = apply_rewrites_to_dataset(dataset, completed_rewrite_log_by_key(log_path))
    write_json(output_path, output)
    print(
        f"Wrote {output_path}: "
        f"{len(output.get('correct_proofs', []))} correct, "
        f"{len(output.get('incorrect_proofs', []))} incorrect",
        flush=True,
    )


def apply_rewrites_to_dataset(
    dataset: dict[str, Any],
    rewrite_rows: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"correct_proofs": [], "incorrect_proofs": []}
    for section in ("correct_proofs", "incorrect_proofs"):
        for proof in dataset.get(section, []):
            row = rewrite_rows.get(proof_section_key(section, proof))
            if row and row.get("status") == "ok" and isinstance(row.get("proof"), dict):
                output[section].append(row["proof"])
            else:
                output[section].append(proof)
    return output


def statement_hardness_score(statement: dict[str, Any]) -> float:
    src = str(statement.get("lean_statement", ""))
    category = str(statement.get("category", ""))
    category_bonus = {
        "existence_construction": 4.0,
        "iff_logic": 3.5,
        "divisibility_modular": 3.0,
        "equality_identity": 2.0,
        "inequality": 1.0,
    }.get(category, 1.5)
    structural = sum(src.count(token) for token in ("∀", "∃", "↔", "→", "∧", "∨"))
    hypotheses = src.count(") (") + src.count("} {")
    return category_bonus + min(len(src) / 120.0, 5.0) + structural + hypotheses * 0.4


def hard_statement_order(statements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for statement in statements:
        by_category.setdefault(str(statement.get("category", "unknown")), []).append(statement)
    for rows in by_category.values():
        rows.sort(key=statement_hardness_score, reverse=True)
    category_order = sorted(
        by_category,
        key=lambda cat: (cat == "inequality", -len(by_category[cat]), cat),
    )
    ordered: list[dict[str, Any]] = []
    while any(by_category.values()):
        for category in category_order:
            rows = by_category[category]
            if rows:
                ordered.append(rows.pop(0))
    return ordered


def run_two_step_pipeline(args: argparse.Namespace) -> None:
    load_env_file(ENV_PATH)
    run_dir = resolve_resume_run(args.resume_run) if args.resume_run else make_run_dir()
    records_path = run_dir / "attempts.jsonl"
    final_path = run_dir / "final_dataset.json"
    print(f"{'Resuming' if args.resume_run else 'Run'} directory: {run_dir}", flush=True)

    input_path = resolve_input_path(args.input or "final_valid_lean_statements_hf_balanced_4000.json")
    print(f"Input statements: {input_path}", flush=True)
    statements = hard_statement_order([normalize_statement(row) for row in load_json(input_path)])
    allowed_ids = {str(statement["id"]) for statement in statements}

    def current_input_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in records if str(row.get("id")) in allowed_ids]

    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.touch(exist_ok=True)
    rows = current_input_rows(read_jsonl(records_path))
    write_final_dataset(final_path, rows)
    already = completed_verification_keys(rows)

    initial_provider = args.initial_provider
    openai_key_value = openai_api_key() if initial_provider == "openai" else ""
    together_key_value = together_api_key_from_env() if initial_provider == "together" else ""
    azure_endpoint, azure_key, azure_api_version = (
        azure_openai_config()
        if initial_provider == "azure-openai" or args.review_provider == "azure-openai"
        else ("", "", DEFAULT_AZURE_OPENAI_API_VERSION)
    )
    initial_anthropic_client = open_anthropic_client(args.api_timeout) if initial_provider == "anthropic" else None
    review_provider = args.review_provider
    review_client = open_anthropic_client(args.api_timeout) if review_provider == "anthropic" else None
    review_together_key = together_api_key_from_env() if review_provider == "together" else ""
    initial_model = args.initial_model or (
        DEFAULT_AZURE_OPENAI_DEPLOYMENT
        if initial_provider == "azure-openai"
        else DEFAULT_TOGETHER_MODEL
        if initial_provider == "together"
        else DEFAULT_ANTHROPIC_MODEL
        if initial_provider == "anthropic"
        else DEFAULT_OPENAI_MODEL
    )
    review_model = args.review_model or (
        DEFAULT_AZURE_OPENAI_REVIEW_DEPLOYMENT
        if review_provider == "azure-openai"
        else DEFAULT_TOGETHER_MODEL
        if review_provider == "together"
        else DEFAULT_ANTHROPIC_MODEL
    )

    ensure_lake_on_path()
    try:
        from lean_interact import Command as Cmd  # pyright: ignore[reportMissingImports]
        from lean_interact import (  # pyright: ignore[reportMissingImports]
            AutoLeanServer,
            LeanREPLConfig,
            TempRequireProject,
        )
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install lean-interact") from exc
    config = LeanREPLConfig(
        project=TempRequireProject(
            directory=LEAN_PROJECT_DIR,
            lean_version=LEAN_VERSION_DEFAULT,
            require="mathlib",
            auto_build=True,
            verbose=False,
        ),
        cache_dir=LEAN_INTERACT_CACHE_DIR,
        memory_hard_limit_mb=LEAN_MEMORY_HARD_LIMIT_MB,
        enable_parallel_elaboration=False,
        verbose=False,
    )
    server = AutoLeanServer(
        config,
        max_total_memory=LEAN_MAX_TOTAL_MEMORY_FRACTION,
        max_process_memory=LEAN_MAX_PROCESS_MEMORY_FRACTION,
    )

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lean_lock = threading.Lock()   # Lean server is single-threaded; serialize it
    io_lock = threading.Lock()     # serialize file writes + shared counters
    lean_checks_since_restart = 0

    def guarded_verify_steps(
        steps: list[dict[str, Any]], lean_context: str
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[tuple[int, int]]]:
        nonlocal lean_checks_since_restart
        with lean_lock:
            result = verify_steps(server, Cmd, steps, lean_context)
            lean_checks_since_restart += 1
            if lean_checks_since_restart >= LEAN_REPL_RECYCLE_EVERY_CHECKS:
                server.restart()
                lean_checks_since_restart = 0
            return result

    _cp, _ip = pools_from_records(rows)
    state = {
        "attempted": len(completed_verification_keys(rows)),
        "kept": len(_cp) + len(_ip),
    }
    stop_flag = threading.Event()
    workers = max(1, getattr(args, "workers", 1))

    # Build to-do list: independent proof slots not already attempted.
    todo: list[tuple[dict[str, Any], int]] = []
    # Cover every theorem once before requesting second and third trajectories.
    # This guarantees broad theorem coverage even when target_kept stops the run
    # before every possible slot is attempted.
    for proof_index in range(
        args.proof_index_start,
        args.proof_index_start + args.proofs_per_statement,
    ):
        for statement in statements:
            if (str(statement["id"]), proof_index) in already:
                continue
            todo.append((statement, proof_index))
    remaining_budget = max(0, args.max_attempts - state["attempted"])
    todo = todo[:remaining_budget]
    print(f"Processing {len(todo)} statements with {workers} worker(s); "
          f"target_kept={args.target_kept}.", flush=True)

    def process_one(statement: dict[str, Any], proof_index: int) -> None:
        if stop_flag.is_set():
            return
        sid = str(statement["id"])
        key = (sid, proof_index)
        # Official proofs may remain in private source metadata, but generation
        # must be blind. Remove them before constructing any solver prompt.
        generation_statement = {
            key: value for key, value in statement.items() if key != "reference_proof"
        }
        generation_statement["_sample_index"] = proof_index
        record: dict[str, Any] = {
            "id": sid,
            "lean_statement": statement["lean_statement"],
            "proof_index": proof_index,
            "provider": initial_provider,
            "initial_provider": initial_provider,
            "model": initial_model,
            "review_model": review_model,
            "review_provider": review_provider,
            "generated_at": now_iso(),
            "category": statement.get("category"),
            "two_step": True,
            "error_source": "natural_solver_failure",
        }
        with io_lock:
            append_jsonl(records_path, {**record, "outcome": "started"})
        print(
            f"{sid}[{proof_index}]: two-step generating with "
            f"{initial_provider}:{initial_model}...",
            flush=True,
        )

        steps: list[dict[str, Any]] = []
        raw_text = ""
        for _retry in range(MAX_RETRIES):
            try:
                if initial_provider == "anthropic":
                    steps, raw_text = generate_one_proof_anthropic(
                        initial_anthropic_client,
                        model=initial_model,
                        max_tokens=args.max_tokens,
                        thinking_budget=args.thinking_budget,
                        incorrect=False,
                        statement=generation_statement,
                        error_style=None,
                    )
                elif initial_provider == "together":
                    steps, raw_text = generate_one_proof_together(
                        api_key=together_key_value,
                        model=initial_model,
                        max_tokens=args.max_tokens,
                        api_timeout=args.api_timeout,
                        incorrect=False,
                        statement=generation_statement,
                        error_style=None,
                    )
                elif initial_provider == "azure-openai":
                    steps, raw_text = generate_one_proof_azure_openai(
                        endpoint=azure_endpoint,
                        api_key=azure_key,
                        api_version=azure_api_version,
                        deployment=initial_model,
                        max_tokens=args.max_tokens,
                        api_timeout=args.api_timeout,
                        statement=generation_statement,
                    )
                else:
                    steps, raw_text = generate_one_proof_openai(
                        api_key=openai_key_value,
                        model=initial_model,
                        max_tokens=args.max_tokens,
                        api_timeout=args.api_timeout,
                        statement=generation_statement,
                    )
                record.update(
                    {
                        "raw_text": raw_text,
                        "original_steps": [dict(step) for step in steps],
                        "steps": [dict(step) for step in steps],
                    }
                )
                proof_src, active_steps, lean_result, ranges = guarded_verify_steps(
                    steps, str(statement.get("lean_context", ""))
                )
                review = two_step_review_rewrite(
                    client=review_client,
                    together_key=review_together_key,
                    azure_endpoint=azure_endpoint,
                    azure_key=azure_key,
                    azure_api_version=azure_api_version,
                    provider=review_provider,
                    model=review_model,
                    max_tokens=args.max_tokens,
                    api_timeout=args.api_timeout,
                    thinking_budget=args.thinking_budget,
                    lean_statement=statement["lean_statement"],
                    lean_context=str(statement.get("lean_context", "")),
                    steps=active_steps,
                    lean_result=lean_result,
                    ranges=ranges,
                )
                # Lean-repair path: step 2 returned corrected lean_code for a
                # proof whose math is valid but whose Lean failed mechanically.
                # We MUST re-verify before trusting it: only accept if the
                # repaired proof compiles clean, keeps step 0 (the theorem)
                # unchanged, and uses no cheat tactics (validate_attempt_shape).
                repaired = False
                if review["verdict"] == "repair":
                    ok, reason, norm_steps = validate_attempt_shape(
                        {"steps": review.get("repair_steps", [])}, statement
                    )
                    if not ok:
                        review["verdict"] = "discard"
                        review["discard_reason"] = f"repair rejected: {reason}"
                    else:
                        r_src, r_steps, r_result, r_ranges = guarded_verify_steps(
                            norm_steps, str(statement.get("lean_context", ""))
                        )
                        if not r_result.get("has_errors"):
                            review["verdict"] = "correct"
                            proof_src, active_steps, lean_result, ranges = (
                                r_src, r_steps, r_result, r_ranges,
                            )
                            repaired = True
                        else:
                            review["verdict"] = "discard"
                            review["discard_reason"] = "repair did not compile"
                record.update(
                    {
                        "raw_text": raw_text,
                        "steps": apply_review_rewrite_steps(active_steps, review),
                        "full_proof": proof_src,
                        "lean_result": lean_result,
                        "review_rewrite": review,
                        "verified_at": now_iso(),
                    }
                )
                if review["verdict"] == "correct":
                    record["outcome"] = "correct"
                    if repaired:
                        record["bucket"] = "repaired_from_lean_error"
                elif review["verdict"] == "reasoning_error":
                    record["outcome"] = "reasoning_error"
                    record["bucket"] = "model_confirmed_reasoning_error"
                    record["error_step"] = review["first_error_step"]
                    record["error_class"] = review.get("error_class")
                else:
                    record["outcome"] = "discarded_after_review"
                    record["bucket"] = review.get("discard_reason", "discard")
                break  # success, exit retry loop
            except Exception as exc:
                # Transient API errors deserve the full retry budget with backoff.
                # Deterministic generation-format errors (step-0 mismatch, placeholder
                # tactic, JSON parse) rarely self-heal, so cap them low to avoid
                # burning expensive thinking calls.
                msg = str(exc).lower()
                # Lean verification timeouts are deterministic (the same proof will
                # time out again) and each retry triggers a costly Mathlib restart,
                # so fail those fast.
                lean_timeout = "did not respond in time" in msg or "is now killed" in msg
                lean_resource_error = lean_timeout or any(
                    text in msg
                    for text in (
                        "lean server closed unexpectedly",
                        "memory usage is too high",
                        "cannot allocate memory",
                    )
                )
                transient = (not lean_resource_error) and any(t in msg for t in (
                    "overloaded", "529", "500", "503", "timeout", "timed out",
                    "connection", "rate limit", "429",
                ))
                cap = MAX_RETRIES if transient else (1 if lean_resource_error else 2)
                if _retry < cap - 1:
                    print(f"  {sid}: attempt {_retry+1}/{cap} failed: {exc}", flush=True)
                    time.sleep(RETRY_SLEEP_S * (2 ** _retry) if transient else 1.0)
                else:
                    record.update(
                        {
                            "outcome": (
                                "verification_resource_exceeded"
                                if lean_resource_error and steps
                                else "generation_failed"
                            ),
                            "bucket": (
                                "lean_memory_or_timeout"
                                if lean_resource_error and steps
                                else "generation_failure"
                            ),
                            "error": str(exc),
                            "verified_at": now_iso(),
                        }
                    )
                    break

        with io_lock:
            append_jsonl(records_path, record)
            already.add(key)
            state["attempted"] += 1
            if record.get("outcome") in ("correct", "reasoning_error"):
                state["kept"] += 1
            if state["attempted"] % 10 == 0 or state["kept"] >= args.target_kept:
                write_final_dataset(final_path, current_input_rows(read_jsonl(records_path)))
            print(f"  [{state['attempted']} attempted | {state['kept']} kept] {sid} -> {record.get('outcome')}", flush=True)
            if state["kept"] >= args.target_kept:
                stop_flag.set()

    if workers == 1:
        for statement, proof_index in todo:
            if stop_flag.is_set():
                break
            process_one(statement, proof_index)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_one, s, i) for s, i in todo]
            for _ in as_completed(futures):
                pass

    rows = current_input_rows(read_jsonl(records_path))
    write_final_dataset(final_path, rows)
    correct_pool, incorrect_pool = pools_from_records(rows)
    write_json(
        run_dir / "run_summary.json",
        {
            "finished_at": now_iso(),
            "initial_model": initial_model,
            "initial_provider": initial_provider,
            "review_model": review_model,
            "review_provider": review_provider,
            "target_kept": args.target_kept,
            "max_attempts": args.max_attempts,
            "attempt_records": len([r for r in rows if r.get("outcome") != "started"]),
            "correct": len(correct_pool),
            "incorrect": len(incorrect_pool),
            "query_budget_per_successful_attempt": 2,
        },
    )
    print(
        f"Wrote {final_path}: {len(correct_pool)} correct, {len(incorrect_pool)} incorrect.",
        flush=True,
    )


def run_pipeline(args: argparse.Namespace) -> None:
    load_env_file(ENV_PATH)

    run_dir = resolve_resume_run(args.resume_run) if args.resume_run else make_run_dir()
    records_path = run_dir / "attempts.jsonl"
    final_path = run_dir / "final_dataset.json"
    print(f"{'Resuming' if args.resume_run else 'Run'} directory: {run_dir}", flush=True)

    input_path = resolve_input_path(args.input)
    print(f"Input statements: {input_path}", flush=True)
    statements = [normalize_statement(row) for row in load_json(input_path)]
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.touch(exist_ok=True)
    initial_rows = read_jsonl(records_path)
    write_final_dataset(final_path, initial_rows)
    print(f"Checkpoint files: {records_path} and {final_path}", flush=True)
    already = completed_verification_keys(initial_rows)

    client: Any | None = None
    server: Any | None = None
    Command: Any | None = None

    def open_anthropic() -> Any:
        nonlocal client
        if client is not None:
            return client
        client = open_anthropic_client(args.api_timeout)
        return client

    def google_api_key() -> str:
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit(
                "GOOGLE_API_KEY is missing. Set it in the environment "
                f"or in {ENV_PATH}."
            )
        return key

    def together_api_key() -> str:
        key = os.environ.get("TOGETHER_API_KEY")
        if not key:
            raise SystemExit(
                "TOGETHER_API_KEY is missing. Set it in the environment "
                f"or in {ENV_PATH}."
            )
        return key

    def open_lean() -> tuple[Any, Any]:
        nonlocal server, Command
        if server is not None and Command is not None:
            return server, Command
        ensure_lake_on_path()
        try:
            from lean_interact import Command as Cmd  # pyright: ignore[reportMissingImports]
            from lean_interact import (  # pyright: ignore[reportMissingImports]
                LeanREPLConfig,
                LeanServer,
                TempRequireProject,
            )
        except ImportError as exc:
            raise SystemExit("Missing dependency: pip install lean-interact") from exc
        config = LeanREPLConfig(
            project=TempRequireProject(
                directory=LEAN_PROJECT_DIR,
                lean_version=LEAN_VERSION_DEFAULT,
                require="mathlib",
                auto_build=True,
                verbose=False,
            ),
            cache_dir=LEAN_INTERACT_CACHE_DIR,
            memory_hard_limit_mb=LEAN_MEMORY_HARD_LIMIT_MB,
            enable_parallel_elaboration=False,
            verbose=False,
        )
        Command = Cmd
        server = LeanServer(config)
        return server, Command

    new_completed = 0
    new_incorrect = 0
    provider = args.provider
    if args.model:
        model = args.model
    elif provider == "gemini":
        model = DEFAULT_GEMINI_MODEL
    elif provider == "together":
        model = DEFAULT_TOGETHER_MODEL
    else:
        model = DEFAULT_ANTHROPIC_MODEL
    # Guard removed — now supporting any Anthropic model (Opus 4.8, etc.)
    stop_all = False

    def limit_reached(rows: list[dict[str, Any]]) -> bool:
        if args.limit is None:
            return False
        if args.incorrect:
            _, incorrect_pool = pools_from_records(rows)
            return len(incorrect_pool) >= args.limit
        return new_completed >= args.limit

    def target_pool_full(rows: list[dict[str, Any]]) -> bool:
        correct_pool, incorrect_pool = pools_from_records(rows)
        if args.incorrect:
            return len(incorrect_pool) >= MAX_INCORRECT_ENTRIES
        return len(correct_pool) >= MAX_CORRECT_ENTRIES

    for statement in statements:
        rows_now = read_jsonl(records_path)
        if pools_full(rows_now):
            print("Reached target pool sizes — stopping.", flush=True)
            break
        if target_pool_full(rows_now):
            print(
                "Reached "
                f"{'incorrect' if args.incorrect else 'correct'} pool cap — stopping.",
                flush=True,
            )
            break
        if limit_reached(rows_now):
            print(
                f"Reached --limit={args.limit}"
                f"{' incorrect proofs' if args.incorrect else ''}.",
                flush=True,
            )
            break

        sid = str(statement["id"])
        for proof_index in range(
            args.proof_index_start,
            args.proof_index_start + args.proofs_per_statement,
        ):
            rows_now = read_jsonl(records_path)
            if pools_full(rows_now):
                break
            if target_pool_full(rows_now):
                print(
                    "Reached "
                    f"{'incorrect' if args.incorrect else 'correct'} pool cap — stopping.",
                    flush=True,
                )
                stop_all = True
                break
            if limit_reached(rows_now):
                print(
                    f"Reached --limit={args.limit}"
                    f"{' incorrect proofs' if args.incorrect else ''}.",
                    flush=True,
                )
                stop_all = True
                break

            key = (sid, proof_index)
            if key in already:
                continue
            if not args.incorrect and args.limit is not None and new_completed >= args.limit:
                print(f"Reached --limit={args.limit}.", flush=True)
                stop_all = True
                break

            record: dict[str, Any] = {
                "id": sid,
                "lean_statement": statement["lean_statement"],
                "proof_index": proof_index,
                "provider": provider,
                "model": model,
                "generated_at": now_iso(),
            }
            error_style = requested_error_style(statement, proof_index) if args.incorrect else None
            if error_style is not None:
                record["requested_error_style"] = error_style[0]

            append_jsonl(records_path, {**record, "outcome": "started"})
            write_final_dataset(final_path, read_jsonl(records_path))
            print(f"{sid} proof_index={proof_index}: generating...", flush=True)
            steps: list[dict[str, Any]] | None = None
            raw_text = ""
            gen_ok = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    if provider == "gemini":
                        steps, raw_text = generate_one_proof_gemini(
                            api_key=google_api_key(),
                            model=model,
                            max_tokens=args.max_tokens,
                            api_timeout=args.api_timeout,
                            incorrect=args.incorrect,
                            statement=statement,
                            error_style=error_style,
                        )
                    elif provider == "together":
                        steps, raw_text = generate_one_proof_together(
                            api_key=together_api_key(),
                            model=model,
                            max_tokens=args.max_tokens,
                            api_timeout=args.api_timeout,
                            incorrect=args.incorrect,
                            statement=statement,
                            error_style=error_style,
                        )
                    else:
                        steps, raw_text = generate_one_proof_anthropic(
                            open_anthropic(),
                            model=model,
                            max_tokens=args.max_tokens,
                            thinking_budget=args.thinking_budget,
                            incorrect=args.incorrect,
                            statement=statement,
                            error_style=error_style,
                        )
                    gen_ok = True
                    break
                except Exception as exc:
                    print(f"  gen attempt {attempt}/{MAX_RETRIES}: {exc}", flush=True)
                    if attempt == MAX_RETRIES:
                        record.update(
                            {
                                "outcome": "generation_failed",
                                "error": str(exc),
                                "api_attempt": attempt,
                            }
                        )
                        append_jsonl(records_path, record)
                        write_final_dataset(final_path, read_jsonl(records_path))
                        new_completed += 1
                    else:
                        time.sleep(RETRY_SLEEP_S)

            if not gen_ok:
                continue

            record["steps"] = steps
            record["raw_text"] = raw_text

            print("  verifying...", flush=True)
            srv, Cmd = open_lean()
            proof_src, active_steps, lean_result, ranges = verify_steps(
                srv, Cmd, steps, str(statement.get("lean_context", ""))
            )

            errs_all = [
                m for m in lean_result.get("messages", []) if m.get("severity") == "error"
            ]

            record.update(
                {
                    "verified_at": now_iso(),
                    "full_proof": proof_src,
                    "steps": active_steps,
                    "lean_result": lean_result,
                }
            )

            policy_rejected_step = next(
                (
                    i
                    for i, step in enumerate(active_steps)
                    if not str(step.get("lean_code", "")).strip()
                    or FORBIDDEN_LEAN_RE.search(str(step.get("lean_code", "")))
                ),
                None,
            )
            compiler_error_step = (
                error_step_from_messages(errs_all, ranges) if errs_all else None
            )
            if policy_rejected_step is not None and (
                compiler_error_step is None or policy_rejected_step <= compiler_error_step
            ):
                record.update(
                    {
                        "outcome": "forbidden_tactic_rejected",
                        "bucket": "invalid_or_forbidden_step",
                        "error_step": policy_rejected_step,
                    }
                )
                append_jsonl(records_path, record)
                write_final_dataset(final_path, read_jsonl(records_path))
                already.add(key)
                new_completed += 1
                continue

            if is_dead_tactic_only_failure(lean_result.get("messages", [])) and lean_result[
                "has_errors"
            ]:
                record.update(
                    {
                        "outcome": "dead_tactic_discarded",
                        "error_step": error_step_from_messages(errs_all, ranges),
                    }
                )
                append_jsonl(records_path, record)
                write_final_dataset(final_path, read_jsonl(records_path))
                already.add(key)
                new_completed += 1
                continue

            if lean_result["has_errors"]:
                candidate = error_step_from_messages(errs_all, ranges)
                syn = classify_messages(errs_all)
                if syn == "syntax_error":
                    record.update(
                        {
                            "outcome": "syntax_excluded",
                            "bucket": syn,
                            "error_step": candidate,
                        }
                    )
                elif args.review_failures:
                    review = review_failed_attempt_with_model(
                        open_anthropic(),
                        model=model if provider == "anthropic" else DEFAULT_ANTHROPIC_MODEL,
                        max_tokens=args.max_tokens,
                        thinking_budget=args.thinking_budget,
                        lean_statement=statement["lean_statement"],
                        steps=active_steps,
                        errors=errs_all,
                        candidate_error_step=candidate,
                    )
                    record["failure_review"] = review
                    if review["verdict"] == "reasoning_error":
                        record.update(
                            {
                                "outcome": "reasoning_error",
                                "bucket": "model_confirmed_reasoning_error",
                                "error_step": review["first_error_step"],
                            }
                        )
                        new_incorrect += 1
                    else:
                        record.update(
                            {
                                "outcome": "ambiguous_excluded",
                                "bucket": review["verdict"],
                                "error_step": candidate,
                            }
                        )
                else:
                    record.update(
                        {
                            "outcome": "reasoning_error",
                            "bucket": syn,
                            "error_step": candidate,
                        }
                    )
                    new_incorrect += 1
            else:
                record.update({"outcome": "correct"})

            append_jsonl(records_path, record)
            write_final_dataset(final_path, read_jsonl(records_path))
            already.add(key)
            new_completed += 1
            if args.incorrect and args.limit is not None and new_incorrect >= args.limit:
                print(f"Reached --limit={args.limit} new incorrect proofs.", flush=True)
                stop_all = True
                break

        if stop_all:
            break

    all_rows = read_jsonl(records_path)
    finalize_dataset = assemble_final_dataset(all_rows)
    cc = finalize_dataset["correct_proofs"]
    ic = finalize_dataset["incorrect_proofs"]
    pool_c, pool_i = pools_from_records(all_rows)
    write_final_dataset(final_path, all_rows)
    print(
        f"Wrote {final_path}: {len(cc)} correct (cap {MAX_CORRECT_ENTRIES}), "
        f"{len(ic)} incorrect (cap {MAX_INCORRECT_ENTRIES}). "
        f"Eligible pools: correct={len(pool_c)}, incorrect={len(pool_i)}.",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "audit", "rewrite-nl", "two-step"),
        default="run",
        help=(
            "Pipeline stage to run. Default 'run' generates/verifies attempts; "
            "'audit' model-reviews incorrect labels; 'rewrite-nl' rewrites NL as "
            "a legacy standalone proof pass; 'two-step' uses one blind solver "
            "call, Lean verification, then classification with syntax-only patches."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("anthropic", "gemini", "together"),
        default=DEFAULT_PROVIDER,
        help=f"Generation provider (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override for the selected provider.",
    )
    parser.add_argument(
        "--initial-model",
        default=None,
        help=f"Initial solver model for two-step mode (default: {DEFAULT_OPENAI_MODEL}).",
    )
    parser.add_argument(
        "--initial-provider",
        choices=("openai", "together", "azure-openai", "anthropic"),
        default="anthropic",
        help="Initial solver provider for two-step mode (default: anthropic).",
    )
    parser.add_argument(
        "--review-model",
        default=None,
        help=f"Classifier/syntax-patch model for two-step mode (default: {DEFAULT_ANTHROPIC_MODEL}).",
    )
    parser.add_argument(
        "--review-provider",
        choices=("anthropic", "together", "azure-openai"),
        default="anthropic",
        help="Classifier/syntax-patch provider for two-step mode (default: anthropic).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Input JSON. For run: statement JSON override, absolute or relative "
            "to lean_statement_selection/. For audit/rewrite-nl: dataset JSON."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output dataset path for audit or rewrite-nl stages.",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="JSONL checkpoint log for audit or rewrite-nl stages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of new proof-slot attempts. With --incorrect, "
            "stop after this many accepted reasoning-error proofs."
        ),
    )
    parser.add_argument(
        "--target-kept",
        type=int,
        default=TWO_STEP_TARGET_KEPT,
        help=f"Kept sample target for two-step mode (default: {TWO_STEP_TARGET_KEPT}).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=TWO_STEP_MAX_ATTEMPTS,
        help=f"Maximum two-step attempted datapoints (default: {TWO_STEP_MAX_ATTEMPTS}).",
    )
    parser.add_argument(
        "--proofs-per-statement",
        type=int,
        default=PROOFS_PER_STATEMENT,
        help=f"Proof slots to try per statement (default: {PROOFS_PER_STATEMENT}).",
    )
    parser.add_argument(
        "--proof-index-start",
        type=int,
        default=0,
        help=(
            "First proof_index to use. Useful when resuming the same run with "
            "--incorrect so correct and incorrect attempts do not share slot keys."
        ),
    )
    parser.add_argument(
        "--incorrect",
        action="store_true",
        help="Append prompt instructions for intentional reasoning-error proof generation.",
    )
    parser.add_argument(
        "--review-failures",
        action="store_true",
        help=(
            "During run, send ambiguous Lean failures through a model review "
            "before accepting them as reasoning_error negatives."
        ),
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        help=(
            "Existing run directory to continue, either absolute, relative to "
            "lean_statement_selection/, or a bare name under runs/."
        ),
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=API_TIMEOUT_S,
        help=f"Provider request timeout seconds (default: {API_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Provider max output tokens (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=DEFAULT_THINKING_BUDGET,
        help=f"Anthropic thinking budget tokens; use 0 to disable (default: {DEFAULT_THINKING_BUDGET}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent two-step workers. Lean verification is serialized; API calls run in parallel (default: 1).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "audit":
        audit_dataset_command(args)
    elif args.command == "rewrite-nl":
        rewrite_nl_command(args)
    elif args.command == "two-step":
        run_two_step_pipeline(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
