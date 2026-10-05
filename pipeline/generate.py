"""Run blind generation followed by Lean checking and model review."""

import argparse
from pathlib import Path

from . import final_dataset_pipeline as engine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/generation"))
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--review-model", default=None)
    p.add_argument("--target-kept", type=int, default=100)
    p.add_argument("--max-attempts", type=int, default=300)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume-run", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.input.is_file():
        p.error("Input JSON does not exist")
    if min(a.workers, a.target_kept, a.max_attempts) < 1:
        p.error("Counts must be positive")
    engine.RUNS_DIR = a.output_dir.resolve()
    argv = [
        "two-step",
        "--input",
        str(a.input.resolve()),
        "--initial-provider",
        "anthropic",
        "--initial-model",
        a.model,
        "--review-provider",
        "anthropic",
        "--review-model",
        a.review_model or a.model,
        "--proofs-per-statement",
        "1",
        "--target-kept",
        str(a.target_kept),
        "--max-attempts",
        str(a.max_attempts),
        "--workers",
        str(a.workers),
        "--max-tokens",
        "64000",
        "--thinking-budget",
        "12000",
        "--api-timeout",
        "600",
    ]
    if a.resume_run:
        argv += ["--resume-run", str(a.resume_run.resolve())]
    args = engine.build_parser().parse_args(argv)
    if a.dry_run:
        print(vars(args))
        return
    engine.run_two_step_pipeline(args)


if __name__ == "__main__":
    main()
