"""
Classify NuminaMath-LEAN problems into difficulty buckets using Gemini Flash.

Buckets:
  1 = easy      (routine textbook, single-step or standard technique)
  2 = medium    (multi-step, requires combining ideas)
  3 = hard      (competition-level, research-adjacent, non-trivial insight)

Usage:
  uv run python classify_difficulty.py --sample 20000 --workers 32
  uv run python classify_difficulty.py --sample 5000 --workers 16  # smaller test

Resumes automatically: skips any uuid already in the output JSONL.
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MODEL = "gemini-3.1-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

RUN_DIR = Path(__file__).parent / "classifier_run"
RUN_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = RUN_DIR / "difficulty_labels.jsonl"
LOG_PATH = RUN_DIR / "classify.log"


def log(msg: str):
    """Write to both stdout and log file."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

PROMPT_TEMPLATE = """Classify this math problem's difficulty into exactly one bucket:
1 = easy (routine textbook, single-step, standard technique)
2 = medium (multi-step, requires combining ideas, undergraduate competition)
3 = hard (olympiad-level, research-adjacent, requires non-trivial insight)

Respond with ONLY a JSON object: {{"difficulty": 1|2|3}}

Problem:
{problem}

Formal statement:
{formal_statement}"""


def load_done_ids(path: Path) -> set:
    """Load already-labeled uuids from output JSONL."""
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["uuid"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


async def classify_one(
    client: httpx.AsyncClient,
    item: dict,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> dict | None:
    """Call Gemini Flash to classify a single problem."""
    prompt = PROMPT_TEMPLATE.format(
        problem=item["problem"][:2000],
        formal_statement=item["formal_statement"][:1500],
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 32,
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(retries):
        async with semaphore:
            try:
                resp = await client.post(API_URL, headers={"x-goog-api-key": GOOGLE_API_KEY}, json=payload, timeout=30)
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt + random.random())
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text)
                difficulty = int(parsed["difficulty"])
                if difficulty not in (1, 2, 3):
                    difficulty = 2  # default to medium if weird
                return {
                    "uuid": item["uuid"],
                    "difficulty": difficulty,
                    "problem_type": item.get("problem_type", ""),
                }
            except (httpx.HTTPStatusError, httpx.ReadTimeout) as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt + random.random())
                else:
                    return {
                        "uuid": item["uuid"],
                        "difficulty": -1,
                        "problem_type": item.get("problem_type", ""),
                        "error": str(e),
                    }
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                return {
                    "uuid": item["uuid"],
                    "difficulty": -1,
                    "problem_type": item.get("problem_type", ""),
                    "error": f"parse: {e}",
                }
    return None


async def main(sample_size: int, workers: int):
    from datasets import load_dataset

    log(f"Loading NuminaMath-LEAN from HuggingFace...")
    ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train")
    log(f"Total problems: {len(ds)}")

    # Load already-done
    done_ids = load_done_ids(OUTPUT_PATH)
    log(f"Already labeled: {len(done_ids)}")

    # Filter out done, then sample
    candidates = [row for row in ds if row["uuid"] not in done_ids]
    log(f"Unlabeled candidates: {len(candidates)}")

    if len(candidates) > sample_size:
        random.seed(42)
        batch = random.sample(candidates, sample_size)
    else:
        batch = candidates
    log(f"Processing batch: {len(batch)} problems with {workers} workers")

    semaphore = asyncio.Semaphore(workers)
    completed = 0
    errors = 0
    t0 = time.time()

    async with httpx.AsyncClient() as client:
        # Process in chunks to write incrementally
        chunk_size = 200
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i : i + chunk_size]
            tasks = [classify_one(client, item, semaphore) for item in chunk]
            results = await asyncio.gather(*tasks)

            with open(OUTPUT_PATH, "a") as f:
                for r in results:
                    if r is not None:
                        f.write(json.dumps(r) + "\n")
                        if r["difficulty"] == -1:
                            errors += 1
                        completed += 1

            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            log(
                f"[{completed}/{len(batch)}] "
                f"{rate:.1f} items/sec | "
                f"errors: {errors} | "
                f"elapsed: {elapsed:.0f}s"
            )

    elapsed = time.time() - t0
    log(f"Done. {completed} labeled in {elapsed:.0f}s ({errors} errors)")
    log(f"Output: {OUTPUT_PATH}")

    # Print distribution
    from collections import Counter
    counts = Counter()
    with open(OUTPUT_PATH) as f:
        for line in f:
            d = json.loads(line)
            counts[d["difficulty"]] += 1
    log(f"Distribution: easy={counts[1]}, medium={counts[2]}, hard={counts[3]}, errors={counts[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify NuminaMath problems by difficulty")
    parser.add_argument("--sample", type=int, default=20000, help="Number of problems to label")
    parser.add_argument("--workers", type=int, default=32, help="Concurrent API requests")
    args = parser.parse_args()
    if not GOOGLE_API_KEY:
        parser.error("Set GOOGLE_API_KEY before running the classifier")
    asyncio.run(main(args.sample, args.workers))
