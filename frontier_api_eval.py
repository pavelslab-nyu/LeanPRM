#!/usr/bin/env python3
"""Prompted frontier-model rows for the verifier evaluation suite (Anthropic API).

Uses the exact prompts of the trained-model evaluations (training layout for
Hard2Verify / ProcessBench / PRMBench / IMO-GradingBench, and the stored
in-domain test prompts) and the exact parsers and metric functions. The model's
extended thinking is wrapped as <think>...</think> so the identical scorers run
unchanged. A markdown-stripping normalization is tried only when the strict
parse fails and every such row is counted in the summary.

Subcommands: pilot (synchronous sample, usage and cost projection), submit
(one Message Batch per suite), fetch (poll, score, write artifacts).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from external_eval import hard2verify_rows, processbench_rows  # noqa: E402
from format_aligned_external_eval import (  # noqa: E402
    format_aligned_messages,
    parse_training_labels,
    prmbench_rows,
    score_generated,
)
from imo_gradingbench_eval import load_rows as imo_rows  # noqa: E402
from imo_gradingbench_eval import metrics as imo_metrics  # noqa: E402
from imo_gradingbench_eval import ours_messages, parse_ours  # noqa: E402
from verifier_task import SYSTEM_PROMPT, _parse_truth, _score_one  # noqa: E402

# Fable 5.1's safeguards refuse prompts that mention </think>; this rewording keeps the task identical.
SYSTEM_VARIANTS = {
    "original": SYSTEM_PROMPT,
    "no_think_tag": SYSTEM_PROMPT.replace(
        "Use your thinking section for the analysis. After </think>, return only the requested labels with no explanation.",
        "Think it through carefully, then return only the requested labels with no explanation.",
    ),
}

SUITES = ("hard2verify", "processbench", "prmbench", "imo", "indomain", "v4test")
# USD per million tokens, Message Batches API (50% of standard), platform.claude.com pricing 2026-09-14
BATCH_PRICES = {
    "claude-opus-5": (2.5, 12.5),
    "claude-sonnet-5": (1.0, 5.0),
    "claude-fable-5-1": (5.0, 25.0),
}


def load_suite(name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    if name == "hard2verify":
        rows = hard2verify_rows(args.hard2verify_repo)
    elif name == "processbench":
        rows = processbench_rows(args.processbench_repo)
    elif name == "prmbench":
        rows = prmbench_rows()
    elif name == "imo":
        rows = imo_rows(None)
    elif name in ("indomain", "v4test"):
        import pyarrow.parquet as pq

        rows = pq.read_table(args.indomain_parquet if name == "indomain" else args.v4test_parquet).to_pylist()
    else:
        raise ValueError(name)
    out = []
    for index, row in enumerate(rows):
        if name in ("hard2verify", "processbench", "prmbench"):
            messages = format_aligned_messages(row["problem"], row["proof_steps"])
            row = {k: v for k, v in row.items() if k != "messages"}
        elif name == "imo":
            messages = ours_messages(row["problem"], row["steps"])
        else:
            messages = row["prompt"]
        out.append({"suite": name, "index": index, "custom_id": f"{name}__{index}", "messages": messages, "row": row})
    return out


def request_params(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    system = [m["content"] for m in item["messages"] if m["role"] == "system"]
    user = [m for m in item["messages"] if m["role"] != "system"]
    params: dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": args.effort},
        "messages": [{"role": m["role"], "content": m["content"]} for m in user],
    }
    if system:
        joined = "\n\n".join(system)
        if args.system_variant != "original" and joined == SYSTEM_PROMPT:
            joined = SYSTEM_VARIANTS[args.system_variant]
        params["system"] = joined
    return params


def wrap_generation(message: Any) -> tuple[str, dict[str, Any]]:
    thinking = []
    text = []
    for block in message.content:
        kind = getattr(block, "type", None)
        if kind == "thinking":
            thinking.append(block.thinking)
        elif kind == "redacted_thinking":
            thinking.append("[redacted thinking]")
        elif kind == "text":
            text.append(block.text)
    reasoning = "\n".join(thinking).strip() or "[thinking not returned by API]"
    generation = f"<think>\n{reasoning}\n</think>\n{''.join(text).strip()}\n"
    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "stop_reason": message.stop_reason,
    }
    return generation, usage


_STEP_LOOSE = re.compile(r"^step\s*(\d+)\s*[:.\-–]\s*(correct|incorrect)\b.*$", re.IGNORECASE)
_OVERALL_LOOSE = re.compile(r"^overall(?:\s+(?:proof\s+)?verdict)?\s*[:.\-–]\s*(correct|incorrect)\b.*$", re.IGNORECASE)


def normalize(generation: str) -> str:
    end = generation.rfind("</think>")
    head, answer = generation[: end + len("</think>")], generation[end + len("</think>") :]
    lines = []
    for raw in answer.splitlines():
        line = raw.strip()
        line = re.sub(r"^[-*•]\s+", "", line)
        line = line.replace("**", "").replace("`", "").replace("*", "").replace("_", " ")
        if match := _STEP_LOOSE.match(line):
            line = f"Step {match.group(1)}: {match.group(2).lower()}"
        elif match := _OVERALL_LOOSE.match(line):
            line = f"Overall: {match.group(1).lower()}"
        else:
            continue
        lines.append(line)
    return head + "\n" + "\n".join(lines) + "\n"


def strict_ok(suite: str, generation: str, item: dict[str, Any]) -> bool:
    row = item["row"]
    if suite in ("hard2verify", "processbench", "prmbench"):
        return parse_training_labels(generation, row["step_count"])[0] is not None
    if suite == "imo":
        return parse_ours(generation, len(row["steps"]))[0] is not None
    return _score_one(generation, _parse_truth(row["reward_model"]["ground_truth"]))["format_valid"] == 1.0  # indomain / v4test


def choose_generation(suite: str, generation: str, item: dict[str, Any]) -> tuple[str, bool]:
    if strict_ok(suite, generation, item):
        return generation, False
    fixed = normalize(generation)
    return (fixed, True) if strict_ok(suite, fixed, item) else (generation, False)


def score_suite(suite: str, items: list[dict[str, Any]], outputs: dict[str, tuple[str, dict[str, Any]]], args: argparse.Namespace) -> dict[str, Any]:
    generations, normalized, usages, errors = [], 0, [], 0
    for item in items:
        generation, usage = outputs.get(item["custom_id"], ("", {"error": "missing"}))
        if "error" in usage:
            errors += 1
        chosen, was_normalized = choose_generation(suite, generation, item)
        normalized += was_normalized
        item["_generation"] = chosen
        item["_raw_generation"] = generation
        item["_usage"] = usage
        item["_normalized"] = was_normalized
        generations.append(chosen)
        usages.append(usage)
    common = {
        "name": args.name,
        "model": args.model,
        "protocol": "format_aligned" if suite not in ("indomain", "v4test") else "training_prompt",
        "decoding": {"temperature": "api default (1.0 with extended thinking)", "max_tokens": args.max_tokens, "thinking": "adaptive", "effort": args.effort, "system_variant": args.system_variant, "api": "anthropic_message_batches"},
        "examples": len(items),
        "api_errors": errors,
        "rows_parsed_only_after_markdown_normalization": normalized,
        "stop_reasons": dict(Counter(u.get("stop_reason", "error") for u in usages)),
        "usage": {
            "input_tokens": sum(u.get("input_tokens", 0) for u in usages),
            "output_tokens": sum(u.get("output_tokens", 0) for u in usages),
        },
    }
    if suite in ("hard2verify", "processbench", "prmbench"):
        scored, summary = score_generated([it["row"] for it in items], generations)
        for s, it in zip(scored, items, strict=True):
            s["normalized"] = it["_normalized"]
            s["raw_generation"] = it["_raw_generation"]
            s["usage"] = it["_usage"]
        summary.update(common)
        return {"summary": summary, "rows": scored}
    if suite == "imo":
        rows = []
        for it in items:
            row = dict(it["row"])
            prediction, status = parse_ours(it["_generation"], len(row["steps"]))
            row.update({"generation": it["_generation"], "raw_generation": it["_raw_generation"], "prediction": prediction, "parser": status, "normalized": it["_normalized"], "usage": it["_usage"]})
            rows.append(row)
        summary = {**common, "step_cap": 30, "imo_gradingbench": imo_metrics(rows)}
        return {"summary": summary, "rows": rows}
    totals: dict[str, list[float]] = {}
    rows = []
    for it in items:
        score = _score_one(it["_generation"], _parse_truth(it["row"]["reward_model"]["ground_truth"]))
        for key, value in score.items():
            if isinstance(value, (int, float)):
                totals.setdefault(key, []).append(float(value))
        rows.append({"extra_info": it["row"]["extra_info"], "output": it["_generation"], "raw_output": it["_raw_generation"], "metrics": score, "normalized": it["_normalized"], "usage": it["_usage"]})
    summary = {**common, **{k: sum(v) / len(v) for k, v in totals.items()}}
    return {"summary": summary, "rows": rows}


def write_suite(out_dir: Path, suite: str, result: dict[str, Any]) -> None:
    out = out_dir / suite
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n")
    with (out / "results.jsonl").open("w") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def spaced(items: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def cmd_pilot(args: argparse.Namespace) -> None:
    import anthropic

    client = anthropic.Anthropic(max_retries=5, timeout=600)
    prices = BATCH_PRICES[args.model]
    projection = 0.0
    for suite in args.suites:
        items = load_suite(suite, args)
        sample = spaced(items, args.pilot_rows)
        outputs = {}
        for item in sample:
            message = client.messages.create(**request_params(item, args))
            outputs[item["custom_id"]] = wrap_generation(message)
        result = score_suite(suite, sample, outputs, args)
        s = result["summary"]
        mean_in = s["usage"]["input_tokens"] / len(sample)
        mean_out = s["usage"]["output_tokens"] / len(sample)
        cost = len(items) * (mean_in * prices[0] + mean_out * prices[1]) / 1e6
        projection += cost
        parse = s.get("failure_categories") or s.get("imo_gradingbench", {}).get("failure_categories") or {"format_valid_mean": s.get("format_valid")}
        print(f"{suite}: n={len(sample)}/{len(items)} mean_in={mean_in:.0f} mean_out={mean_out:.0f} stop={s['stop_reasons']} normalized={s['rows_parsed_only_after_markdown_normalization']} parse={parse} projected_batch_cost=${cost:.0f}")
        for row in result["rows"][:2]:
            gen = row.get("generation") or row.get("output")
            print("   sample answer:", gen[gen.rfind("</think>") + 8 :].strip().replace("\n", " | ")[:200])
        if args.pilot_out:
            write_suite(args.pilot_out, suite, result)
    print(f"PROJECTED_TOTAL_BATCH_COST_USD={projection:.0f}")


def cmd_submit(args: argparse.Namespace) -> None:
    import anthropic

    client = anthropic.Anthropic(max_retries=5, timeout=600)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for suite in args.suites:
        if suite in state:
            print(f"{suite}: already submitted as {state[suite]['id']}")
            continue
        items = load_suite(suite, args)
        requests = [{"custom_id": it["custom_id"], "params": request_params(it, args)} for it in items]
        batch = client.messages.batches.create(requests=requests)
        state[suite] = {"id": batch.id, "rows": len(items), "model": args.model, "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(f"{suite}: submitted {batch.id} rows={len(items)}")


def cmd_fetch(args: argparse.Namespace) -> None:
    import anthropic

    client = anthropic.Anthropic(max_retries=5, timeout=600)
    state = json.loads((args.output_dir / "batches.json").read_text())
    done = True
    for suite, info in state.items():
        if suite not in SUITES:
            continue
        summary_path = args.output_dir / suite / "summary.json"
        if summary_path.exists():
            print(f"{suite}: scored")
            continue
        batch = client.messages.batches.retrieve(info["id"])
        counts = batch.request_counts
        print(f"{suite}: {batch.processing_status} succeeded={counts.succeeded} errored={counts.errored} processing={counts.processing} expired={counts.expired}")
        if batch.processing_status != "ended":
            done = False
            continue
        outputs: dict[str, tuple[str, dict[str, Any]]] = {}
        raw_path = args.output_dir / suite
        raw_path.mkdir(parents=True, exist_ok=True)
        with (raw_path / "raw_results.jsonl").open("w") as handle:
            for entry in client.messages.batches.results(info["id"]):
                handle.write(entry.model_dump_json() + "\n")
                if entry.result.type == "succeeded":
                    outputs[entry.custom_id] = wrap_generation(entry.result.message)
                else:
                    outputs[entry.custom_id] = ("", {"error": entry.result.type})
        items = load_suite(suite, args)
        result = score_suite(suite, items, outputs, args)
        write_suite(args.output_dir, suite, result)
        s = result["summary"]
        headline = {k: v for k, v in s.items() if k in ("hard2verify", "processbench", "prmbench", "imo_gradingbench", "exact_proof_accuracy", "verdict_accuracy", "incorrect_step_recall", "format_valid", "rows_parsed_only_after_markdown_normalization", "api_errors", "usage", "stop_reasons") and v is not None}
        print(json.dumps({suite: headline}, indent=2))
    print("ALL_SCORED" if done else "PENDING")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pilot", "submit", "fetch", "retry-submit", "retry-fetch"])
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", default=[s for s in SUITES if s != "v4test"], choices=SUITES)
    parser.add_argument("--hard2verify-repo", type=Path, required=True)
    parser.add_argument("--processbench-repo", type=Path, required=True)
    parser.add_argument("--indomain-parquet", type=Path, required=True)
    parser.add_argument("--v4test-parquet", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--system-variant", default="original", choices=list(SYSTEM_VARIANTS))
    parser.add_argument("--pilot-rows", type=int, default=6)
    parser.add_argument("--pilot-out", type=Path)
    args = parser.parse_args()
    {"pilot": cmd_pilot, "submit": cmd_submit, "fetch": cmd_fetch, "retry-submit": cmd_retry_submit, "retry-fetch": cmd_retry_fetch}[args.command](args)



# --- one retry pass for truncated or refused rows at a larger output cap ---------------------

RETRY_STOPS = ("max_tokens", "refusal")


def _read_raw(path_base: Path) -> list[Any]:
    import gzip

    from anthropic.types.messages import MessageBatchIndividualResponse

    plain, gz = path_base / "raw_results.jsonl", path_base / "raw_results.jsonl.gz"
    opener = (lambda: gz.open(gz, "rt")) if gz.exists() else (lambda: plain.open())
    with opener() as handle:
        return [MessageBatchIndividualResponse.model_validate_json(line) for line in handle if line.strip()]


def _outputs_from_raw(entries: list[Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    outputs: dict[str, tuple[str, dict[str, Any]]] = {}
    for entry in entries:
        if entry.result.type == "succeeded":
            outputs[entry.custom_id] = wrap_generation(entry.result.message)
        else:
            outputs[entry.custom_id] = ("", {"error": entry.result.type})
    return outputs


def cmd_retry_submit(args: argparse.Namespace) -> None:
    import anthropic

    client = anthropic.Anthropic(max_retries=5, timeout=600)
    state_path = args.output_dir / "batches.json"
    state = json.loads(state_path.read_text())
    if "retry" in state:
        print(f"retry already submitted as {state['retry']['id']}")
        return
    requests = []
    per_suite: dict[str, int] = {}
    for suite in [s for s in state if s in SUITES]:
        outputs = _outputs_from_raw(_read_raw(args.output_dir / suite))
        items = load_suite(suite, args)
        for item in items:
            generation, usage = outputs.get(item["custom_id"], ("", {"error": "missing"}))
            if usage.get("stop_reason") in RETRY_STOPS or "error" in usage:
                requests.append({"custom_id": item["custom_id"], "params": request_params(item, args)})
                per_suite[suite] = per_suite.get(suite, 0) + 1
    if not requests:
        print("nothing to retry")
        return
    batch = client.messages.batches.create(requests=requests)
    state["retry"] = {"id": batch.id, "rows": len(requests), "per_suite": per_suite, "max_tokens": args.max_tokens, "reason": list(RETRY_STOPS), "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(f"retry: submitted {batch.id} rows={len(requests)} {per_suite}")


def cmd_retry_fetch(args: argparse.Namespace) -> None:
    import anthropic

    client = anthropic.Anthropic(max_retries=5, timeout=600)
    state = json.loads((args.output_dir / "batches.json").read_text())
    info = state["retry"]
    batch = client.messages.batches.retrieve(info["id"])
    counts = batch.request_counts
    print(f"retry: {batch.processing_status} succeeded={counts.succeeded} errored={counts.errored} processing={counts.processing}")
    if batch.processing_status != "ended":
        print("PENDING")
        return
    retry_dir = args.output_dir / "retry"
    retry_dir.mkdir(exist_ok=True)
    with (retry_dir / "raw_results.jsonl").open("w") as handle:
        for entry in client.messages.batches.results(info["id"]):
            handle.write(entry.model_dump_json() + "\n")
    retry_outputs = _outputs_from_raw(_read_raw(retry_dir))
    for suite in [s for s in state if s in SUITES]:
        suite_dir = args.output_dir / suite
        outputs = _outputs_from_raw(_read_raw(suite_dir))
        replaced = still_bad = 0
        for custom_id, (generation, usage) in retry_outputs.items():
            if not custom_id.startswith(f"{suite}__"):
                continue
            if usage.get("stop_reason") in RETRY_STOPS or "error" in usage:
                still_bad += 1
                continue
            outputs[custom_id] = (generation, {**usage, "retried_at_max_tokens": info["max_tokens"]})
            replaced += 1
        first_pass = suite_dir / "summary_first_pass.json"
        if not first_pass.exists():
            (suite_dir / "summary.json").replace(first_pass)
        items = load_suite(suite, args)
        result = score_suite(suite, items, outputs, args)
        result["summary"]["retry"] = {"policy": "rows that stopped on max_tokens or refusal were rerun once", "max_tokens": info["max_tokens"], "rows_replaced": replaced, "rows_still_truncated_or_refused": still_bad}
        write_suite(args.output_dir, suite, result)
        print(f"{suite}: replaced={replaced} still_bad={still_bad}")
    print("RETRY_SCORED")


if __name__ == "__main__":
    main()
