#!/usr/bin/env python3
"""Prompted frontier row through the Codex CLI (ChatGPT subscription).

Same prompts, parsers and metrics as frontier_api_eval.py; each row is one non-interactive
`codex exec` call (ephemeral session, read-only sandbox, web search off) whose final agent
message is scored. Codex prepends its own agent instructions and does not return reasoning,
so generations are wrapped as <think>[not returned]</think> for the shared parsers.
Raw outputs append to <out>/<suite>/raw.jsonl keyed by custom_id, so runs resume.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontier_api_eval as F  # noqa: E402

CODEX_FLAGS = [
    "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--ephemeral",
    "-s", "read-only", "--json", "-c", 'web_search="disabled"',
    "-c", "features.shell_tool=false", "-c", "tools.view_image=false", "-c", "mcp_servers={}",
    "--disable", "apps", "--disable", "browser_use", "--disable", "browser_use_external",
    "--disable", "computer_use", "--disable", "image_generation",
]
TOOL_ITEM_TYPES = ("mcp_tool_call", "command_execution", "web_search", "file_change", "computer_use", "browser_use")
LIMIT_MARKERS = ("usage limit", "rate limit", "rate_limit", "too many requests", "429", "quota", "usage_limit")


def prompt_text(item: dict) -> str:
    system = [m["content"] for m in item["messages"] if m["role"] == "system"]
    user = [m["content"] for m in item["messages"] if m["role"] != "system"]
    return "\n\n".join(system + user)


class RateLimited(Exception):
    pass


def run_codex(text: str, args: argparse.Namespace) -> tuple[str, dict]:
    cmd = ["codex", "exec", *CODEX_FLAGS, *args.extra_flag, "-m", args.model,
           "-c", f'model_reasoning_effort="{args.effort}"', "-C", str(args.workdir), "-"]
    t0 = time.time()
    proc = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=args.timeout)
    events = []
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    messages = [e["item"].get("text", "") for e in events
                if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message"]
    other_items = [{k: (str(v)[:120] if not isinstance(v, (int, float, bool)) else v) for k, v in e["item"].items() if k != "text"}
                   for e in events if e.get("type") == "item.completed" and e.get("item", {}).get("type") != "agent_message"]
    usage = next((e.get("usage") or {} for e in events if e.get("type") == "turn.completed"), {})
    errors = [e for e in events if e.get("type") in ("error", "turn.failed")]
    blob = (proc.stderr + " " + json.dumps(errors)).lower()
    if proc.returncode != 0 or errors or not messages:
        if any(m in blob for m in LIMIT_MARKERS):
            raise RateLimited(blob[-400:])
        raise RuntimeError(f"rc={proc.returncode} errors={json.dumps(errors)[:300]} stderr={proc.stderr[-300:]}")
    text_out = messages[-1].strip()
    generation = f"<think>\n[reasoning not returned by Codex]\n</think>\n{text_out}\n"
    info = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0) + usage.get("reasoning_output_tokens", 0),
        "reasoning_output_tokens": usage.get("reasoning_output_tokens", 0),
        "stop_reason": "end_turn",
        "tool_items": other_items,
        "agent_messages": len(messages),
        "elapsed_s": round(time.time() - t0, 1),
    }
    return generation, info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-5.6-sol")
    p.add_argument("--effort", default="high")
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--suites", nargs="+", default=["indomain", "hard2verify", "imo", "processbench", "prmbench"], choices=F.SUITES)
    p.add_argument("--hard2verify-repo", type=Path, required=True)
    p.add_argument("--processbench-repo", type=Path, required=True)
    p.add_argument("--indomain-parquet", type=Path, required=True)
    p.add_argument("--v4test-parquet", type=Path)
    p.add_argument("--workdir", type=Path, required=True, help="empty directory Codex uses as its working root")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="first N rows of each suite (pilot only)")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--backoff", type=int, default=900, help="seconds to wait when Codex reports a usage limit")
    p.add_argument("--extra-flag", action="append", default=[], help="additional codex exec flag (repeatable)")
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()
    args.max_tokens = 0
    args.system_variant = "original"
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(["codex", "--version"], capture_output=True, text=True).stdout.strip()
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "model": args.model, "effort": args.effort, "codex": version, "flags": CODEX_FLAGS + args.extra_flag,
        "note": "prompted through the Codex CLI on a ChatGPT subscription; Codex agent instructions precede the paper's prompt",
    }, indent=2) + "\n")
    log = open(args.output_dir / "progress.log", "a")

    def say(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    for suite in args.suites:
        items = F.load_suite(suite, args)
        if args.limit:
            items = items[: args.limit]
        out = args.output_dir / suite
        out.mkdir(parents=True, exist_ok=True)
        raw_path = out / "raw.jsonl"
        done: dict[str, tuple[str, dict]] = {}
        if raw_path.exists():
            for line in raw_path.read_text().splitlines():
                rec = json.loads(line)
                if "error" not in rec["usage"]:
                    done[rec["custom_id"]] = (rec["generation"], rec["usage"])
        todo = [it for it in items if it["custom_id"] not in done]
        say(f"{suite}: {len(items)} rows, {len(done)} done, {len(todo)} to run")
        lock = threading.Lock()
        pause_until = [0.0]
        counter = [0]

        def work(item: dict) -> None:
            for attempt in range(6):
                wait = pause_until[0] - time.time()
                if wait > 0:
                    time.sleep(wait)
                try:
                    generation, usage = run_codex(prompt_text(item), args)
                except RateLimited as exc:
                    with lock:
                        pause_until[0] = max(pause_until[0], time.time() + args.backoff)
                        say(f"usage limit reported, pausing {args.backoff}s: {str(exc)[:160]}")
                    continue
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    with lock:
                        say(f"{item['custom_id']} attempt {attempt + 1} failed: {str(exc)[:200]}")
                    time.sleep(30)
                    continue
                with lock:
                    done[item["custom_id"]] = (generation, usage)
                    with raw_path.open("a") as handle:
                        handle.write(json.dumps({"custom_id": item["custom_id"], "generation": generation, "usage": usage}, ensure_ascii=False) + "\n")
                    counter[0] += 1
                    if counter[0] % 10 == 0 or counter[0] == len(todo):
                        say(f"{suite}: {len(done)}/{len(items)} done")
                return
            with lock:
                say(f"{item['custom_id']} gave up")

        if todo and not args.score_only:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for future in as_completed([pool.submit(work, it) for it in todo]):
                    future.result()
        outputs = {cid: val for cid, val in done.items()}
        result = F.score_suite(suite, items, outputs, args)
        result["summary"]["decoding"] = {"temperature": "Codex default", "effort": args.effort, "api": "codex exec (ChatGPT subscription)", "codex": version}
        result["summary"]["rows_with_tool_use"] = sum(
            1 for _, u in outputs.values()
            if any((it.get("type") if isinstance(it, dict) else it) in TOOL_ITEM_TYPES for it in u.get("tool_items") or []))
        F.write_suite(args.output_dir, suite, result)
        headline = {k: v for k, v in result["summary"].items() if isinstance(v, (int, float)) and k not in ("examples",)}
        say(f"{suite}: scored {len(outputs)}/{len(items)}; {json.dumps(headline)[:400]}")


if __name__ == "__main__":
    main()
