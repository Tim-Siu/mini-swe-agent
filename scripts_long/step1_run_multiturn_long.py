#!/usr/bin/env python3
"""Run multi-turn long-context degradation study on math datasets via OpenRouter.

Output format:
- One run directory (`--output-dir`)
- One JSONL per round: round_{k}.jsonl
- Run metadata/config: run_config.json
- Aggregated summary: summary.json

Resumable behavior:
- Existing round JSONL files are loaded to recover completed (session_id, round_idx).
- Re-running with larger --rounds appends missing rounds in-place.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from minisweagent.utils.grader import math_equal


DEFAULT_DATASET = "MathArena/aime_2025"
DEFAULT_SPLIT = "train"
USER_PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_answer(text: str) -> str:
    matches = BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def normalize_model_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def extract_assistant_text_from_api_response(api_response: dict[str, Any]) -> str:
    choices = api_response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return normalize_model_text(message.get("content"))


def get_record_assistant_text(rec: dict[str, Any]) -> str:
    api_response = rec.get("api_response")
    if isinstance(api_response, dict):
        return extract_assistant_text_from_api_response(api_response)
    if "response_text" in rec:  # backward compatibility
        return str(rec.get("response_text", ""))
    return ""


def provider_label(provider_cfg: dict[str, Any] | None) -> str:
    if provider_cfg is None:
        return "auto"
    order = provider_cfg.get("order")
    if isinstance(order, list) and order:
        label = str(order[0])
    else:
        label = "custom"
    q = provider_cfg.get("quantizations")
    if isinstance(q, list) and q:
        label = f"{label}/{q[0]}"
    return label


def parse_context_limit_error(error_text: str) -> dict[str, int] | None:
    limit_match = re.search(r"maximum context length is\s*([\d,]+)\s*tokens", error_text, re.IGNORECASE)
    input_match = re.search(r"([\d,]+)\s*of text input", error_text, re.IGNORECASE)
    output_match = re.search(r"([\d,]+)\s*in the output", error_text, re.IGNORECASE)
    if not limit_match or not input_match:
        return None
    context_limit = int(limit_match.group(1).replace(",", ""))
    input_tokens = int(input_match.group(1).replace(",", ""))
    requested_output_tokens = int(output_match.group(1).replace(",", "")) if output_match else -1
    return {
        "context_limit": context_limit,
        "input_tokens": input_tokens,
        "requested_output_tokens": requested_output_tokens,
    }


def format_user_prompt(problem: str) -> str:
    return f"{problem}{USER_PROMPT_SUFFIX}"


def build_problem_messages(history: list[dict[str, str]], problem: str) -> list[dict[str, str]]:
    return [*history, {"role": "user", "content": format_user_prompt(problem)}]


def build_round_robin_session_qids(qids: list[int], session_id: int, max_rounds: int, seed: int) -> list[int]:
    if max_rounds > len(qids):
        raise ValueError(f"rounds={max_rounds} exceeds dataset size={len(qids)}")
    base_qid = qids[session_id]
    others = [q for q in qids if q != base_qid]
    rng = random.Random(seed + session_id)
    rng.shuffle(others)
    return [base_qid, *others[: max_rounds - 1]]


def load_problems(dataset_name: str, split: str) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)
    required = {"problem_idx", "problem", "answer"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset {dataset_name} split {split} missing columns: {sorted(missing)}; got {ds.column_names}"
        )

    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in ds:
        qid = int(row["problem_idx"])
        if qid in seen:
            raise ValueError(f"Duplicate qid detected: {qid}")
        seen.add(qid)
        rows.append({"qid": qid, "problem": str(row["problem"]), "answer": str(row["answer"]).strip(), "raw": row})

    if not rows:
        raise ValueError(f"Dataset {dataset_name} split {split} is empty")
    rows.sort(key=lambda x: x["qid"])
    return rows


def summarize_records(records: list[dict[str, Any]], rounds: int) -> dict[str, Any]:
    per_round: dict[int, dict[str, float | int]] = {}
    for r in range(1, rounds + 1):
        rr = [x for x in records if int(x["round_idx"]) == r]
        if not rr:
            per_round[r] = {"count": 0, "accuracy": 0.0}
            continue
        correct = sum(1 for x in rr if bool(x["is_correct"]))
        per_round[r] = {"count": len(rr), "accuracy": correct / len(rr)}

    cumulative: dict[int, dict[str, float | int]] = {}
    for r in range(1, rounds + 1):
        rr = [x for x in records if int(x["round_idx"]) <= r]
        if not rr:
            cumulative[r] = {"count": 0, "avg_accuracy_first_k_rounds": 0.0}
            continue
        correct = sum(1 for x in rr if bool(x["is_correct"]))
        cumulative[r] = {"count": len(rr), "avg_accuracy_first_k_rounds": correct / len(rr)}

    return {
        "total_records": len(records),
        "round_1_accuracy": per_round.get(1, {}).get("accuracy", 0.0),
        "per_round": per_round,
        "cumulative": cumulative,
    }


def parse_reasoning_enabled(raw: str) -> bool | None:
    s = raw.strip().lower()
    if s in {"none", "null", ""}:
        return None
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid --reasoning-enabled value: {raw}")


def parse_provider_order_json(raw: str) -> list[dict[str, Any]]:
    if not raw.strip():
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("--provider-order-json must be a JSON list")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"provider entry at index {i} must be an object")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-turn long-context study runner (OpenRouter)")
    p.add_argument("--model", required=True)
    p.add_argument("--provider-order-json", default="[]")
    p.add_argument("--reasoning-enabled", default="none")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-workers", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-output-tokens", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--retry-backoff-sec", type=float, default=2.0)
    p.add_argument("--context-safety-margin", type=int, default=256)
    p.add_argument("--dry-run-plan", action="store_true")
    return p.parse_args()


def run_config_path(output_dir: Path) -> Path:
    return output_dir / "run_config.json"


def summary_path(output_dir: Path) -> Path:
    return output_dir / "summary.json"


def round_file(output_dir: Path, round_idx: int) -> Path:
    return output_dir / f"round_{round_idx}.jsonl"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def load_existing_records(output_dir: Path, rounds: int) -> dict[tuple[int, int], dict[str, Any]]:
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for r in range(1, rounds + 1):
        fp = round_file(output_dir, r)
        if not fp.exists():
            continue
        with fp.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at {fp}:{line_no}: {e}") from e
                sid = int(rec["session_id"])
                ridx = int(rec["round_idx"])
                if ridx != r:
                    raise ValueError(f"Round mismatch in {fp}:{line_no}; row round_idx={ridx}, file round={r}")
                by_pair[(sid, ridx)] = rec
    return by_pair


def validate_or_write_run_config(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        with path.open() as f:
            existing = json.load(f)
        keys = [
            "model",
            "dataset",
            "split",
            "seed",
            "temperature",
            "top_p",
            "max_output_tokens",
            "reasoning_enabled",
            "provider_order",
            "context_safety_margin",
        ]
        for k in keys:
            old_v = existing.get(k)
            new_v = config.get(k)
            if old_v != new_v:
                raise ValueError(f"Resume config mismatch for '{k}': existing={old_v} new={new_v}")
        existing["rounds_requested"] = max(int(existing.get("rounds_requested", 0)), int(config["rounds_requested"]))
        save_json(path, existing)
        return existing

    save_json(path, config)
    return config


def main() -> None:
    args = parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be > 0")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be > 0")

    provider_order = parse_provider_order_json(args.provider_order_json)
    reasoning_enabled = parse_reasoning_enabled(args.reasoning_enabled)

    problems = load_problems(args.dataset, args.split)
    qid_to_problem = {p["qid"]: p for p in problems}
    qids = [p["qid"] for p in problems]
    if args.rounds > len(qids):
        raise ValueError(f"Requested rounds={args.rounds}, but dataset has only {len(qids)} questions")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "rounds_requested": args.rounds,
        "max_workers": args.max_workers,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "provider_order": provider_order,
        "reasoning_enabled": reasoning_enabled,
        "context_safety_margin": args.context_safety_margin,
    }
    cfg = validate_or_write_run_config(run_config_path(args.output_dir), run_cfg)

    records_by_pair = load_existing_records(args.output_dir, args.rounds)
    done_pairs = set(records_by_pair.keys())

    jobs: list[tuple[int, int]] = []
    for session_id in range(len(qids)):
        for round_idx in range(1, args.rounds + 1):
            if (session_id, round_idx) not in done_pairs:
                jobs.append((session_id, round_idx))

    if args.dry_run_plan:
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "existing_records": len(records_by_pair),
                    "missing_jobs": len(jobs),
                    "rounds_requested": args.rounds,
                    "sessions": len(qids),
                },
                indent=2,
            )
        )
        return

    if not jobs:
        summary = summarize_records(list(records_by_pair.values()), args.rounds)
        save_json(summary_path(args.output_dir), summary)
        cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
        cfg["rounds_completed"] = args.rounds
        save_json(run_config_path(args.output_dir), cfg)
        print("No missing jobs; summary refreshed.")
        return

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    provider_try_order = [*provider_order, None]

    session_histories: dict[int, list[dict[str, str]]] = {}
    session_lock = threading.Lock()
    file_lock = threading.Lock()

    for session_id in range(len(qids)):
        hist: list[dict[str, str]] = []
        for r in range(1, args.rounds + 1):
            rec = records_by_pair.get((session_id, r))
            if rec is None:
                break
            hist.append({"role": "user", "content": format_user_prompt(rec["problem"])})
            hist.append({"role": "assistant", "content": get_record_assistant_text(rec)})
        session_histories[session_id] = hist

    def append_round_record(rec: dict[str, Any]) -> None:
        fp = round_file(args.output_dir, int(rec["round_idx"]))
        line = json.dumps(rec, ensure_ascii=True)
        with file_lock:
            with fp.open("a") as f:
                f.write(line + "\n")

    def run_one(session_id: int, round_idx: int) -> dict[str, Any]:
        session_qids = build_round_robin_session_qids(qids=qids, session_id=session_id, max_rounds=args.rounds, seed=args.seed)
        qid = session_qids[round_idx - 1]
        q = qid_to_problem[qid]

        with session_lock:
            local_history = list(session_histories[session_id])
            expected_items = (round_idx - 1) * 2
            if len(local_history) != expected_items:
                local_history = []
                for prev in range(1, round_idx):
                    prev_rec = records_by_pair.get((session_id, prev))
                    if prev_rec is None:
                        raise RuntimeError(
                            f"Missing dependency for session={session_id} round={round_idx}: prev round {prev} not found"
                        )
                    local_history.append({"role": "user", "content": format_user_prompt(prev_rec["problem"])})
                    local_history.append({"role": "assistant", "content": get_record_assistant_text(prev_rec)})

        messages = build_problem_messages(local_history, q["problem"])

        final_error: str | None = None
        for provider_idx, provider_cfg in enumerate(provider_try_order):
            if provider_idx > 0:
                tqdm.write(
                    f"[fallback] session={session_id} round={round_idx} switching provider={provider_label(provider_cfg)}"
                )
            provider_max_tokens = args.max_output_tokens
            for attempt in range(1, args.max_retries + 1):
                request_body: dict[str, Any] = {
                    "model": args.model,
                    "messages": messages,
                    "max_tokens": provider_max_tokens,
                }
                if args.temperature is not None:
                    request_body["temperature"] = args.temperature
                if args.top_p is not None:
                    request_body["top_p"] = args.top_p

                extra_body: dict[str, Any] = {"usage": {"include": True}}
                if provider_cfg is not None:
                    extra_body["provider"] = provider_cfg
                if reasoning_enabled is not None:
                    extra_body["reasoning"] = {"enabled": bool(reasoning_enabled)}
                request_body["extra_body"] = extra_body

                t0 = time.time()
                try:
                    response = client.chat.completions.create(**request_body)
                    latency = time.time() - t0
                    api_response = response.model_dump()
                    response_text = extract_assistant_text_from_api_response(api_response)
                    pred_answer = extract_answer(response_text)
                    is_correct = math_equal(pred_answer, q["answer"], timeout=False)

                    rec = {
                        "session_id": session_id,
                        "round_idx": round_idx,
                        "qid": qid,
                        "problem": q["problem"],
                        "gold_answer": q["answer"],
                        "pred_answer": pred_answer,
                        "is_correct": bool(is_correct),
                        "provider_requested": provider_cfg,
                        "request_body": request_body,
                        "api_response": api_response,
                        "latency_seconds": latency,
                        "attempt_index": attempt,
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                    }

                    with session_lock:
                        records_by_pair[(session_id, round_idx)] = rec
                        expected = round_idx * 2 - 2
                        if len(session_histories[session_id]) == expected:
                            session_histories[session_id].append({"role": "user", "content": format_user_prompt(q["problem"])})
                            session_histories[session_id].append({"role": "assistant", "content": response_text})

                    append_round_record(rec)
                    return rec
                except Exception as e:  # noqa: BLE001
                    err_text = str(e)
                    final_error = f"provider={provider_cfg or 'auto'} attempt={attempt}: {err_text}"
                    parsed_ctx = parse_context_limit_error(err_text)
                    if parsed_ctx is not None:
                        suggested = parsed_ctx["context_limit"] - parsed_ctx["input_tokens"] - args.context_safety_margin
                        suggested = max(1, suggested)
                        if suggested < provider_max_tokens:
                            tqdm.write(
                                f"[budget] session={session_id} round={round_idx} provider={provider_label(provider_cfg)} "
                                f"context_limit={parsed_ctx['context_limit']} input_tokens={parsed_ctx['input_tokens']} "
                                f"requested_output={parsed_ctx['requested_output_tokens']} "
                                f"max_tokens: {provider_max_tokens}->{suggested} (margin={args.context_safety_margin})"
                            )
                            provider_max_tokens = suggested
                    elif "maximum context length" in err_text.lower():
                        tqdm.write(
                            f"[budget] session={session_id} round={round_idx} provider={provider_label(provider_cfg)} "
                            f"parse-miss for context error; keeping max_tokens={provider_max_tokens}"
                        )

                    if attempt < args.max_retries:
                        wait_s = args.retry_backoff_sec * attempt
                        tqdm.write(
                            f"[retry] session={session_id} round={round_idx} provider={provider_label(provider_cfg)} "
                            f"attempt={attempt}/{args.max_retries} wait={wait_s:.1f}s error={e}"
                        )
                        time.sleep(wait_s)
                    else:
                        tqdm.write(
                            f"[retry] session={session_id} round={round_idx} provider={provider_label(provider_cfg)} "
                            f"attempt={attempt}/{args.max_retries} exhausted"
                        )

        raise RuntimeError(f"API call failed for session={session_id} round={round_idx}; last_error={final_error}")

    newly_done = 0
    for round_idx in range(1, args.rounds + 1):
        round_jobs = [(sid, ridx) for sid, ridx in jobs if ridx == round_idx]
        if not round_jobs:
            continue

        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(run_one, sid, ridx): (sid, ridx) for sid, ridx in round_jobs}
            for fut in tqdm(as_completed(futures), total=len(round_jobs), desc=f"round {round_idx}"):
                _ = fut.result()
                newly_done += 1
                if newly_done % 20 == 0:
                    summary = summarize_records(list(records_by_pair.values()), args.rounds)
                    save_json(summary_path(args.output_dir), summary)
                    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
                    cfg["rounds_completed"] = max(int(cfg.get("rounds_completed", 0)), round_idx)
                    save_json(run_config_path(args.output_dir), cfg)

    summary = summarize_records(list(records_by_pair.values()), args.rounds)
    save_json(summary_path(args.output_dir), summary)
    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    cfg["rounds_completed"] = args.rounds
    save_json(run_config_path(args.output_dir), cfg)

    print(f"Saved records in {args.output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
