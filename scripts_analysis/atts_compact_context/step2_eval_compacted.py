#!/usr/bin/env python3
"""
Step 2: Run single-turn LLM evaluation on compacted prompts and compute metrics.

This script reads compacted_inputs.jsonl from step1, skips early-stopped
questions by default, and calls a single LLM completion per question.
"""

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

"""
Step 2 intentionally does *no* context processing.

It expects step1 to have produced a fully-formed natural-language prompt per question
under the `final_prompt` field.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate compacted prompts with a single LLM call")
    parser.add_argument("--compacted-file", required=True, type=Path, help="Path to compacted_inputs.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--model", required=True, help="Litellm model name")
    parser.add_argument("--temperature", type=float, required=True, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, required=True, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, required=True, help="Max new tokens for completion")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent LLM calls (default: 1)",
    )
    parser.add_argument(
        "--litellm-verbose",
        action="store_true",
        help="Enable litellm verbose logging (may print requests/responses)",
    )
    parser.add_argument("--api-base", default=None, help="Override API base URL")
    parser.add_argument("--api-key", default=None, help="Override API key")
    parser.add_argument(
        "--extra-body",
        default=None,
        help="JSON string to pass as extra_body to litellm.completion",
    )
    parser.add_argument(
        "--include-early-stopped",
        action="store_true",
        help="Also run LLM on early-stopped questions (default: skip)",
    )
    parser.add_argument(
        "--question-ids",
        default="",
        help="Comma-separated list of question IDs to process (default: all)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (after filtering)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume and skip question_ids already in output file",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_existing_ids(path: Path) -> set[int]:
    """Load IDs of questions that are already processed (ok or early_stopped).
    
    On resume, error entries are removed from the file and their IDs are returned
    for reprocessing.
    """
    if not path.exists():
        return set()
    
    ok_rows = []
    error_ids = set()
    
    for row in _iter_jsonl(path):
        qid = row.get("question_id")
        if qid is None:
            continue
        status = row.get("status", "")
        if status == "error":
            error_ids.add(int(qid))
        else:
            ok_rows.append(row)
    
    # Rewrite file without error entries
    if error_ids:
        path.write_text("".join(json.dumps(r) + "\n" for r in ok_rows))
        logging.info(f"Removed {len(error_ids)} error entries for reprocessing")
    
    # Return IDs that are already done (ok + early_stopped)
    return set(int(r["question_id"]) for r in ok_rows if r.get("question_id") is not None)


def _extract_answer(text: str) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""

    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", raw)
    if boxed:
        return boxed[-1].strip()

    lowered = raw.lower()
    for marker in ["final answer:", "answer:"]:
        if marker in lowered:
            idx = lowered.rfind(marker)
            candidate = raw[idx + len(marker):].strip()
            if candidate:
                raw = candidate
                break

    # Take last non-empty line if multiline
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        raw = lines[-1]

    # Strip surrounding quotes/backticks
    raw = raw.strip().strip("`'\"")
    return raw


def _safe_math_equal(pred: str, gold: str) -> bool:
    try:
        from minisweagent.utils.grader import math_equal  # type: ignore

        return math_equal(pred, gold, timeout=True)
    except Exception:
        return str(pred).strip() == str(gold).strip()


def _call_llm(
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    litellm_verbose: bool,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    question_id: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        import litellm  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "litellm is required to run this script. Install project dependencies "
            "(see pyproject.toml) or run inside the project's configured environment."
        ) from e

    # litellm.set_verbose is deprecated; use env var per litellm warning.
    if litellm_verbose and os.environ.get("LITELLM_LOG", "").upper() != "DEBUG":
        os.environ["LITELLM_LOG"] = "DEBUG"

    messages = [
        {"role": "user", "content": prompt},
    ]

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if extra_body:
        kwargs["extra_body"] = extra_body

    # Retry logic: 3 attempts total with exponential backoff
    max_retries = 2
    backoff_delays = [1.0, 2.0]  # delays in seconds after 1st and 2nd failure
    
    for attempt in range(max_retries + 1):
        try:
            return litellm.completion(**kwargs)
        except Exception as e:
            qid_str = f"Q{question_id}: " if question_id is not None else ""
            if attempt < max_retries:
                logging.warning(f"{qid_str}API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                delay = backoff_delays[attempt]
                logging.warning(f"{qid_str}Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logging.error(f"{qid_str}API call failed after {max_retries + 1} attempts: {e}")
                raise

def _evaluate_one(row: Dict[str, Any], args: argparse.Namespace, extra_body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    qid = int(row["question_id"])
    prompt = row.get("final_prompt", "")
    if not prompt:
        # Backward compatibility if the input JSONL came from an older step1.
        prompt = row.get("user_task_full", "") or row.get("problem", "")

    start = time.time()
    try:
        response = _call_llm(
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            litellm_verbose=args.litellm_verbose,
            api_base=args.api_base,
            api_key=args.api_key,
            extra_body=extra_body,
            question_id=qid,
        )
        elapsed = time.time() - start
        text = response["choices"][0]["message"]["content"]
        answer = _extract_answer(text)

        gold = row.get("gold", "")
        correct = _safe_math_equal(answer, gold) if gold != "" else False
        original_answer = row.get("original_answer", "")
        agreement = None
        if original_answer != "":
            agreement = _safe_math_equal(answer, original_answer)

        usage = response.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        
        # Runtime logging of token stats
        logging.info(f"Q{qid}: prompt={prompt_tokens}, completion={completion_tokens}")

        # Extract reasoning_content from response (for models like GLM that provide it)
        reasoning_content = None
        try:
            reasoning_content = response["choices"][0]["message"].get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            pass

        return {
            "question_id": qid,
            "status": "ok",
            "response_text": text,
            "reasoning_content": reasoning_content,
            "answer": answer,
            "correct": correct,
            "agreement_with_original": agreement,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens"),
            "latency_sec": elapsed,
            "original_answer": original_answer,
            "gold": gold,
            "original_correct": row.get("original_correct", False),
            "original_exit_status": row.get("original_exit_status", ""),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "question_id": qid,
            "status": "error",
            "error": str(e),
            "latency_sec": elapsed,
            "original_answer": row.get("original_answer", ""),
            "gold": row.get("gold", ""),
            "original_correct": row.get("original_correct", False),
            "original_exit_status": row.get("original_exit_status", ""),
        }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be >= 1")

    logging.basicConfig(level=logging.INFO)
    if args.litellm_verbose and os.environ.get("LITELLM_LOG", "").upper() != "DEBUG":
        os.environ["LITELLM_LOG"] = "DEBUG"

    extra_body = None
    if args.extra_body:
        extra_body = json.loads(args.extra_body)

    selected_qids: Optional[set[int]] = None
    if args.question_ids:
        selected_qids = {int(x.strip()) for x in args.question_ids.split(",") if x.strip()}

    results_path = output_dir / "compact_eval_results.jsonl"
    done_ids = _load_existing_ids(results_path) if args.resume else set()

    processed = 0
    skipped_early = 0
    total = 0

    # Pre-load rows so tqdm can show a real progress bar with a total.
    rows: List[Dict[str, Any]] = []
    for row in _iter_jsonl(args.compacted_file):
        qid = int(row["question_id"])
        if selected_qids is not None and qid not in selected_qids:
            continue
        if qid in done_ids:
            continue
        rows.append(row)

    if args.max_questions is not None:
        rows = rows[: args.max_questions]

    try:
        from tqdm import tqdm  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("tqdm is required for progress bars. Install project dependencies.") from e

    # Separate early-stopped (skipped) from LLM-evaluated items
    skipped_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for row in rows:
        early_stopped = bool(row.get("early_stopped"))
        if early_stopped and not args.include_early_stopped:
            skipped_rows.append(row)
        else:
            eval_rows.append(row)

    total = len(rows)

    with results_path.open("a") as f_out, tqdm(total=total, desc="compact-eval") as pbar:
        for row in skipped_rows:
            qid = int(row["question_id"])
            skipped_early += 1
            record = {
                "question_id": qid,
                "status": "early_stopped",
                "original_answer": row.get("original_answer", ""),
                "gold": row.get("gold", ""),
                "original_correct": row.get("original_correct", False),
                "original_exit_status": row.get("original_exit_status", ""),
            }
            f_out.write(json.dumps(record) + "\n")
            processed += 1
            pbar.update(1)

        if args.concurrency == 1:
            for row in eval_rows:
                record = _evaluate_one(row, args, extra_body)
                f_out.write(json.dumps(record) + "\n")
                processed += 1
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [ex.submit(_evaluate_one, row, args, extra_body) for row in eval_rows]
                for fut in as_completed(futures):
                    record = fut.result()
                    f_out.write(json.dumps(record) + "\n")
                    processed += 1
                    pbar.update(1)

    # Aggregate metrics
    results = list(_iter_jsonl(results_path))
    evaluated = [r for r in results if r.get("status") == "ok"]
    early = [r for r in results if r.get("status") == "early_stopped"]
    errors = [r for r in results if r.get("status") == "error"]

    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    acc_evaluated = mean([1.0 if r.get("correct") else 0.0 for r in evaluated])
    # Overall accuracy treats early-stopped and errors as incorrect
    acc_overall = mean([1.0 if r.get("correct") else 0.0 for r in results]) if results else 0.0
    agree_rate = mean([
        1.0 if r.get("agreement_with_original") else 0.0
        for r in evaluated
        if r.get("agreement_with_original") is not None
    ])

    summary = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "litellm_verbose": bool(args.litellm_verbose),
        "include_early_stopped": bool(args.include_early_stopped),
        "num_total": len(results),
        "num_evaluated": len(evaluated),
        "num_early_stopped": len(early),
        "num_errors": len(errors),
        "accuracy_evaluated": acc_evaluated,
        "accuracy_overall": acc_overall,
        "agreement_rate": agree_rate,
        "avg_prompt_tokens": mean([r.get("prompt_tokens", 0) or 0 for r in evaluated]),
        "avg_completion_tokens": mean([r.get("completion_tokens", 0) or 0 for r in evaluated]),
        "avg_total_tokens": mean([r.get("total_tokens", 0) or 0 for r in evaluated]),
    }
    (output_dir / "compact_eval_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
