#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Dict, List, Tuple


DEFAULT_PRICING = {
    "total_context": 200_000,
    "max_output": 131_100,
    "input_price_per_million": 0.60,
    "output_price_per_million": 2.20,
    "cache_read_price_per_million": 0.11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token usage for ATTS rollouts and write a report.")
    parser.add_argument(
        "--atts-run-dir",
        required=True,
        type=Path,
        help="Path to an ATTS rollout directory (e.g., atts_rollout/<run_name>)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Base output directory (a subdir with run name will be created)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override run name (default: directory name of atts-run-dir)",
    )
    parser.add_argument(
        "--input-price",
        type=float,
        default=DEFAULT_PRICING["input_price_per_million"],
        help="Input price per 1M tokens",
    )
    parser.add_argument(
        "--output-price",
        type=float,
        default=DEFAULT_PRICING["output_price_per_million"],
        help="Output price per 1M tokens",
    )
    parser.add_argument(
        "--cache-read-price",
        type=float,
        default=DEFAULT_PRICING["cache_read_price_per_million"],
        help="Cache read price per 1M tokens",
    )
    parser.add_argument(
        "--total-context",
        type=int,
        default=DEFAULT_PRICING["total_context"],
        help="Total context size (tokens)",
    )
    parser.add_argument(
        "--max-output",
        type=int,
        default=DEFAULT_PRICING["max_output"],
        help="Max output size (tokens)",
    )
    return parser.parse_args()


def find_traj_files(atts_run_dir: Path) -> List[Path]:
    traj_files = sorted(atts_run_dir.rglob("*.traj.json"))
    return traj_files


def extract_usage_from_messages(messages: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """Return (total_prompt, max_prompt, total_completion, call_count)."""
    prompt_tokens_list: List[int] = []
    completion_tokens_list: List[int] = []

    for msg in messages:
        usage = (
            msg.get("extra", {})
            .get("response", {})
            .get("usage", {})
        )
        if not usage:
            continue
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None:
            prompt_tokens_list.append(int(prompt_tokens))
        if completion_tokens is not None:
            completion_tokens_list.append(int(completion_tokens))

    total_prompt = sum(prompt_tokens_list)
    max_prompt = max(prompt_tokens_list) if prompt_tokens_list else 0
    total_completion = sum(completion_tokens_list)
    call_count = max(len(prompt_tokens_list), len(completion_tokens_list))
    return total_prompt, max_prompt, total_completion, call_count


def compute_problem_stats(traj_path: Path) -> Dict[str, Any]:
    data = json.loads(traj_path.read_text())
    messages = data.get("messages", [])
    total_prompt, max_prompt, total_completion, call_count = extract_usage_from_messages(messages)

    input_cache = total_prompt - max_prompt

    return {
        "question_id": data.get("question_id"),
        "traj_path": str(traj_path),
        "total_prompt_tokens": total_prompt,
        "max_prompt_tokens": max_prompt,
        "input_cache_tokens": input_cache,
        "total_completion_tokens": total_completion,
        "call_count": call_count,
    }


def summarize_metric(values: List[int]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "variance": 0.0,
            "stddev": 0.0,
        }
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p50 = sorted_vals[int(0.5 * (n - 1))]
    p90 = sorted_vals[int(0.9 * (n - 1))]
    variance = statistics.pvariance(sorted_vals)
    stddev = statistics.pstdev(sorted_vals)
    return {
        "mean": statistics.mean(sorted_vals),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "p50": float(p50),
        "p90": float(p90),
        "variance": variance,
        "stddev": stddev,
    }


def format_money(value: float) -> str:
    return f"${value:,.4f}"


def compute_prices(total_input: int, total_cache: int, total_output: int, pricing: Dict[str, float]) -> Dict[str, float]:
    input_price = pricing["input_price_per_million"]
    output_price = pricing["output_price_per_million"]
    cache_price = pricing["cache_read_price_per_million"]

    price_with_cache = (
        total_input * input_price + total_cache * cache_price + total_output * output_price
    ) / 1_000_000

    price_without_cache = (
        (total_input + total_cache) * input_price + total_output * output_price
    ) / 1_000_000

    return {
        "with_cache": price_with_cache,
        "without_cache": price_without_cache,
    }


def write_report(
    output_dir: Path,
    run_name: str,
    problem_stats: List[Dict[str, Any]],
    pricing: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_input = sum(p["max_prompt_tokens"] for p in problem_stats)
    total_cache = sum(p["input_cache_tokens"] for p in problem_stats)
    total_output = sum(p["total_completion_tokens"] for p in problem_stats)
    total_prompt = sum(p["total_prompt_tokens"] for p in problem_stats)
    total_calls = sum(p["call_count"] for p in problem_stats)

    prices = compute_prices(total_input, total_cache, total_output, pricing)

    metrics = {
        "max_prompt_tokens": [p["max_prompt_tokens"] for p in problem_stats],
        "input_cache_tokens": [p["input_cache_tokens"] for p in problem_stats],
        "total_completion_tokens": [p["total_completion_tokens"] for p in problem_stats],
        "total_prompt_tokens": [p["total_prompt_tokens"] for p in problem_stats],
        "call_count": [p["call_count"] for p in problem_stats],
    }

    metric_summaries = {name: summarize_metric(vals) for name, vals in metrics.items()}

    nonempty_stats = [p for p in problem_stats if p["call_count"] > 0]
    nonempty_metrics = {
        "max_prompt_tokens": [p["max_prompt_tokens"] for p in nonempty_stats],
        "input_cache_tokens": [p["input_cache_tokens"] for p in nonempty_stats],
        "total_completion_tokens": [p["total_completion_tokens"] for p in nonempty_stats],
        "total_prompt_tokens": [p["total_prompt_tokens"] for p in nonempty_stats],
        "call_count": [p["call_count"] for p in nonempty_stats],
    }
    nonempty_metric_summaries = {
        name: summarize_metric(vals) for name, vals in nonempty_metrics.items()
    }

    report = {
        "run_name": run_name,
        "problem_count": len(problem_stats),
        "totals": {
            "input_tokens": total_input,
            "input_cache_tokens": total_cache,
            "output_tokens": total_output,
            "total_prompt_tokens": total_prompt,
            "call_count": total_calls,
        },
        "pricing": {
            **pricing,
            "price_with_cache": prices["with_cache"],
            "price_without_cache": prices["without_cache"],
        },
        "metric_summaries": metric_summaries,
        "metric_summaries_nonempty": nonempty_metric_summaries,
        "nonempty_problem_count": len(nonempty_stats),
        "problems": problem_stats,
    }

    (output_dir / "report.json").write_text(json.dumps(report, indent=2))

    # CSV per-problem
    csv_path = output_dir / "problem_stats.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id",
            "total_prompt_tokens",
            "max_prompt_tokens",
            "input_cache_tokens",
            "total_completion_tokens",
            "call_count",
        ])
        for p in problem_stats:
            writer.writerow([
                p["question_id"],
                p["total_prompt_tokens"],
                p["max_prompt_tokens"],
                p["input_cache_tokens"],
                p["total_completion_tokens"],
                p["call_count"],
            ])

    # Markdown report
    md_lines = []
    md_lines.append(f"# ATTS Token Usage Report: {run_name}")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append(f"- Problems: {len(problem_stats)}")
    md_lines.append(f"- Total input tokens (max per problem): {total_input}")
    md_lines.append(f"- Total cache input tokens: {total_cache}")
    md_lines.append(f"- Total output tokens: {total_output}")
    md_lines.append(f"- Total prompt tokens (all calls): {total_prompt}")
    md_lines.append(f"- Total model calls: {total_calls}")
    md_lines.append("")

    md_lines.append("## Pricing Assumptions")
    md_lines.append("")
    md_lines.append("| Total Context | Max Output | Input Price | Output Price | Cache Read |")
    md_lines.append("| --- | --- | --- | --- | --- |")
    md_lines.append(
        f"| {pricing['total_context']:,} | {pricing['max_output']:,} | ${pricing['input_price_per_million']:.2f} | ${pricing['output_price_per_million']:.2f} | ${pricing['cache_read_price_per_million']:.2f} |"
    )
    md_lines.append("")
    md_lines.append("## Price Estimates")
    md_lines.append("")
    md_lines.append(f"- With cache: {format_money(prices['with_cache'])}")
    md_lines.append(f"- Without cache: {format_money(prices['without_cache'])}")
    md_lines.append("")

    md_lines.append("## Problem-Level Statistics")
    md_lines.append("")
    md_lines.append("| Metric | Mean | P50 | P90 | Min | Max | Variance | Std Dev |")
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, summary in metric_summaries.items():
        md_lines.append(
            "| {name} | {mean:.2f} | {p50:.2f} | {p90:.2f} | {min:.2f} | {max:.2f} | {variance:.2f} | {stddev:.2f} |".format(
                name=name,
                **summary,
            )
        )
    md_lines.append("")

    md_lines.append("## Problem-Level Statistics (Non-Empty Only)")
    md_lines.append("")
    md_lines.append(
        f"- Non-empty problems (call_count > 0): {len(nonempty_stats)} / {len(problem_stats)}"
    )
    md_lines.append("")
    md_lines.append("| Metric | Mean | P50 | P90 | Min | Max | Variance | Std Dev |")
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, summary in nonempty_metric_summaries.items():
        md_lines.append(
            "| {name} | {mean:.2f} | {p50:.2f} | {p90:.2f} | {min:.2f} | {max:.2f} | {variance:.2f} | {stddev:.2f} |".format(
                name=name,
                **summary,
            )
        )
    md_lines.append("")

    md_lines.append("## Per-Problem Details")
    md_lines.append("")
    md_lines.append("| Question | Total Prompt | Max Prompt | Cache Input | Output | Calls |")
    md_lines.append("| --- | --- | --- | --- | --- | --- |")
    for p in problem_stats:
        md_lines.append(
            f"| {p['question_id']} | {p['total_prompt_tokens']} | {p['max_prompt_tokens']} | {p['input_cache_tokens']} | {p['total_completion_tokens']} | {p['call_count']} |"
        )
    md_lines.append("")

    (output_dir / "report.md").write_text("\n".join(md_lines))


def main() -> None:
    args = parse_args()

    run_name = args.run_name or args.atts_run_dir.name
    traj_files = find_traj_files(args.atts_run_dir)
    if not traj_files:
        raise SystemExit(f"No trajectory files found under {args.atts_run_dir}")

    problem_stats = [compute_problem_stats(p) for p in traj_files]
    problem_stats.sort(key=lambda x: (x["question_id"] is None, x["question_id"]))

    pricing = {
        "total_context": args.total_context,
        "max_output": args.max_output,
        "input_price_per_million": args.input_price,
        "output_price_per_million": args.output_price,
        "cache_read_price_per_million": args.cache_read_price,
    }

    output_dir = args.output_dir / run_name
    write_report(output_dir, run_name, problem_stats, pricing)


if __name__ == "__main__":
    main()
