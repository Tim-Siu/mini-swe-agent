#!/usr/bin/env python3
"""Step 2 analysis for multi-turn long-context runs (round-based JSONL format)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze a run directory with round_*.jsonl files")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory created by step1")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory (default: <run-dir>/analysis_step2_<timestamp>)",
    )
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def collect_round_files(run_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for fp in sorted(run_dir.glob("round_*.jsonl")):
        stem = fp.stem
        try:
            ridx = int(stem.split("_")[1])
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Unexpected round file name: {fp.name}") from e
        files.append((ridx, fp))
    return files


def safe_div(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def get_usage(rec: dict[str, Any]) -> tuple[int, int, int]:
    api_response = rec.get("api_response", {})
    usage = api_response.get("usage", {}) if isinstance(api_response, dict) else {}
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)
    tt = int(usage.get("total_tokens", 0) or 0)
    return pt, ct, tt


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing run config: {cfg_path}")
    run_cfg = load_json(cfg_path)

    round_files = collect_round_files(run_dir)
    if not round_files:
        raise FileNotFoundError(f"no round_*.jsonl files in {run_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (run_dir / f"analysis_step2_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_session_round: dict[tuple[int, int], dict[str, Any]] = {}

    for ridx, fp in round_files:
        with fp.open() as f:
            for line_no, line in tqdm(enumerate(f, start=1), desc=f"load round_{ridx}"):
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {fp}:{line_no}: {e}") from e

                if int(rec.get("round_idx", -1)) != ridx:
                    raise ValueError(
                        f"Round index mismatch at {fp}:{line_no}; row has {rec.get('round_idx')} expected {ridx}"
                    )
                if "session_id" not in rec or "is_correct" not in rec:
                    raise ValueError(f"Missing required keys at {fp}:{line_no}")

                sid = int(rec["session_id"])
                key = (sid, ridx)
                if key in by_session_round:
                    raise ValueError(f"Duplicate record for session={sid}, round={ridx} (line {line_no} in {fp})")
                by_session_round[key] = rec
                by_round[ridx].append(rec)

    rounds_present = sorted(by_round.keys())
    max_round = max(rounds_present)

    # Per-round metrics.
    per_round_rows: list[dict[str, Any]] = []
    for ridx in range(1, max_round + 1):
        rows = by_round.get(ridx, [])
        n = len(rows)
        correct = sum(1 for r in rows if bool(r["is_correct"]))

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        provider_counts: dict[str, int] = defaultdict(int)
        for r in rows:
            pt, ct, tt = get_usage(r)
            prompt_tokens += pt
            completion_tokens += ct
            total_tokens += tt
            p = r.get("provider_requested")
            provider_key = json.dumps(p, sort_keys=True) if p is not None else "auto"
            provider_counts[provider_key] += 1

        per_round_rows.append(
            {
                "round_idx": ridx,
                "num_records": n,
                "num_correct": correct,
                "accuracy": safe_div(correct, n),
                "sum_prompt_tokens": prompt_tokens,
                "sum_completion_tokens": completion_tokens,
                "sum_total_tokens": total_tokens,
                "avg_total_tokens": safe_div(total_tokens, n),
                "provider_counts": dict(sorted(provider_counts.items())),
            }
        )

    # Cumulative metrics through each round.
    cumulative_rows: list[dict[str, Any]] = []
    cum_n = 0
    cum_correct = 0
    cum_tokens = 0
    for row in per_round_rows:
        cum_n += int(row["num_records"])
        cum_correct += int(row["num_correct"])
        cum_tokens += int(row["sum_total_tokens"])
        cumulative_rows.append(
            {
                "through_round": int(row["round_idx"]),
                "num_records": cum_n,
                "num_correct": cum_correct,
                "avg_accuracy_first_k_rounds": safe_div(cum_correct, cum_n),
                "sum_total_tokens": cum_tokens,
                "avg_total_tokens": safe_div(cum_tokens, cum_n),
            }
        )

    # Per-session trajectories.
    all_sessions = sorted({sid for sid, _ in by_session_round})
    per_session_rows: list[dict[str, Any]] = []
    for sid in all_sessions:
        correctness: list[int | None] = []
        pred_answers: list[str | None] = []
        for ridx in range(1, max_round + 1):
            rec = by_session_round.get((sid, ridx))
            if rec is None:
                correctness.append(None)
                pred_answers.append(None)
            else:
                correctness.append(1 if bool(rec["is_correct"]) else 0)
                pred_answers.append(str(rec.get("pred_answer", "")))

        valid = [x for x in correctness if x is not None]
        per_session_rows.append(
            {
                "session_id": sid,
                "round_correctness": correctness,
                "round_pred_answers": pred_answers,
                "num_answered_rounds": len(valid),
                "avg_accuracy_answered_rounds": safe_div(sum(valid), len(valid)),
            }
        )

    # Persist outputs.
    summary = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(),
        "model": run_cfg.get("model"),
        "dataset": run_cfg.get("dataset"),
        "split": run_cfg.get("split"),
        "seed": run_cfg.get("seed"),
        "rounds_requested": run_cfg.get("rounds_requested"),
        "rounds_present": rounds_present,
        "total_records": sum(len(v) for v in by_round.values()),
        "num_sessions": len(all_sessions),
        "round_1_accuracy": next((r["accuracy"] for r in per_round_rows if r["round_idx"] == 1), 0.0),
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "per_round_metrics.json", {"rows": per_round_rows})
    write_json(output_dir / "cumulative_metrics.json", {"rows": cumulative_rows})
    write_jsonl(output_dir / "per_session_trajectory.jsonl", per_session_rows)

    md_lines = [
        f"# Step2 Analysis",
        "",
        f"- run_dir: `{run_dir}`",
        f"- model: `{summary['model']}`",
        f"- dataset: `{summary['dataset']}`",
        f"- seed: `{summary['seed']}`",
        f"- rounds_present: `{summary['rounds_present']}`",
        f"- total_records: `{summary['total_records']}`",
        f"- num_sessions: `{summary['num_sessions']}`",
        "",
        "## Per-Round",
        "",
        "| round | n | correct | acc | total_tokens | avg_total_tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in per_round_rows:
        md_lines.append(
            f"| {r['round_idx']} | {r['num_records']} | {r['num_correct']} | {r['accuracy']:.4f} | {r['sum_total_tokens']} | {r['avg_total_tokens']:.1f} |"
        )

    md_lines.extend(
        [
            "",
            "## Cumulative",
            "",
            "| through_round | n | correct | avg_accuracy_first_k_rounds |",
            "|---|---:|---:|---:|",
        ]
    )
    for r in cumulative_rows:
        md_lines.append(
            f"| {r['through_round']} | {r['num_records']} | {r['num_correct']} | {r['avg_accuracy_first_k_rounds']:.4f} |"
        )

    write_markdown(output_dir / "report.md", "\n".join(md_lines) + "\n")
    print(f"Analysis written to: {output_dir}")


if __name__ == "__main__":
    main()
