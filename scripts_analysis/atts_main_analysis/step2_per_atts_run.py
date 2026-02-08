#!/usr/bin/env python3
"""
Step 2: Per ATTS Run Analysis

Generates per-question table with:
- Gold answer
- Majority answer (from full pool)
- ATTS chosen answer
- Whether ATTS chose majority
- Whether ATTS was correct
- Whether majority was correct

Plus summary stats on majority agreement rates.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Per ATTS Run Analysis")
    p.add_argument("--atts-dir", required=True, type=Path, help="Single ATTS run directory")
    p.add_argument("--rollout-files", required=True, action="append")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_rollout_data(rollout_files: List[str]) -> Dict[int, List[dict]]:
    """Load rollout pool data."""
    data = defaultdict(list)
    for fp in rollout_files:
        path = Path(fp)
        if not path.exists():
            raise FileNotFoundError(f"Rollout file not found: {fp}")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                data[r["question_id"]].append(r)
    return dict(data)


def compute_majority_answer(responses: List[dict], seed: int) -> tuple[str, int]:
    """
    Compute majority answer from full pool.
    Returns (majority_answer, majority_count).
    Random tie-breaking.
    """
    if not responses:
        return "", 0

    groups = defaultdict(list)
    for r in responses:
        groups[r["pred_answer"]].append(r)

    max_count = max(len(g) for g in groups.values())
    majority_groups = [ans for ans, g in groups.items() if len(g) == max_count]

    rng = random.Random(seed)
    majority_answer = rng.choice(majority_groups)
    return majority_answer, max_count


def parse_answer_dist_from_traj(traj: dict) -> dict | None:
    """Parse answer distribution from trajectory for maj@budget."""
    info = traj.get("info", {})
    srm = info.get("sampled_rollout_metadata")
    esm = info.get("early_stop_metadata")

    if srm:
        return srm.get("sampled_answers")
    elif esm:
        return esm.get("sampled_answers")

    # Parse from tool response messages
    answer_dist = {}
    for msg in traj.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if "Answer Statistics:" not in content or "<output>" not in content:
            continue
        try:
            start = content.index("<output>")
            end = content.index("</output>", start)
            output_text = content[start + 8 : end]
        except ValueError:
            continue
        if "Answer Statistics:" not in output_text:
            continue
        for line in output_text.split("\n"):
            line = line.strip()
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) != 2:
                continue
            idx_str, answer = parts
            if idx_str in ("Index", "-------", "---"):
                continue
            if "-" in idx_str:
                try:
                    lo, hi = idx_str.split("-")
                    count = int(hi) - int(lo) + 1
                except ValueError:
                    continue
            else:
                try:
                    int(idx_str)
                    count = 1
                except ValueError:
                    continue
            answer_dist[answer] = answer_dist.get(answer, 0) + count
    return answer_dist if answer_dist else None


def compute_maj_budget_answer(dist: dict, seed: int) -> str:
    """Compute majority answer from sampled distribution (maj@budget)."""
    if not dist:
        return ""
    max_count = max(dist.values())
    majority_answers = [a for a, c in dist.items() if c == max_count]
    rng = random.Random(seed)
    return rng.choice(majority_answers)


def build_answer_correctness_map(responses: List[dict]) -> Dict[str, bool]:
    """Build pred_answer->correctness map from rollout labels."""
    out: Dict[str, bool] = {}
    for r in responses:
        ans = str(r["pred_answer"])
        out[ans] = out.get(ans, False) or bool(r.get("label"))
    return out


def analyze_atts_run(atts_dir: Path, rollout_data: Dict[int, List[dict]], seed: int) -> dict:
    """Analyze a single ATTS run."""
    preds_file = atts_dir / "preds.json"
    if not preds_file.exists():
        raise FileNotFoundError(f"Not found: {preds_file}")

    with open(preds_file) as f:
        preds = json.load(f)

    per_question = []

    sorted_qids = sorted((int(qid_str), qid_str) for qid_str in preds.keys())
    for qid, qid_str in tqdm(sorted_qids, desc="Processing questions"):
        pred = preds[qid_str]
        gold = pred["gold"]
        atts_answer = pred["answer"]
        atts_correct = pred["correct"]

        # Get pool responses
        pool_responses = rollout_data.get(qid, [])

        # Compute majority from full pool
        majority_answer, majority_count = compute_majority_answer(pool_responses, seed)
        majority_correct = (majority_answer == gold)
        answer_correct_map = build_answer_correctness_map(pool_responses)

        # Did ATTS choose majority?
        atts_chose_majority = (atts_answer == majority_answer)

        # Load trajectory for maj@budget
        traj_file = atts_dir / f"question_{qid}" / f"{qid}.traj.json"
        if not traj_file.exists():
            traj_file = atts_dir / f"question_{qid}" / "0.traj.json"

        maj_budget_answer = ""
        if traj_file.exists():
            with open(traj_file) as f:
                traj = json.load(f)
            dist = parse_answer_dist_from_traj(traj)
            if dist:
                maj_budget_answer = compute_maj_budget_answer(dist, seed)

        maj_budget_correct = answer_correct_map.get(str(maj_budget_answer), False) if maj_budget_answer else False

        per_question.append({
            "qid": qid,
            "gold": gold,
            "majority_answer": majority_answer,
            "majority_count": majority_count,
            "majority_correct": majority_correct,
            "atts_answer": atts_answer,
            "atts_correct": atts_correct,
            "atts_chose_majority": atts_chose_majority,
            "maj_budget_answer": maj_budget_answer,
            "maj_budget_correct": maj_budget_correct,
            "exit_status": pred.get("exit_status", "Submitted"),
        })

    # Ensure stable numeric ordering by qid in outputs.
    per_question.sort(key=lambda d: d["qid"])

    # Summary stats
    n_total = len(per_question)
    n_atts_correct = sum(1 for d in per_question if d["atts_correct"])
    n_majority_correct = sum(1 for d in per_question if d["majority_correct"])
    n_maj_budget_correct = sum(1 for d in per_question if d["maj_budget_correct"])
    n_maj_budget_valid = sum(1 for d in per_question if d["maj_budget_answer"])

    # Agreement analysis
    n_atts_chose_majority = sum(1 for d in per_question if d["atts_chose_majority"])

    # When ATTS agrees with majority
    agreed_correct = sum(1 for d in per_question if d["atts_chose_majority"] and d["atts_correct"])
    agreed_total = n_atts_chose_majority

    # When ATTS disagrees with majority
    disagreed_correct = sum(1 for d in per_question if not d["atts_chose_majority"] and d["atts_correct"])
    disagreed_total = n_total - n_atts_chose_majority

    return {
        "name": atts_dir.name,
        "per_question": per_question,
        "summary": {
            "n_total": n_total,
            "atts_accuracy": n_atts_correct / n_total if n_total else 0.0,
            "majority_accuracy": n_majority_correct / n_total if n_total else 0.0,
            "maj_budget_accuracy": n_maj_budget_correct / n_total if n_total else 0.0,
            "n_maj_budget_valid": n_maj_budget_valid,
            "atts_chose_majority_count": n_atts_chose_majority,
            "atts_chose_majority_pct": n_atts_chose_majority / n_total * 100 if n_total else 0.0,
            "when_agreed_accuracy": agreed_correct / agreed_total if agreed_total else 0.0,
            "when_agreed_total": agreed_total,
            "when_disagreed_accuracy": disagreed_correct / disagreed_total if disagreed_total else 0.0,
            "when_disagreed_total": disagreed_total,
        }
    }


def write_per_question_table(data: dict, output_dir: Path):
    """Write per-question markdown table."""
    lines = [
        f"# Per-Question Analysis: {data['name']}",
        "",
        "| QID | Gold | Majority | ATTS Answer | ATTS Chose Maj | ATTS Correct | Maj Correct | Maj@budget | Maj@budget Correct |",
        "|-----|------|----------|-------------|----------------|--------------|-------------|------------|-------------------|",
    ]

    for d in sorted(data["per_question"], key=lambda x: x["qid"]):
        chose_majority = "Yes" if d["atts_chose_majority"] else "No"
        atts_correct = "Yes" if d["atts_correct"] else "No"
        maj_correct = "Yes" if d["majority_correct"] else "No"
        mb = d["maj_budget_answer"] if d["maj_budget_answer"] else "-"
        mb_correct = "Yes" if d["maj_budget_correct"] else "No" if d["maj_budget_answer"] else "-"

        lines.append(
            f"| {d['qid']} | {d['gold']} | {d['majority_answer']} | {d['atts_answer']} | "
            f"{chose_majority} | {atts_correct} | {maj_correct} | {mb} | {mb_correct} |"
        )

    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")

    s = data["summary"]
    lines.append(f"- **Total Questions**: {s['n_total']}")
    lines.append(f"- **ATTS Accuracy**: {s['atts_accuracy']*100:.2f}%")
    lines.append(f"- **Majority Vote Accuracy**: {s['majority_accuracy']*100:.2f}%")
    lines.append(f"- **Maj@budget Accuracy**: {s['maj_budget_accuracy']*100:.2f}% ({s['n_maj_budget_valid']}/{s['n_total']} valid)")
    lines.append("")
    lines.append(f"- **ATTS Chose Majority**: {s['atts_chose_majority_count']}/{s['n_total']} ({s['atts_chose_majority_pct']:.1f}%)")
    lines.append(f"- **When Agreed**: {s['when_agreed_accuracy']*100:.2f}% correct ({s['when_agreed_total']} questions)")
    lines.append(f"- **When Disagreed**: {s['when_disagreed_accuracy']*100:.2f}% correct ({s['when_disagreed_total']} questions)")
    lines.append("")

    md_path = output_dir / f"{data['name']}_per_question.md"
    md_path.write_text("\n".join(lines))
    print(f"  Saved per-question table to {md_path}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Step 2: Per ATTS Run Analysis - {args.atts_dir.name}")
    print("=" * 70)

    # Load rollout data
    print("\n1. Loading rollout data...")
    rollout_data = load_rollout_data(args.rollout_files)
    print(f"   {len(rollout_data)} questions")

    # Analyze ATTS run
    print(f"\n2. Analyzing ATTS run: {args.atts_dir}")
    if not args.atts_dir.exists():
        print(f"ERROR: Directory not found: {args.atts_dir}")
        sys.exit(1)

    result = analyze_atts_run(args.atts_dir, rollout_data, args.seed)

    # Print summary
    s = result["summary"]
    print(f"\n3. Results for {result['name']}:")
    print(f"   ATTS Accuracy: {s['atts_accuracy']*100:.2f}%")
    print(f"   Majority Accuracy: {s['majority_accuracy']*100:.2f}%")
    print(f"   Maj@budget Accuracy: {s['maj_budget_accuracy']*100:.2f}%")
    print(f"   ATTS Chose Majority: {s['atts_chose_majority_pct']:.1f}%")
    print(f"   When Agreed: {s['when_agreed_accuracy']*100:.2f}% correct")
    print(f"   When Disagreed: {s['when_disagreed_accuracy']*100:.2f}% correct")

    # Write outputs
    print("\n4. Writing outputs...")
    write_per_question_table(result, args.output_dir)

    # Save JSON
    json_path = args.output_dir / f"{result['name']}_results.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved JSON to {json_path}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
