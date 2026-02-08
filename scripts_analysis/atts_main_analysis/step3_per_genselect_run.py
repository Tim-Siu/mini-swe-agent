#!/usr/bin/env python3
"""
Step 3: Per GenSelect Run Analysis

Generates per-question table with:
- Gold answer
- Majority answer (from full pool)
- GenSelect chosen answer
- Whether GenSelect chose majority
- Whether GenSelect was correct

Plus summary stats on majority agreement rates.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Per GenSelect Run Analysis")
    p.add_argument("--genselect-dir", required=True, type=Path, help="Single GenSelect run directory")
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


def analyze_genselect_run(genselect_dir: Path, rollout_data: Dict[int, List[dict]], seed: int) -> dict:
    """Analyze a single GenSelect run."""
    results_file = genselect_dir / "results_per_question.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Not found: {results_file}")

    with open(results_file) as f:
        per_q_results = json.load(f)

    per_question = []

    for r in tqdm(sorted(per_q_results, key=lambda x: x["question_id"]), desc="Processing questions"):
        qid = r["question_id"]
        gold = r["gold_answer"]
        genselect_answer = r.get("selected_answer", "")
        genselect_correct = r["is_correct"]

        # Get pool responses
        pool_responses = rollout_data.get(qid, [])

        # Compute majority from full pool
        majority_answer, majority_count = compute_majority_answer(pool_responses, seed)
        majority_correct = (majority_answer == gold)

        # Did GenSelect choose majority?
        genselect_chose_majority = (genselect_answer == majority_answer)

        per_question.append({
            "qid": qid,
            "gold": gold,
            "majority_answer": majority_answer,
            "majority_count": majority_count,
            "majority_correct": majority_correct,
            "genselect_answer": genselect_answer,
            "genselect_correct": genselect_correct,
            "genselect_chose_majority": genselect_chose_majority,
            "selector_prompt": r.get("selector_prompt", ""),
            "selector_response": r.get("selector_response", ""),
        })

    # Ensure stable numeric ordering by qid in outputs.
    per_question.sort(key=lambda d: d["qid"])

    # Summary stats
    n_total = len(per_question)
    n_genselect_correct = sum(1 for d in per_question if d["genselect_correct"])
    n_majority_correct = sum(1 for d in per_question if d["majority_correct"])

    # Agreement analysis
    n_genselect_chose_majority = sum(1 for d in per_question if d["genselect_chose_majority"])

    # When GenSelect agrees with majority
    agreed_correct = sum(1 for d in per_question if d["genselect_chose_majority"] and d["genselect_correct"])
    agreed_total = n_genselect_chose_majority

    # When GenSelect disagrees with majority
    disagreed_correct = sum(1 for d in per_question if not d["genselect_chose_majority"] and d["genselect_correct"])
    disagreed_total = n_total - n_genselect_chose_majority

    return {
        "name": genselect_dir.name,
        "per_question": per_question,
        "summary": {
            "n_total": n_total,
            "genselect_accuracy": n_genselect_correct / n_total if n_total else 0.0,
            "majority_accuracy": n_majority_correct / n_total if n_total else 0.0,
            "genselect_chose_majority_count": n_genselect_chose_majority,
            "genselect_chose_majority_pct": n_genselect_chose_majority / n_total * 100 if n_total else 0.0,
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
        "| QID | Gold | Majority | GenSelect Answer | Chose Maj | GenSelect Correct | Maj Correct |",
        "|-----|------|----------|------------------|-----------|-------------------|-------------|",
    ]

    for d in sorted(data["per_question"], key=lambda x: x["qid"]):
        chose_majority = "Yes" if d["genselect_chose_majority"] else "No"
        genselect_correct = "Yes" if d["genselect_correct"] else "No"
        maj_correct = "Yes" if d["majority_correct"] else "No"

        lines.append(
            f"| {d['qid']} | {d['gold']} | {d['majority_answer']} | {d['genselect_answer']} | "
            f"{chose_majority} | {genselect_correct} | {maj_correct} |"
        )

    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")

    s = data["summary"]
    lines.append(f"- **Total Questions**: {s['n_total']}")
    lines.append(f"- **GenSelect Accuracy**: {s['genselect_accuracy']*100:.2f}%")
    lines.append(f"- **Majority Vote Accuracy**: {s['majority_accuracy']*100:.2f}%")
    lines.append("")
    lines.append(f"- **GenSelect Chose Majority**: {s['genselect_chose_majority_count']}/{s['n_total']} ({s['genselect_chose_majority_pct']:.1f}%)")
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
    print(f"Step 3: Per GenSelect Run Analysis - {args.genselect_dir.name}")
    print("=" * 70)

    # Load rollout data
    print("\n1. Loading rollout data...")
    rollout_data = load_rollout_data(args.rollout_files)
    print(f"   {len(rollout_data)} questions")

    # Analyze GenSelect run
    print(f"\n2. Analyzing GenSelect run: {args.genselect_dir}")
    if not args.genselect_dir.exists():
        print(f"ERROR: Directory not found: {args.genselect_dir}")
        sys.exit(1)

    result = analyze_genselect_run(args.genselect_dir, rollout_data, args.seed)

    # Print summary
    s = result["summary"]
    print(f"\n3. Results for {result['name']}:")
    print(f"   GenSelect Accuracy: {s['genselect_accuracy']*100:.2f}%")
    print(f"   Majority Accuracy: {s['majority_accuracy']*100:.2f}%")
    print(f"   GenSelect Chose Majority: {s['genselect_chose_majority_pct']:.1f}%")
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
