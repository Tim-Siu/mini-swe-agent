#!/usr/bin/env python3
"""
Step 3 Analysis: Generate Tables

Generates tables comparing:
- Majority voting
- Pass@k
- ATTS Agent

Columns: Different k values + Agent
Rows: Methods
Values: Accuracy (x.x format) and Budget ($)
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Step 3: Generate Tables")
    parser.add_argument(
        "--step1-results",
        required=True,
        type=Path,
        help="Path to step1_results.json",
    )
    parser.add_argument(
        "--step2-results",
        required=True,
        type=Path,
        help="Path to step2_results.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for tables",
    )
    return parser.parse_args()


def load_results(step1_path: Path, step2_path: Path) -> dict:
    """Load results from step1 and step2."""
    with open(step1_path) as f:
        step1 = json.load(f)
    
    with open(step2_path) as f:
        step2 = json.load(f)
    
    return {
        "step1": step1,
        "step2": step2,
    }


def generate_accuracy_table(results: dict) -> str:
    """Generate markdown table for accuracy."""
    step1 = results["step1"]
    k_values = step1["k_values"]
    
    # Build header
    header = "| Method | " + " | ".join([f"k={k}" for k in k_values]) + " | Agent |"
    separator = "|" + "|".join([" --- " for _ in range(len(k_values) + 2)]) + "|"
    
    # Build rows (multiply by 100 for percentage format)
    maj_row = "| Majority Vote | " + " | ".join(
        [f"{acc*100:.1f}" for acc in step1["majority_vote"]["accuracies"]]
    ) + " | - |"
    
    pass_row = "| Pass@k | " + " | ".join(
        [f"{acc*100:.1f}" for acc in step1["pass_at_k"]["accuracies"]]
    ) + " | - |"
    
    agent_row = "| ATTS Agent | " + " | ".join(["-"] * len(k_values)) + f" | {step1['agent']['accuracy']*100:.1f} |"
    
    lines = [
        "# Accuracy Comparison",
        "",
        "Values shown as x.x (percentage without % symbol)",
        "",
        header,
        separator,
        maj_row,
        pass_row,
        agent_row,
        "",
    ]
    
    return "\n".join(lines)


def generate_budget_table(results: dict) -> str:
    """Generate markdown table for budget."""
    step2 = results["step2"]
    k_values = step2["k_values"]
    
    # Build header
    header = "| Method | " + " | ".join([f"k={k}" for k in k_values]) + " | Agent |"
    separator = "|" + "|".join([" --- " for _ in range(len(k_values) + 2)]) + "|"
    
    # Build rows
    maj_row = "| Majority Vote | " + " | ".join(
        [f"${budget:.2f}" for budget in step2["majority_vote"]["budgets"]]
    ) + " | - |"
    
    pass_row = "| Pass@k | " + " | ".join(
        [f"${budget:.2f}" for budget in step2["pass_at_k"]["budgets"]]
    ) + " | - |"
    
    agent_row = "| ATTS Agent | " + " | ".join(["-"] * len(k_values)) + f" | ${step2['agent']['budget']:.2f} |"
    
    lines = [
        "# Budget Comparison",
        "",
        "Values shown as $x.xx (USD)",
        "",
        header,
        separator,
        maj_row,
        pass_row,
        agent_row,
        "",
    ]
    
    return "\n".join(lines)


def generate_combined_table(results: dict) -> str:
    """Generate combined markdown table with accuracy and budget."""
    step1 = results["step1"]
    step2 = results["step2"]
    k_values = step1["k_values"]
    
    lines = [
        "# Combined Results: Accuracy and Budget",
        "",
        "Format: Accuracy (x.x) / Budget ($x.xx)",
        "",
    ]
    
    # Header
    header = "| Method | " + " | ".join([f"k={k}" for k in k_values]) + " | Agent |"
    separator = "|" + "|".join([" --- " for _ in range(len(k_values) + 2)]) + "|"
    
    lines.extend([header, separator])
    
    # Majority Vote row
    maj_cells = []
    for acc, budget in zip(step1["majority_vote"]["accuracies"], step2["majority_vote"]["budgets"]):
        maj_cells.append(f"{acc*100:.1f} / ${budget:.2f}")
    maj_row = "| Majority Vote | " + " | ".join(maj_cells) + " | - |"
    lines.append(maj_row)
    
    # Pass@k row
    pass_cells = []
    for acc, budget in zip(step1["pass_at_k"]["accuracies"], step2["pass_at_k"]["budgets"]):
        pass_cells.append(f"{acc*100:.1f} / ${budget:.2f}")
    pass_row = "| Pass@k | " + " | ".join(pass_cells) + " | - |"
    lines.append(pass_row)
    
    # Agent row
    agent_acc = step1["agent"]["accuracy"]
    agent_budget = step2["agent"]["budget"]
    agent_row = "| ATTS Agent | " + " | ".join(["-"] * len(k_values)) + f" | {agent_acc*100:.1f} / ${agent_budget:.2f} |"
    lines.append(agent_row)
    
    lines.append("")
    
    return "\n".join(lines)


def save_csv_tables(results: dict, output_dir: Path):
    """Save tables as CSV files."""
    step1 = results["step1"]
    step2 = results["step2"]
    k_values = step1["k_values"]
    
    # Accuracy CSV
    acc_csv_path = output_dir / "accuracy_table.csv"
    with open(acc_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method"] + [f"k={k}" for k in k_values] + ["Agent"])
        writer.writerow(
            ["Majority Vote"] + 
            [f"{acc*100:.1f}" for acc in step1["majority_vote"]["accuracies"]] + 
            ["-"]
        )
        writer.writerow(
            ["Pass@k"] + 
            [f"{acc*100:.1f}" for acc in step1["pass_at_k"]["accuracies"]] + 
            ["-"]
        )
        writer.writerow(
            ["ATTS Agent"] + 
            ["-"] * len(k_values) + 
            [f"{step1['agent']['accuracy']*100:.1f}"]
        )
    
    print(f"Saved accuracy CSV to {acc_csv_path}")
    
    # Budget CSV
    budget_csv_path = output_dir / "budget_table.csv"
    with open(budget_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method"] + [f"k={k}" for k in k_values] + ["Agent"])
        writer.writerow(
            ["Majority Vote"] + 
            [f"{budget:.4f}" for budget in step2["majority_vote"]["budgets"]] + 
            ["-"]
        )
        writer.writerow(
            ["Pass@k"] + 
            [f"{budget:.4f}" for budget in step2["pass_at_k"]["budgets"]] + 
            ["-"]
        )
        writer.writerow(
            ["ATTS Agent"] + 
            ["-"] * len(k_values) + 
            [f"{step2['agent']['budget']:.4f}"]
        )
    
    print(f"Saved budget CSV to {budget_csv_path}")


def main():
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Step 3: Generate Tables")
    print("=" * 80)
    
    # Load results
    print("\n1. Loading results...")
    results = load_results(args.step1_results, args.step2_results)
    print(f"   Loaded results for run: {results['step1']['run_name']}")
    
    # Generate tables
    print("\n2. Generating tables...")
    
    # Markdown report
    md_lines = [
        f"# Analysis Results: {results['step1']['run_name']}",
        "",
    ]
    
    md_lines.append(generate_accuracy_table(results))
    md_lines.append(generate_budget_table(results))
    md_lines.append(generate_combined_table(results))
    
    # Add summary
    agent = results['step2']['agent']
    orch_cost = agent.get('orchestrator_cost', 0)
    roll_cost = agent.get('rollout_cost', 0)
    
    md_lines.extend([
        "# Summary",
        "",
        "## ATTS Agent Performance",
        f"- Accuracy: {results['step1']['agent']['accuracy']*100:.1f}",
        f"- Average k: {results['step1']['agent']['avg_k']:.1f}",
        f"- Total Budget: ${agent['budget']:.2f}",
    ])
    
    if orch_cost > 0 or roll_cost > 0:
        md_lines.extend([
            f"  - Orchestrator cost: ${orch_cost:.2f}",
            f"  - Sampled rollout cost: ${roll_cost:.2f}",
        ])
    
    md_lines.extend([
        "",
        "## Comparison with Baselines",
        f"At k={results['step1']['k_values'][-1]}:",
        f"- Majority Vote accuracy: {results['step1']['majority_vote']['accuracies'][-1]*100:.1f}",
        f"- Pass@k accuracy: {results['step1']['pass_at_k']['accuracies'][-1]*100:.1f}",
        f"- Budget for k={results['step1']['k_values'][-1]}: ${results['step2']['majority_vote']['budgets'][-1]:.2f}",
        "",
    ])
    
    md_path = args.output_dir / "step3_tables.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"   Saved markdown report to {md_path}")
    
    # CSV tables
    print("\n3. Saving CSV tables...")
    save_csv_tables(results, args.output_dir)
    
    print("\n" + "=" * 80)
    print("Step 3 Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
