#!/usr/bin/env python3
"""
Step 1 Analysis: Majority Voting vs Pass@k vs ATTS Agent

Compares accuracy of:
- Majority voting (with different k values)
- Pass@k (best-of-k)
- ATTS Agent (adaptive test-time scaling)

The agent's k is the average number of subagents used across all problems.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Majority Voting vs Pass@k vs Agent")
    parser.add_argument(
        "--rollout-files",
        required=True,
        nargs="+",
        help="Path(s) to rollout JSONL files (e.g., output_rollouts/glm-4p7/*.jsonl)",
    )
    parser.add_argument(
        "--atts-dir",
        required=True,
        type=Path,
        help="Path to ATTS rollout directory (e.g., atts_rollout/hle_full_v24_...)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for results",
    )
    parser.add_argument(
        "--k-values",
        default="1,2,4,8,16,32,64",
        help="Comma-separated list of k values to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for majority voting tie-breaking",
    )
    return parser.parse_args()


def load_rollout_data(rollout_files: List[str]) -> Dict[int, List[dict]]:
    """Load rollout data grouped by question_id."""
    data_by_question = defaultdict(list)
    
    for filepath in rollout_files:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Rollout file not found: {filepath}")
        
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                qid = record["question_id"]
                data_by_question[qid].append(record)
    
    return dict(data_by_question)


def load_atts_data(atts_dir: Path) -> Dict[int, dict]:
    """Load ATTS agent data from preds.json and trajectory files."""
    preds_file = atts_dir / "preds.json"
    if not preds_file.exists():
        raise FileNotFoundError(f"ATTS preds file not found: {preds_file}")
    
    with open(preds_file) as f:
        preds = json.load(f)
    
    atts_data = {}
    for qid_str, pred_info in preds.items():
        qid = int(qid_str)
        
        # Load trajectory to get subagent count
        traj_file = atts_dir / f"question_{qid}" / f"{qid}.traj.json"
        if not traj_file.exists():
            traj_file = atts_dir / f"question_{qid}" / "0.traj.json"
        
        subagent_count = 0
        if traj_file.exists():
            with open(traj_file) as f:
                traj = json.load(f)
            
            # Count subagent calls from tool_calls
            for msg in traj.get("messages", []):
                if msg.get("role") == "assistant":
                    extra = msg.get("extra", {})
                    response = extra.get("response", {})
                    choices = response.get("choices", [])
                    if choices:
                        tool_calls = choices[0].get("message", {}).get("tool_calls", [])
                        if tool_calls:
                            for tc in tool_calls:
                                if tc.get("function", {}).get("name") == "subagent":
                                    args = json.loads(tc["function"]["arguments"])
                                    subagent_count += args.get("count", 0)
            
            # For early stopped questions, use the budget_used from metadata
            early_stop_metadata = traj.get("info", {}).get("early_stop_metadata", {})
            if early_stop_metadata:
                subagent_count = early_stop_metadata.get("budget_used", subagent_count)
        
        atts_data[qid] = {
            "question_id": qid,
            "answer": pred_info["answer"],
            "gold": pred_info["gold"],
            "correct": pred_info["correct"],
            "exit_status": pred_info.get("exit_status", "submitted"),
            "subagent_count": subagent_count,
        }
    
    return atts_data


def compute_majority_vote(responses: List[dict], k: int, seed: int) -> Tuple[bool, int]:
    """
    Compute majority voting accuracy for k samples.
    
    Returns:
        (is_correct, actual_k) - actual_k may be less than k if not enough responses
    """
    if not responses:
        return False, 0
    
    actual_k = min(k, len(responses))
    
    # Sample k responses (deterministic with seed)
    rng = random.Random(seed)
    sampled = rng.sample(responses, actual_k)
    
    # Group by predicted answer
    answer_groups = defaultdict(list)
    for r in sampled:
        answer_groups[r["pred_answer"]].append(r)
    
    if not answer_groups:
        return False, actual_k
    
    # Find majority (max count)
    max_count = max(len(group) for group in answer_groups.values())
    majority_groups = [group for group in answer_groups.values() if len(group) == max_count]
    
    # Random tie-breaking
    rng = random.Random(seed)
    majority_group = rng.choice(majority_groups)
    
    # Return correctness of majority answer
    return majority_group[0]["label"], actual_k


def compute_pass_at_k(responses: List[dict], k: int) -> Tuple[bool, int]:
    """
    Compute pass@k (whether any of k samples is correct).
    
    Returns:
        (is_correct, actual_k)
    """
    if not responses:
        return False, 0
    
    actual_k = min(k, len(responses))
    
    # Check if any of the first k responses is correct
    # (assuming responses are already shuffled in the rollout)
    for i in range(actual_k):
        if responses[i]["label"]:
            return True, actual_k
    
    return False, actual_k


def compute_metrics_for_k(
    data_by_question: Dict[int, List[dict]],
    k: int,
    seed: int,
) -> Tuple[float, float, Dict[int, bool], Dict[int, bool]]:
    """
    Compute majority vote and pass@k metrics for a given k.
    
    Returns:
        (maj_acc, pass_acc, maj_correctness, pass_correctness)
    """
    maj_results = {}
    pass_results = {}
    
    for qid, responses in data_by_question.items():
        # Majority vote
        maj_correct, _ = compute_majority_vote(responses, k, seed)
        maj_results[qid] = maj_correct
        
        # Pass@k
        pass_correct, _ = compute_pass_at_k(responses, k)
        pass_results[qid] = pass_correct
    
    maj_acc = np.mean(list(maj_results.values())) if maj_results else 0.0
    pass_acc = np.mean(list(pass_results.values())) if pass_results else 0.0
    
    return maj_acc, pass_acc, maj_results, pass_results


def compute_atts_agent_accuracy(atts_data: Dict[int, dict]) -> Tuple[float, float]:
    """
    Compute ATTS agent accuracy and average k.
    
    Returns:
        (accuracy, avg_k)
    """
    correct_count = sum(1 for d in atts_data.values() if d["correct"])
    total = len(atts_data)
    avg_k = np.mean([d["subagent_count"] for d in atts_data.values()]) if atts_data else 0.0
    
    return correct_count / total if total > 0 else 0.0, avg_k


def plot_comparison(
    k_values: List[int],
    maj_accuracies: List[float],
    pass_accuracies: List[float],
    agent_accuracy: float,
    agent_k: float,
    output_path: Path,
    run_name: str,
):
    """Plot comparison of majority voting, pass@k, and agent."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot majority voting
    ax.plot(
        k_values,
        maj_accuracies,
        "o-",
        color="#3B82F6",
        linewidth=2.5,
        markersize=10,
        label="Majority Vote@k",
    )
    
    # Plot pass@k
    ax.plot(
        k_values,
        pass_accuracies,
        "s-",
        color="#10B981",
        linewidth=2.5,
        markersize=10,
        label="Pass@k",
    )
    
    # Plot agent as horizontal line with marker
    ax.axhline(
        y=agent_accuracy,
        color="#EF4444",
        linestyle="--",
        linewidth=2.5,
        label=f"ATTS Agent (k={agent_k:.1f})",
    )
    # Add a marker at the agent's average k position
    ax.plot(
        agent_k,
        agent_accuracy,
        "D",
        color="#EF4444",
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=2,
    )
    
    ax.set_xlabel("k (Number of Samples)", fontsize=13, fontweight="600")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="600")
    ax.set_title(f"Majority Vote vs Pass@k vs ATTS Agent\n{run_name}", fontsize=14, fontweight="bold")
    
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved plot to {output_path}")


def save_results(
    output_dir: Path,
    k_values: List[int],
    maj_accuracies: List[float],
    pass_accuracies: List[float],
    agent_accuracy: float,
    agent_k: float,
    run_name: str,
):
    """Save numerical results to JSON."""
    results = {
        "run_name": run_name,
        "k_values": k_values,
        "majority_vote": {
            "accuracies": maj_accuracies,
        },
        "pass_at_k": {
            "accuracies": pass_accuracies,
        },
        "agent": {
            "accuracy": agent_accuracy,
            "avg_k": agent_k,
        },
    }
    
    output_path = output_dir / "step1_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {output_path}")


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse k values
    k_values = sorted([int(k.strip()) for k in args.k_values.split(",")])
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = args.atts_dir.name
    
    print("=" * 80)
    print("Step 1: Majority Voting vs Pass@k vs ATTS Agent")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading rollout data...")
    data_by_question = load_rollout_data(args.rollout_files)
    print(f"   Loaded {len(data_by_question)} questions")
    
    print("\n2. Loading ATTS agent data...")
    atts_data = load_atts_data(args.atts_dir)
    print(f"   Loaded {len(atts_data)} ATTS predictions")
    
    # Compute ATTS agent metrics
    print("\n3. Computing ATTS agent metrics...")
    agent_accuracy, agent_k = compute_atts_agent_accuracy(atts_data)
    print(f"   Agent accuracy: {agent_accuracy:.3f}")
    print(f"   Agent average k: {agent_k:.2f}")
    
    # Compute metrics for each k
    print("\n4. Computing Majority Vote and Pass@k metrics...")
    maj_accuracies = []
    pass_accuracies = []
    
    for k in tqdm(k_values, desc="Processing k values"):
        maj_acc, pass_acc, _, _ = compute_metrics_for_k(data_by_question, k, args.seed)
        maj_accuracies.append(maj_acc)
        pass_accuracies.append(pass_acc)
        print(f"   k={k:2d}: Majority={maj_acc:.3f}, Pass@k={pass_acc:.3f}")
    
    # Plot comparison
    print("\n5. Generating plot...")
    plot_path = args.output_dir / "step1_comparison.png"
    plot_comparison(
        k_values,
        maj_accuracies,
        pass_accuracies,
        agent_accuracy,
        agent_k,
        plot_path,
        run_name,
    )
    
    # Save results
    print("\n6. Saving results...")
    save_results(
        args.output_dir,
        k_values,
        maj_accuracies,
        pass_accuracies,
        agent_accuracy,
        agent_k,
        run_name,
    )
    
    print("\n" + "=" * 80)
    print("Step 1 Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
