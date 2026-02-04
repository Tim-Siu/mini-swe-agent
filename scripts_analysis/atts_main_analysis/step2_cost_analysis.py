#!/usr/bin/env python3
"""
Step 2 Analysis: Cost-Based Comparison

Compares accuracy vs cost (budget) for:
- Majority voting (with different k values)
- Pass@k (best-of-k)
- ATTS Agent (adaptive test-time scaling)

Cost is calculated using token counts with cache pricing.
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


# Pricing for GLM-4.7 (from analyze_token_usage.py)
DEFAULT_PRICING = {
    "input_price_per_million": 0.60,
    "output_price_per_million": 2.20,
    "cache_read_price_per_million": 0.11,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: Cost-Based Comparison")
    parser.add_argument(
        "--rollout-files",
        required=True,
        nargs="+",
        help="Path(s) to rollout JSONL files",
    )
    parser.add_argument(
        "--token-counts-files",
        required=True,
        nargs="+",
        help="Path(s) to token counts JSON files (corresponding to rollout files)",
    )
    parser.add_argument(
        "--atts-dir",
        required=True,
        type=Path,
        help="Path to ATTS rollout directory",
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
        help="Random seed",
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


def load_token_counts(token_counts_files: List[str]) -> Dict[int, List[int]]:
    """Load token counts grouped by question_id."""
    tokens_by_question = defaultdict(list)
    
    for filepath in token_counts_files:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Token counts file not found: {filepath}")
        
        with open(path) as f:
            data = json.load(f)
        
        for record in data.get("records", []):
            qid = record["question_id"]
            token_count = record.get("post_think_token_count", 0)
            tokens_by_question[qid].append(token_count)
    
    return dict(tokens_by_question)


def load_atts_data(atts_dir: Path, tokens_by_question: Dict[int, List[int]]) -> Dict[int, dict]:
    """Load ATTS agent data with token usage including sampled rollout cost."""
    preds_file = atts_dir / "preds.json"
    if not preds_file.exists():
        raise FileNotFoundError(f"ATTS preds file not found: {preds_file}")
    
    with open(preds_file) as f:
        preds = json.load(f)
    
    atts_data = {}
    for qid_str, pred_info in preds.items():
        qid = int(qid_str)
        
        # Load trajectory
        traj_file = atts_dir / f"question_{qid}" / f"{qid}.traj.json"
        if not traj_file.exists():
            traj_file = atts_dir / f"question_{qid}" / "0.traj.json"
        
        subagent_count = 0
        total_prompt_tokens = 0
        max_prompt_tokens = 0
        total_completion_tokens = 0
        
        if traj_file.exists():
            with open(traj_file) as f:
                traj = json.load(f)
            
            # Extract usage from messages
            for msg in traj.get("messages", []):
                if msg.get("role") == "assistant":
                    # Count subagent calls
                    extra = msg.get("extra", {})
                    response = extra.get("response", {})
                    choices = response.get("choices", [])
                    
                    # Extract token usage
                    usage = response.get("usage", {})
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_prompt_tokens += prompt_tokens
                        total_completion_tokens += completion_tokens
                        max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
                    
                    # Count subagents
                    if choices:
                        tool_calls = choices[0].get("message", {}).get("tool_calls", [])
                        if tool_calls:
                            for tc in tool_calls:
                                if tc.get("function", {}).get("name") == "subagent":
                                    args = json.loads(tc["function"]["arguments"])
                                    subagent_count += args.get("count", 0)
            
            # For early stopped questions
            early_stop_metadata = traj.get("info", {}).get("early_stop_metadata", {})
            if early_stop_metadata:
                subagent_count = early_stop_metadata.get("budget_used", subagent_count)
        
        # Calculate cache tokens (same method as analyze_token_usage.py)
        input_cache_tokens = total_prompt_tokens - max_prompt_tokens
        
        # Calculate rollout cost for the sampled responses
        # ATTS agent samples 'subagent_count' responses from the rollout pool
        rollout_tokens = tokens_by_question.get(qid, [])
        sampled_rollout_tokens = sum(rollout_tokens[:subagent_count]) if rollout_tokens else 0
        
        atts_data[qid] = {
            "question_id": qid,
            "answer": pred_info["answer"],
            "gold": pred_info["gold"],
            "correct": pred_info["correct"],
            "exit_status": pred_info.get("exit_status", "submitted"),
            "subagent_count": subagent_count,
            "total_prompt_tokens": total_prompt_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "input_cache_tokens": input_cache_tokens,
            "total_completion_tokens": total_completion_tokens,
            "sampled_rollout_tokens": sampled_rollout_tokens,
        }
    
    return atts_data


def compute_price(
    input_tokens: int,
    cache_tokens: int,
    output_tokens: int,
    pricing: Dict[str, float],
) -> float:
    """Compute price with cache."""
    input_price = pricing["input_price_per_million"]
    output_price = pricing["output_price_per_million"]
    cache_price = pricing["cache_read_price_per_million"]
    
    price = (
        input_tokens * input_price +
        cache_tokens * cache_price +
        output_tokens * output_price
    ) / 1_000_000
    
    return price


def compute_rollout_cost_for_k(
    tokens_by_question: Dict[int, List[int]],
    k: int,
    pricing: Dict[str, float],
) -> float:
    """
    Compute total cost for rollout with k samples per question.
    
    For rollout, we assume:
    - Input tokens: proportional to the output tokens (using a ratio)
    - Cache is used (we have max_prompt and cache calculations)
    
    Actually, for rollout, we only have output token counts. We need to estimate
    input tokens. Looking at the ATTS analysis, the input tokens are typically
    much larger than output tokens (due to long prompts).
    
    For simplicity, we'll use a ratio based on the ATTS data or use a fixed ratio.
    """
    total_cost = 0.0
    
    for qid, token_counts in tokens_by_question.items():
        # Use first k token counts for this question
        actual_k = min(k, len(token_counts))
        
        # For rollout cost calculation:
        # We only have output token counts from the token_counts.json
        # We need to estimate input tokens. Based on ATTS data, input is typically
        # 5-10x output. Let's use a conservative estimate.
        
        # Actually, let's use the response_token_count from rollout data
        # which we loaded separately
        total_output_tokens = sum(token_counts[:actual_k])
        
        # Estimate input tokens: for each generation, input is roughly the same
        # (the prompt). Let's estimate based on typical prompt length.
        # From ATTS data, max_prompt_tokens is typically 5000-20000
        # For a fair comparison, we should use actual data if available
        # For now, estimate input as 1x output (conservative for cost)
        estimated_input_per_gen = 5000  # Conservative estimate
        max_input = estimated_input_per_gen
        total_input = estimated_input_per_gen * actual_k
        cache_tokens = total_input - max_input
        
        cost = compute_price(max_input, cache_tokens, total_output_tokens, pricing)
        total_cost += cost
    
    return total_cost


def compute_atts_cost(
    atts_data: Dict[int, dict],
    pricing: Dict[str, float],
) -> Tuple[float, float, float]:
    """
    Compute total cost for ATTS agent.
    
    Returns:
        (total_cost, orchestrator_cost, rollout_cost)
    """
    orchestrator_cost = 0.0
    rollout_cost = 0.0
    
    for qid, data in atts_data.items():
        # Orchestrator agent cost (with cache)
        orch_cost = compute_price(
            data["max_prompt_tokens"],
            data["input_cache_tokens"],
            data["total_completion_tokens"],
            pricing,
        )
        orchestrator_cost += orch_cost
        
        # Sampled rollout cost (no cache for rollout responses)
        # Estimate: each sampled response has ~5000 input tokens (prompt)
        # and the output tokens from token_counts
        sampled_tokens = data["sampled_rollout_tokens"]
        if sampled_tokens > 0:
            # Estimate input tokens per response (conservative)
            input_per_response = 5000
            num_samples = data["subagent_count"]
            total_input = input_per_response * num_samples
            max_input = input_per_response
            cache_tokens = total_input - max_input
            
            roll_cost = compute_price(
                max_input,
                cache_tokens,
                sampled_tokens,
                pricing,
            )
            rollout_cost += roll_cost
    
    total_cost = orchestrator_cost + rollout_cost
    return total_cost, orchestrator_cost, rollout_cost


def compute_majority_vote(responses: List[dict], k: int, seed: int) -> bool:
    """Compute majority voting accuracy for k samples."""
    if not responses:
        return False
    
    actual_k = min(k, len(responses))
    rng = random.Random(seed)
    sampled = rng.sample(responses, actual_k)
    
    answer_groups = defaultdict(list)
    for r in sampled:
        answer_groups[r["pred_answer"]].append(r)
    
    if not answer_groups:
        return False
    
    max_count = max(len(group) for group in answer_groups.values())
    majority_groups = [group for group in answer_groups.values() if len(group) == max_count]
    
    rng = random.Random(seed)
    majority_group = rng.choice(majority_groups)
    
    return majority_group[0]["label"]


def compute_pass_at_k(responses: List[dict], k: int) -> bool:
    """Compute pass@k (whether any of k samples is correct)."""
    if not responses:
        return False
    
    actual_k = min(k, len(responses))
    
    for i in range(actual_k):
        if responses[i]["label"]:
            return True
    
    return False


def compute_metrics_for_k(
    data_by_question: Dict[int, List[dict]],
    k: int,
    seed: int,
) -> Tuple[float, float]:
    """Compute majority vote and pass@k accuracy for a given k."""
    maj_results = []
    pass_results = []
    
    for qid, responses in data_by_question.items():
        maj_results.append(compute_majority_vote(responses, k, seed))
        pass_results.append(compute_pass_at_k(responses, k))
    
    maj_acc = np.mean(maj_results) if maj_results else 0.0
    pass_acc = np.mean(pass_results) if pass_results else 0.0
    
    return maj_acc, pass_acc


def compute_atts_agent_accuracy(atts_data: Dict[int, dict]) -> float:
    """Compute ATTS agent accuracy."""
    correct_count = sum(1 for d in atts_data.values() if d["correct"])
    total = len(atts_data)
    return correct_count / total if total > 0 else 0.0


def plot_cost_comparison(
    maj_budgets: List[float],
    maj_accuracies: List[float],
    pass_budgets: List[float],
    pass_accuracies: List[float],
    agent_budget: float,
    agent_accuracy: float,
    output_path: Path,
    run_name: str,
):
    """Plot comparison with budget on x-axis."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot majority voting
    ax.plot(
        maj_budgets,
        maj_accuracies,
        "o-",
        color="#3B82F6",
        linewidth=2.5,
        markersize=10,
        label="Majority Vote@k",
    )
    
    # Plot pass@k
    ax.plot(
        pass_budgets,
        pass_accuracies,
        "s-",
        color="#10B981",
        linewidth=2.5,
        markersize=10,
        label="Pass@k",
    )
    
    # Plot agent
    ax.plot(
        agent_budget,
        agent_accuracy,
        "D",
        color="#EF4444",
        markersize=14,
        markeredgecolor="white",
        markeredgewidth=2,
        label="ATTS Agent",
        zorder=10,
    )
    
    ax.set_xlabel("Budget ($)", fontsize=13, fontweight="600")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="600")
    ax.set_title(f"Accuracy vs Budget: Majority Vote vs Pass@k vs ATTS Agent\n{run_name}", 
                 fontsize=14, fontweight="bold")
    
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    
    # Format x-axis as currency
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:.2f}"))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved plot to {output_path}")


def save_results(
    output_dir: Path,
    k_values: List[int],
    maj_budgets: List[float],
    maj_accuracies: List[float],
    pass_budgets: List[float],
    pass_accuracies: List[float],
    agent_budget: float,
    agent_accuracy: float,
    agent_orch_cost: float,
    agent_rollout_cost: float,
    run_name: str,
):
    """Save numerical results to JSON."""
    results = {
        "run_name": run_name,
        "k_values": k_values,
        "majority_vote": {
            "budgets": maj_budgets,
            "accuracies": maj_accuracies,
        },
        "pass_at_k": {
            "budgets": pass_budgets,
            "accuracies": pass_accuracies,
        },
        "agent": {
            "budget": agent_budget,
            "accuracy": agent_accuracy,
            "orchestrator_cost": agent_orch_cost,
            "rollout_cost": agent_rollout_cost,
        },
    }
    
    output_path = output_dir / "step2_results.json"
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
    
    pricing = {
        "input_price_per_million": args.input_price,
        "output_price_per_million": args.output_price,
        "cache_read_price_per_million": args.cache_read_price,
    }
    
    print("=" * 80)
    print("Step 2: Cost-Based Comparison")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading rollout data...")
    data_by_question = load_rollout_data(args.rollout_files)
    print(f"   Loaded {len(data_by_question)} questions")
    
    print("\n2. Loading token counts...")
    tokens_by_question = load_token_counts(args.token_counts_files)
    print(f"   Loaded token counts for {len(tokens_by_question)} questions")
    
    print("\n3. Loading ATTS agent data...")
    atts_data = load_atts_data(args.atts_dir, tokens_by_question)
    print(f"   Loaded {len(atts_data)} ATTS predictions")
    
    # Compute ATTS agent metrics
    print("\n4. Computing ATTS agent metrics...")
    agent_accuracy = compute_atts_agent_accuracy(atts_data)
    agent_budget, agent_orch_cost, agent_rollout_cost = compute_atts_cost(atts_data, pricing)
    print(f"   Agent accuracy: {agent_accuracy:.3f}")
    print(f"   Agent total budget: ${agent_budget:.4f}")
    print(f"     - Orchestrator cost: ${agent_orch_cost:.4f}")
    print(f"     - Sampled rollout cost: ${agent_rollout_cost:.4f}")
    
    # Compute metrics and costs for each k
    print("\n5. computing Majority Vote and Pass@k metrics with costs...")
    maj_budgets = []
    maj_accuracies = []
    pass_budgets = []
    pass_accuracies = []
    
    for k in tqdm(k_values, desc="Processing k values"):
        # Compute accuracy
        maj_acc, pass_acc = compute_metrics_for_k(data_by_question, k, args.seed)
        
        # Compute cost (same for both methods since they use same rollout)
        cost = compute_rollout_cost_for_k(tokens_by_question, k, pricing)
        
        maj_budgets.append(cost)
        maj_accuracies.append(maj_acc)
        pass_budgets.append(cost)
        pass_accuracies.append(pass_acc)
        
        print(f"   k={k:2d}: Budget=${cost:.4f}, Majority={maj_acc:.3f}, Pass@k={pass_acc:.3f}")
    
    # Plot comparison
    print("\n6. Generating plot...")
    plot_path = args.output_dir / "step2_cost_comparison.png"
    plot_cost_comparison(
        maj_budgets,
        maj_accuracies,
        pass_budgets,
        pass_accuracies,
        agent_budget,
        agent_accuracy,
        plot_path,
        run_name,
    )
    
    # Save results
    print("\n7. Saving results...")
    save_results(
        args.output_dir,
        k_values,
        maj_budgets,
        maj_accuracies,
        pass_budgets,
        pass_accuracies,
        agent_budget,
        agent_accuracy,
        agent_orch_cost,
        agent_rollout_cost,
        run_name,
    )
    
    print("\n" + "=" * 80)
    print("Step 2 Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
