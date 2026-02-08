#!/usr/bin/env python3
"""
Step 1: Aggregated Analysis

Generates:
- Plot 1: Accuracy vs k (pass@k, maj@k, genselect, ATTS, maj@budget)
- Plot 2: Accuracy vs cost (same groups)
- Dedicated tables mirroring the plots
- Per-question table with correct freq, maj@64, ATTS, genselect@64, maj@budget
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

# ── Pricing ──────────────────────────────────────────────────────────────────
PRICING = {
    "input_price_per_million": 0.60,
    "output_price_per_million": 2.20,
    "cache_read_price_per_million": 0.11,
}


def parse_args():
    p = argparse.ArgumentParser(description="Aggregated ATTS Analysis")
    p.add_argument("--rollout-files", required=True, action="append")
    p.add_argument("--atts-dirs", required=True, action="append", type=Path)
    p.add_argument("--genselect-dirs", required=True, action="append", type=Path)
    p.add_argument("--k-values", default="1,2,4,8,16,32,64")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


# ── Rollout data loading ─────────────────────────────────────────────────────
def load_rollout_data(rollout_files: List[str]) -> Dict[int, List[dict]]:
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


def build_rollout_tokens(data: Dict[int, List[dict]]) -> Dict[int, List[int]]:
    out = {}
    for qid, responses in data.items():
        tokens = []
        for r in responses:
            tc = int(r["response_token_count"])
            assert tc > 0, f"Non-positive token count for qid={qid}"
            tokens.append(tc)
        out[qid] = tokens
    return out


def build_answer_correctness(data: Dict[int, List[dict]]) -> Dict[int, Dict[str, bool]]:
    """Build answer->correctness map per question from rollout labels."""
    out: Dict[int, Dict[str, bool]] = {}
    for qid, rows in data.items():
        ans_map: Dict[str, bool] = {}
        for r in rows:
            ans = str(r["pred_answer"])
            ans_map[ans] = ans_map.get(ans, False) or bool(r.get("label"))
        out[qid] = ans_map
    return out


# ── Pass@k / Maj@k ──────────────────────────────────────────────────────────
def pass_at_k_estimator(responses: List[dict], k: int) -> float:
    n = len(responses)
    if n < k:
        raise ValueError(f"Need {k} responses, have {n}")
    c = sum(1 for r in responses if r.get("label"))
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def majority_vote_mc(responses: List[dict], k: int, seed: int, n_samples: int = 100) -> float:
    if len(responses) < k:
        raise ValueError(f"Need {k} responses, have {len(responses)}")
    rng = random.Random(seed)
    correct = 0
    for _ in range(n_samples):
        sampled = rng.sample(responses, k)
        groups = defaultdict(list)
        for r in sampled:
            # Ignore empty extracted answers in majority voting.
            # This avoids counting parser-failure blanks as a valid answer option.
            ans = str(r.get("pred_answer", "")).strip()
            if not ans:
                continue
            groups[ans].append(r)
        if not groups:
            continue
        max_count = max(len(g) for g in groups.values())
        majority_groups = [g for g in groups.values() if len(g) == max_count]
        if rng.choice(majority_groups)[0]["label"]:
            correct += 1
    return correct / n_samples


def compute_rollout_cost_for_k(tokens_by_q: Dict[int, List[int]], k: int) -> float:
    total = 0.0
    for qid, toks in tokens_by_q.items():
        assert len(toks) >= k, f"qid={qid}: need {k}, have {len(toks)}"
        scaled = sum(toks) * (k / len(toks))
        total += scaled * PRICING["output_price_per_million"] / 1_000_000
    return total


# ── ATTS data loading ────────────────────────────────────────────────────────
def load_atts_run(
    atts_dir: Path,
    tokens_by_q: Dict[int, List[int]],
    answer_correct_by_q: Dict[int, Dict[str, bool]],
    seed: int,
) -> dict:
    """Load a single ATTS run. Returns dict with per-question data and aggregates."""
    preds_file = atts_dir / "preds.json"
    if not preds_file.exists():
        raise FileNotFoundError(f"Not found: {preds_file}")
    with open(preds_file) as f:
        preds = json.load(f)

    per_question = {}
    for qid_str, pred in preds.items():
        qid = int(qid_str)
        traj_file = atts_dir / f"question_{qid}" / f"{qid}.traj.json"
        if not traj_file.exists():
            traj_file = atts_dir / f"question_{qid}" / "0.traj.json"

        subagent_count = 0
        total_prompt_tokens = 0
        max_prompt_tokens = 0
        total_completion_tokens = 0
        sampled_answer_dist = None  # answer -> count

        if traj_file.exists():
            with open(traj_file) as f:
                traj = json.load(f)

            for msg in traj.get("messages", []):
                if msg.get("role") == "assistant":
                    extra = msg.get("extra", {})
                    response = extra.get("response", {})
                    usage = response.get("usage", {})
                    if usage:
                        pt = usage.get("prompt_tokens", 0)
                        total_prompt_tokens += pt
                        total_completion_tokens += usage.get("completion_tokens", 0)
                        max_prompt_tokens = max(max_prompt_tokens, pt)
                    choices = response.get("choices", [])
                    if choices:
                        tcs = choices[0].get("message", {}).get("tool_calls", [])
                        for tc in (tcs or []):
                            if tc.get("function", {}).get("name") == "subagent":
                                args = json.loads(tc["function"]["arguments"])
                                subagent_count += args.get("count", 0)

            info = traj.get("info", {})
            # Handle new format (sampled_rollout_metadata) and old format (early_stop_metadata)
            srm = info.get("sampled_rollout_metadata")
            esm = info.get("early_stop_metadata")
            if srm:
                subagent_count = srm.get("total_budget_used", subagent_count)
                sampled_answer_dist = srm.get("sampled_answers")
            elif esm:
                subagent_count = esm.get("budget_used", subagent_count)
                sampled_answer_dist = esm.get("sampled_answers")

            # For non-early-stopped without metadata, parse tool response
            if sampled_answer_dist is None and pred.get("exit_status") != "early_stopped":
                sampled_answer_dist = _parse_answer_stats_from_traj(traj)

        input_cache_tokens = total_prompt_tokens - max_prompt_tokens
        sampled_rollout_tokens = sum(tokens_by_q.get(qid, [])[:subagent_count]) if qid in tokens_by_q else 0

        per_question[qid] = {
            "answer": pred["answer"],
            "gold": pred["gold"],
            "correct": pred["correct"],
            "exit_status": pred.get("exit_status", "Submitted"),
            "subagent_count": subagent_count,
            "total_prompt_tokens": total_prompt_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "input_cache_tokens": input_cache_tokens,
            "total_completion_tokens": total_completion_tokens,
            "sampled_rollout_tokens": sampled_rollout_tokens,
            "sampled_answer_dist": sampled_answer_dist,
        }

    # Aggregates
    n_total = len(per_question)
    n_correct = sum(1 for d in per_question.values() if d["correct"])
    avg_k = np.mean([d["subagent_count"] for d in per_question.values()])
    accuracy = n_correct / n_total if n_total else 0.0

    # Cost (extrapolate from non-early-stopped)
    orch_cost_ne = 0.0
    roll_cost_ne = 0.0
    n_ne = 0
    for d in per_question.values():
        if d["exit_status"] != "early_stopped":
            orch_cost_ne += _compute_price(
                d["max_prompt_tokens"], d["input_cache_tokens"], d["total_completion_tokens"]
            )
            roll_cost_ne += d["sampled_rollout_tokens"] * PRICING["output_price_per_million"] / 1_000_000
            n_ne += 1
    if n_ne > 0:
        factor = n_total / n_ne
        orch_cost = orch_cost_ne * factor
        roll_cost = roll_cost_ne * factor
    else:
        orch_cost = roll_cost = 0.0

    # Maj@budget: majority vote from the same rollouts the agent saw
    # Use Monte Carlo with random tie-breaking for consistency with maj@k
    def maj_budget_mc(dist: dict, qid: int, seed: int, n_samples: int = 100) -> float:
        if not dist:
            return 0.0
        max_count = max(dist.values())
        majority_answers = [a for a, c in dist.items() if c == max_count]
        answer_map = answer_correct_by_q.get(qid, {})
        rng = random.Random(seed)
        correct = 0
        for _ in range(n_samples):
            chosen = rng.choice(majority_answers)
            if answer_map.get(str(chosen), False):
                correct += 1
        return correct / n_samples

    maj_budget_results = []
    maj_budget_valid = 0
    for qid, d in per_question.items():
        dist = d.get("sampled_answer_dist")
        if dist is None:
            # Count missing distribution as 0 for denominator consistency with all questions.
            maj_budget_results.append(0.0)
            continue
        maj_budget_valid += 1
        acc = maj_budget_mc(dist, qid, seed)
        maj_budget_results.append(acc)

    maj_budget_acc = float(np.mean(maj_budget_results)) if maj_budget_results else 0.0
    maj_budget_total = n_total

    return {
        "name": atts_dir.name,
        "per_question": per_question,
        "accuracy": accuracy,
        "avg_k": avg_k,
        "total_cost": orch_cost + roll_cost,
        "orch_cost": orch_cost,
        "roll_cost": roll_cost,
        "n_early_stopped": n_total - n_ne,
        "n_total": n_total,
        "maj_budget_accuracy": maj_budget_acc,
        "maj_budget_total": maj_budget_total,
        "maj_budget_valid": maj_budget_valid,
    }


def _parse_answer_stats_from_traj(traj: dict) -> dict | None:
    """Parse answer distribution from tool response messages in trajectory (option B)."""
    answer_dist = {}
    for msg in traj.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if "Answer Statistics:" not in content or "<output>" not in content:
            continue
        # Extract the output section
        try:
            start = content.index("<output>")
            end = content.index("</output>", start)
            output_text = content[start + 8 : end]
        except ValueError:
            continue
        if "Answer Statistics:" not in output_text:
            continue
        # Parse table rows: | 0-3     | 10               |
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
            # Parse index range to count
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


def _compute_price(input_tokens: int, cache_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * PRICING["input_price_per_million"]
        + cache_tokens * PRICING["cache_read_price_per_million"]
        + output_tokens * PRICING["output_price_per_million"]
    ) / 1_000_000


# ── GenSelect data loading ───────────────────────────────────────────────────
def load_genselect_runs(dirs: List[Path]) -> Dict[int, List[dict]]:
    """Load genselect runs grouped by k. Returns {k: [run_data, ...]}."""
    by_k = defaultdict(list)
    for d in dirs:
        m = re.match(r"k(\d+)_", d.name)
        if not m:
            raise ValueError(f"Cannot parse k from genselect dir name: {d.name}")
        k = int(m.group(1))

        results_file = d / "results_per_question.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Not found: {results_file}")
        with open(results_file) as f:
            per_q = json.load(f)

        summary_file = d / "results_summary.json"
        summary = {}
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)

        # Load trace token usage
        traces_file = d / "selector_traces.jsonl"
        total_prompt = 0
        total_completion = 0
        n_traces = 0
        if traces_file.exists():
            with open(traces_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    n_traces += 1
                    api_resp = rec.get("api_response") or rec.get("response")
                    if isinstance(api_resp, dict):
                        usage = api_resp.get("usage", {})
                        total_prompt += usage.get("prompt_tokens", 0)
                        total_completion += usage.get("completion_tokens", 0)

        correctness_by_qid = {}
        answers_by_qid = {}
        for r in per_q:
            correctness_by_qid[r["question_id"]] = r["is_correct"]
            answers_by_qid[r["question_id"]] = r.get("selected_answer", "")

        accuracy = np.mean(list(correctness_by_qid.values())) if correctness_by_qid else 0.0

        by_k[k].append({
            "name": d.name,
            "k": k,
            "accuracy": accuracy,
            "correctness_by_qid": correctness_by_qid,
            "answers_by_qid": answers_by_qid,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "n_traces": n_traces,
            "n_questions": len(correctness_by_qid),
            "summary": summary,
        })
    return dict(by_k)


# ── Plotting ─────────────────────────────────────────────────────────────────
def make_plots(
    k_values, maj_accs, pass_accs, maj_costs, pass_costs,
    gs_k_values, gs_accs, gs_accs_std, gs_costs, gs_costs_std,
    atts_avg_k, atts_acc, atts_acc_std, atts_cost, atts_cost_std,
    mb_acc, mb_acc_std,
    output_dir: Path,
):
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # ── Plot 1: accuracy vs k ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, pass_accs, "s-", color="#10B981", linewidth=2.5, markersize=10, label="Pass@k")
    ax.plot(k_values, maj_accs, "o-", color="#3B82F6", linewidth=2.5, markersize=10, label="Maj@k")
    if gs_k_values:
        ax.errorbar(gs_k_values, gs_accs, yerr=gs_accs_std, fmt="^-", color="#F59E0B",
                     linewidth=2.5, markersize=10, capsize=5, label="GenSelect")
    ax.plot(atts_avg_k, atts_acc, "D", color="#EF4444", markersize=14,
            markeredgecolor="white", markeredgewidth=2, label=f"ATTS (k={atts_avg_k:.1f})", zorder=10)
    ax.plot(atts_avg_k, mb_acc, "P", color="#8B5CF6", markersize=14,
            markeredgecolor="white", markeredgewidth=2, label=f"Maj@budget (k={atts_avg_k:.1f})", zorder=10)

    ax.set_xlabel("k (Number of Samples)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs k")
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_accuracy_vs_k.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot 2: accuracy vs cost ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pass_costs, pass_accs, "s-", color="#10B981", linewidth=2.5, markersize=10, label="Pass@k")
    ax.plot(maj_costs, maj_accs, "o-", color="#3B82F6", linewidth=2.5, markersize=10, label="Maj@k")
    if gs_k_values:
        ax.errorbar(gs_costs, gs_accs, xerr=gs_costs_std, yerr=gs_accs_std, fmt="^-",
                     color="#F59E0B", linewidth=2.5, markersize=10, capsize=5, label="GenSelect")
    ax.errorbar([atts_cost], [atts_acc], xerr=[atts_cost_std], yerr=[atts_acc_std],
                fmt="D", color="#EF4444", markersize=14, markeredgecolor="white",
                markeredgewidth=2, capsize=5, label="ATTS", zorder=10)
    ax.errorbar([atts_cost], [mb_acc], xerr=[atts_cost_std], yerr=[mb_acc_std],
                fmt="P", color="#8B5CF6", markersize=14, markeredgecolor="white",
                markeredgewidth=2, capsize=5, label="Maj@budget", zorder=10)

    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Cost")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:.2f}"))
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_accuracy_vs_cost.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plots to {output_dir}")


# ── Tables ───────────────────────────────────────────────────────────────────
def write_dedicated_tables(
    k_values, maj_accs, pass_accs, maj_costs, pass_costs,
    gs_k_values, gs_accs, gs_accs_std, gs_costs, gs_costs_std,
    atts_avg_k, atts_acc, atts_acc_std, atts_cost, atts_cost_std,
    mb_acc, mb_acc_std,
    output_dir: Path,
):
    gs_by_k = {k: (a, s) for k, a, s in zip(gs_k_values, gs_accs, gs_accs_std)}
    gs_cost_by_k = {k: (c, s) for k, c, s in zip(gs_k_values, gs_costs, gs_costs_std)}

    lines = ["# Accuracy Table", ""]
    header = "| Method |" + "".join(f" k={k} |" for k in k_values) + " Avg |"
    sep = "|---|" + "---|" * (len(k_values) + 1)
    lines += [header, sep]

    # Pass@k row
    row = "| Pass@k |"
    for i, k in enumerate(k_values):
        row += f" {pass_accs[i]*100:.2f} |"
    row += " - |"
    lines.append(row)

    # Maj@k row
    row = "| Maj@k |"
    for i, k in enumerate(k_values):
        row += f" {maj_accs[i]*100:.2f} |"
    row += " - |"
    lines.append(row)

    # GenSelect row
    row = "| GenSelect |"
    for k in k_values:
        if k in gs_by_k:
            a, s = gs_by_k[k]
            row += f" {a*100:.2f}±{s*100:.2f} |"
        else:
            row += " - |"
    row += " - |"
    lines.append(row)

    # ATTS row
    row = "| ATTS |"
    for k in k_values:
        row += " - |"
    row += f" {atts_acc*100:.2f}±{atts_acc_std*100:.2f} (k={atts_avg_k:.1f}) |"
    lines.append(row)

    # Maj@budget row
    row = "| Maj@budget |"
    for k in k_values:
        row += " - |"
    row += f" {mb_acc*100:.2f}±{mb_acc_std*100:.2f} (k={atts_avg_k:.1f}) |"
    lines.append(row)

    # Cost table
    lines += ["", "# Cost Table ($)", ""]
    header_c = "| Method |" + "".join(f" k={k} |" for k in k_values) + " Avg |"
    lines += [header_c, sep]

    row = "| Pass@k / Maj@k |"
    for i, k in enumerate(k_values):
        row += f" {pass_costs[i]:.4f} |"
    row += " - |"
    lines.append(row)

    row = "| GenSelect |"
    for k in k_values:
        if k in gs_cost_by_k:
            c, s = gs_cost_by_k[k]
            row += f" {c:.4f}±{s:.4f} |"
        else:
            row += " - |"
    row += " - |"
    lines.append(row)

    row = "| ATTS / Maj@budget |"
    for k in k_values:
        row += " - |"
    row += f" {atts_cost:.4f}±{atts_cost_std:.4f} |"
    lines.append(row)

    lines.append("")
    md_path = output_dir / "tables_dedicated.md"
    md_path.write_text("\n".join(lines))
    print(f"  Saved dedicated tables to {md_path}")


def write_per_question_table(
    rollout_data: Dict[int, List[dict]],
    atts_runs: List[dict],
    gs_by_k: Dict[int, List[dict]],
    output_dir: Path,
    seed: int = 42,
):
    """Per-question table: correct freq, maj@64, ATTS success rate, genselect@64, maj@budget."""
    all_qids = sorted(rollout_data.keys())
    n_atts = len(atts_runs)

    # GenSelect at k=64
    gs64_runs = gs_by_k.get(64, [])
    n_gs64 = len(gs64_runs)

    lines = [
        "# Per-Question Results",
        "",
        f"ATTS: {n_atts} runs, GenSelect@64: {n_gs64} runs",
        "",
        "| QID | Gold | Correct Freq | Maj@64 | ATTS | GenSelect@64 | Maj@budget |",
        "|-----|------|-------------|--------|------|-------------|-----------|",
    ]
    advantage_rows = []

    for qid in all_qids:
        responses = rollout_data[qid]
        gold = responses[0].get("gold_answer", "?") if responses else "?"
        n_pool = len(responses)
        n_correct_pool = sum(1 for r in responses if r.get("label"))
        correct_freq = f"{n_correct_pool}/{n_pool}"

        # Maj@64
        if n_pool >= 64:
            maj64 = majority_vote_mc(responses, 64, seed)
            maj64_str = f"{maj64*100:.2f}"
        else:
            maj64_str = "-"

        # ATTS success rate
        atts_correct = sum(
            1 for run in atts_runs
            if qid in run["per_question"] and run["per_question"][qid]["correct"]
        )
        atts_total = sum(1 for run in atts_runs if qid in run["per_question"])
        atts_str = f"{atts_correct}/{atts_total}" if atts_total > 0 else "-"

        # GenSelect@64
        gs_correct = sum(
            1 for run in gs64_runs
            if run["correctness_by_qid"].get(qid, False)
        )
        gs_total = sum(1 for run in gs64_runs if qid in run["correctness_by_qid"])
        gs_str = f"{gs_correct}/{gs_total}" if gs_total > 0 else "-"

        # Maj@budget
        mb_correct = 0
        mb_total = 0
        answer_map = {}
        for r in responses:
            ans = str(r["pred_answer"])
            answer_map[ans] = answer_map.get(ans, False) or bool(r.get("label"))
        for run in atts_runs:
            pq = run["per_question"].get(qid)
            if pq is None:
                continue
            dist = pq.get("sampled_answer_dist")
            if dist is None:
                continue
            mb_total += 1
            max_c = max(dist.values())
            majority_ans = [a for a, c in dist.items() if c == max_c]
            if any(answer_map.get(str(a), False) for a in majority_ans):
                mb_correct += 1
        mb_str = f"{mb_correct}/{mb_total}" if mb_total > 0 else "-"
        atts_rate = (atts_correct / atts_total) if atts_total > 0 else 0.0
        mb_rate = (mb_correct / mb_total) if mb_total > 0 else 0.0
        advantage_pp = (atts_rate - mb_rate) * 100.0
        advantage_rows.append({
            "qid": qid,
            "gold": gold,
            "atts_str": atts_str,
            "mb_str": mb_str,
            "atts_rate": atts_rate,
            "mb_rate": mb_rate,
            "advantage_pp": advantage_pp,
        })

        lines.append(f"| {qid} | {gold} | {correct_freq} | {maj64_str} | {atts_str} | {gs_str} | {mb_str} |")

    advantage_rows.sort(key=lambda d: (-d["advantage_pp"], d["qid"]))

    lines.extend([
        "",
        "## ATTS Advantage Over Maj@budget (Sorted)",
        "",
        "| Rank | QID | Gold | ATTS | Maj@budget | Advantage (pp) |",
        "|------|-----|------|------|------------|----------------|",
    ])
    for rank, row in enumerate(advantage_rows, start=1):
        lines.append(
            f"| {rank} | {row['qid']} | {row['gold']} | {row['atts_str']} | "
            f"{row['mb_str']} | {row['advantage_pp']:+.2f} |"
        )

    lines.append("")
    md_path = output_dir / "table_per_question.md"
    md_path.write_text("\n".join(lines))
    print(f"  Saved per-question table to {md_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    k_values = sorted(int(k.strip()) for k in args.k_values.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 1: Aggregated Analysis")
    print("=" * 70)

    # 1. Load rollout data
    print("\n1. Loading rollout data...")
    rollout_data = load_rollout_data(args.rollout_files)
    tokens_by_q = build_rollout_tokens(rollout_data)
    answer_correct_by_q = build_answer_correctness(rollout_data)
    print(f"   {len(rollout_data)} questions, {sum(len(v) for v in rollout_data.values())} total responses")

    # 2. Pass@k and Maj@k
    print("\n2. Computing Pass@k and Maj@k...")
    maj_accs, pass_accs, maj_costs, pass_costs = [], [], [], []
    for k in tqdm(k_values, desc="k values"):
        maj_results, pass_results = [], []
        for qid, responses in rollout_data.items():
            maj_results.append(majority_vote_mc(responses, k, args.seed))
            pass_results.append(pass_at_k_estimator(responses, k))
        maj_accs.append(float(np.mean(maj_results)))
        pass_accs.append(float(np.mean(pass_results)))
        cost = compute_rollout_cost_for_k(tokens_by_q, k)
        maj_costs.append(cost)
        pass_costs.append(cost)
        print(f"   k={k:2d}: Maj={maj_accs[-1]*100:.2f}, Pass={pass_accs[-1]*100:.2f}, Cost=${cost:.4f}")

    # 3. ATTS runs
    print(f"\n3. Loading {len(args.atts_dirs)} ATTS runs...")
    atts_runs = []
    for d in tqdm(args.atts_dirs, desc="ATTS runs"):
        if not d.exists():
            print(f"   WARNING: {d} not found, skipping")
            continue
        run = load_atts_run(d, tokens_by_q, answer_correct_by_q, args.seed)
        atts_runs.append(run)
        print(f"   {run['name']}: acc={run['accuracy']*100:.2f}, avg_k={run['avg_k']:.1f}, "
              f"cost=${run['total_cost']:.4f}, maj@budget={run['maj_budget_accuracy']*100:.2f} "
              f"(valid={run['maj_budget_valid']}/{run['n_total']})")

    if not atts_runs:
        print("ERROR: No ATTS runs loaded")
        sys.exit(1)

    atts_acc = float(np.mean([r["accuracy"] for r in atts_runs]))
    atts_acc_std = float(np.std([r["accuracy"] for r in atts_runs]))
    atts_avg_k = float(np.mean([r["avg_k"] for r in atts_runs]))
    atts_cost = float(np.mean([r["total_cost"] for r in atts_runs]))
    atts_cost_std = float(np.std([r["total_cost"] for r in atts_runs]))
    mb_acc = float(np.mean([r["maj_budget_accuracy"] for r in atts_runs]))
    mb_acc_std = float(np.std([r["maj_budget_accuracy"] for r in atts_runs]))

    print(f"   Aggregated ATTS: acc={atts_acc*100:.2f}±{atts_acc_std*100:.2f}, "
          f"avg_k={atts_avg_k:.1f}, cost=${atts_cost:.4f}")
    print(f"   Aggregated Maj@budget: acc={mb_acc*100:.2f}±{mb_acc_std*100:.2f}")

    # 4. GenSelect runs
    print(f"\n4. Loading {len(args.genselect_dirs)} GenSelect runs...")
    gs_by_k = load_genselect_runs(args.genselect_dirs)
    gs_k_values_sorted = sorted(gs_by_k.keys())
    gs_accs, gs_accs_std, gs_costs, gs_costs_std = [], [], [], []
    for k in gs_k_values_sorted:
        runs = gs_by_k[k]
        accs = [r["accuracy"] for r in runs]
        rollout_cost = compute_rollout_cost_for_k(tokens_by_q, k)
        costs = []
        for r in runs:
            sel_cost = (
                r["total_prompt_tokens"] * PRICING["input_price_per_million"]
                + r["total_completion_tokens"] * PRICING["output_price_per_million"]
            ) / 1_000_000
            costs.append(rollout_cost + sel_cost)
        gs_accs.append(float(np.mean(accs)))
        gs_accs_std.append(float(np.std(accs)))
        gs_costs.append(float(np.mean(costs)))
        gs_costs_std.append(float(np.std(costs)))
        print(f"   k={k}: acc={gs_accs[-1]*100:.2f}±{gs_accs_std[-1]*100:.2f}, "
              f"cost=${gs_costs[-1]:.4f}±{gs_costs_std[-1]:.4f} ({len(runs)} seeds)")

    # 5. Plots
    print("\n5. Generating plots...")
    make_plots(
        k_values, maj_accs, pass_accs, maj_costs, pass_costs,
        gs_k_values_sorted, gs_accs, gs_accs_std, gs_costs, gs_costs_std,
        atts_avg_k, atts_acc, atts_acc_std, atts_cost, atts_cost_std,
        mb_acc, mb_acc_std,
        args.output_dir,
    )

    # 6. Dedicated tables
    print("\n6. Writing dedicated tables...")
    write_dedicated_tables(
        k_values, maj_accs, pass_accs, maj_costs, pass_costs,
        gs_k_values_sorted, gs_accs, gs_accs_std, gs_costs, gs_costs_std,
        atts_avg_k, atts_acc, atts_acc_std, atts_cost, atts_cost_std,
        mb_acc, mb_acc_std,
        args.output_dir,
    )

    # 7. Per-question table
    print("\n7. Writing per-question table...")
    write_per_question_table(rollout_data, atts_runs, gs_by_k, args.output_dir, args.seed)

    # 8. Save raw results JSON
    results = {
        "k_values": k_values,
        "majority_vote": {"accuracies": maj_accs, "costs": maj_costs},
        "pass_at_k": {"accuracies": pass_accs, "costs": pass_costs},
        "atts": {
            "accuracy": atts_acc, "accuracy_std": atts_acc_std,
            "avg_k": atts_avg_k, "cost": atts_cost, "cost_std": atts_cost_std,
            "n_runs": len(atts_runs),
            "per_run": [{"name": r["name"], "accuracy": r["accuracy"], "avg_k": r["avg_k"],
                         "cost": r["total_cost"]} for r in atts_runs],
        },
        "maj_at_budget": {"accuracy": mb_acc, "accuracy_std": mb_acc_std},
        "genselect": {
            "k_values": gs_k_values_sorted,
            "accuracies": gs_accs, "accuracies_std": gs_accs_std,
            "costs": gs_costs, "costs_std": gs_costs_std,
        },
    }
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results.json")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
