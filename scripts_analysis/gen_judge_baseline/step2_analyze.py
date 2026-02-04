#!/usr/bin/env python3
"""
Analyze results from gen_judge baseline runs.

Generates a markdown report with statistics and comparisons across different
k values, modes (dedup/non-dedup), and configurations.

Usage:
    python step2_analyze.py --results_dir ./scripts_analysis_output/gen_judge_baseline
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def load_results(results_dir: str) -> dict[str, Any]:
    """Load results from a single results directory."""
    summary_path = Path(results_dir) / 'results_summary.json'
    dry_run_path = Path(results_dir) / 'dry_run_stats.json'
    per_question_path = Path(results_dir) / 'results_per_question.json'
    
    results = {'dir': results_dir, 'name': Path(results_dir).name}
    
    if summary_path.exists():
        with open(summary_path) as f:
            results['summary'] = json.load(f)
    
    if dry_run_path.exists():
        with open(dry_run_path) as f:
            results['dry_run'] = json.load(f)
    
    if per_question_path.exists():
        with open(per_question_path) as f:
            results['per_question'] = json.load(f)
    
    return results


def parse_run_name(name: str) -> dict[str, Any]:
    """Parse run directory name to extract parameters.
    
    Expected format: k{K}_{comment}_{mode}_seed{SEED} or k{K}_{mode}_seed{SEED}
    """
    # Pattern: k{N}_*_{mode}_seed{S}
    pattern = r'^k(\d+)_(?:([^_]+)_)?(dedup|non-dedup)_seed(\d+)$'
    match = re.match(pattern, name)
    
    if match:
        return {
            'k': int(match.group(1)),
            'comment': match.group(2),
            'mode': match.group(3),
            'seed': int(match.group(4)),
        }
    
    # Fallback: try to extract k and mode
    k_match = re.search(r'k(\d+)', name)
    mode_match = re.search(r'(dedup|non-dedup)', name)
    
    return {
        'k': int(k_match.group(1)) if k_match else None,
        'comment': None,
        'mode': mode_match.group(1) if mode_match else 'unknown',
        'seed': None,
    }


def compute_baseline_accuracy(per_question: list[dict]) -> float:
    """Compute random baseline accuracy: 1/k for each question."""
    if not per_question:
        return 0.0
    
    total = 0.0
    for q in per_question:
        k = q.get('num_candidates', 1)
        total += 1.0 / k
    
    return total / len(per_question)


def compute_breakdown_by_has_correct(per_question: list[dict], dry_run: dict) -> dict:
    """Compute accuracy breakdown by whether a correct trace was available."""
    if not per_question or not dry_run:
        return {}
    
    dry_run_by_qid = {s['question_id']: s for s in dry_run.get('per_question', [])}
    
    with_correct = []
    without_correct = []
    
    for q in per_question:
        qid = q['question_id']
        dry = dry_run_by_qid.get(qid, {})
        has_correct = dry.get('has_correct_trace', False)
        
        if has_correct:
            with_correct.append(q)
        else:
            without_correct.append(q)
    
    result = {
        'count_with_correct': len(with_correct),
        'count_without_correct': len(without_correct),
    }
    
    if with_correct:
        correct = sum(1 for q in with_correct if q.get('is_correct', False))
        result['accuracy_when_correct_available'] = correct / len(with_correct)
    else:
        result['accuracy_when_correct_available'] = None
    
    if without_correct:
        correct = sum(1 for q in without_correct if q.get('is_correct', False))
        result['accuracy_when_no_correct'] = correct / len(without_correct)
    else:
        result['accuracy_when_no_correct'] = None
    
    return result


def format_pct(value: Optional[float], digits: int = 2) -> str:
    """Format a value as percentage string."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def generate_report(all_results: list[dict]) -> str:
    """Generate markdown report from all results."""
    lines = []
    lines.append("# Generative Judge Baseline Results\n")
    
    # Group by k value and mode
    grouped = defaultdict(list)
    for r in all_results:
        parsed = parse_run_name(r['name'])
        k = parsed.get('k', 'unknown')
        mode = parsed.get('mode', 'unknown')
        grouped[(k, mode)].append(r)
    
    # Summary table
    lines.append("## Summary by Configuration\n")
    lines.append("| Run | k | Mode | Comment | Accuracy | Random Baseline | Optimal UB | API Fail | Parse Fail |")
    lines.append("|-----|---|------|---------|----------|-----------------|------------|----------|------------|")
    
    for (k, mode), results in sorted(grouped.items()):
        for r in results:
            run_name = r['name']
            parsed = parse_run_name(run_name)
            comment = parsed.get('comment', '') or ''
            
            summary = r.get('summary', {})
            dry_run = r.get('dry_run', {})
            per_question = r.get('per_question', [])
            
            if summary:
                accuracy = format_pct(summary.get('accuracy'))
                api_fail = format_pct(summary.get('api_failure_rate', 0))
                parse_fail = format_pct(summary.get('parse_failure_rate', 0))
            else:
                accuracy = "N/A (dry run)"
                api_fail = "N/A"
                parse_fail = "N/A"
            
            # Random baseline
            if per_question:
                random_baseline = format_pct(compute_baseline_accuracy(per_question))
            else:
                random_baseline = "N/A"
            
            # Optimal upper bound
            if dry_run:
                optimal_ub = format_pct(dry_run.get('summary', {}).get('optimal_upper_bound_accuracy'))
            else:
                optimal_ub = "N/A"
            
            lines.append(f"| {run_name} | {k} | {mode} | {comment} | {accuracy} | {random_baseline} | {optimal_ub} | {api_fail} | {parse_fail} |")
    
    lines.append("")
    
    # Detailed results per run
    lines.append("## Detailed Results\n")
    
    for r in sorted(all_results, key=lambda x: x['name']):
        run_name = r['name']
        parsed = parse_run_name(run_name)
        lines.append(f"### {run_name}\n")
        lines.append(f"- **k**: {parsed.get('k', 'N/A')}")
        lines.append(f"- **Mode**: {parsed.get('mode', 'N/A')}")
        lines.append(f"- **Comment**: {parsed.get('comment', 'N/A')}")
        lines.append(f"- **Seed**: {parsed.get('seed', 'N/A')}")
        lines.append("")
        
        summary = r.get('summary', {})
        dry_run = r.get('dry_run', {})
        per_question = r.get('per_question', [])
        
        # Dry run stats
        if dry_run:
            dr_summary = dry_run.get('summary', {})
            lines.append("#### Dry Run Statistics\n")
            lines.append(f"- **k**: {dr_summary.get('k', 'N/A')}")
            lines.append(f"- **Max input tokens**: {dr_summary.get('max_input_tokens', 'N/A')}")
            lines.append(f"- **Seed**: {dr_summary.get('seed', 'N/A')}")
            lines.append(f"- **Questions**: {dr_summary.get('num_questions', 'N/A')}")
            
            traces = dr_summary.get('traces_selected', {})
            lines.append(f"- **Traces selected**: {traces.get('mean', 0):.1f} +/- {traces.get('std', 0):.1f} (range: {traces.get('min', 0)}-{traces.get('max', 0)})")
            
            tokens = dr_summary.get('tokens_used', {})
            lines.append(f"- **Tokens used**: {tokens.get('mean', 0):.0f} +/- {tokens.get('std', 0):.0f}")
            
            lines.append(f"- **Optimal upper bound accuracy**: {format_pct(dr_summary.get('optimal_upper_bound_accuracy'))}")
            lines.append("")
        
        # Actual run stats
        if summary:
            lines.append("#### Actual Run Statistics\n")
            lines.append(f"- **Model**: {summary.get('model_name', 'N/A')}")
            lines.append(f"- **Questions processed**: {summary.get('num_questions', 'N/A')}")
            lines.append(f"- **Correct**: {summary.get('correct_count', 'N/A')}")
            lines.append(f"- **Accuracy**: {format_pct(summary.get('accuracy'))}")
            lines.append(f"- **API failures**: {summary.get('api_failure_rate', 0):.4f}")
            lines.append(f"- **Parse failures**: {summary.get('parse_failures', 'N/A')} ({format_pct(summary.get('parse_failure_rate'))})")
            lines.append(f"- **Timestamp**: {summary.get('timestamp', 'N/A')}")
            lines.append("")
            
            # Breakdown
            if per_question and dry_run:
                breakdown = compute_breakdown_by_has_correct(per_question, dry_run)
                if breakdown:
                    lines.append("#### Accuracy Breakdown\n")
                    lines.append(f"- **Questions with correct trace available**: {breakdown.get('count_with_correct', 0)}")
                    lines.append(f"- **Accuracy when correct available**: {format_pct(breakdown.get('accuracy_when_correct_available'))}")
                    lines.append(f"- **Questions without correct trace**: {breakdown.get('count_without_correct', 0)}")
                    lines.append(f"- **Accuracy when no correct (should be 0%)**: {format_pct(breakdown.get('accuracy_when_no_correct'))}")
                    lines.append("")
            
            # Random baseline comparison
            if per_question:
                random_baseline = compute_baseline_accuracy(per_question)
                actual_acc = summary.get('accuracy', 0)
                improvement = actual_acc - random_baseline if actual_acc else 0
                lines.append("#### vs Random Baseline\n")
                lines.append(f"- **Random baseline (1/k)**: {format_pct(random_baseline)}")
                lines.append(f"- **Actual accuracy**: {format_pct(actual_acc)}")
                if random_baseline > 0:
                    lines.append(f"- **Improvement**: {format_pct(improvement)} ({improvement/random_baseline*100:.1f}x relative)")
                lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze gen_judge baseline results')
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing results subdirectories'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output markdown file (default: print to stdout)'
    )
    
    args = parser.parse_args()
    
    # Find all result directories
    result_dirs = []
    results_path = Path(args.results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {args.results_dir}")
        return
    
    for item in sorted(results_path.iterdir()):
        if item.is_dir():
            # Check if it has results files
            if (item / 'results_summary.json').exists() or (item / 'dry_run_stats.json').exists():
                result_dirs.append(str(item))
    
    if not result_dirs:
        print(f"No result directories found in {args.results_dir}")
        return
    
    print(f"Found {len(result_dirs)} result directories")
    
    # Load all results
    all_results = []
    for d in result_dirs:
        results = load_results(d)
        all_results.append(results)
    
    # Generate report
    report = generate_report(all_results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()
