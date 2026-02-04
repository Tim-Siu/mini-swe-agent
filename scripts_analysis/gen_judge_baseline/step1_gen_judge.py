#!/usr/bin/env python3
"""
Generative Judge Baseline with Fixed-k Sampling.

Uses a judge model (via LiteLLM) to perform listwise selection across k sampled traces.
Similar to ATTS implementation, samples k traces from the pool first, then optionally
filters by token budget.

Supports two modes:
- dedup: Keep only one trace per unique answer group
- non-dedup: Keep all sampled traces regardless of answer

Usage:
    # Dry run (compute stats without API calls)
    python step1_gen_judge.py \
        --pool_path "/path/to/rollouts/*.jsonl" \
        --output_dir "./results" \
        --k 16 \
        --mode non-dedup \
        --seed 0 \
        --comment c70k \
        --dry_run

    # Actual run with LiteLLM
    python step1_gen_judge.py \
        --pool_path "/path/to/rollouts/*.jsonl" \
        --output_dir "./results" \
        --k 16 \
        --mode non-dedup \
        --seed 0 \
        --comment c70k \
        --model "zhipuai/public-glm-4.7" \
        --max_input_tokens 70000
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.utils.response_pool import ResponsePool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_post_think_content(response: str) -> str:
    """Extract content after </think> tag, or return full content if no tag."""
    marker = "</think>"
    idx = response.find(marker)
    if idx >= 0:
        return response[idx + len(marker):].strip()
    return response.strip()


def extract_answer(response: str) -> str:
    """Extract the last \\boxed{} answer from a response."""
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, response)
    return matches[-1] if matches else "NO_ANSWER"


def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens in text. Uses simple approximation if no tokenizer provided."""
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Simple approximation: ~4 characters per token
    return len(text) // 4


def deduplicate_by_answer(
    traces: list[tuple[str, bool]],
    rng,
) -> list[tuple[str, bool]]:
    """Keep only one trace per unique answer.
    
    Args:
        traces: List of (response_text, label) tuples
        rng: Random number generator for shuffling within groups
    
    Returns:
        List with one trace per unique answer
    """
    # Group by extracted answer
    groups = defaultdict(list)
    for resp, label in traces:
        answer = extract_answer(resp)
        groups[answer].append((resp, label))
    
    # Shuffle within each group and take one
    result = []
    for answer in groups:
        group = groups[answer]
        rng.shuffle(group)
        result.append(group[0])
    
    return result


def filter_by_token_count(
    traces: list[tuple[str, bool]],
    max_tokens: int,
    tokenizer=None,
) -> list[tuple[str, bool]]:
    """Filter traces to fit within token budget.
    
    Args:
        traces: List of (response_text, label) tuples
        max_tokens: Maximum tokens allowed
        tokenizer: Optional tokenizer for accurate counting
    
    Returns:
        Traces that fit within budget (in original order)
    """
    selected = []
    current_tokens = 0
    
    for resp, label in traces:
        post_think = extract_post_think_content(resp)
        tokens = count_tokens(post_think, tokenizer)
        
        if current_tokens + tokens <= max_tokens:
            selected.append((resp, label))
            current_tokens += tokens
    
    return selected


def build_judge_prompt(question: str, candidates: list[tuple[str, bool]]) -> str:
    """Build the judge prompt for selecting the best trace.
    
    Args:
        question: The question text
        candidates: List of (response_text, label) tuples
    
    Returns:
        Formatted prompt string
    """
    num_candidates = len(candidates)
    
    # Build candidate responses section with numerical indices
    candidate_texts = []
    for idx, (resp, _) in enumerate(candidates, start=1):
        post_think = extract_post_think_content(resp)
        if not post_think:
            answer = extract_answer(resp)
            post_think = f"[Empty response - extracted answer: {answer}]"
        candidate_texts.append(f"Response {idx}:\n{post_think}\n")
    
    responses_section = "\n".join(candidate_texts)
    index_range = f"1-{num_candidates}" if num_candidates > 1 else "1"
    
    prompt = f"""You are given {num_candidates} attempts to answer a reasoning question. Each attempt provides a response with reasoning. Your task is to pick the best attempt.

Question: {question}

{responses_section}

Instructions:
- Carefully review all attempts and their reasoning
- Respond with \\boxed{{N}} where N is the attempt number ({index_range})
- You must respond with exactly one choice in the format \\boxed{{N}}"""

    return prompt


def parse_judge_choice(judge_response: str, num_options: int, fallback_seed: int = None) -> tuple[int, bool]:
    """Extract numerical choice from judge response.
    
    Args:
        judge_response: The LLM judge's response text
        num_options: Number of options (1 to num_options)
        fallback_seed: Optional seed for deterministic fallback
    
    Returns:
        Tuple of (choice index 0-based, parse_failed flag)
    """
    valid_choices = set(range(1, num_options + 1))
    
    # Strategy 1: Look for \boxed{N} pattern
    match = re.search(r'\\boxed\{(\d+)\}', judge_response)
    if match:
        choice = int(match.group(1))
        if choice in valid_choices:
            return choice - 1, False
    
    # Strategy 2: Look for "Response N" or similar patterns in last few lines
    lines = judge_response.strip().split('\n')
    for line in reversed(lines[-10:]):
        match = re.search(r'(?:Response|Attempt|Choice|Answer|Option)\s*(\d+)', line, re.IGNORECASE)
        if match:
            choice = int(match.group(1))
            if choice in valid_choices:
                return choice - 1, False
    
    # Strategy 3: Find any valid number in the last line
    last_line = lines[-1] if lines else ''
    numbers = re.findall(r'\b(\d+)\b', last_line)
    for num_str in numbers:
        choice = int(num_str)
        if choice in valid_choices:
            return choice - 1, False
    
    # Strategy 4: Find last occurrence of any valid number in entire response
    all_numbers = re.findall(r'\b(\d+)\b', judge_response)
    for num_str in reversed(all_numbers):
        choice = int(num_str)
        if choice in valid_choices:
            return choice - 1, False
    
    # Fallback: Random choice (deterministic if seed provided)
    import random
    if fallback_seed is not None:
        rng = random.Random(fallback_seed)
        choice_idx = rng.randint(0, num_options - 1)
    else:
        choice_idx = random.randint(0, num_options - 1)
    
    return choice_idx, True


def run_dry_run(
    response_pool: ResponsePool,
    k: int,
    mode: str,
    max_input_tokens: int | None,
    seed: int,
    output_dir: str,
    comment: str | None,
):
    """Run in dry-run mode: compute stats without API calls.
    
    Args:
        response_pool: Loaded ResponsePool instance
        k: Number of traces to sample
        mode: 'dedup' or 'non-dedup'
        max_input_tokens: Optional token budget
        seed: Random seed
        output_dir: Base output directory
        comment: Optional comment to append to output dir name
    """
    # Create output directory name
    mode_str = mode
    dir_name = f"k{k}_{mode_str}_seed{seed}"
    if comment:
        dir_name = f"k{k}_{comment}_{mode_str}_seed{seed}"
    full_output_dir = Path(output_dir) / dir_name
    full_output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = response_pool.rng
    stats = []
    
    question_ids = response_pool.get_question_ids()
    
    for qid in tqdm(question_ids, desc="Dry run"):
        # Sample k traces from pool
        sampled = response_pool.sample(qid, k)
        
        # Apply mode
        if mode == "dedup":
            sampled = deduplicate_by_answer(sampled, rng)
        
        # Apply token budget if specified
        if max_input_tokens is not None:
            sampled = filter_by_token_count(sampled, max_input_tokens)
        
        # Compute stats
        num_unique_answers = len(set(extract_answer(resp) for resp, _ in sampled))
        has_correct = any(label for _, label in sampled)
        optimal_upper_bound = 1.0 if has_correct else 0.0
        
        # Count tokens
        total_tokens = sum(
            count_tokens(extract_post_think_content(resp))
            for resp, _ in sampled
        )
        
        stat = {
            'question_id': qid,
            'num_traces_selected': len(sampled),
            'total_tokens_used': total_tokens,
            'num_unique_answers': num_unique_answers,
            'has_correct_trace': has_correct,
            'optimal_upper_bound': optimal_upper_bound,
        }
        stats.append(stat)
    
    # Compute summary
    num_questions = len(stats)
    traces_selected = [s['num_traces_selected'] for s in stats]
    tokens_used = [s['total_tokens_used'] for s in stats]
    optimal_bounds = [s['optimal_upper_bound'] for s in stats]
    
    summary = {
        'mode': mode,
        'k': k,
        'max_input_tokens': max_input_tokens,
        'seed': seed,
        'comment': comment,
        'num_questions': num_questions,
        'traces_selected': {
            'mean': sum(traces_selected) / num_questions,
            'min': min(traces_selected),
            'max': max(traces_selected),
            'std': (sum((x - sum(traces_selected)/num_questions)**2 for x in traces_selected) / num_questions) ** 0.5,
        },
        'tokens_used': {
            'mean': sum(tokens_used) / num_questions,
            'min': min(tokens_used),
            'max': max(tokens_used),
            'std': (sum((x - sum(tokens_used)/num_questions)**2 for x in tokens_used) / num_questions) ** 0.5,
        },
        'optimal_upper_bound_accuracy': sum(optimal_bounds) / num_questions,
    }
    
    # Save results
    with open(full_output_dir / 'dry_run_stats.json', 'w') as f:
        json.dump({
            'summary': summary,
            'per_question': stats,
        }, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("DRY RUN SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode}")
    logger.info(f"k: {k}")
    logger.info(f"Max input tokens: {max_input_tokens}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Comment: {comment}")
    logger.info(f"Number of questions: {num_questions}")
    logger.info(f"Traces selected: {summary['traces_selected']['mean']:.1f} +/- {summary['traces_selected']['std']:.1f} (range: {summary['traces_selected']['min']}-{summary['traces_selected']['max']})")
    logger.info(f"Tokens used: {summary['tokens_used']['mean']:.0f} +/- {summary['tokens_used']['std']:.0f}")
    logger.info(f"Optimal upper bound accuracy: {summary['optimal_upper_bound_accuracy']:.4f}")
    logger.info("=" * 60)
    
    return summary


def run_actual(
    response_pool: ResponsePool,
    k: int,
    mode: str,
    max_input_tokens: int | None,
    seed: int,
    output_dir: str,
    comment: str | None,
    model_name: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 131072,
    enable_thinking: bool = True,
):
    """Run actual selection with LiteLLM API calls.
    
    Args:
        response_pool: Loaded ResponsePool instance
        k: Number of traces to sample
        mode: 'dedup' or 'non-dedup'
        max_input_tokens: Optional token budget
        seed: Random seed
        output_dir: Base output directory
        comment: Optional comment to append to output dir name
        model_name: LiteLLM model name (e.g., "zhipuai/public-glm-4.7")
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Max tokens for judge response
        enable_thinking: Whether to enable thinking mode
    """
    # Create output directory name
    mode_str = mode
    dir_name = f"k{k}_{mode_str}_seed{seed}"
    if comment:
        dir_name = f"k{k}_{comment}_{mode_str}_seed{seed}"
    full_output_dir = Path(output_dir) / dir_name
    full_output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = response_pool.rng
    
    # Initialize LiteLLM model with retry logic
    # Hardcode max 3 retry attempts via environment variable
    os.environ['MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT'] = '3'
    
    model_kwargs = {
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
    }
    
    # Add thinking parameter if supported (GLM-4.7 specific)
    if enable_thinking and 'glm' in model_name.lower():
        model_kwargs['extra_body'] = {'enable_thinking': True}
    
    judge_model = LitellmModel(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )
    
    # Phase 1: Prepare all prompts
    logger.info("Phase 1: Preparing prompts for all questions...")
    question_ids = response_pool.get_question_ids()
    prompts = []
    metadata = []
    
    for qid in tqdm(question_ids, desc="Preparing prompts"):
        # Sample k traces from pool
        sampled = response_pool.sample(qid, k)
        
        # Apply mode
        if mode == "dedup":
            sampled = deduplicate_by_answer(sampled, rng)
        
        # Apply token budget if specified
        if max_input_tokens is not None:
            sampled = filter_by_token_count(sampled, max_input_tokens)
        
        if not sampled:
            logger.warning(f"No traces selected for question {qid}")
            continue
        
        # Build prompt
        question_text = response_pool.get_question(qid)
        prompt = build_judge_prompt(question_text, sampled)
        prompts.append(prompt)
        
        metadata.append({
            'question_id': qid,
            'sampled': sampled,
            'gold_answer': response_pool.get_gold_answer(qid),
        })
    
    logger.info(f"Prepared {len(prompts)} prompts")
    
    # Phase 2: Call judge model
    logger.info("Phase 2: Calling judge model...")
    results = []
    correct_count = 0
    total_count = 0
    parse_failures = 0
    
    with open(full_output_dir / 'selector_traces.jsonl', 'w') as traces_file:
        for idx, (meta, prompt) in enumerate(tqdm(zip(metadata, prompts), total=len(prompts), desc="Judging")):
            qid = meta['question_id']
            sampled = meta['sampled']
            gold_answer = meta['gold_answer']
            
            # Call judge model
            try:
                response = judge_model.query([
                    {'role': 'user', 'content': prompt}
                ])
                judge_content = response['content']
                api_success = True
            except Exception as e:
                logger.warning(f"API call failed for question {qid}: {e}")
                api_success = False
                judge_content = ""
            
            # Parse response
            if api_success:
                choice_idx, parse_failed = parse_judge_choice(
                    judge_content,
                    len(sampled),
                    fallback_seed=seed + qid
                )
                if parse_failed:
                    parse_failures += 1
            else:
                # API failure - random select
                import random
                choice_idx = random.randint(0, len(sampled) - 1)
                parse_failed = True
            
            # Get selected trace
            if choice_idx >= len(sampled):
                choice_idx = 0
            chosen_resp, chosen_label = sampled[choice_idx]
            chosen_answer = extract_answer(chosen_resp)
            
            # Check if correct
            is_correct = response_pool.is_correct_answer(qid, chosen_answer)
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Record result
            result = {
                'question_id': qid,
                'num_candidates': len(sampled),
                'selected_choice': choice_idx + 1,
                'selected_idx': choice_idx,
                'is_correct': is_correct,
                'api_success': api_success,
                'parse_failed': parse_failed,
                'selected_answer': chosen_answer,
                'gold_answer': gold_answer,
            }
            results.append(result)
            
            # Save full trace
            trace_record = {
                'question_id': qid,
                'prompt': prompt,
                'response': judge_content if api_success else None,
                'candidates': [
                    {
                        'idx': i,
                        'label': i + 1,
                        'answer': extract_answer(resp),
                        'is_correct': response_pool.is_correct_answer(qid, extract_answer(resp)),
                    }
                    for i, (resp, _) in enumerate(sampled)
                ],
                'result': result,
            }
            traces_file.write(json.dumps(trace_record) + '\n')
    
    # Compute summary
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    summary = {
        'mode': mode,
        'k': k,
        'max_input_tokens': max_input_tokens,
        'seed': seed,
        'comment': comment,
        'model_name': model_name,
        'num_questions': total_count,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'api_failure_rate': (total_count - sum(1 for r in results if r['api_success'])) / total_count if total_count > 0 else 0.0,
        'parse_failures': parse_failures,
        'parse_failure_rate': parse_failures / total_count if total_count > 0 else 0.0,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save summary
    with open(full_output_dir / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save per-question results
    with open(full_output_dir / 'results_per_question.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode}")
    logger.info(f"k: {k}")
    logger.info(f"Max input tokens: {max_input_tokens}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Comment: {comment}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Questions processed: {total_count}")
    logger.info(f"Correct: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"API failures: {total_count - sum(1 for r in results if r['api_success'])}")
    logger.info(f"Parse failures: {parse_failures} ({summary['parse_failure_rate']:.4f})")
    logger.info("=" * 60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Generative Judge Baseline with Fixed-k Sampling')
    parser.add_argument(
        '--pool_path',
        type=str,
        required=True,
        help='Path to response pool JSONL file(s). Can be glob pattern.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Base output directory'
    )
    parser.add_argument(
        '--k',
        type=int,
        required=True,
        help='Number of traces to sample from pool'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['dedup', 'non-dedup'],
        required=True,
        help='Selection mode: dedup (one per answer) or non-dedup (keep all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--max_input_tokens',
        type=int,
        default=None,
        help='Maximum tokens for input context (optional, filters after sampling)'
    )
    parser.add_argument(
        '--comment',
        type=str,
        default=None,
        help='Comment to append to output directory name (e.g., c70k)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only compute stats without API calls'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='zhipuai/public-glm-4.7',
        help='LiteLLM model name (default: zhipuai/public-glm-4.7)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help='Top-p sampling parameter'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=131072,
        help='Max tokens for judge response'
    )
    parser.add_argument(
        '--enable_thinking',
        action='store_true',
        default=True,
        help='Enable thinking mode for judge model'
    )
    
    args = parser.parse_args()
    
    # Load response pool
    logger.info(f"Loading response pool from: {args.pool_path}")
    response_pool = ResponsePool(args.pool_path, seed=args.seed)
    logger.info(f"Loaded {len(response_pool)} questions")
    
    if args.dry_run:
        run_dry_run(
            response_pool,
            args.k,
            args.mode,
            args.max_input_tokens,
            args.seed,
            args.output_dir,
            args.comment,
        )
    else:
        run_actual(
            response_pool,
            args.k,
            args.mode,
            args.max_input_tokens,
            args.seed,
            args.output_dir,
            args.comment,
            args.model,
            args.temperature,
            args.top_p,
            args.max_tokens,
            args.enable_thinking,
        )


if __name__ == '__main__':
    main()
