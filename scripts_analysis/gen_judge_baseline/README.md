# Generative Judge Baseline

This directory contains the generative judge baseline implementation for Test-Time Scaling (TTS) evaluation. It uses a judge model (via LiteLLM) to select the best answer from k sampled traces.

## Overview

The baseline works by:
1. Sampling k traces from a pre-computed response pool
2. Optionally filtering by token budget (`max_input_tokens`)
3. Using a judge model to select the best trace
4. Comparing against random baseline and optimal upper bound

## Modes

- **dedup**: Keep only one trace per unique answer group (diverse selection)
- **non-dedup**: Keep all sampled traces regardless of answer

## Directory Structure

```
scripts_analysis/gen_judge_baseline/
├── step1_gen_judge.py      # Main selection script
├── step2_analyze.py        # Results analysis
├── run_gen_judge.sh        # Launch script
└── README.md              # This file

scripts_analysis_output/gen_judge_baseline/
├── k{N}_{comment}_{mode}_seed{S}/   # Results directories
│   ├── results_summary.json
│   ├── results_per_question.json
│   ├── dry_run_stats.json
│   └── selector_traces.jsonl
└── ...
```

## Usage

### Quick Start

```bash
# Dry run to compute statistics without API calls
./run_gen_judge.sh --k 16 --mode non-dedup --comment c70k --seed 0 --dry_run

# Actual run with API calls
./run_gen_judge.sh --k 16 --mode non-dedup --comment c70k --seed 0

# With token budget
./run_gen_judge.sh --k 32 --mode dedup --comment c70k --max_input_tokens 70000 --seed 0
```

### Direct Python Usage

```bash
# Dry run
python step1_gen_judge.py \
    --pool_path "/path/to/rollouts/*.jsonl" \
    --output_dir "./results" \
    --k 16 \
    --mode non-dedup \
    --seed 0 \
    --comment c70k \
    --dry_run

# Actual run
python step1_gen_judge.py \
    --pool_path "/path/to/rollouts/*.jsonl" \
    --output_dir "./results" \
    --k 16 \
    --mode non-dedup \
    --seed 0 \
    --comment c70k \
    --model "zhipuai/public-glm-4.7"
```

### Analysis

```bash
# Generate markdown report
python step2_analyze.py \
    --results_dir ./scripts_analysis_output/gen_judge_baseline \
    --output report.md
```

## Parameters

### Required
- `--k`: Number of traces to sample from pool
- `--mode`: Selection mode (`dedup` or `non-dedup`)

### Optional
- `--seed`: Random seed (default: 42)
- `--comment`: Comment appended to output dir (e.g., `c70k`)
- `--max_input_tokens`: Token budget for filtering after sampling
- `--dry_run`: Compute stats without API calls
- `--model`: LiteLLM model name (default: `zhipuai/public-glm-4.7`)
- `--retry`: Retry failed API calls from existing results
- `--max-retries`: Maximum retry attempts per API call (default: 5)
- `--concurrency`: Number of concurrent API calls (default: 1)

## Output Directory Naming

Output directories follow the pattern:
```
k{K}_{comment}_{mode}_seed{SEED}
```

Examples:
- `k16_c70k_non-dedup_seed0`
- `k32_c70k_dedup_seed42`
- `k8_non-dedup_seed0` (no comment)

## Ported Results

The following baseline results have been ported from the original implementation:

- `c70k_non-dedup_seed0`: Previously `70k_random_seed0` (random mode)
- `c70k_dedup_seed0`: Previously `70k_diverse_seed0` (diverse mode)

These use the token-based selection from the original baseline and are documented with the `c70k` comment to indicate they were run with approximately 70k token budget.

## Key Differences from Original Baseline

1. **Fixed-k sampling**: Sample exactly k traces first, then filter by tokens (vs. filling by tokens)
2. **ResponsePool integration**: Uses ATTS ResponsePool for consistent sampling
3. **LiteLLM backend**: Replaces custom API client with LiteLLM (with retry logic)
4. **Renamed modes**: `random` → `non-dedup`, `diverse` → `dedup`
5. **Comment support**: Output directories include optional comment (e.g., `c70k`)

## Retry Logic

The LiteLLM backend is configured with:
- Maximum 5 retry attempts per API call (configurable via `--max-retries`)
- Exponential backoff (1s, 2s, 4s, 8s, 16s delays)
- Specific exceptions that don't trigger retry (auth errors, context window exceeded, etc.)

### Retry Mode for Failed Questions

If some API calls fail during a run, you can retry only the failed questions without re-processing successful ones:

```bash
# Retry failed questions for a specific experiment
python step1_gen_judge.py \
    --pool_path "/path/to/rollouts/*.jsonl" \
    --output_dir "./results" \
    --k 16 \
    --mode non-dedup \
    --seed 42 \
    --comment c131k \
    --retry

# Using the batch launch script with retry mode
RETRY=1 ./launch_batch_experiments.sh
RETRY=1 ./launch_batch_experiments_nondedup.sh
```

**How it works:**
- Loads existing `results_per_question.json` and identifies questions with `api_success: false`
- Regenerates prompts for failed questions only (using same seed for consistent sampling)
- Re-calls API for failed questions
- Merges new results with existing successful results
- Rewrites output files with updated statistics

**Notes:**
- Safe to run multiple times (idempotent)
- Won't re-process already successful questions
- If no failures found, exits with "Nothing to retry"
