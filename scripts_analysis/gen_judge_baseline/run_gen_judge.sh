#!/bin/bash
# Launch script for Generative Judge Baseline with Fixed-k Sampling
#
# Usage:
#   ./run_gen_judge.sh --k 16 --mode non-dedup --comment c70k --seed 0
#   ./run_gen_judge.sh --k 32 --mode dedup --comment c70k --seed 0 --dry_run
#
# The comment parameter (e.g., c70k) will be appended to the output directory name:
#   results/k16_c70k_non-dedup_seed0/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default parameters
K=""
MODE=""
SEED=0
COMMENT=""
MAX_INPUT_TOKENS=""
DRY_RUN=""
MODEL="zhipuai/public-glm-4.7"
POOL_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --k)
            K="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --comment)
            COMMENT="$2"
            shift 2
            ;;
        --max_input_tokens)
            MAX_INPUT_TOKENS="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN="--dry_run"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --pool_path)
            POOL_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required Options:"
            echo "  --k N                    Number of traces to sample from pool"
            echo "  --mode MODE              Selection mode: dedup or non-dedup"
            echo ""
            echo "Optional Options:"
            echo "  --seed SEED              Random seed (default: 0)"
            echo "  --comment COMMENT        Comment to append to output dir (e.g., c70k)"
            echo "  --max_input_tokens N     Maximum tokens for input context"
            echo "  --dry_run                Only compute stats without API calls"
            echo "  --model MODEL            LiteLLM model name (default: zhipuai/public-glm-4.7)"
            echo "  --pool_path PATH         Path to response pool (default: auto-detect)"
            echo ""
            echo "Examples:"
            echo "  $0 --k 16 --mode non-dedup --comment c70k --seed 0"
            echo "  $0 --k 32 --mode dedup --comment c70k --seed 0 --dry_run"
            echo "  $0 --k 8 --mode non-dedup --comment c70k --max_input_tokens 70000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$K" ]]; then
    echo "Error: --k is required"
    exit 1
fi

if [[ -z "$MODE" ]]; then
    echo "Error: --mode is required (dedup or non-dedup)"
    exit 1
fi

# Auto-detect pool path if not provided
if [[ -z "$POOL_PATH" ]]; then
    # Try to find pool in standard locations
    if [[ -d "${PROJECT_ROOT}/scripts_initial_generation_output" ]]; then
        # Look for GLM-4.7 rollouts
        POOL_PATTERN="${PROJECT_ROOT}/scripts_initial_generation_output/glm-4p7/*.jsonl"
        if compgen -G "$POOL_PATTERN" > /dev/null; then
            POOL_PATH="$POOL_PATTERN"
        fi
    fi
    
    # Fallback to output_rollouts
    if [[ -z "$POOL_PATH" && -d "${PROJECT_ROOT}/../trustworthy/output_rollouts" ]]; then
        POOL_PATTERN="${PROJECT_ROOT}/../trustworthy/output_rollouts/glm-4p7/*.jsonl"
        if compgen -G "$POOL_PATTERN" > /dev/null; then
            POOL_PATH="$POOL_PATTERN"
        fi
    fi
fi

if [[ -z "$POOL_PATH" ]]; then
    echo "Error: Could not auto-detect pool path. Please specify with --pool_path"
    exit 1
fi

# Build output directory path
OUTPUT_DIR="${SCRIPT_DIR}/../scripts_analysis_output/gen_judge_baseline"

# Build comment argument
COMMENT_ARG=""
if [[ -n "$COMMENT" ]]; then
    COMMENT_ARG="--comment $COMMENT"
fi

# Build max_input_tokens argument
TOKENS_ARG=""
if [[ -n "$MAX_INPUT_TOKENS" ]]; then
    TOKENS_ARG="--max_input_tokens $MAX_INPUT_TOKENS"
fi

# Display configuration
echo "=========================================="
echo "Generative Judge Baseline"
echo "=========================================="
echo "k: $K"
echo "Mode: $MODE"
echo "Seed: $SEED"
if [[ -n "$COMMENT" ]]; then
    echo "Comment: $COMMENT"
fi
if [[ -n "$MAX_INPUT_TOKENS" ]]; then
    echo "Max input tokens: $MAX_INPUT_TOKENS"
fi
if [[ -n "$DRY_RUN" ]]; then
    echo "Mode: DRY RUN (no API calls)"
else
    echo "Model: $MODEL"
fi
echo "Pool path: $POOL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

# Activate virtual environment
source "${PROJECT_ROOT}/.venv/bin/activate" 2>/dev/null || source "${PROJECT_ROOT}/.py3.10/bin/activate" 2>/dev/null || true

# Run the script
python "${SCRIPT_DIR}/step1_gen_judge.py" \
    --pool_path "$POOL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --k "$K" \
    --mode "$MODE" \
    --seed "$SEED" \
    $COMMENT_ARG \
    $TOKENS_ARG \
    $DRY_RUN \
    --model "$MODEL"

echo "=========================================="
echo "Done!"
echo "=========================================="
