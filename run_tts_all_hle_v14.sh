#!/bin/bash
# Run TTS v14 with forced budget mode
set -e

# Configuration
# Can be a single file, a glob pattern, or multiple files
# Single file example:
# RESPONSE_JSONL="/path/to/seed_0.jsonl"
# Glob pattern example (will load all matching files):
RESPONSE_JSONL="/Users/tim/Documents/research/trustworthy/output_rollouts/nemotron-nano-30b/hle_integer_100-test-temp_1.0-top_p_1.0-top_k_-1-seed_*-20251231-103834_a100_128k_hle.jsonl"
OUTPUT_DIR="./hle_full_v14_32_40_s42"
WORKERS=4
SEED=42
FORCE_BUDGET=32  # Force agent to spawn this many agents on first turn

# Set your NVIDIA API key here (or export it before running)
export MSWEA_MODEL_API_KEY="${MSWEA_MODEL_API_KEY:-nvapi-YOUR_KEY_HERE}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -b|--budget)
            FORCE_BUDGET="$2"
            shift 2
            ;;
        --redo)
            REDO="--redo-existing"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -w, --workers N    Number of parallel workers (default: 4)"
            echo "  -o, --output DIR   Output directory (default: ./hle_full_v14_32_40_s42)"
            echo "  -s, --seed N       Random seed (default: 42)"
            echo "  -b, --budget N     Forced budget for first turn (default: 32)"
            echo "  --redo             Redo existing questions"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running TTS v14 with forced budget mode..."
echo "  Response file: $RESPONSE_JSONL"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Workers:       $WORKERS"
echo "  Seed:          $SEED"
echo "  Forced budget: $FORCE_BUDGET"
echo ""

# Run TTS batch with forced budget
mini-extra tts-batch \
    "$RESPONSE_JSONL" \
    "$OUTPUT_DIR" \
    --config tts-v14.yaml \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --force-budget "$FORCE_BUDGET" \
    $REDO

echo ""
echo "Done! Check results:"
echo "  Predictions: $OUTPUT_DIR/preds.json"
echo "  Trajectories: $OUTPUT_DIR/question_*/*.traj.json"
