#!/bin/bash
# Test TTS on a single question
set -e

# Configuration
RESPONSE_JSONL="/Users/tim/Documents/research/trustworthy/output_rollouts/nemotron-nano-30b/aime2025-test-temp_1.0-top_p_1.0-top_k_-1-seed_0-20251231-102840_a100_128k.jsonl"
OUTPUT_DIR="./tts_test_output_v4"
QUESTION_ID="12"

# Set your NVIDIA API key here (or export it before running)
export MSWEA_MODEL_API_KEY="${MSWEA_MODEL_API_KEY:-nvapi-YOUR_KEY_HERE}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running TTS on question $QUESTION_ID..."
echo "Output will be saved to: $OUTPUT_DIR"
echo ""

# Run TTS batch on single question
mini-extra tts-batch \
    "$RESPONSE_JSONL" \
    "$OUTPUT_DIR" \
    --questions "$QUESTION_ID" \
    --config tts.yaml \
    --seed 42

echo ""
echo "Done! Check results:"
echo "  Predictions: $OUTPUT_DIR/preds.json"
echo "  Trajectory:  $OUTPUT_DIR/question_${QUESTION_ID}/${QUESTION_ID}.traj.json"
