#!/bin/bash

# ATTS Main Analysis Launch Script
# Analyzes: /Users/tim/Documents/research/mini-swe-agent/atts_rollout/hle_full_v24_glm4p7_think_64_40_s42
#
# This script runs three steps:
#   Step 1: Majority Voting vs Pass@k vs ATTS Agent (accuracy comparison)
#   Step 2: Cost-based comparison (budget on x-axis)
#   Step 3: Generate tables with accuracy and budget

set -e  # Exit on error

# Configuration
ATTS_DIR="/Users/tim/Documents/research/mini-swe-agent/atts_rollout/hle_full_v24_glm4p7_think_64_40_s42"
OUTPUT_DIR="/Users/tim/Documents/research/mini-swe-agent/scripts_analysis_output/atts_main_analysis/hle_full_v24_glm4p7_think_64_40_s42"

# Rollout files (for pass@k and majority voting)
ROLLOUT_FILES=(
    "/Users/tim/Documents/research/trustworthy/output_rollouts/glm-4p7/hle_integer_100-test-temp_1.0-top_p_0.95-top_k_-1-seed_0-thinking-20260122-233312_seed_0.jsonl"
    "/Users/tim/Documents/research/trustworthy/output_rollouts/glm-4p7/hle_integer_100-test-temp_1.0-top_p_0.95-top_k_-1-seed_4-thinking-20260124-020401_seed_4.jsonl"
)

# Token counts files (corresponding to rollout files)
TOKEN_COUNTS_FILES=(
    "/Users/tim/Documents/research/trustworthy/output_rollouts/glm-4p7/hle_integer_100-test-temp_1.0-top_p_0.95-top_k_-1-seed_0-thinking-20260122-233312_seed_0.jsonl.token_counts.json"
    "/Users/tim/Documents/research/trustworthy/output_rollouts/glm-4p7/hle_integer_100-test-temp_1.0-top_p_0.95-top_k_-1-seed_4-thinking-20260124-020401_seed_4.jsonl.token_counts.json"
)

# K values to evaluate
K_VALUES="1,2,4,8,16,32,64"

# Python interpreter
PYTHON="/Users/tim/Documents/research/mini-swe-agent/.venv/bin/python"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================================"
echo "ATTS Main Analysis"
echo "================================================================================"
echo "ATTS Directory: $ATTS_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "K Values: $K_VALUES"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Majority Voting vs Pass@k vs ATTS Agent
echo "================================================================================"
echo "Step 1: Majority Voting vs Pass@k vs ATTS Agent"
echo "================================================================================"
$PYTHON "$SCRIPT_DIR/step1_analysis.py" \
    --rollout-files "${ROLLOUT_FILES[@]}" \
    --atts-dir "$ATTS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --k-values "$K_VALUES" \
    --seed 42

echo ""
echo "Step 1 complete. Results saved to: $OUTPUT_DIR/step1_comparison.png"
echo ""

# Step 2: Cost-based Comparison
echo "================================================================================"
echo "Step 2: Cost-based Comparison"
echo "================================================================================"
$PYTHON "$SCRIPT_DIR/step2_cost_analysis.py" \
    --rollout-files "${ROLLOUT_FILES[@]}" \
    --token-counts-files "${TOKEN_COUNTS_FILES[@]}" \
    --atts-dir "$ATTS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --k-values "$K_VALUES" \
    --seed 42

echo ""
echo "Step 2 complete. Results saved to: $OUTPUT_DIR/step2_cost_comparison.png"
echo ""

# Step 3: Generate Tables
echo "================================================================================"
echo "Step 3: Generate Tables"
echo "================================================================================"
$PYTHON "$SCRIPT_DIR/step3_tables.py" \
    --step1-results "$OUTPUT_DIR/step1_results.json" \
    --step2-results "$OUTPUT_DIR/step2_results.json" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Step 3 complete. Tables saved to: $OUTPUT_DIR/step3_tables.md"
echo ""

echo "================================================================================"
echo "All Steps Complete!"
echo "================================================================================"
echo "Output files:"
echo "  - $OUTPUT_DIR/step1_comparison.png"
echo "  - $OUTPUT_DIR/step2_cost_comparison.png"
echo "  - $OUTPUT_DIR/step3_tables.md"
echo "  - $OUTPUT_DIR/accuracy_table.csv"
echo "  - $OUTPUT_DIR/budget_table.csv"
echo "================================================================================"
