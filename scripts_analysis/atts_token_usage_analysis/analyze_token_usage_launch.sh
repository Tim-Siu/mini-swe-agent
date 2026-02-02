#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-hle_full_v23_glm4p7_think_f32_40_s42}"
ATTS_DIR="atts_rollout/${RUN_NAME}"
OUTPUT_BASE="scripts_analysis_output/atts_token_usage_analysis"

python scripts_analysis/atts_token_usage_analysis/analyze_token_usage.py \
  --atts-run-dir "${ATTS_DIR}" \
  --output-dir "${OUTPUT_BASE}" \
  --run-name "${RUN_NAME}"
