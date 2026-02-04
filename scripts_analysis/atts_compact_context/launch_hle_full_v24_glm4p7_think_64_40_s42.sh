#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="hle_full_v24_glm4p7_think_64_40_s42"
ATTS_DIR="atts_rollout/${RUN_NAME}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-scripts_analysis_output/atts_compact_context/${RUN_NAME}__${TS}}"

MODEL="${MODEL:-openai/z-ai/glm4.7}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-131072}"
API_BASE="${API_BASE:-https://integrate.api.nvidia.com/v1}"
API_KEY="${API_KEY:-}"
CONCURRENCY="${CONCURRENCY:-4}"
LITELLM_VERBOSE="${LITELLM_VERBOSE:-1}"

mkdir -p "$OUT_DIR"

PYTHONPATH=src python scripts_analysis/atts_compact_context/step1_build_compacted.py \
  --atts-dir "$ATTS_DIR" \
  --output-dir "$OUT_DIR"

EXTRA_BODY_ARGS=()
if [[ -n "${EXTRA_BODY:-}" ]]; then
  EXTRA_BODY_ARGS=(--extra-body "$EXTRA_BODY")
fi

API_ARGS=()
if [[ -n "$API_BASE" ]]; then
  API_ARGS+=(--api-base "$API_BASE")
fi
if [[ -n "$API_KEY" ]]; then
  API_ARGS+=(--api-key "$API_KEY")
fi

VERBOSE_ARGS=()
if [[ "$LITELLM_VERBOSE" == "1" ]]; then
  VERBOSE_ARGS+=(--litellm-verbose)
fi

PYTHONPATH=src python scripts_analysis/atts_compact_context/step2_eval_compacted.py \
  --compacted-file "$OUT_DIR/compacted_inputs.jsonl" \
  --output-dir "$OUT_DIR" \
  --model "$MODEL" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --concurrency "$CONCURRENCY" \
  --resume \
  ${API_ARGS[@]+"${API_ARGS[@]}"} \
  ${VERBOSE_ARGS[@]+"${VERBOSE_ARGS[@]}"} \
  ${EXTRA_BODY_ARGS[@]+"${EXTRA_BODY_ARGS[@]}"}
