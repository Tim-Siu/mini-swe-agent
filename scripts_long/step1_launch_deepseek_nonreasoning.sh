#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=".venv/bin/python"
SCRIPT="scripts_long/step1_run_multiturn_long.py"

MODEL="deepseek/deepseek-v3.2"
REASONING_ENABLED="false"
PROVIDER_ORDER_JSON='[{"order":["deepseek"],"allow_fallbacks":false},{"order":["novita"],"quantizations":["fp8"],"allow_fallbacks":false},{"order":["fireworks"],"allow_fallbacks":false},{"order":["together"],"allow_fallbacks":false}]'

DATASET="${DATASET:-MathArena/aime_2025}"
SPLIT="${SPLIT:-train}"
ROUNDS="${ROUNDS:-3}"
MAX_WORKERS="${MAX_WORKERS:-30}"
SEED="${SEED:-42}"
MAX_RETRIES="${MAX_RETRIES:-4}"
RETRY_BACKOFF_SEC="${RETRY_BACKOFF_SEC:-2.0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-32000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"

if [[ -n "${RESUME_DIR:-}" ]]; then
  OUTPUT_DIR="$RESUME_DIR"
elif [[ -n "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$OUTPUT_DIR"
else
  TS="$(date +%Y%m%d_%H%M%S)"
  DATASET_ESCAPED="${DATASET//\//__}"
  MODEL_ESCAPED="${MODEL//\//__}"
  OUTPUT_DIR="scripts_long_output/${MODEL_ESCAPED}/${DATASET_ESCAPED}/run_rounds_${ROUNDS}_seed${SEED}_${TS}"
fi

if [[ "${HF_DOWNLOAD_DATASET:-0}" == "1" ]]; then
  HF_LOCAL_DIR="tmp/hf_dataset/${DATASET//\//__}"
  hf download "$DATASET" --repo-type dataset --local-dir "$HF_LOCAL_DIR" --quiet
fi

CMD=( "$PYTHON_BIN" "$SCRIPT" \
  --model "$MODEL" \
  --provider-order-json "$PROVIDER_ORDER_JSON" \
  --reasoning-enabled "$REASONING_ENABLED" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --rounds "$ROUNDS" \
  --max-workers "$MAX_WORKERS" \
  --seed "$SEED" \
  --max-retries "$MAX_RETRIES" \
  --retry-backoff-sec "$RETRY_BACKOFF_SEC" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --output-dir "$OUTPUT_DIR" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" )

if [[ -n "${CONTEXT_SAFETY_MARGIN:-}" ]]; then
  CMD+=( --context-safety-margin "$CONTEXT_SAFETY_MARGIN" )
fi
if [[ "${DRY_RUN_PLAN:-0}" == "1" ]]; then
  CMD+=( --dry-run-plan )
fi

printf 'Running model=%s dataset=%s split=%s rounds=%s max_workers=%s\n' "$MODEL" "$DATASET" "$SPLIT" "$ROUNDS" "$MAX_WORKERS"
printf 'Output dir: %s\n' "$OUTPUT_DIR"
"${CMD[@]}"
