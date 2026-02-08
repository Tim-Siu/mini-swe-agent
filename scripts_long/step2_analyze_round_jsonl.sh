#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=".venv/bin/python"
SCRIPT="scripts_long/step2_analyze_round_jsonl.py"

: "${RUN_DIR:?RUN_DIR is required (path to run_rounds_* directory)}"

if [[ -n "${ANALYSIS_OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$ANALYSIS_OUTPUT_DIR"
else
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="${RUN_DIR}/analysis_step2_${TS}"
fi

"$PYTHON_BIN" "$SCRIPT" --run-dir "$RUN_DIR" --output-dir "$OUTPUT_DIR"
