import os
#!/usr/bin/env bash
# Matched reference-corrected 20-loop launcher.
# ALWAYS run preflight first. This script refuses to start if preflight fails.
set -euo pipefail
ROOT="${PROJECT_ROOT}/reports/rdagent_matched_reference_rerun_20260916"
# shellcheck disable=SC1091
source "${ROOT}/env.sh"

bash "${ROOT}/preflight_matched_rerun.sh"

STAMP=$(date +%Y%m%d_%H%M%S)
export LOG_TRACE_PATH="${ROOT}/sessions/matched_20loop_${STAMP}"
mkdir -p "$PICKLE_CACHE_FOLDER_PATH_STR" "$LOG_TRACE_PATH" "${ROOT}/replication_20loop"
TEE="${ROOT}/replication_20loop/tee_matched_20loop_${STAMP}.log"
printf '%s\n' "$LOG_TRACE_PATH" > "${ROOT}/replication_20loop/latest_log_trace_path.txt"
printf '%s\n' "$TEE" > "${ROOT}/replication_20loop/latest_tee_path.txt"

cd "$RDAGENT_ROOT"
{
  echo "START $(date -Iseconds)"
  echo "RUN=matched_reference_corrected_20loop"
  echo "LOG_TRACE_PATH=$LOG_TRACE_PATH"
  echo "PICKLE_CACHE_FOLDER_PATH_STR=$PICKLE_CACHE_FOLDER_PATH_STR"
  echo "FACTOR_CoSTEER_DATA_FOLDER=$FACTOR_CoSTEER_DATA_FOLDER"
  echo "FACTOR_CoSTEER_DATA_FOLDER_DEBUG=$FACTOR_CoSTEER_DATA_FOLDER_DEBUG"
  echo "QLIB_FACTOR_MARKET=$QLIB_FACTOR_MARKET TOPK=$QLIB_FACTOR_TOPK N_DROP=$QLIB_FACTOR_N_DROP"
  echo "QLIB_FACTOR_ACCUMULATE_ALL=$QLIB_FACTOR_ACCUMULATE_ALL"
  echo "TEST=$QLIB_FACTOR_TEST_START..$QLIB_FACTOR_TEST_END (no 2024-2025)"
  echo "LOOP_N=20"
  echo "PROMPT_MODE=overlay_golden_2f02043b"
  dotenv run --no-override -- "$RDAGENT_PYTHON" "${ROOT}/launch_factor_with_prompt_overlay.py" --loop_n 20
  echo "END $(date -Iseconds)"
} 2>&1 | tee "$TEE"

echo "tee=$TEE"
echo "session=$LOG_TRACE_PATH"
# Alternative (same principal designation): run_first10.sh then resume_next10.sh — see run_10_then_resume.md
