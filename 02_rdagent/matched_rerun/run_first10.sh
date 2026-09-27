import os
#!/usr/bin/env bash
# Phase A: start the principal matched session and run loops 0–9 only.
# After audit, continue with resume_next10.sh (same session → loops 10–19).
# Does NOT clear pickle_cache. Does NOT start until preflight passes.
set -euo pipefail
ROOT="${PROJECT_ROOT}/reports/rdagent_matched_reference_rerun_20260916"
# shellcheck disable=SC1091
source "${ROOT}/env.sh"

bash "${ROOT}/preflight_matched_rerun.sh"

META="${ROOT}/replication_20loop"
mkdir -p "$META" "$PICKLE_CACHE_FOLDER_PATH_STR"
CANON="${META}/canonical_log_trace_path.txt"

if [[ -f "$CANON" ]]; then
  EXISTING=$(cat "$CANON")
  if [[ -d "$EXISTING/__session__" ]]; then
    echo "FAIL: canonical session already exists: $EXISTING" >&2
    echo "Use resume_next10.sh to continue, or deliberately retire this session first." >&2
    exit 1
  fi
fi

STAMP=$(date +%Y%m%d_%H%M%S)
export LOG_TRACE_PATH="${ROOT}/sessions/matched_20loop_${STAMP}"
mkdir -p "$LOG_TRACE_PATH"
TEE="${META}/tee_matched_first10_${STAMP}.log"

printf '%s\n' "$LOG_TRACE_PATH" > "$CANON"
printf '%s\n' "$LOG_TRACE_PATH" > "${META}/latest_log_trace_path.txt"
printf '%s\n' "$TEE" > "${META}/latest_tee_path.txt"
printf 'phase=first10_running\nloop_n_this_leg=10\ntarget_total_loops=20\n' > "${META}/phase_status.txt"

cd "$RDAGENT_ROOT"
{
  echo "START $(date -Iseconds)"
  echo "RUN=matched_reference_corrected_20loop_phase_first10"
  echo "LOG_TRACE_PATH=$LOG_TRACE_PATH"
  echo "PICKLE_CACHE_FOLDER_PATH_STR=$PICKLE_CACHE_FOLDER_PATH_STR"
  echo "FACTOR_CoSTEER_DATA_FOLDER=$FACTOR_CoSTEER_DATA_FOLDER"
  echo "FACTOR_CoSTEER_DATA_FOLDER_DEBUG=$FACTOR_CoSTEER_DATA_FOLDER_DEBUG"
  echo "QLIB_FACTOR_MARKET=$QLIB_FACTOR_MARKET TOPK=$QLIB_FACTOR_TOPK N_DROP=$QLIB_FACTOR_N_DROP"
  echo "QLIB_FACTOR_ACCUMULATE_ALL=$QLIB_FACTOR_ACCUMULATE_ALL"
  echo "TEST=$QLIB_FACTOR_TEST_START..$QLIB_FACTOR_TEST_END (no 2024-2025)"
  echo "LOOP_N=10 (phase A of 20; resume via resume_next10.sh)"
  echo "PROMPT_MODE=overlay_golden_2f02043b"
  dotenv run --no-override -- "$RDAGENT_PYTHON" "${ROOT}/launch_factor_with_prompt_overlay.py" --loop_n 10
  echo "END $(date -Iseconds)"
} 2>&1 | tee "$TEE"

printf 'phase=first10_done\nloop_n_this_leg=10\ntarget_total_loops=20\nended_at=%s\n' "$(date -Iseconds)" > "${META}/phase_status.txt"
echo "tee=$TEE"
echo "session=$LOG_TRACE_PATH"
echo "Next: read-only audit, then bash ${ROOT}/resume_next10.sh"
