import os
#!/usr/bin/env bash
# Phase B: resume the SAME matched session for loops 10–19 (total 20).
# Requires run_first10.sh to have finished. Does NOT clear pickle_cache.
# Does NOT start until preflight passes and session gates pass.
#
# RD-Agent run() always resets loop_idx=0 and treats --loop_n as remaining
# kickoffs (finished loops are no-ops). After Loops 0–9 are complete, the CLI
# must be --loop_n 20 so kickoffs 0–9 no-op and 10–19 actually run.
set -euo pipefail
ROOT="${PROJECT_ROOT}/reports/rdagent_matched_reference_rerun_20260916"
# shellcheck disable=SC1091
source "${ROOT}/env.sh"

bash "${ROOT}/preflight_matched_rerun.sh"

META="${ROOT}/replication_20loop"
CANON="${META}/canonical_log_trace_path.txt"

if [[ ! -f "$CANON" ]]; then
  echo "FAIL: missing $CANON — run run_first10.sh first" >&2
  exit 1
fi

export LOG_TRACE_PATH
LOG_TRACE_PATH=$(cat "$CANON")
if [[ ! -d "$LOG_TRACE_PATH/__session__" ]]; then
  echo "FAIL: session has no __session__: $LOG_TRACE_PATH" >&2
  exit 1
fi

# Expect loops 0–9 present; refuse if already past loop 9 or incomplete.
if [[ ! -d "$LOG_TRACE_PATH/Loop_9" && ! -d "$LOG_TRACE_PATH/__session__/9" ]]; then
  echo "FAIL: Loop_9 / __session__/9 not found — first10 incomplete?" >&2
  ls "$LOG_TRACE_PATH" | head -40 >&2
  exit 1
fi
if [[ -d "$LOG_TRACE_PATH/Loop_19" || -d "$LOG_TRACE_PATH/__session__/19" ]]; then
  echo "FAIL: Loop_19 already exists — session already completed 20 loops" >&2
  exit 1
fi
if [[ -d "$LOG_TRACE_PATH/Loop_10" || -d "$LOG_TRACE_PATH/__session__/10" ]]; then
  echo "FAIL: Loop_10 already present — resume already started; inspect session before re-running" >&2
  exit 1
fi

# Clear any false-positive complete_20 left by a prior no-op resume.
STAMP=$(date +%Y%m%d_%H%M%S)
TEE="${META}/tee_matched_resume10_${STAMP}.log"
printf '%s\n' "$LOG_TRACE_PATH" > "${META}/latest_log_trace_path.txt"
printf '%s\n' "$TEE" > "${META}/latest_tee_path.txt"
printf 'phase=resume10_running\nloop_n_cli=20\nnew_loops_expected=10\ntarget_total_loops=20\nnote=RD-Agent_run_resets_loop_idx_to_0_so_cli_loop_n_must_equal_target_total\n' > "${META}/phase_status.txt"

cd "$RDAGENT_ROOT"
{
  echo "START $(date -Iseconds)"
  echo "RUN=matched_reference_corrected_20loop_phase_resume10"
  echo "RESUME_FROM=$LOG_TRACE_PATH"
  echo "PICKLE_CACHE_FOLDER_PATH_STR=$PICKLE_CACHE_FOLDER_PATH_STR"
  echo "FACTOR_CoSTEER_DATA_FOLDER=$FACTOR_CoSTEER_DATA_FOLDER"
  echo "FACTOR_CoSTEER_DATA_FOLDER_DEBUG=$FACTOR_CoSTEER_DATA_FOLDER_DEBUG"
  echo "QLIB_FACTOR_MARKET=$QLIB_FACTOR_MARKET TOPK=$QLIB_FACTOR_TOPK N_DROP=$QLIB_FACTOR_N_DROP"
  echo "QLIB_FACTOR_ACCUMULATE_ALL=$QLIB_FACTOR_ACCUMULATE_ALL"
  echo "TEST=$QLIB_FACTOR_TEST_START..$QLIB_FACTOR_TEST_END (no 2024-2025)"
  echo "LOOP_N=20 (CLI kickoffs from loop_idx=0; 0-9 no-op, 10-19 new; same session)"
  echo "PROMPT_MODE=overlay_golden_2f02043b"
  # Passing the session directory loads the latest __session__ dump and continues in-place.
  dotenv run --no-override -- "$RDAGENT_PYTHON" "${ROOT}/launch_factor_with_prompt_overlay.py" \
    "$LOG_TRACE_PATH" --loop_n 20 --checkout True
  echo "END $(date -Iseconds)"
} 2>&1 | tee "$TEE"

printf 'phase=complete_20\nloop_n_cli=20\nnew_loops_expected=10\ntarget_total_loops=20\nended_at=%s\n' "$(date -Iseconds)" > "${META}/phase_status.txt"
echo "tee=$TEE"
echo "session=$LOG_TRACE_PATH"
echo "Principal matched 20-loop trajectory should now cover Loop_0..Loop_19"
