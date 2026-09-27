import os
#!/usr/bin/env bash
# Isolated corrected semantic 10-loop launcher.
# Endpoint: Loop9. Does not write principal matched/ablation/downstream.
set -euo pipefail
ROOT="${PROJECT_ROOT}/reports/rdagent_corrected_semantic_rerun_20260918"
# shellcheck disable=SC1091
source "${ROOT}/env.sh"

cd ${RDAGENT_ROOT}

STAMP=$(date +%Y%m%d_%H%M%S)
TEE="${ROOT}/logs/tee_corrected_semantic_10loop_${STAMP}.log"
printf '%s\n' "$TEE" > "${ROOT}/logs/latest_tee_path.txt"

{
  echo "START $(date -Iseconds)"
  echo "RUN=corrected_semantic_10loop"
  echo "ROOT=$ROOT"
  echo "PICKLE_CACHE_FOLDER_PATH_STR=$PICKLE_CACHE_FOLDER_PATH_STR"
  echo "FACTOR_CoSTEER_DATA_FOLDER=$FACTOR_CoSTEER_DATA_FOLDER"
  echo "QLIB_FACTOR_ACCUMULATE_ALL=$QLIB_FACTOR_ACCUMULATE_ALL"
  echo "TEST=$QLIB_FACTOR_TEST_START..$QLIB_FACTOR_TEST_END"
  echo "LOOP_N_SEMANTICS=see LOOP_N_SEMANTICS.md; CLI target total=10"
  echo "FORK=principal __session__/1/4_record"
  echo "CORRECTED_INTERACTION=rel_ATR20 * VR20_orig"

  echo "=== correction gate ==="
  "$RDAGENT_PYTHON" "${ROOT}/scripts/correction_gate.py"

  echo "=== launch 10-loop pipeline ==="
  # dotenv preserves API keys; --no-override keeps our isolation env
  dotenv run --no-override -- "$RDAGENT_PYTHON" "${ROOT}/scripts/run_corrected_semantic_10loop.py"

  echo "END $(date -Iseconds)"
} 2>&1 | tee "$TEE"

echo "tee=$TEE"
