import os
#!/usr/bin/env bash
# Continue corrected semantic branch Loop10-19 from Loop9 checkpoint.
# Endpoint: Loop19. Does NOT overwrite corrected_semantic_10loop.
# Does NOT touch principal matched 20-loop, ablation, or holdout.
set -euo pipefail
ROOT="${PROJECT_ROOT}/reports/rdagent_corrected_semantic_rerun_20260918"
# shellcheck disable=SC1091
source "${ROOT}/env.sh"

SRC="${ROOT}/sessions/corrected_semantic_10loop"
DUMP="${SRC}/__session__/9/4_record"
DEST="${ROOT}/sessions/corrected_semantic_20loop"

if [[ ! -f "$DUMP" ]]; then
  echo "FAIL: missing Loop9 dump: $DUMP" >&2
  exit 1
fi
if [[ -d "${DEST}/__session__/10" || -d "${DEST}/Loop_10" ]]; then
  echo "FAIL: destination already has Loop10+ — refuse overwrite: $DEST" >&2
  exit 1
fi
if [[ -d "${SRC}/__session__/10" ]]; then
  echo "FAIL: source 10loop session unexpectedly has Loop10+" >&2
  exit 1
fi

cd ${RDAGENT_ROOT}

STAMP=$(date +%Y%m%d_%H%M%S)
TEE="${ROOT}/logs/tee_corrected_semantic_20loop_${STAMP}.log"
printf '%s\n' "$TEE" > "${ROOT}/logs/latest_tee_path_20loop.txt"

{
  echo "START $(date -Iseconds)"
  echo "RUN=corrected_semantic_20loop_continuation"
  echo "ROOT=$ROOT"
  echo "SOURCE_CHECKPOINT=$DUMP"
  echo "DEST_SESSION=$DEST"
  echo "PICKLE_CACHE_FOLDER_PATH_STR=$PICKLE_CACHE_FOLDER_PATH_STR"
  echo "FACTOR_CoSTEER_DATA_FOLDER=$FACTOR_CoSTEER_DATA_FOLDER"
  echo "QLIB_FACTOR_ACCUMULATE_ALL=$QLIB_FACTOR_ACCUMULATE_ALL"
  echo "TEST=$QLIB_FACTOR_TEST_START..$QLIB_FACTOR_TEST_END"
  echo "LOOP_N_SEMANTICS=target_total=20; kickoffs 0-9 no-op; newly execute 10-19; stop at Loop19"
  echo "HOLDOUT=CLOSED"
  echo "=== launch Loop10-19 continuation ==="
  dotenv run --no-override -- "$RDAGENT_PYTHON" "${ROOT}/scripts/run_corrected_semantic_20loop.py"
  echo "END $(date -Iseconds)"
} 2>&1 | tee "$TEE"

echo "tee=$TEE"
