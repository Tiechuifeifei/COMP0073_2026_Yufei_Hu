#!/usr/bin/env bash
# Preflight checks for the matched reference-corrected 20-loop run.
# Does not start RD-Agent; exits 1 if any gate fails.
set -euo pipefail
ROOT="${PROJECT_ROOT}/reports/rdagent_matched_reference_rerun_20260916"
RDAGENT_ROOT=os.environ["RDAGENT_ROOT"]
CORRECTED_0915="${PROJECT_ROOT}/reports/rdagent_corrected_replication_20260915"
OLD_CACHE="${RDAGENT_ROOT}/pickle_cache"
OLD_EMPTY_PKL="${OLD_CACHE}/rdagent.scenarios.qlib.developer.factor_runner.develop/d41d8cd98f00b204e9800998ecf8427e.pkl"
GOLDEN="${ROOT}/golden_prompts"
FAIL=0

fail() { echo "FAIL: $*" >&2; FAIL=1; }
ok() { echo "OK: $*"; }

echo "=== preflight_matched_rerun $(date -Iseconds) ==="
# shellcheck disable=SC1091
source "${ROOT}/env.sh"

# --- 1. cache isolation + empty before first run (or only contains this run's keys) ---
if [[ "${PICKLE_CACHE_FOLDER_PATH_STR}" != "${ROOT}/pickle_cache" ]]; then
  fail "PICKLE_CACHE_FOLDER_PATH_STR=${PICKLE_CACHE_FOLDER_PATH_STR} != ${ROOT}/pickle_cache"
else
  ok "pickle cache path isolated"
fi
if [[ "${PICKLE_CACHE_FOLDER_PATH_STR}" == "${OLD_CACHE}" ]]; then
  fail "pickle cache falls back to default RD-Agent cache"
fi
if [[ -e "${PICKLE_CACHE_FOLDER_PATH_STR}" ]]; then
  # Allow empty OR only artifacts from a prior successful baseline in THIS root.
  # Forbid any path that looks like 09-15 or CSI300 leak reuse.
  if find "${PICKLE_CACHE_FOLDER_PATH_STR}" -type f -name '*.pkl' 2>/dev/null | rg -q "corrected_replication_20260915|rd13|fundamental"; then
    fail "pickle cache contains forbidden name markers"
  fi
  # Before main 20-loop, require either empty or baseline-only (documented).
  n_pkl=$(find "${PICKLE_CACHE_FOLDER_PATH_STR}" -type f -name '*.pkl' 2>/dev/null | wc -l | tr -d ' ')
  ok "pickle cache pkl count=${n_pkl} (must not be default RD-Agent cache)"
else
  fail "pickle cache dir missing"
fi
if [[ -f "$OLD_EMPTY_PKL" ]]; then
  ok "original leaked empty-baseline pkl still present at default path (must remain untouched by this run)"
fi

# --- 2. source data: no RD13/F1C/fundamental/sentiment ---
for d in "${FACTOR_CoSTEER_DATA_FOLDER}" "${FACTOR_CoSTEER_DATA_FOLDER_DEBUG}"; do
  if [[ ! -d "$d" ]]; then fail "missing source dir $d"; continue; fi
  if find "$d" -maxdepth 1 -type f | rg -q "rd13|fundamental|sentiment|F1C|f1c"; then
    fail "forbidden file in $d"
  fi
  if [[ ! -f "$d/daily_pv.h5" ]]; then fail "daily_pv.h5 missing in $d"; fi
  # Only allow daily_pv.h5 + README.md
  while IFS= read -r f; do
    base=$(basename "$f")
    case "$base" in
      daily_pv.h5|README.md) ;;
      *) fail "unexpected file in clean source_data: $f" ;;
    esac
  done < <(find "$d" -maxdepth 1 -type f)
done
"$RDAGENT_PYTHON" - <<'PY' || fail "column audit failed"
from pathlib import Path
import pandas as pd
import os, sys
root = (Path(os.environ["PROJECT_ROOT"]) / "reports/rdagent_matched_reference_rerun_20260916") if os.environ.get("PROJECT_ROOT") else (_repo_root() / "reports/rdagent_matched_reference_rerun_20260916")
allowed = {"$open","$close","$high","$low","$volume","$factor"}
for d in [root/"source_data_clean", root/"source_data_clean_debug"]:
    df = pd.read_hdf(d/"daily_pv.h5")
    cols = set(map(str, df.columns))
    if cols != allowed:
        print("BAD cols", d, cols); sys.exit(1)
    blob = " ".join(cols).lower()
    for bad in ("rd13","fundamental","sentiment","f1c","gross_profitability"):
        if bad in blob:
            print("BAD token", bad); sys.exit(1)
print("columns ok")
PY
ok "source_data clean (daily_pv only)"

# --- 3. prompts: golden overlay (required) + other files hash-match ---
# Host may block writes to RD-Agent; matched launchers use launch_factor_with_prompt_overlay.py.
expect_qlib="f8a213cb35c6677a9a4a38ccbdfb4b05fe4798090da83764b8fafe0f1dc98a1c"
golden_qlib_hash=$(shasum -a 256 "${GOLDEN}/scenarios_qlib_prompts.yaml" | awk '{print $1}')
if [[ "$golden_qlib_hash" != "$expect_qlib" ]]; then
  fail "golden scenarios_qlib_prompts.yaml hash drifted (got $golden_qlib_hash)"
fi
export LOG_TRACE_PATH="${ROOT}/sessions/preflight_probe"
mkdir -p "$LOG_TRACE_PATH"
cd "$RDAGENT_ROOT"
if dotenv run --no-override -- "$RDAGENT_PYTHON" "${ROOT}/check_prompt_overlay.py"; then
  ok "prompt overlay loads 07-07 hypothesis_specification (no diversity)"
else
  fail "prompt overlay probe failed"
fi
active_qlib_hash=$(shasum -a 256 "${RDAGENT_ROOT}/rdagent/scenarios/qlib/prompts.yaml" | awk '{print $1}')
if [[ "$active_qlib_hash" == "$expect_qlib" ]]; then
  ok "worktree prompts.yaml also matches golden"
else
  ok "worktree prompts.yaml differs; overlay launcher is mandatory (active=$active_qlib_hash)"
fi

declare -a MAP=(
  "scenarios_qlib_experiment_prompts.yaml:rdagent/scenarios/qlib/experiment/prompts.yaml:6325f0366075ffc0519a983ac0049c5bf30714479932a7b4d9ba5651ae86d004"
  "components_proposal_prompts.yaml:rdagent/components/proposal/prompts.yaml:a5187101d3b281ea0be9e0d974f3bd65706a9e7235221e108a92f689c4e5e44c"
  "components_coder_factor_coder_prompts.yaml:rdagent/components/coder/factor_coder/prompts.yaml:8c6cfc174915bff8b173e959611ffc4e946a19aa406634f0a1eb261725a0efce"
)
cd "$ROOT"
for entry in "${MAP[@]}"; do
  gname="${entry%%:*}"
  rest="${entry#*:}"
  rel="${rest%%:*}"
  expect="${rest##*:}"
  active_hash=$(shasum -a 256 "${RDAGENT_ROOT}/${rel}" | awk '{print $1}')
  golden_hash=$(shasum -a 256 "${GOLDEN}/${gname}" | awk '{print $1}')
  if [[ "$golden_hash" != "$expect" ]]; then
    fail "golden $gname hash drifted (got $golden_hash)"
  fi
  if [[ "$active_hash" != "$expect" ]]; then
    fail "active $rel hash $active_hash != golden $expect"
  else
    ok "prompt hash match $rel"
  fi
done

# --- 4. fresh baseline present + consistent ---
BM="${ROOT}/baseline/fresh_baseline_metrics.json"
if [[ ! -f "$BM" ]]; then
  fail "missing $BM — run: bash ${ROOT}/run_fresh_baseline.sh"
else
  if "$RDAGENT_PYTHON" "${ROOT}/check_baseline_metrics.py"; then
    ok "fresh baseline metrics present and within anchors"
  else
    fail "baseline metrics gate failed"
  fi
fi

# --- 5. session/output paths under new ROOT ---
if [[ "${ROOT}" == *corrected_replication_20260915* ]]; then
  fail "ROOT points at 09-15 corrected run"
fi
ok "ROOT is matched_reference_rerun_20260916"

# --- 6. must not reference 09-15 factor workspaces / 07-07 runtime cache ---
# env.sh may name CORRECTED_0915_ROOT only as a forbidden-path constant — that is OK.
WIRE_HITS=$(rg -n "sessions/corrected_20loop|corrected_replication_20260915/pickle_cache|corrected_replication_20260915/sessions" \
  "${ROOT}/env.sh" "${ROOT}/run_20loop.sh" "${ROOT}/run_fresh_baseline.sh" "${ROOT}/run_fresh_baseline.py" 2>/dev/null || true)
if [[ -n "$WIRE_HITS" ]]; then
  echo "$WIRE_HITS" >&2
  fail "launcher references 09-15 runtime paths"
else
  ok "launchers do not wire 09-15 factor workspaces"
fi
if [[ "${PICKLE_CACHE_FOLDER_PATH_STR}" == "${OLD_CACHE}" ]]; then
  fail "using original 07-07 pickle_cache"
fi
ok "not using original RD-Agent pickle_cache"

# --- 7. loop_n / accumulation ---
if ! rg -q -- '--loop_n 20' "${ROOT}/run_20loop.sh"; then
  fail "run_20loop.sh must pass --loop_n 20"
else
  ok "loop_n=20 in run_20loop.sh"
fi
if ! rg -q 'QLIB_FACTOR_ACCUMULATE_ALL=false' "${ROOT}/env.sh"; then
  fail "accumulation must be false to match 07-07 runtime gate"
else
  ok "QLIB_FACTOR_ACCUMULATE_ALL=false (07-07 runtime matched)"
fi
# Live settings check with env loaded (force log under ROOT so we never write default RD-Agent/log)
export LOG_TRACE_PATH="${ROOT}/sessions/preflight_probe"
mkdir -p "$LOG_TRACE_PATH"
cd "$RDAGENT_ROOT"
if dotenv run --no-override -- "$RDAGENT_PYTHON" "${ROOT}/check_live_settings.py"; then
  ok "live settings match audited 07-07-intended mechanism + isolation"
else
  fail "live FactorBasePropSetting / data_folder / pickle isolation mismatch"
fi

# --- 8. decisions pre-registration present ---
if ! rg -q "principal matched reference-corrected RD-Agent run" "${ROOT}/decisions.md"; then
  fail "decisions.md missing pre-registration text"
else
  ok "decisions.md pre-registration present"
fi

echo "=== preflight summary ==="
if [[ "$FAIL" -ne 0 ]]; then
  echo "PREFLIGHT FAILED — do not start RD-Agent" >&2
  exit 1
fi
echo "PREFLIGHT PASSED — safe to start via run_20loop.sh (not executed by this script)"
exit 0
