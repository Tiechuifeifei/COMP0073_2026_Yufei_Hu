import os
# Matched reference-corrected RD-Agent env (07-07 config restored + contamination removed).
# Source from launchers / preflight only. Does NOT start the experiment by itself.
# Does not read 2024-2025 holdout. Does not touch RD-Agent/pickle_cache or 09-15 ROOT.

ROOT="${PROJECT_ROOT}/reports/rdagent_matched_reference_rerun_20260916"
RDAGENT_ROOT=os.environ["RDAGENT_ROOT"]
RDAGENT_PYTHON="/opt/anaconda3/envs/qlib/bin/python"
OLD_EMPTY_PKL="${RDAGENT_ROOT}/pickle_cache/rdagent.scenarios.qlib.developer.factor_runner.develop/d41d8cd98f00b204e9800998ecf8427e.pkl"
CORRECTED_0915_ROOT="${PROJECT_ROOT}/reports/rdagent_corrected_replication_20260915"

# --- isolation (mandatory) ---
export PICKLE_CACHE_FOLDER_PATH_STR="${ROOT}/pickle_cache"
# Absolute paths so discovery cannot fall back to polluted git_ignore_folder defaults.
export FACTOR_CoSTEER_DATA_FOLDER="${ROOT}/source_data_clean"
export FACTOR_CoSTEER_DATA_FOLDER_DEBUG="${ROOT}/source_data_clean_debug"

# --- Qlib factor splits / portfolio (07-07 session + workspace evidence) ---
export QLIB_FACTOR_TRAIN_START=2008-01-01
export QLIB_FACTOR_TRAIN_END=2017-12-31
export QLIB_FACTOR_VALID_START=2018-01-01
export QLIB_FACTOR_VALID_END=2019-12-31
export QLIB_FACTOR_TEST_START=2020-01-01
export QLIB_FACTOR_TEST_END=2023-12-31
export QLIB_FACTOR_MARKET=sp500
export QLIB_FACTOR_TOPK=20
export QLIB_FACTOR_N_DROP=2

# Accumulation: match 07-07 RUNTIME (tee: zero "SOTA factor processing" while all Replace=False).
# Feedback.__bool__ == decision → based_experiments only includes Replace=yes loops.
# QLIB_FACTOR_ACCUMULATE_ALL did not exist on 07-07; current default False reproduces that gate.
# Keep false for the US matched rerun (true enables the sector accumulate-all extension).
export QLIB_FACTOR_ACCUMULATE_ALL=false

# Explicitly unset X1 / custom scenario overrides if present in parent shell.
unset QLIB_FACTOR_SCEN || true
unset QLIB_FACTOR_HYPOTHESIS_GEN || true
