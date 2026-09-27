"""Configuration for final 2024–2025 holdout evaluation."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "data" / "portfolio_experiments" / "final_holdout"
REPORT_ROOT = PROJECT_ROOT / "reports" / "portfolio_experiments" / "final_holdout"

MAIN_HMM_ROOT = PROJECT_ROOT / "data" / "portfolio_experiments" / "hmm_regime"
R2_ROOT = PROJECT_ROOT / "data" / "portfolio_experiments" / "hmm_3state_robustness"

TRAIN_START = "2008-01-01"
TRAIN_END = "2017-12-31"
VALID_START = "2018-01-01"
VALID_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"
HOLDOUT_END = "2025-12-31"
HARD_CUTOFF_PRIOR = "2023-12-31"

PANEL_PATH = (
    PROJECT_ROOT
    / "data/rd13_v2_downstream_replication/stage_1_delayed_panel/delayed_common_sample_rd13_v2.parquet"
)
D2_PRED_PATH = (
    PROJECT_ROOT
    / "data/rd13_v2_downstream_replication/stage_2_delayed_models/pred_D2_ALPHA20_RD13_V2.parquet"
)
F1C_PRED_PATH = (
    PROJECT_ROOT
    / "data/fundamental_experiments/F1_fundamental_lightgbm/main/F1C_no_operating_profitability_ensemble_pred.pkl"
)
SPY_CSV_PATH = PROJECT_ROOT / "staging/csv_benchmark/p84398.csv"

TOPK = 20
N_DROP = 2
HOLD_THRESH = 1
OPEN_COST = 0.0001
CLOSE_COST = 0.0001
BENCHMARK = "P84398"
BASE_RISK_DEGREE = 0.95

R2_LABEL_MAP = {0: "neutral", 1: "bear", 2: "bull"}
R2_FEATURE_COLS = ["spy_ret", "spy_vol20", "spy_vol_ratio_20_60"]
R2_SEED = 42
HMM_2STATE_SEED = 42
