"""Frozen configuration for HMM regime-aware portfolio experiments."""

from __future__ import annotations
import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = Path(__file__).resolve().parent
OUT_ROOT = PROJECT_ROOT / "data" / "portfolio_experiments" / "hmm_regime"
REPORT_ROOT = PROJECT_ROOT / "reports" / "portfolio_experiments" / "hmm_regime"

# Canonical Qlib runtime used by frozen D/F portfolio backtests.
QLIB_PYTHON = Path("/opt/anaconda3/envs/qlib/bin/python")
QLIB_INIT = {
    "provider_uri": str(PROJECT_ROOT / "staging" / "qlib_data"),
    "region": "us",
    "kernels": 1,
}

WORKFLOW_YAML = PROJECT_ROOT / "experiments" / "conf_alpha20_sp500_transfer.yaml"
PHASE3_SCRIPTS = (Path(os.environ["RDAGENT_ROOT"]) / "phase3_portfolio_ablation/scripts")

# Time splits — holdout 2024+ is forbidden for this experiment line.
TRAIN_START = "2008-01-01"
TRAIN_END = "2017-12-31"
VALID_START = "2018-01-01"
VALID_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2023-12-31"
HARD_CUTOFF = "2023-12-31"
HOLDOUT_START = "2024-01-01"

SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "valid": (VALID_START, VALID_END),
    "test": (TEST_START, TEST_END),
}

# Portfolio constants — aligned with frozen R1R2P / F2P specification.
TOPK = 20
N_DROP = 2
HOLD_THRESH = 1
OPEN_COST = 0.0001
CLOSE_COST = 0.0001
BENCHMARK = "P84398"
ACCOUNT = 100_000_000
BASE_RISK_DEGREE = 0.95  # frozen D2/D3 portfolio default

# Dynamic exposure candidates (Bull max = BASE_RISK_DEGREE).
EXPOSURE_3STATE_A = {"bull": 0.95, "neutral": 0.70, "bear": 0.30}
EXPOSURE_3STATE_B = {"bull": 0.95, "neutral": 0.50, "bear": 0.00}
EXPOSURE_2STATE_A = {"bull": 0.95, "bear": 0.30}
EXPOSURE_2STATE_B = {"bull": 0.95, "bear": 0.00}

# Frozen prediction inputs.
PANEL_PATH = (
    PROJECT_ROOT
    / "data/rd13_v2_downstream_replication/stage_1_delayed_panel/delayed_common_sample_rd13_v2.parquet"
)
D2_PRED_PATH = (
    PROJECT_ROOT
    / "data/rd13_v2_downstream_replication/stage_2_delayed_models/pred_D2_ALPHA20_RD13_V2.parquet"
)
D3_PRED_PATH = (
    PROJECT_ROOT
    / "data/rd13_v2_downstream_replication/stage_2_delayed_models/pred_D3_ALPHA20_RD13_V2_FUND.parquet"
)
F1C_PRED_PATH = (
    PROJECT_ROOT
    / "data/fundamental_experiments/F1_fundamental_lightgbm/main/F1C_no_operating_profitability_ensemble_pred.pkl"
)
S5R_D1_PATH = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/av_only/delayed/D1_results.pkl"
SPY_CSV_PATH = PROJECT_ROOT / "staging/csv_benchmark/p84398.csv"

# Reference frozen portfolio metrics for parity checks.
D2_FROZEN_PORT_METRICS = PROJECT_ROOT / (
    "data/rd13_v2_downstream_replication/stage_2_delayed_models/delayed_portfolio_metrics.csv"
)

RANDOM_SEED = 42
HMM_TRAIN_SEED = 42

# Smoke test window (validation segment, no holdout).
SMOKE_START = "2019-01-02"
SMOKE_END = "2019-03-15"  # ~50 calendar days / ~50 trading days
