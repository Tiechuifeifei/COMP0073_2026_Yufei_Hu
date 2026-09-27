"""Write frozen precommit specification before any holdout data access."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from portfolio_experiments.final_holdout.config import (
    BASE_RISK_DEGREE,
    BENCHMARK,
    CLOSE_COST,
    D2_PRED_PATH,
    F1C_PRED_PATH,
    HOLDOUT_END,
    HOLDOUT_START,
    HOLD_THRESH,
    N_DROP,
    OPEN_COST,
    OUT_ROOT,
    PANEL_PATH,
    R2_FEATURE_COLS,
    R2_LABEL_MAP,
    R2_SEED,
    HMM_2STATE_SEED,
    TOPK,
    TRAIN_END,
    TRAIN_START,
)

W1 = {
    "bull": {"w_d2": 0.7, "w_f1c": 0.3},
    "neutral": {"w_d2": 0.7, "w_f1c": 0.3},
    "bear": {"w_d2": 0.7, "w_f1c": 0.3},
}
W2 = {
    "bull": {"w_d2": 0.8, "w_f1c": 0.2},
    "neutral": {"w_d2": 0.6, "w_f1c": 0.4},
    "bear": {"w_d2": 0.4, "w_f1c": 0.6},
}


def build_precommit() -> dict:
    created = datetime.now(timezone.utc).isoformat()
    return {
        "created_at": created,
        "status": "FROZEN_PRECOMMIT",
        "holdout_period": [HOLDOUT_START, HOLDOUT_END],
        "sample": "panel_intersect_D2_intersect_F1C",
        "sample_paths": {
            "panel": str(PANEL_PATH),
            "d2_pred": str(D2_PRED_PATH),
            "f1c_pred": str(F1C_PRED_PATH),
        },
        "w_sentiment": 0,
        "sentiment_excluded": True,
        "no_post_run_tuning": True,
        "no_new_candidates_after_precommit": True,
        "portfolio_constants": {
            "topk": TOPK,
            "n_drop": N_DROP,
            "hold_thresh": HOLD_THRESH,
            "open_cost": OPEN_COST,
            "close_cost": CLOSE_COST,
            "benchmark": BENCHMARK,
            "base_risk_degree": BASE_RISK_DEGREE,
        },
        "hmm_train_period": [TRAIN_START, TRAIN_END],
        "inference": {
            "method": "strict_online_forward_filter",
            "lag": 1,
            "no_full_sample_smoothing": True,
            "no_holdout_retraining": True,
            "no_holdout_rescaling_of_train_stats": True,
            "cross_section_preprocess": "daily_winsorize_zscore",
        },
        "candidates": {
            "H0": {
                "identity": "Original validation-selected primary specification",
                "maps_to_main_experiment": "T0",
                "hmm": {"n_states": 2, "seed": HMM_2STATE_SEED, "features": ["spy_ret", "spy_vol20"]},
                "weights": W1,
                "exposure": BASE_RISK_DEGREE,
            },
            "H1": {
                "identity": "Original regime-dependent factor-weighting hypothesis",
                "maps_to_main_experiment": "T1",
                "hmm": {"n_states": 2, "seed": HMM_2STATE_SEED, "features": ["spy_ret", "spy_vol20"]},
                "weights": W2,
                "exposure": BASE_RISK_DEGREE,
            },
            "H2": {
                "identity": "Fixed-weight control for revised three-state specification",
                "hmm": {
                    "n_states": 3,
                    "seed": R2_SEED,
                    "spec_id": "R2",
                    "features": R2_FEATURE_COLS,
                    "label_map": {str(k): v for k, v in R2_LABEL_MAP.items()},
                },
                "weights": W1,
                "exposure": BASE_RISK_DEGREE,
                "parity_expectation": "Must match H0 portfolio scores (state-independent W1)",
            },
            "H3": {
                "identity": "Post-hoc revised candidate requiring untouched holdout confirmation",
                "hmm": {
                    "n_states": 3,
                    "seed": R2_SEED,
                    "spec_id": "R2",
                    "features": R2_FEATURE_COLS,
                    "label_map": {str(k): v for k, v in R2_LABEL_MAP.items()},
                },
                "weights": W2,
                "exposure": BASE_RISK_DEGREE,
            },
        },
        "metrics": [
            "arr",
            "benchmark_arr",
            "excess_arr_net",
            "annual_volatility",
            "sharpe",
            "ir_net",
            "mdd",
            "calmar",
            "turnover",
            "transaction_cost_total",
            "transaction_cost_daily_mean",
            "average_holdings",
            "average_exposure",
            "topk_overlap",
            "yearly_2024_2025",
            "regime_breakdown",
            "monthly_excess_return",
            "best_worst_month",
        ],
        "pairwise_comparisons_required": [
            "H1_vs_H0",
            "H3_vs_H2",
            "H3_vs_H0",
            "2state_vs_3state_summary",
        ],
        "decision_rules_frozen": {
            "revised_candidate_holdout_supported": [
                "H3 excess ARR > H2",
                "H3 Sharpe or IR > H2",
                "improvement not from single month only",
                "turnover/cost not abnormally higher",
                "MDD not severely worse without compensating return",
                "not entirely dependent on one holdout year",
            ],
            "mixed_evidence_template": (
                "The revised candidate achieved higher holdout return, but the improvement was "
                "accompanied by weaker downside-risk performance and limited temporal stability."
            ),
            "not_replicated_template": (
                "The validation and 2020–2023 test advantage of the revised three-state dynamic "
                "specification did not replicate in the untouched 2024–2025 holdout."
            ),
            "primary_status_preserved_if": "H0 risk-adjusted performance remains superior",
            "no_winner_rewriting_history": True,
        },
        "prerequisite_gate": {
            "requires_frozen_d2_predictions_through_holdout_end": True,
            "requires_frozen_f1c_predictions_through_holdout_end": True,
            "requires_panel_holdout_rows": True,
            "forbidden_if_missing": "retrain_on_holdout_or_impute_scores",
        },
    }


def load_or_write_precommit() -> tuple[Path, dict]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "final_holdout_precommit.json"
    if path.exists():
        return path, json.loads(path.read_text(encoding="utf-8"))
    payload = build_precommit()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, payload


def write_precommit() -> Path:
    path, _ = load_or_write_precommit()
    return path
