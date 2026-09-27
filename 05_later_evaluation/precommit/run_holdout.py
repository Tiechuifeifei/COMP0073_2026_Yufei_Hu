#!/usr/bin/env python3
"""Final holdout evaluation gate and runner (2024–2025)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "portfolio_experiments"))

from portfolio_experiments.final_holdout.config import (
    D2_PRED_PATH,
    F1C_PRED_PATH,
    HOLDOUT_END,
    HOLDOUT_START,
    OUT_ROOT,
    PANEL_PATH,
    REPORT_ROOT,
)
from portfolio_experiments.final_holdout.precommit import load_or_write_precommit


def _load_dates(path: Path, *, is_pkl: bool = False) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    if is_pkl:
        df = pd.read_pickle(path)
        if hasattr(df, "reset_index"):
            df = df.reset_index()
        col = "datetime"
    else:
        df = pd.read_parquet(path, columns=["datetime"])
        col = "datetime"
    df[col] = pd.to_datetime(df[col])
    holdout_n = int((df[col] >= HOLDOUT_START).sum())
    return df[col].min(), df[col].max(), holdout_n


def check_prerequisites(precommit_created_at: str) -> dict:
    panel_min, panel_max, panel_ho = _load_dates(PANEL_PATH)
    d2_min, d2_max, d2_ho = _load_dates(D2_PRED_PATH)
    f1c_min, f1c_max, f1c_ho = _load_dates(F1C_PRED_PATH, is_pkl=True)

    blockers = []
    if panel_ho == 0:
        blockers.append(f"Panel has 0 rows >= {HOLDOUT_START} (max={panel_max.date()})")
    if d2_ho == 0:
        blockers.append(f"D2 predictions have 0 rows >= {HOLDOUT_START} (max={d2_max.date()})")
    if f1c_ho == 0:
        blockers.append(f"F1C predictions have 0 rows >= {HOLDOUT_START} (max={f1c_max.date()})")

    common_ho = 0
    if panel_ho and d2_ho and f1c_ho:
        panel = pd.read_parquet(PANEL_PATH, columns=["datetime", "instrument"])
        d2 = pd.read_parquet(D2_PRED_PATH, columns=["datetime", "instrument"])
        f1c = pd.read_pickle(F1C_PRED_PATH).reset_index()
        for df in (panel, d2, f1c):
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
            df["instrument"] = df["instrument"].astype(str)
        common = panel.merge(d2, on=["datetime", "instrument"], how="inner")
        common = common.merge(f1c, on=["datetime", "instrument"], how="inner")
        common_ho = int((common["datetime"] >= HOLDOUT_START).sum())

    audit = {
        "precommit_created_at": precommit_created_at,
        "prerequisite_checked_at": datetime.now(timezone.utc).isoformat(),
        "holdout_first_portfolio_access_at": None,
        "holdout_portfolio_rows_accessed": 0,
        "status": "BLOCKED_MISSING_FROZEN_PREDICTIONS" if blockers else "READY_TO_RUN",
        "blockers": blockers,
        "panel_date_range": [str(panel_min.date()), str(panel_max.date())],
        "d2_date_range": [str(d2_min.date()), str(d2_max.date())],
        "f1c_date_range": [str(f1c_min.date()), str(f1c_max.date())],
        "holdout_rows_panel": panel_ho,
        "holdout_rows_d2": d2_ho,
        "holdout_rows_f1c": f1c_ho,
        "holdout_rows_panel_intersect_d2_intersect_f1c": common_ho,
        "data_2026_plus_accessed": 0,
        "train_only_through": "2017-12-31",
        "validation_only_through": "2019-12-31",
        "prior_test_only_through": "2023-12-31",
        "holdout_period": [HOLDOUT_START, HOLDOUT_END],
        "online_filter_required": True,
        "parameters_changed_after_precommit": False,
        "sentiment_included": False,
        "w_sentiment": 0,
        "holdout_used_for_model_selection": False,
        "forbidden_actions_if_blocked": [
            "retrain_D2_on_holdout",
            "retrain_F1C_on_holdout",
            "retrain_HMM_on_holdout",
            "impute_scores_from_holdout_labels",
        ],
    }
    return audit


def write_blocked_outputs(audit: dict, precommit: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    sample_rows = [
        {
            "component": "panel",
            "min_date": audit["panel_date_range"][0],
            "max_date": audit["panel_date_range"][1],
            "holdout_rows": audit["holdout_rows_panel"],
        },
        {
            "component": "d2_pred",
            "min_date": audit["d2_date_range"][0],
            "max_date": audit["d2_date_range"][1],
            "holdout_rows": audit["holdout_rows_d2"],
        },
        {
            "component": "f1c_pred",
            "min_date": audit["f1c_date_range"][0],
            "max_date": audit["f1c_date_range"][1],
            "holdout_rows": audit["holdout_rows_f1c"],
        },
        {
            "component": "panel_intersect_d2_intersect_f1c",
            "min_date": audit["panel_date_range"][0],
            "max_date": audit["panel_date_range"][1],
            "holdout_rows": audit["holdout_rows_panel_intersect_d2_intersect_f1c"],
        },
    ]
    pd.DataFrame(sample_rows).to_csv(OUT_ROOT / "final_holdout_sample_audit.csv", index=False)

    empty_summary = pd.DataFrame(
        columns=[
            "arm_id",
            "status",
            "arr",
            "excess_arr_net",
            "sharpe",
            "ir_net",
            "mdd",
            "calmar",
            "turnover",
        ]
    )
    empty_summary.to_csv(OUT_ROOT / "final_holdout_summary.csv", index=False)
    pd.DataFrame(columns=["arm_id", "year"]).to_csv(OUT_ROOT / "final_holdout_yearly.csv", index=False)
    pd.DataFrame(columns=["arm_id", "month"]).to_csv(OUT_ROOT / "final_holdout_monthly.csv", index=False)
    pd.DataFrame(columns=["arm_id", "state"]).to_csv(OUT_ROOT / "final_holdout_regime_breakdown.csv", index=False)
    pd.DataFrame(columns=["pair", "overlap_ratio"]).to_csv(
        OUT_ROOT / "final_holdout_holdings_overlap.csv", index=False
    )
    pd.DataFrame(columns=["comparison", "note"]).to_csv(OUT_ROOT / "final_holdout_pairwise_comparison.csv", index=False)

    (OUT_ROOT / "final_holdout_leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": audit["status"],
        "holdout_evaluated": False,
        "candidates_run": [],
        "primary_model_unchanged": "H0 / T0",
        "revised_candidate_status": "NOT_EVALUATED_ON_HOLDOUT",
        "conclusion_text": (
            "Final holdout evaluation (2024–2025) was **not executed**. Frozen D2 and F1C prediction "
            "artifacts, and the common panel, end at 2023-12-29 with zero holdout-period rows. "
            "Per protocol, holdout labels and returns must not be used to retrain or impute scores. "
            "The original 2-state validation-selected primary model (H0/T0) remains the deployable "
            "specification based on validation and 2020–2023 test evidence. The revised R2 3-state "
            "dynamic candidate (H3) has not received its first untouched holdout assessment."
        ),
        "blockers": audit["blockers"],
        "required_next_step": (
            "Generate legally frozen out-of-sample D2 and F1C predictions for 2024–2025 using "
            "models trained only through 2017 (or earlier frozen train cutoff), extend panel with "
            "holdout split, then re-run this script without modifying precommit candidate definitions."
        ),
        "research_questions_unanswered_on_holdout": {
            "H1_vs_H0_original_dynamic": "not_run",
            "H3_vs_H2_revised_dynamic": "not_run",
            "H3_vs_H0_revised_vs_primary": "not_run",
            "2state_vs_3state_holdout": "not_run",
        },
        "precommit_hash_reference": precommit.get("created_at"),
    }
    (OUT_ROOT / "final_holdout_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    report = f"""# Final Holdout Evaluation Report (2024–2025)

Generated: {decision["generated_at"]}

## Status: BLOCKED — NOT EXECUTED

The precommit specification was frozen before any holdout portfolio evaluation. The prerequisite gate failed because **no frozen D2/F1C predictions or panel rows exist for 2024–2025**.

## Identity of candidates (frozen, not run)

| Arm | Identity | HMM | Weights | Exposure |
|-----|----------|-----|---------|----------|
| **H0** | Original validation-selected primary (T0) | 2-state, seed=42 | W1 fixed 0.7/0.3 all states | 0.95 |
| **H1** | Original 2-state dynamic hypothesis (T1) | 2-state, seed=42 | W2 Bull 0.8/0.2, Bear 0.4/0.6 | 0.95 |
| **H2** | R2 3-state fixed control | R2 3-state, seed=42 | W1 fixed all states | 0.95 |
| **H3** | Post-hoc revised R2 dynamic candidate | R2 3-state, seed=42 | W2 with Neutral | 0.95 |

## Prerequisite audit

| Component | Max date | Holdout rows (≥2024-01-01) |
|-----------|----------|---------------------------|
| Panel | {audit["panel_date_range"][1]} | {audit["holdout_rows_panel"]} |
| D2 pred | {audit["d2_date_range"][1]} | {audit["holdout_rows_d2"]} |
| F1C pred | {audit["f1c_date_range"][1]} | {audit["holdout_rows_f1c"]} |
| panel ∩ D2 ∩ F1C | — | {audit["holdout_rows_panel_intersect_d2_intersect_f1c"]} |

## Protocol compliance

- Precommit written at: `{audit["precommit_created_at"]}`
- Holdout portfolio data accessed: **no** (0 rows)
- 2026+ data accessed: **no**
- Sentiment: excluded (`w_sentiment=0`)
- Post-run tuning / new candidates: **forbidden**

## Conclusion

{decision["conclusion_text"]}

## Required next step

{decision["required_next_step"]}

```json
{json.dumps(decision, indent=2)}
```
"""
    (REPORT_ROOT / "final_holdout_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    precommit_path, precommit = load_or_write_precommit()
    audit = check_prerequisites(precommit["created_at"])

    if audit["status"] != "READY_TO_RUN":
        write_blocked_outputs(audit, precommit)
        print(f"BLOCKED: {audit['blockers']}")
        print(f"Wrote blocked holdout artifacts under {OUT_ROOT}")
        return

    from portfolio_experiments.final_holdout.holdout_runner import run_holdout_evaluation

    audit["holdout_first_portfolio_access_at"] = datetime.now(timezone.utc).isoformat()
    run_holdout_evaluation(precommit, audit)
    print(f"Holdout evaluation complete. Artifacts under {OUT_ROOT}")


if __name__ == "__main__":
    main()
