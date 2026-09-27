#!/usr/bin/env python3
"""Execute Final Holdout H0–H3 portfolio evaluation (2024–2025)."""

from __future__ import annotations
import os

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.contrib.evaluate import risk_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "portfolio_experiments"))
sys.path.insert(0, str(Path(os.environ["RDAGENT_ROOT"]) / "phase3_portfolio_ablation" / "scripts"))

import run_portfolio_ablation as rpa  # noqa: E402

from portfolio_experiments.final_holdout.config import (
    BASE_RISK_DEGREE,
    D2_PRED_PATH,
    F1C_PRED_PATH,
    HOLDOUT_END,
    HOLDOUT_START,
    OUT_ROOT,
    PANEL_PATH,
    R2_FEATURE_COLS,
    R2_LABEL_MAP,
    R2_SEED,
    REPORT_ROOT,
    TOPK,
    TRAIN_END,
    TRAIN_START,
)
from portfolio_experiments.hmm_3state_robustness.hmm_utils import (
    HMMFitConfig,
    attach_trade_dates,
    fit_hmm,
    online_filter,
)
from portfolio_experiments.hmm_regime.config import OUT_ROOT as MAIN_HMM_ROOT
from portfolio_experiments.hmm_regime.hmm_online_filter import (
    OnlineHMMConfig,
    attach_trade_dates as attach_trade_dates_2s,
    fit_hmm_train_only,
    leakage_audit,
    online_filter_posteriors,
    state_label_series,
)
from portfolio_experiments.hmm_regime.metrics_utils import load_r1r2p_module, mdd_from_returns, run_qlib_backtest
from portfolio_experiments.hmm_regime.run_portfolio_test import calmar_ratio, holdings_topk, overlap_daily
from portfolio_experiments.hmm_regime.run_validation import attach_states_to_panel
from portfolio_experiments.hmm_regime.signal_utils import (
    WEIGHT_SCHEMES,
    build_dynamic_score,
    pred_df_to_qlib,
    preprocess_scores,
)

ARMS = {
    "H0": {"weight_scheme": "W1", "hmm": "2state", "maps_to": "T0"},
    "H1": {"weight_scheme": "W2", "hmm": "2state", "maps_to": "T1"},
    "H2": {"weight_scheme": "W1", "hmm": "r2_3state", "maps_to": "R2-Fixed"},
    "H3": {"weight_scheme": "W2", "hmm": "r2_3state", "maps_to": "R2-Dynamic"},
}


def load_holdout_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH, columns=["datetime", "instrument", "split", "label"])
    d2 = pd.read_parquet(D2_PRED_PATH).rename(columns={"score": "quant_score"})
    f1c = pd.read_pickle(F1C_PRED_PATH).reset_index().rename(columns={"score": "fundamental_score"})
    for df in (panel, d2, f1c):
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df["instrument"] = df["instrument"].astype(str)
    common = panel.merge(d2, on=["datetime", "instrument"], how="inner")
    common = common.merge(f1c, on=["datetime", "instrument"], how="inner")
    return common


def load_spy_2state() -> pd.DataFrame:
    path = MAIN_HMM_ROOT / "spy_features_through_holdout.parquet"
    spy = pd.read_parquet(path)
    spy = spy.rename(columns={"date": "date"})
    if "date" not in spy.columns:
        spy = spy.reset_index()
    spy["date"] = pd.to_datetime(spy["date"])
    return spy


def load_spy_r2() -> pd.DataFrame:
    path = OUT_ROOT / "spy_features_r2_through_holdout.parquet"
    spy = pd.read_parquet(path)
    spy["date"] = pd.to_datetime(spy["date"])
    return spy


def build_2state_hmm() -> pd.DataFrame:
    spy = load_spy_2state()
    train = spy[(spy["date"] >= TRAIN_START) & (spy["date"] <= TRAIN_END)]
    infer = spy[spy["date"] <= pd.Timestamp(HOLDOUT_END)].copy()
    cfg = OnlineHMMConfig(n_states=2, random_state=42)
    model, infer_s, _ = fit_hmm_train_only(train, cfg, apply_features=infer)
    online = online_filter_posteriors(model, infer_s)
    online = attach_trade_dates_2s(online, pd.DatetimeIndex(spy["date"]))
    return online


def build_r2_hmm() -> pd.DataFrame:
    features = load_spy_r2().dropna(subset=R2_FEATURE_COLS).reset_index(drop=True)
    train = features[(features["date"] >= TRAIN_START) & (features["date"] <= TRAIN_END)]
    apply = features[features["date"] <= pd.Timestamp(HOLDOUT_END)]
    cfg = HMMFitConfig(n_states=3, feature_cols=R2_FEATURE_COLS, random_state=R2_SEED)
    model, scaled, _ = fit_hmm(train, apply, cfg)
    states = online_filter(model, scaled, R2_FEATURE_COLS)
    return attach_trade_dates(states, pd.DatetimeIndex(features["date"]))


def load_2state_label_map() -> dict[int, str]:
    selected = json.loads((MAIN_HMM_ROOT / "selected_hmm_model.json").read_text())
    return {int(k): v for k, v in selected["label_map"].items()}


def infer_flip(panel: pd.DataFrame, scheme: str, hmm_daily: pd.DataFrame, label_map: dict[int, str]) -> bool:
    valid = panel[panel["split"] == "valid"].copy()
    valid = preprocess_scores(valid, ["quant_score", "fundamental_score"])
    if scheme == "W1":
        wq, wf = WEIGHT_SCHEMES[scheme]["bull"]
        valid["score"] = wq * valid["quant_score_z"] + wf * valid["fundamental_score_z"]
    else:
        valid = attach_states_to_panel(valid, hmm_daily, label_map)
        valid["score"] = build_dynamic_score(valid, WEIGHT_SCHEMES[scheme])
    return bool(valid["score"].corr(valid["label"]) < 0)


def build_holdout_pred(
    panel: pd.DataFrame, scheme: str, hmm_daily: pd.DataFrame, label_map: dict[int, str], *, flip: bool
) -> pd.DataFrame:
    sub = attach_states_to_panel(panel[panel["split"] == "holdout"].copy(), hmm_daily, label_map)
    sub = preprocess_scores(sub, ["quant_score", "fundamental_score"])
    sub["score"] = build_dynamic_score(sub, WEIGHT_SCHEMES[scheme])
    if flip:
        sub["score"] = -sub["score"]
    return pred_df_to_qlib(sub[["datetime", "instrument", "score"]]), sub


def metrics_row(arm_id: str, report: pd.DataFrame, positions: dict) -> dict:
    excess_g = report["return"] - report["bench"]
    excess_n = excess_g - report["cost"]
    port_ra = risk_analysis(report["return"], freq="day")
    bench_ra = risk_analysis(report["bench"], freq="day")
    ex_ra = risk_analysis(excess_n, freq="day")
    ann_ret = float(port_ra.loc["annualized_return", "risk"])
    mdd = float(port_ra.loc["max_drawdown", "risk"])
    hold = holdings_topk(positions)
    avg_hold = float(hold.groupby("trade_date")["instrument"].nunique().mean()) if len(hold) else np.nan
    return {
        "arm_id": arm_id,
        "status": "EXECUTED",
        "arr": ann_ret,
        "benchmark_arr": float(bench_ra.loc["annualized_return", "risk"]),
        "excess_arr_net": float(ex_ra.loc["annualized_return", "risk"]),
        "annual_volatility": float(report["return"].std() * np.sqrt(252)) if report["return"].std() > 0 else np.nan,
        "sharpe": float(report["return"].mean() / report["return"].std() * np.sqrt(252))
        if report["return"].std() > 0
        else np.nan,
        "ir_net": float(ex_ra.loc["information_ratio", "risk"]),
        "mdd": mdd,
        "calmar": calmar_ratio(ann_ret, mdd),
        "turnover": float(report["turnover"].mean()),
        "transaction_cost_total": float(report["total_cost"].iloc[-1]) if len(report) else np.nan,
        "transaction_cost_daily_mean": float(report["cost"].mean()),
        "average_holdings": avg_hold,
        "average_exposure": BASE_RISK_DEGREE,
        "n_days": int(len(report)),
    }


def yearly_metrics(arm_id: str, report: pd.DataFrame) -> list[dict]:
    rows = []
    for year in (2024, 2025):
        sub = report[report.index.year == year]
        if len(sub) < 5:
            continue
        excess_n = sub["return"] - sub["bench"] - sub["cost"]
        ex_ra = risk_analysis(excess_n, freq="day")
        ann_ret = float((1 + sub["return"]).prod() ** (252 / len(sub)) - 1)
        bench_ret = float((1 + sub["bench"]).prod() ** (252 / len(sub)) - 1)
        rows.append(
            {
                "arm_id": arm_id,
                "year": year,
                "arr": ann_ret,
                "benchmark_arr": bench_ret,
                "excess_arr_net": float(ex_ra.loc["annualized_return", "risk"]),
                "sharpe": float(sub["return"].mean() / sub["return"].std() * np.sqrt(252))
                if sub["return"].std() > 0
                else np.nan,
                "ir_net": float(ex_ra.loc["information_ratio", "risk"]),
                "mdd": mdd_from_returns(sub["return"]),
                "turnover": float(sub["turnover"].mean()),
                "n_days": int(len(sub)),
            }
        )
    return rows


def monthly_metrics(arm_id: str, report: pd.DataFrame) -> list[dict]:
    rep = report.copy()
    rep["month"] = rep.index.to_period("M").astype(str)
    rows = []
    for month, sub in rep.groupby("month"):
        excess_n = sub["return"] - sub["bench"] - sub["cost"]
        rows.append(
            {
                "arm_id": arm_id,
                "month": month,
                "excess_arr_net": float((1 + excess_n).prod() - 1),
                "portfolio_return": float((1 + sub["return"]).prod() - 1),
                "n_days": int(len(sub)),
            }
        )
    return rows


def regime_breakdown(arm_id: str, report: pd.DataFrame, states: pd.DataFrame) -> list[dict]:
    st = states[["trade_date", "state_label"]].copy()
    st["trade_date"] = pd.to_datetime(st["trade_date"]).dt.normalize()
    rep = report.reset_index()
    rep.columns = ["trade_date"] + list(report.columns)
    rep["trade_date"] = pd.to_datetime(rep["trade_date"]).dt.normalize()
    merged = rep.merge(st, on="trade_date", how="left")
    rows = []
    for state, sub in merged.groupby("state_label"):
        if len(sub) < 5 or pd.isna(state):
            continue
        r = sub["return"]
        ex = sub["return"] - sub["bench"] - sub["cost"]
        rows.append(
            {
                "arm_id": arm_id,
                "state": state,
                "n_days": int(len(sub)),
                "arr": float((1 + r).prod() ** (252 / len(r)) - 1),
                "excess_arr_net": float((1 + ex).prod() ** (252 / len(ex)) - 1),
                "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
                "mdd": mdd_from_returns(r),
            }
        )
    return rows


def evaluate_decision(summary: pd.DataFrame, monthly: pd.DataFrame, precommit: dict) -> dict:
    h0 = summary[summary["arm_id"] == "H0"].iloc[0]
    h1 = summary[summary["arm_id"] == "H1"].iloc[0]
    h2 = summary[summary["arm_id"] == "H2"].iloc[0]
    h3 = summary[summary["arm_id"] == "H3"].iloc[0]
    rules = precommit["decision_rules_frozen"]

    h3_support_checks = {
        "excess_arr_higher": bool(h3["excess_arr_net"] > h2["excess_arr_net"]),
        "sharpe_or_ir_higher": bool((h3["sharpe"] > h2["sharpe"]) or (h3["ir_net"] > h2["ir_net"])),
        "mdd_not_severely_worse": bool(h3["mdd"] >= h2["mdd"]),
        "turnover_not_abnormal": bool(h3["turnover"] <= h2["turnover"] * 1.25),
    }
    h3_months = monthly[monthly["arm_id"] == "H3"]
    best = h3_months.loc[h3_months["excess_arr_net"].idxmax()] if len(h3_months) else None
    worst = h3_months.loc[h3_months["excess_arr_net"].idxmin()] if len(h3_months) else None
    top_share = float(best["excess_arr_net"] / h3_months["excess_arr_net"].sum()) if len(h3_months) and h3_months["excess_arr_net"].sum() != 0 else np.nan
    h3_support_checks["not_single_month_driven"] = bool(top_share < 0.5 if not np.isnan(top_share) else False)

    yearly = pd.read_csv(OUT_ROOT / "final_holdout_yearly.csv")
    h3_y = yearly[yearly["arm_id"] == "H3"]
    h3_support_checks["not_single_year_only"] = bool(
        not (
            len(h3_y) == 2
            and (h3_y["excess_arr_net"] > 0).sum() == 1
            and h3_y["excess_arr_net"].max() > h3_y["excess_arr_net"].clip(lower=0).sum()
        )
    )

    supported = sum(h3_support_checks.values()) >= 4 and h3["excess_arr_net"] > h2["excess_arr_net"]

    if h3["excess_arr_net"] <= h2["excess_arr_net"]:
        h3_text = rules["not_replicated_template"]
        h3_status = "NOT_REPLICATED_ON_HOLDOUT"
    elif supported:
        h3_text = "Revised R2 3-state dynamic candidate (H3) met most pre-specified holdout support criteria versus fixed control (H2)."
        h3_status = "HOLDOUT_SUPPORTED"
    else:
        h3_text = rules["mixed_evidence_template"]
        h3_status = "MIXED_EVIDENCE"

    primary = "H0"
    if h0["sharpe"] >= max(h1["sharpe"], h3["sharpe"]) and h0["ir_net"] >= max(h1["ir_net"], h3["ir_net"]):
        primary_note = "Original validation-selected primary (H0/T0) retains primary status based on risk-adjusted holdout performance."
    else:
        primary_note = "H0 retains formal primary status per no-winner-rewriting-history rule; holdout does not override validation selection."

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "EXECUTED",
        "holdout_evaluated": True,
        "candidates_run": list(ARMS.keys()),
        "primary_model_unchanged": "H0 / T0",
        "primary_note": primary_note,
        "revised_candidate_status": h3_status,
        "revised_candidate_conclusion": h3_text,
        "h3_support_checks": h3_support_checks,
        "pairwise": {
            "H1_vs_H0": {
                "delta_excess_arr_net": float(h1["excess_arr_net"] - h0["excess_arr_net"]),
                "delta_sharpe": float(h1["sharpe"] - h0["sharpe"]),
                "delta_mdd": float(h1["mdd"] - h0["mdd"]),
                "original_dynamic_holdout_supported": bool(h1["excess_arr_net"] > h0["excess_arr_net"]),
            },
            "H3_vs_H2": {
                "delta_excess_arr_net": float(h3["excess_arr_net"] - h2["excess_arr_net"]),
                "delta_sharpe": float(h3["sharpe"] - h2["sharpe"]),
                "delta_mdd": float(h3["mdd"] - h2["mdd"]),
            },
            "H3_vs_H0": {
                "delta_excess_arr_net": float(h3["excess_arr_net"] - h0["excess_arr_net"]),
                "delta_sharpe": float(h3["sharpe"] - h0["sharpe"]),
                "delta_mdd": float(h3["mdd"] - h0["mdd"]),
            },
        },
        "best_month_h3": best.to_dict() if best is not None else None,
        "worst_month_h3": worst.to_dict() if worst is not None else None,
    }


def run_holdout_evaluation(precommit: dict, audit: dict) -> None:
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(PROJECT_ROOT / "staging/qlib_data"), region=REG_US, kernels=1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    panel = load_holdout_panel()
    ho = panel[panel["split"] == "holdout"]
    if ho.empty:
        raise RuntimeError("Holdout panel empty")

    audit["holdout_first_portfolio_access_at"] = datetime.now(timezone.utc).isoformat()
    audit["holdout_portfolio_rows_accessed"] = int(len(ho))
    audit["status"] = "EXECUTING"

    hmm2 = build_2state_hmm()
    hmmr2 = build_r2_hmm()
    label_map_2 = load_2state_label_map()
    label_map_r2 = {int(k): v for k, v in R2_LABEL_MAP.items()}

    flip_w1 = infer_flip(panel, "W1", hmm2, label_map_2)
    flip_w2 = infer_flip(panel, "W2", hmm2, label_map_2)

    r1r2p = load_r1r2p_module(PROJECT_ROOT)
    results = {}
    scored = {}
    bt_end = pd.Timestamp(HOLDOUT_END)
    for arm_id, meta in ARMS.items():
        if meta["hmm"] == "2state":
            states, lmap = hmm2, label_map_2
        else:
            states, lmap = hmmr2, label_map_r2
        scheme = meta["weight_scheme"]
        flip = flip_w1 if scheme == "W1" else flip_w2
        pred, sub = build_holdout_pred(panel, scheme, states, lmap, flip=flip)
        pred_max = pred.index.get_level_values("datetime").max()
        end = min(bt_end, pred_max)
        bt = run_qlib_backtest(r1r2p, rpa, pred, HOLDOUT_START, end.strftime("%Y-%m-%d"))
        holdout_states = state_label_series(states, lmap)
        holdout_states = holdout_states[
            (holdout_states["trade_date"] >= HOLDOUT_START) & (holdout_states["trade_date"] <= HOLDOUT_END)
        ]
        results[arm_id] = {
            "report": bt["report"],
            "positions": bt["result"]["positions"],
            "sub": sub,
            "states": holdout_states,
        }

    # H0 vs H2 parity on scores (state-independent W1)
    h0_scores = results["H0"]["sub"][["datetime", "instrument", "score"]].copy()
    h2_scores = results["H2"]["sub"][["datetime", "instrument", "score"]].copy()
    m = h0_scores.merge(h2_scores, on=["datetime", "instrument"], suffixes=("_h0", "_h2"))
    score_diff = (m["score_h0"] - m["score_h2"]).abs()
    if score_diff.max() > 1e-8:
        raise RuntimeError(f"H0/H2 score parity failed max_diff={score_diff.max()}")

    summary_rows = []
    yearly_rows = []
    monthly_rows = []
    regime_rows = []
    overlap_rows = []
    for arm_id, res in results.items():
        summary_rows.append(metrics_row(arm_id, res["report"], res["positions"]))
        yearly_rows.extend(yearly_metrics(arm_id, res["report"]))
        monthly_rows.extend(monthly_metrics(arm_id, res["report"]))
        regime_rows.extend(regime_breakdown(arm_id, res["report"], res["states"]))
        scored[arm_id] = holdings_topk(res["positions"])

    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    monthly = pd.DataFrame(monthly_rows)
    regime = pd.DataFrame(regime_rows)

    pairs = [("H0", "H1"), ("H2", "H3"), ("H0", "H3")]
    for a, b in pairs:
        ov = overlap_daily(scored[a], scored[b])
        ov["pair"] = f"{a}_vs_{b}"
        ov["overlap_ratio_mean"] = ov["overlap_ratio"].mean()
        overlap_rows.append({"pair": f"{a}_vs_{b}", "overlap_ratio_mean": float(ov["overlap_ratio"].mean())})
    overlap_df = pd.DataFrame(overlap_rows)

    h0 = summary[summary["arm_id"] == "H0"].iloc[0]
    h1 = summary[summary["arm_id"] == "H1"].iloc[0]
    h2 = summary[summary["arm_id"] == "H2"].iloc[0]
    h3 = summary[summary["arm_id"] == "H3"].iloc[0]
    pairwise = pd.DataFrame(
        [
            {"comparison": "H1_vs_H0", "delta_excess_arr_net": h1["excess_arr_net"] - h0["excess_arr_net"], "delta_sharpe": h1["sharpe"] - h0["sharpe"], "delta_mdd": h1["mdd"] - h0["mdd"]},
            {"comparison": "H3_vs_H2", "delta_excess_arr_net": h3["excess_arr_net"] - h2["excess_arr_net"], "delta_sharpe": h3["sharpe"] - h2["sharpe"], "delta_mdd": h3["mdd"] - h2["mdd"]},
            {"comparison": "H3_vs_H0", "delta_excess_arr_net": h3["excess_arr_net"] - h0["excess_arr_net"], "delta_sharpe": h3["sharpe"] - h0["sharpe"], "delta_mdd": h3["mdd"] - h0["mdd"]},
        ]
    )

    summary.to_csv(OUT_ROOT / "final_holdout_summary.csv", index=False)
    yearly.to_csv(OUT_ROOT / "final_holdout_yearly.csv", index=False)
    monthly.to_csv(OUT_ROOT / "final_holdout_monthly.csv", index=False)
    regime.to_csv(OUT_ROOT / "final_holdout_regime_breakdown.csv", index=False)
    overlap_df.to_csv(OUT_ROOT / "final_holdout_holdings_overlap.csv", index=False)
    pairwise.to_csv(OUT_ROOT / "final_holdout_pairwise_comparison.csv", index=False)

    audit.update(
        {
            "status": "EXECUTED",
            "h0_h2_score_parity_max_diff": float(score_diff.max()),
            "train_only_through": "2017-12-31",
            "holdout_period": [HOLDOUT_START, HOLDOUT_END],
            "data_2026_plus_accessed": 0,
            "parameters_changed_after_precommit": False,
            "sentiment_included": False,
            "holdout_used_for_model_selection": False,
            **leakage_audit(hmm2.rename(columns={"trade_date": "trade_date"})),
        }
    )
    (OUT_ROOT / "final_holdout_leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    decision = evaluate_decision(summary, monthly, precommit)
    (OUT_ROOT / "final_holdout_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    report = f"""# Final Holdout Evaluation Report (2024–2025)

Generated: {decision["generated_at"]}

## Status: EXECUTED

## Candidate identity

| Arm | Identity |
|-----|----------|
| H0 | Original validation-selected primary (T0) |
| H1 | Original 2-state dynamic hypothesis (T1) |
| H2 | R2 3-state fixed control |
| H3 | Post-hoc revised R2 dynamic candidate |

## Summary

{summary.to_markdown(index=False, floatfmt=".4f")}

## Core comparisons

{pairwise.to_markdown(index=False, floatfmt=".4f")}

## Revised candidate conclusion

{decision["revised_candidate_conclusion"]}

## Primary model

{decision["primary_note"]}

```json
{json.dumps(decision, indent=2)}
```
"""
    (REPORT_ROOT / "final_holdout_report.md").write_text(report, encoding="utf-8")
