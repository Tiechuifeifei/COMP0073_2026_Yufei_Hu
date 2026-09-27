#!/usr/bin/env python3
"""2020–2023 portfolio test for the frozen T0–T3 arms (development window only)."""

from __future__ import annotations
import os

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.contrib.evaluate import risk_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(os.environ["RDAGENT_ROOT"]) / "phase3_portfolio_ablation" / "scripts"))

import run_portfolio_ablation as rpa  # noqa: E402

from build_spy_features import build_spy_features
from config import (
    BASE_RISK_DEGREE,
    HARD_CUTOFF,
    HOLDOUT_START,
    OUT_ROOT,
    REPORT_ROOT,
    TEST_END,
    TEST_START,
    TOPK,
    TRAIN_END,
    TRAIN_START,
)
from hmm_online_filter import (
    OnlineHMMConfig,
    attach_trade_dates,
    fit_hmm_train_only,
    leakage_audit,
    online_filter_posteriors,
    state_label_series,
)
from metrics_utils import (
    load_r1r2p_module,
    mdd_from_returns,
    portfolio_by_year,
    run_qlib_backtest,
)
from regime_exposure_strategy import GLOBAL_AUDIT_LOG
from run_validation import (
    attach_states_to_panel,
    build_exposure_schedule,
    load_common_panel,
)
from signal_utils import (
    WEIGHT_SCHEMES,
    build_dynamic_score,
    pred_df_to_qlib,
    preprocess_scores,
)

TEST_ARMS = {
    "T0": {
        "label": "Selected baseline",
        "role": "validation_selected",
        "weight_scheme": "W1",
        "exposure_rule": "fixed_095",
        "hypothesis": "validation-selected strategy",
    },
    "T1": {
        "label": "Dynamic factor test",
        "role": "dynamic_factor_hypothesis",
        "weight_scheme": "W2",
        "exposure_rule": "fixed_095",
        "hypothesis": "HMM dynamic factor weights vs fixed fusion",
    },
    "T2": {
        "label": "Dynamic exposure test",
        "role": "dynamic_exposure_hypothesis",
        "weight_scheme": "W1",
        "exposure_rule": "Soft-2A",
        "hypothesis": "Soft-2A not validation-selected; research arm only",
    },
    "T3": {
        "label": "Combined test",
        "role": "combined_hypothesis",
        "weight_scheme": "W2",
        "exposure_rule": "Soft-2A",
        "hypothesis": "combined dynamic factor + dynamic exposure",
    },
}


def calmar_ratio(ann_ret: float, mdd: float) -> float:
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(ann_ret / abs(mdd))


def infer_score_sign(panel: pd.DataFrame, scheme_id: str) -> bool:
    """Return True if scores should be flipped (validation split only)."""
    valid = panel[panel["split"] == "valid"].copy()
    valid = preprocess_scores(valid, ["quant_score", "fundamental_score"])
    valid["score"] = build_dynamic_score(valid, WEIGHT_SCHEMES[scheme_id])
    return bool(valid["score"].corr(valid["label"]) < 0)


def build_scheme_pred(panel: pd.DataFrame, scheme_id: str, *, flip: bool, split: str = "test") -> pd.DataFrame:
    sub = panel[panel["split"] == split].copy()
    sub = preprocess_scores(sub, ["quant_score", "fundamental_score"])
    sub["score"] = build_dynamic_score(sub, WEIGHT_SCHEMES[scheme_id])
    if flip:
        sub["score"] = -sub["score"]
    return pred_df_to_qlib(sub[["datetime", "instrument", "score"]])


def frozen_hmm_timeline() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str], int]:
    """Refit frozen 2-state seed=42 on 2008–2017; online filter through TEST_END."""
    selected = json.loads((OUT_ROOT / "selected_hmm_model.json").read_text())
    label_map = {int(k): v for k, v in selected["label_map"].items()}
    n_states = int(selected["n_states"])
    seed = int(selected["seed"])
    if n_states != 2 or seed != 42:
        raise RuntimeError(f"Frozen HMM mismatch: n_states={n_states} seed={seed}")

    spy = build_spy_features()
    if spy["date"].max() > pd.Timestamp(HARD_CUTOFF):
        raise RuntimeError("SPY features exceed HARD_CUTOFF")
    train = spy[(spy["date"] >= TRAIN_START) & (spy["date"] <= TRAIN_END)]
    infer = spy[spy["date"] <= pd.Timestamp(TEST_END)].copy()
    cfg = OnlineHMMConfig(n_states=n_states, random_state=seed)
    model, infer_s, _ = fit_hmm_train_only(train, cfg, apply_features=infer)
    online = online_filter_posteriors(model, infer_s)
    online = attach_trade_dates(online, pd.DatetimeIndex(spy["date"]))
    online = state_label_series(online, label_map)

    full_path = OUT_ROOT / "hmm_daily_states_through_test.csv"
    online.to_csv(full_path, index=False)

    test_tl = online[(online["trade_date"] >= TEST_START) & (online["trade_date"] <= TEST_END)].copy()
    test_tl = test_tl.dropna(subset=["trade_date"]).sort_values("trade_date")
    test_tl = test_tl.groupby("trade_date", as_index=False).last()
    test_tl["state_label"] = test_tl["state_label"].fillna(test_tl["hmm_state"] if "hmm_state" in test_tl else np.nan)
    return online, test_tl, label_map, n_states


def run_arm(
    arm_id: str,
    pred: pd.DataFrame,
    states_test: pd.DataFrame,
    n_states: int,
    r1r2p,
) -> dict:
    meta = TEST_ARMS[arm_id]
    rule = meta["exposure_rule"]
    audit_log: list = []

    if rule == "fixed_095":
        bt = run_qlib_backtest(r1r2p, rpa, pred, TEST_START, TEST_END)
        schedule = {d.strftime("%Y-%m-%d"): BASE_RISK_DEGREE for d in states_test["trade_date"]}
    else:
        schedule = build_exposure_schedule(states_test, rule, n_states)
        GLOBAL_AUDIT_LOG.clear()
        bt = run_qlib_backtest(
            r1r2p,
            rpa,
            pred,
            TEST_START,
            TEST_END,
            strategy_class="RegimeExposureTopkStrategy",
            strategy_module="regime_exposure_strategy",
            extra_kwargs={
                "exposure_by_date": schedule,
                "use_target_exposure_scaling": True,
                "audit_log": audit_log,
            },
        )
        audit_log = list(GLOBAL_AUDIT_LOG)

    return {
        "arm_id": arm_id,
        "meta": meta,
        "bt": bt,
        "schedule": schedule,
        "audit_log": audit_log,
    }


def metrics_row(arm_id: str, meta: dict, report: pd.DataFrame, period_label: str) -> dict:
    excess_g = report["return"] - report["bench"]
    excess_n = excess_g - report["cost"]
    port_ra = risk_analysis(report["return"], freq="day")
    bench_ra = risk_analysis(report["bench"], freq="day")
    ex_ra = risk_analysis(excess_n, freq="day")
    ann_ret = float(port_ra.loc["annualized_return", "risk"])
    mdd = float(port_ra.loc["max_drawdown", "risk"])
    return {
        "arm_id": arm_id,
        "arm_label": meta["label"],
        "role": meta["role"],
        "weight_scheme": meta["weight_scheme"],
        "exposure_rule": meta["exposure_rule"],
        "period": period_label,
        "arr": ann_ret,
        "benchmark_arr": float(bench_ra.loc["annualized_return", "risk"]),
        "excess_arr_gross": float(risk_analysis(excess_g, freq="day").loc["annualized_return", "risk"]),
        "excess_arr_net": float(ex_ra.loc["annualized_return", "risk"]),
        "annual_volatility": float(report["return"].std() * np.sqrt(252)) if report["return"].std() > 0 else np.nan,
        "sharpe": float(report["return"].mean() / report["return"].std() * np.sqrt(252))
        if report["return"].std() > 0
        else np.nan,
        "ir_gross": float(risk_analysis(excess_g, freq="day").loc["information_ratio", "risk"]),
        "ir_net": float(ex_ra.loc["information_ratio", "risk"]),
        "mdd": mdd,
        "calmar": calmar_ratio(ann_ret, mdd),
        "turnover": float(report["turnover"].mean()),
        "transaction_cost_total": float(report["total_cost"].iloc[-1]) if len(report) else np.nan,
        "transaction_cost_daily_mean": float(report["cost"].mean()),
        "n_days": int(len(report)),
    }


def regime_metrics(arm_id: str, report: pd.DataFrame, states: pd.DataFrame) -> list[dict]:
    st = states[["trade_date", "state_label", "p_bull", "p_bear"]].copy()
    st["trade_date"] = pd.to_datetime(st["trade_date"]).dt.normalize()
    rep = report.copy()
    rep.index = pd.to_datetime(rep.index).normalize()
    rep = rep.reset_index()
    if "trade_date" not in rep.columns:
        rep = rep.rename(columns={rep.columns[0]: "trade_date"})
    rep["trade_date"] = pd.to_datetime(rep["trade_date"]).dt.normalize()
    merged = rep.merge(st, on="trade_date", how="left")
    rows = []
    for state in ("bull", "bear"):
        sub = merged[merged["state_label"] == state]
        if len(sub) < 5:
            continue
        r = sub["return"]
        ex = sub["return"] - sub["bench"] - sub["cost"]
        mdd = mdd_from_returns(r)
        rows.append(
            {
                "arm_id": arm_id,
                "state": state,
                "n_days": int(len(sub)),
                "arr": float((1 + r).prod() ** (252 / len(r)) - 1),
                "excess_arr_net": float((1 + ex).prod() ** (252 / len(ex)) - 1),
                "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
                "mdd": mdd,
                "mean_target_exposure": float((0.95 * sub["p_bull"] + 0.30 * sub["p_bear"]).mean()),
            }
        )
    return rows


def holdings_topk(positions: dict) -> pd.DataFrame:
    rows = []
    for dt, pos in positions.items():
        ts = pd.Timestamp(dt).normalize()
        for inst in pos.get_stock_list():
            rows.append({"trade_date": ts, "instrument": str(inst)})
    return pd.DataFrame(rows)


def overlap_daily(hold_a: pd.DataFrame, hold_b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = sorted(set(hold_a["trade_date"]).intersection(set(hold_b["trade_date"])))
    for dt in dates:
        sa = set(hold_a.loc[hold_a["trade_date"] == dt, "instrument"])
        sb = set(hold_b.loc[hold_b["trade_date"] == dt, "instrument"])
        rows.append(
            {
                "trade_date": dt,
                "overlap_count": len(sa & sb),
                "overlap_ratio": len(sa & sb) / TOPK if TOPK else np.nan,
            }
        )
    return pd.DataFrame(rows)


def transition_table(states: pd.DataFrame) -> pd.DataFrame:
    tl = states.sort_values("trade_date").copy()
    tl["previous_state"] = tl["state_label"].shift(1)
    trans = tl[tl["previous_state"].notna() & (tl["previous_state"] != tl["state_label"])]
    rows = []
    for _, r in trans.iterrows():
        rows.append(
            {
                "trade_date": r["trade_date"],
                "transition": f"{r['previous_state']}→{r['state_label']}",
                "previous_state": r["previous_state"],
                "current_state": r["state_label"],
            }
        )
    return pd.DataFrame(rows)


def transition_forward_perf(report: pd.DataFrame, trans: pd.DataFrame, *, arm_id: str) -> list[dict]:
    rep = report.copy()
    rep.index = pd.to_datetime(rep.index).normalize()
    dates = list(rep.index)
    idx = {d: i for i, d in enumerate(dates)}
    rows = []
    for _, t in trans.iterrows():
        td = pd.Timestamp(t["trade_date"]).normalize()
        if td not in idx:
            continue
        i = idx[td]
        for h, name in [(1, "1d"), (5, "5d"), (20, "20d")]:
            if i + h < len(dates):
                window = rep.iloc[i + 1 : i + h + 1]
                cum = float((1 + window["return"]).prod() - 1)
                rows.append(
                    {
                        "arm_id": arm_id,
                        "trade_date": td,
                        "transition": t["transition"],
                        "horizon": name,
                        "portfolio_forward_return": cum,
                    }
                )
    return rows


def return_attribution_vs_baseline(
    base_report: pd.DataFrame,
    dyn_report: pd.DataFrame,
    states: pd.DataFrame,
    *,
    arm_id: str,
    baseline_id: str = "T0",
) -> pd.DataFrame:
    st = states[["trade_date", "state_label", "p_bull", "p_bear"]].copy()
    st["trade_date"] = pd.to_datetime(st["trade_date"]).dt.normalize()
    st["target_exposure"] = np.clip(0.95 * st["p_bull"] + 0.30 * st["p_bear"], 0, 0.95)

    b = base_report[["return", "cost"]].rename(columns={"return": "return_baseline", "cost": "cost_baseline"})
    d = dyn_report[["return", "cost"]].rename(columns={"return": "return_arm", "cost": "cost_arm"})
    df = b.join(d, how="inner")
    df.index = pd.to_datetime(df.index).normalize()
    df = df.reset_index()
    if "trade_date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df = df.merge(st, on="trade_date", how="left")
    df["return_difference"] = df["return_arm"] - df["return_baseline"]
    df["cost_difference"] = df["cost_arm"] - df["cost_baseline"]
    df["arm_id"] = arm_id
    df["baseline_id"] = baseline_id
    return df


def attribution_summary(df: pd.DataFrame) -> dict:
    return {
        "total_return_diff": float(df["return_difference"].sum()),
        "missed_upside_when_baseline_positive": float(
            (-df.loc[df["return_baseline"] > 0, "return_difference"].clip(upper=0)).sum()
        ),
        "saved_downside_when_baseline_negative": float(
            df.loc[df["return_baseline"] < 0, "return_difference"].clip(lower=0).sum()
        ),
        "extra_cost": float(df["cost_difference"].sum()),
        "mean_target_exposure": float(df["target_exposure"].mean()) if "target_exposure" in df else np.nan,
    }


def test_leakage_audit(
    panel: pd.DataFrame,
    preds: dict[str, pd.DataFrame],
    hmm_online: pd.DataFrame,
    test_tl: pd.DataFrame,
) -> dict:
    test_panel = panel[panel["split"] == "test"]
    pred_max = max(p.index.get_level_values("datetime").max() for p in preds.values())
    leak = leakage_audit(test_tl.rename(columns={"trade_date": "trade_date"}))
    return {
        "test_period": [TEST_START, TEST_END],
        "hard_cutoff": HARD_CUTOFF,
        "holdout_start": HOLDOUT_START,
        "holdout_accessed": False,
        "panel_test_max_date": str(test_panel["datetime"].max().date()),
        "prediction_max_date": str(pd.Timestamp(pred_max).date()),
        "hmm_max_information_cutoff": str(pd.to_datetime(hmm_online["information_cutoff_date"]).max().date()),
        "hmm_max_trade_date": str(pd.to_datetime(test_tl["trade_date"]).max().date()),
        "prediction_dates_within_test": bool(pd.Timestamp(pred_max) <= pd.Timestamp(TEST_END)),
        "no_2024_plus_predictions": bool(pd.Timestamp(pred_max) < pd.Timestamp(HOLDOUT_START)),
        "frozen_hmm_seed": 42,
        "frozen_n_states": 2,
        "frozen_label_map": json.loads((OUT_ROOT / "selected_hmm_model.json").read_text())["label_map"],
        "reselected_on_test": False,
        **leak,
    }


def write_report(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    regime: pd.DataFrame,
    attr_summaries: dict[str, dict],
    trans_summary: pd.DataFrame,
    comparisons: list[dict],
) -> str:
    t0 = summary[(summary["arm_id"] == "T0") & (summary["period"] == "test_all")].iloc[0]

    def row(arm):
        return summary[(summary["arm_id"] == arm) & (summary["period"] == "test_all")].iloc[0]

    lines = [
        "# HMM Regime Portfolio Test Report (2020–2023)",
        "",
        f"*Generated: {datetime.now(timezone.utc).isoformat()}*",
        "",
        "## Frozen configuration (no test-time tuning)",
        "",
        "- HMM: 2-state, seed=42, train 2008–2017, online forward filter",
        "- Sample: `panel ∩ D2 ∩ F1C`",
        "- Selected baseline **T0**: W1 (0.7 D2 + 0.3 F1C) + fixed exposure 0.95",
        "- **T2/T3 Soft-2A**: pre-specified research arm, **not** validation-selected exposure",
        "- 2024–2025 holdout: **not accessed**",
        "",
        "## Test arms",
        "",
        "| Arm | Role | Factor | Exposure |",
        "|-----|------|--------|----------|",
    ]
    for arm, meta in TEST_ARMS.items():
        lines.append(f"| **{arm}** | {meta['label']} | {meta['weight_scheme']} | {meta['exposure_rule']} |")

    lines += ["", "## Overall results (2020–2023)", ""]
    lines.append(
        "| Arm | Excess ARR (net) | Sharpe | IR (net) | MDD | Calmar | Turnover |"
    )
    lines.append("|-----|-----------------:|-------:|---------:|----:|-------:|---------:|")
    for arm in TEST_ARMS:
        s = row(arm)
        lines.append(
            f"| {arm} | {s['excess_arr_net']:.2%} | {s['sharpe']:.2f} | {s['ir_net']:.2f} | "
            f"{s['mdd']:.2%} | {s['calmar']:.2f} | {s['turnover']:.3f} |"
        )

    lines += ["", "## Core comparisons vs T0 (selected baseline)", ""]
    for c in comparisons:
        lines.append(
            f"- **{c['pair']}**: ΔExcess ARR {c['delta_excess_arr']:+.2%}, ΔSharpe {c['delta_sharpe']:+.2f}, "
            f"ΔMDD {c['delta_mdd']:+.2%}"
        )

    lines += ["", "## Year-by-year excess ARR (net)", ""]
    pivot = yearly.pivot(index="year", columns="arm_id", values="excess_arr_net")
    try:
        lines.append(pivot.to_markdown(floatfmt=".2%"))
    except ImportError:
        lines.append(pivot.to_string(float_format=lambda x: f"{x:.2%}"))

    lines += ["", "## Regime breakdown (trade-date states)", ""]
    if not regime.empty:
        try:
            lines.append(regime.to_markdown(index=False, floatfmt=".2%"))
        except ImportError:
            lines.append(regime.to_string(index=False))

    lines += ["", "## Dynamic exposure attribution vs T0", ""]
    for arm, summ in attr_summaries.items():
        lines.append(f"### {arm}")
        lines.append(f"- Total return diff: {summ['total_return_diff']:.2%}")
        lines.append(f"- Missed upside (baseline up days): {summ['missed_upside_when_baseline_positive']:.2%}")
        lines.append(f"- Saved downside (baseline down days): {summ['saved_downside_when_baseline_negative']:.2%}")
        lines.append(f"- Extra cost: {summ['extra_cost']:.4%}")

    if not trans_summary.empty:
        lines += ["", "## State transitions on test (portfolio forward returns)", ""]
        agg = trans_summary.groupby(["transition", "horizon"])["portfolio_forward_return"].agg(["mean", "count"])
        try:
            lines.append(agg.to_markdown(floatfmt=".2%"))
        except ImportError:
            lines.append(agg.to_string(float_format=lambda x: f"{x:.2%}"))

    lines += [
        "",
        "## Interpretation (with validation transition audit)",
        "",
        "1. **T0** is the validation-selected strategy; all hypothesis tests are vs this baseline.",
        "2. **T1 vs T0**: tests whether W2 dynamic factor weights improve stock ranking / portfolio.",
        "3. **T2 vs T0**: tests Soft-2A dynamic cash–equity mix (research arm; validation chose fixed 0.95).",
        "4. Validation audit showed HMM **lag-1** states + Soft-2A tend to **cut exposure after drawdowns** "
        "and miss rebounds; expect T2 lower ARR, possibly lower MDD.",
        "5. Check yearly table for single-year drivers (2020 COVID, 2022 bear, 2023 rally).",
        "",
        "## External reference only (not in main matrix)",
        "",
        "- D2-native / D3-native frozen portfolio results use different samples; do not compare directly.",
        "",
        f"*Baseline T0 excess ARR net: {t0['excess_arr_net']:.2%}*",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    r1r2p = load_r1r2p_module(PROJECT_ROOT)
    rpa.init_qlib({"provider_uri": str(PROJECT_ROOT / "staging/qlib_data"), "region": "us"})

    hmm_online, states_test, label_map, n_states = frozen_hmm_timeline()
    panel = load_common_panel()
    panel = attach_states_to_panel(panel, hmm_online, label_map)

    flip_w1 = infer_score_sign(panel, "W1")
    flip_w2 = infer_score_sign(panel, "W2")
    preds = {
        "W1": build_scheme_pred(panel, "W1", flip=flip_w1),
        "W2": build_scheme_pred(panel, "W2", flip=flip_w2),
    }

    results = {}
    for arm_id, meta in TEST_ARMS.items():
        pred = preds[meta["weight_scheme"]]
        results[arm_id] = run_arm(arm_id, pred, states_test, n_states, r1r2p)
        print(f"Completed {arm_id}: excess_arr={results[arm_id]['bt']['metrics']['excess_annualized_return_net']:.2%}")

    pred_by_arm = {arm: preds[TEST_ARMS[arm]["weight_scheme"]] for arm in TEST_ARMS}
    leak = test_leakage_audit(panel, pred_by_arm, hmm_online, states_test)
    (OUT_ROOT / "test_leakage_audit.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")

    summary_rows, regime_rows = [], []
    daily_frames, exposure_rows = [], []
    overlap_rows, attr_frames, trans_rows = [], [], []

    holdings = {}
    reports = {}

    for arm_id, res in results.items():
        rep = res["bt"]["result"]["report"]
        reports[arm_id] = rep
        meta = res["meta"]
        pos = res["bt"]["result"]["positions"]
        holdings[arm_id] = holdings_topk(pos)

        summary_rows.append(metrics_row(arm_id, meta, rep, "test_all"))
        for yr in sorted(rep.index.year.unique()):
            sub = rep[rep.index.year == yr]
            if len(sub) >= 5:
                summary_rows.append(metrics_row(arm_id, meta, sub, str(int(yr))))

        regime_rows.extend(regime_metrics(arm_id, rep, states_test))

        dr = rep[["return", "bench", "cost", "turnover"]].copy()
        dr["arm_id"] = arm_id
        dr["excess_net"] = dr["return"] - dr["bench"] - dr["cost"]
        dr.index.name = "trade_date"
        daily_frames.append(dr.reset_index())

        sched = res["schedule"]
        for td, exp in sched.items():
            exposure_rows.append({"arm_id": arm_id, "trade_date": td, "target_exposure": exp})
        if res["audit_log"]:
            for a in res["audit_log"]:
                exposure_rows.append(
                    {
                        "arm_id": arm_id,
                        "trade_date": a["trade_date"],
                        "actual_pre_trade_exposure": a.get("pre_trade_equity_exposure"),
                        "target_exposure_audit": a.get("target_risk_degree"),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    yearly_df = summary[summary["period"] != "test_all"].copy()
    yearly_df["year"] = yearly_df["period"].astype(int)

    exp_df = pd.DataFrame(exposure_rows)
    avg_exp = (
        exp_df.groupby("arm_id")["target_exposure"].mean().to_dict()
        if "target_exposure" in exp_df.columns
        else {}
    )
    summary.loc[summary["period"] == "test_all", "avg_equity_exposure"] = summary.loc[
        summary["period"] == "test_all", "arm_id"
    ].map(lambda a: avg_exp.get(a, BASE_RISK_DEGREE if TEST_ARMS[a]["exposure_rule"] == "fixed_095" else np.nan))
    summary["avg_cash_exposure"] = 1.0 - summary["avg_equity_exposure"]

    summary.to_csv(OUT_ROOT / "portfolio_test_summary.csv", index=False)
    yearly_df.to_csv(OUT_ROOT / "portfolio_test_yearly.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(OUT_ROOT / "portfolio_test_regime_breakdown.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_parquet(OUT_ROOT / "portfolio_test_daily_returns.parquet", index=False)
    exp_df.to_csv(OUT_ROOT / "portfolio_test_daily_exposure.csv", index=False)

    pairs = [("T1", "T0"), ("T3", "T1"), ("T3", "T2"), ("T3", "T0"), ("T2", "T0")]
    for a, b in pairs:
        if a in holdings and b in holdings:
            od = overlap_daily(holdings[a], holdings[b])
            od["pair"] = f"{a}_vs_{b}"
            overlap_rows.append(od)
    pd.concat(overlap_rows, ignore_index=True).to_csv(OUT_ROOT / "portfolio_test_holdings_overlap.csv", index=False)

    trans = transition_table(states_test)
    attr_summaries = {}
    for arm in ("T2", "T3"):
        attr = return_attribution_vs_baseline(reports["T0"], reports[arm], states_test, arm_id=arm)
        attr_frames.append(attr)
        attr_summaries[arm] = attribution_summary(attr)
        trans_rows.extend(transition_forward_perf(reports[arm], trans, arm_id=arm))
    pd.concat(attr_frames, ignore_index=True).to_csv(OUT_ROOT / "portfolio_test_return_attribution.csv", index=False)
    trans_summary = pd.DataFrame(trans_rows)
    if not trans_summary.empty:
        trans_summary.to_csv(OUT_ROOT / "portfolio_test_transition_forward.csv", index=False)

    t0s = summary[(summary["arm_id"] == "T0") & (summary["period"] == "test_all")].iloc[0]
    t1 = summary[(summary["arm_id"] == "T1") & (summary["period"] == "test_all")].iloc[0]
    t2 = summary[(summary["arm_id"] == "T2") & (summary["period"] == "test_all")].iloc[0]
    t3 = summary[(summary["arm_id"] == "T3") & (summary["period"] == "test_all")].iloc[0]
    comparisons = [
        {"pair": "T1 vs T0", "delta_excess_arr": t1["excess_arr_net"] - t0s["excess_arr_net"], "delta_sharpe": t1["sharpe"] - t0s["sharpe"], "delta_mdd": t1["mdd"] - t0s["mdd"]},
        {"pair": "T2 vs T0", "delta_excess_arr": t2["excess_arr_net"] - t0s["excess_arr_net"], "delta_sharpe": t2["sharpe"] - t0s["sharpe"], "delta_mdd": t2["mdd"] - t0s["mdd"]},
        {"pair": "T3 vs T1", "delta_excess_arr": t3["excess_arr_net"] - t1["excess_arr_net"], "delta_sharpe": t3["sharpe"] - t1["sharpe"], "delta_mdd": t3["mdd"] - t1["mdd"]},
        {"pair": "T3 vs T2", "delta_excess_arr": t3["excess_arr_net"] - t2["excess_arr_net"], "delta_sharpe": t3["sharpe"] - t2["sharpe"], "delta_mdd": t3["mdd"] - t2["mdd"]},
        {"pair": "T3 vs T0", "delta_excess_arr": t3["excess_arr_net"] - t0s["excess_arr_net"], "delta_sharpe": t3["sharpe"] - t0s["sharpe"], "delta_mdd": t3["mdd"] - t0s["mdd"]},
    ]

    report_md = write_report(summary, yearly_df, pd.DataFrame(regime_rows), attr_summaries, trans_summary, comparisons)
    (REPORT_ROOT / "test_results_report.md").write_text(report_md, encoding="utf-8")

    print(json.dumps({"leakage_ok": leak["prediction_dates_within_test"], "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
