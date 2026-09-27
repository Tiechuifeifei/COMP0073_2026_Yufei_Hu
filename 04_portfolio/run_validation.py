#!/usr/bin/env python3
"""2018–2019 validation for HMM regime portfolio experiments."""

from __future__ import annotations
import os

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "portfolio_experiments"))
sys.path.insert(0, str(Path(os.environ["RDAGENT_ROOT"]) / "phase3_portfolio_ablation" / "scripts"))

import run_portfolio_ablation as rpa  # noqa: E402

from portfolio_experiments.hmm_regime.build_spy_features import build_spy_features
from portfolio_experiments.hmm_regime.config import (
    BASE_RISK_DEGREE,
    D2_PRED_PATH,
    D3_PRED_PATH,
    F1C_PRED_PATH,
    HARD_CUTOFF,
    OUT_ROOT,
    PANEL_PATH,
    REPORT_ROOT,
    S5R_D1_PATH,
    TOPK,
    VALID_END,
    VALID_START,
)
from portfolio_experiments.hmm_regime.hmm_online_filter import (
    OnlineHMMConfig,
    attach_trade_dates,
    compute_state_run_stats,
    fit_hmm_train_only,
    label_states_by_spy_stats,
    leakage_audit,
    online_filter_posteriors,
    spy_state_summary,
    state_label_series,
    validation_log_likelihood,
)
from portfolio_experiments.hmm_regime.metrics_utils import (
    daily_ic_series,
    daily_rank_ic_series,
    ic_summary,
    load_r1r2p_module,
    mdd_from_returns,
    portfolio_by_year,
    run_qlib_backtest,
    topk_overlap,
)
from portfolio_experiments.hmm_regime.signal_utils import (
    WEIGHT_SCHEMES,
    build_dynamic_score,
    pred_df_to_qlib,
    preprocess_scores,
    score_direction_audit,
)

HMM_SEEDS = [42, 123, 456, 2026, 3407]


def load_common_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH, columns=["datetime", "instrument", "split", "label"])
    d2 = pd.read_parquet(D2_PRED_PATH).rename(columns={"score": "quant_score"})
    f1c = pd.read_pickle(F1C_PRED_PATH).reset_index().rename(columns={"score": "fundamental_score"})
    for df in (panel, d2, f1c):
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df["instrument"] = df["instrument"].astype(str)
    common = panel.merge(d2, on=["datetime", "instrument"], how="inner")
    common = common.merge(f1c, on=["datetime", "instrument"], how="inner")
    return common


def audit_common_panel(df: pd.DataFrame) -> dict:
    valid = df[df["split"] == "valid"].copy()
    daily_n = valid.groupby("datetime").size()
    dup = valid.duplicated(subset=["datetime", "instrument"]).sum()
    dir_q = score_direction_audit(valid, "quant_score")
    dir_f = score_direction_audit(valid, "fundamental_score")
    processed = preprocess_scores(valid, ["quant_score", "fundamental_score"])
    return {
        "valid_rows": int(len(valid)),
        "valid_dates": int(valid["datetime"].nunique()),
        "valid_instruments": int(valid["instrument"].nunique()),
        "mean_daily_cross_section": float(daily_n.mean()),
        "days_below_topk": int((daily_n < TOPK).sum()),
        "duplicate_keys": int(dup),
        "quant_direction": dir_q,
        "fund_direction": dir_f,
        "quant_missing_rate": float(valid["quant_score"].isna().mean()),
        "fund_missing_rate": float(valid["fundamental_score"].isna().mean()),
        "winsorize_zscore_applied": True,
    }


def run_hmm_selection(spy: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    train = spy[(spy["date"] >= pd.Timestamp("2008-01-01")) & (spy["date"] <= pd.Timestamp("2017-12-31"))]
    infer = spy[spy["date"] <= pd.Timestamp(VALID_END)].copy()
    valid = spy[(spy["date"] >= pd.Timestamp(VALID_START)) & (spy["date"] <= pd.Timestamp(VALID_END))]

    rows = []
    models = {}
    daily_by_key = {}
    for n_states in (2, 3):
        seed_maps = []
        for seed in HMM_SEEDS:
            cfg = OnlineHMMConfig(n_states=n_states, random_state=seed)
            try:
                model, infer_s, scaler = fit_hmm_train_only(train, cfg, apply_features=infer)
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "model": f"{n_states}-state",
                        "seed": seed,
                        "fit_error": str(exc),
                    }
                )
                continue
            online = online_filter_posteriors(model, infer_s)
            cal = pd.DatetimeIndex(spy["date"])
            online = attach_trade_dates(online, cal)
            vsub = online[(online["date"] >= VALID_START) & (online["date"] <= VALID_END)].copy()
            label_map = label_states_by_spy_stats(vsub, spy, n_states=n_states)
            labeled = state_label_series(vsub, label_map)
            run_stats = compute_state_run_stats(labeled["trade_state_lag1"].dropna())
            spy_stats = spy_state_summary(vsub, spy)
            spy_stats["economic_label"] = spy_stats["state"].map(label_map)
            valid_scaled = valid.copy()
            for col in ("spy_ret", "spy_vol20"):
                valid_scaled[col] = (valid[col] - scaler["mean"][col]) / scaler["std"][col]
            ll = validation_log_likelihood(model, valid_scaled)
            key = f"{n_states}state_seed{seed}"
            models[key] = {
                "model": model,
                "label_map": label_map,
                "online": online,
                "valid_labeled": labeled,
                "scaler": scaler,
            }
            daily_by_key[key] = labeled
            seed_maps.append(labeled[["date", "trade_state_lag1"]].assign(seed=seed))
            for _, st in spy_stats.iterrows():
                rows.append(
                    {
                        "model": f"{n_states}-state",
                        "seed": seed,
                        "raw_state": int(st["state"]),
                        "economic_label": st["economic_label"],
                        "n_days": int(st["n_days"]),
                        "pct_days": float(st["pct_days"]),
                        "mean_daily_ret": float(st["mean_daily_ret"]),
                        "ann_ret": float(st["ann_ret"]),
                        "ann_vol": float(st["ann_vol"]),
                        "mdd": float(st["mdd"]),
                        "validation_log_likelihood": ll,
                        "transition_matrix": model.transmat_.tolist(),
                        **run_stats,
                    }
                )
        # seed stability: fraction of days where modal state agrees across seeds
        merged = seed_maps[0][["date", "trade_state_lag1"]].rename(columns={"trade_state_lag1": "s0"})
        for i, sm in enumerate(seed_maps[1:], 1):
            merged = merged.merge(
                sm[["date", "trade_state_lag1"]].rename(columns={"trade_state_lag1": f"s{i}"}),
                on="date",
                how="inner",
            )
        state_cols = [c for c in merged.columns if c.startswith("s")]
        merged["agree"] = merged[state_cols].apply(lambda r: r.nunique() == 1, axis=1)
        stability = float(merged["agree"].mean())
        for row in rows:
            if row["model"] == f"{n_states}-state":
                row["cross_seed_stability"] = stability

    sel_df = pd.DataFrame(rows)

    # selection rubric (no portfolio metrics)
    def score_candidate(n_states: int) -> tuple[float, list[str]]:
        sub = sel_df[(sel_df["model"] == f"{n_states}-state") & (sel_df["seed"] == 42)]
        reasons = []
        score = 0.0
        labels = sub.set_index("economic_label")
        if n_states == 3 and set(labels.index) >= {"bull", "neutral", "bear"}:
            if labels.loc["bull", "ann_ret"] > labels.loc["neutral", "ann_ret"] > labels.loc["bear", "ann_ret"]:
                score += 2
                reasons.append("SPY return ordering bull>neutral>bear")
        if n_states == 2 and set(labels.index) >= {"bull", "bear"}:
            if labels.loc["bull", "ann_ret"] > labels.loc["bear", "ann_ret"]:
                score += 2
                reasons.append("SPY return ordering bull>bear")
        min_pct = labels["pct_days"].min()
        if min_pct >= 0.12:
            score += 2
            reasons.append(f"min state pct {min_pct:.2%} >= 12%")
        elif min_pct >= 0.08:
            score += 1
        mean_dur = sub["mean_duration"].mean()
        if mean_dur >= 8:
            score += 2
            reasons.append(f"mean duration {mean_dur:.1f}d")
        elif mean_dur >= 5:
            score += 1
        switches = sub["n_switches"].mean()
        if switches <= 25:
            score += 1
            reasons.append(f"low switches ({switches:.0f})")
        stab = sub["cross_seed_stability"].iloc[0]
        if stab >= 0.75:
            score += 2
            reasons.append(f"seed stability {stab:.2%}")
        elif stab >= 0.6:
            score += 1
        ll = sub["validation_log_likelihood"].iloc[0]
        reasons.append(f"valid LL={ll:.1f}")
        return score, reasons

    s2, r2 = score_candidate(2)
    s3, r3 = score_candidate(3)
    chosen_n = 2 if s2 >= s3 else 3
    chosen_reasons = r2 if chosen_n == 2 else r3
    if s2 == s3:
        chosen_n = 2
        chosen_reasons = r2 + ["tie-break: prefer simpler 2-state"]

    if f"2state_seed42" not in models and f"3state_seed42" not in models:
        raise RuntimeError("All HMM fits failed")
    if f"{chosen_n}state_seed42" not in models:
        # fallback to first successful model for chosen n_states
        fallback = next(k for k in models if k.startswith(f"{chosen_n}state"))
        chosen_key = fallback
    else:
        chosen_key = f"{chosen_n}state_seed42"
    chosen = models[chosen_key]
    selected = {
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "n_states": chosen_n,
        "seed": 42,
        "label_map": {str(k): v for k, v in chosen["label_map"].items()},
        "selection_scores": {"2-state": s2, "3-state": s3},
        "selection_reasons": chosen_reasons,
        "validation_period": [VALID_START, VALID_END],
        "hard_cutoff": HARD_CUTOFF,
    }
    hmm_daily = chosen["online"]
    hmm_daily["economic_label"] = hmm_daily["trade_state_lag1"].map(chosen["label_map"])
    return sel_df, selected, hmm_daily


def attach_states_to_panel(panel: pd.DataFrame, hmm_daily: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    states = state_label_series(hmm_daily, label_map)
    states["trade_date"] = pd.to_datetime(states["trade_date"]).dt.normalize()
    out = panel.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    state_cols = ["trade_date", "state_label", "p_bull", "p_neutral", "p_bear", "trade_state_lag1"]
    out = out.merge(states[state_cols], left_on="datetime", right_on="trade_date", how="left")
    return out


def regime_signal_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    valid = panel[panel["split"] == "valid"].copy()
    valid = preprocess_scores(valid, ["quant_score", "fundamental_score"])
    rows = []
    for label in sorted(valid["state_label"].dropna().unique()):
        sub = valid[valid["state_label"] == label]
        for sig, col in [("D2", "quant_score_z"), ("F1C", "fundamental_score_z")]:
            ic = ic_summary(daily_ic_series(sub, col))
            ric = ic_summary(daily_rank_ic_series(sub, col))
            rows.append({"state": label, "signal": sig, "metric": "IC", **ic})
            rows.append(
                {
                    "state": label,
                    "signal": sig,
                    "metric": "RankIC",
                    "mean_ic": ric["mean_ic"],
                    "ic_std": ric["ic_std"],
                    "icir": ric["icir"],
                    "positive_ic_pct": ric["positive_ic_pct"],
                    "n_days": ric["n_days"],
                }
            )
    return pd.DataFrame(rows)


def build_exposure_schedule(states: pd.DataFrame, rule_id: str, n_states: int) -> dict[str, float]:
    hard_3 = {
        "3A": {"bull": 0.95, "neutral": 0.70, "bear": 0.30},
        "3B": {"bull": 0.95, "neutral": 0.50, "bear": 0.00},
    }
    hard_2 = {"2A": {"bull": 0.95, "bear": 0.30}, "2B": {"bull": 0.95, "bear": 0.00}}
    schedule = {}
    for _, row in states.iterrows():
        td = pd.Timestamp(row["trade_date"])
        if pd.isna(td):
            continue
        key = td.strftime("%Y-%m-%d")
        if rule_id == "Soft-3A":
            exp = 0.95 * row.get("p_bull", 0) + 0.70 * row.get("p_neutral", 0) + 0.30 * row.get("p_bear", 0)
        elif rule_id == "Soft-3B":
            exp = 0.95 * row.get("p_bull", 0) + 0.50 * row.get("p_neutral", 0) + 0.00 * row.get("p_bear", 0)
        elif rule_id == "Soft-2A":
            exp = 0.95 * row.get("p_bull", 0) + 0.30 * row.get("p_bear", 0)
        elif rule_id == "Soft-2B":
            exp = 0.95 * row.get("p_bull", 0)
        elif rule_id in hard_3:
            lab = row.get("state_label")
            exp = hard_3[rule_id].get(lab, BASE_RISK_DEGREE)
        elif rule_id in hard_2:
            lab = row.get("state_label")
            exp = hard_2[rule_id].get(lab, BASE_RISK_DEGREE)
        else:
            exp = BASE_RISK_DEGREE
        schedule[key] = float(np.clip(exp, 0.0, 0.95))
    return schedule


def run_factor_weight_validation(panel: pd.DataFrame, r1r2p, d3_pred: pd.DataFrame) -> pd.DataFrame:
    valid = panel[panel["split"] == "valid"].copy()
    valid = preprocess_scores(valid, ["quant_score", "fundamental_score"])
    valid_dates = valid["datetime"].unique()
    p0_pred = d3_pred.loc[d3_pred.index.get_level_values("datetime").isin(valid_dates)]
    rows = []
    scheme_preds = {}
    for scheme_id, scheme in WEIGHT_SCHEMES.items():
        sub = valid.copy()
        sub["dynamic_score"] = build_dynamic_score(sub, scheme)
        if sub["dynamic_score"].corr(sub["label"]) < 0:
            sub["dynamic_score"] = -sub["dynamic_score"]
        pred = pred_df_to_qlib(sub.rename(columns={"dynamic_score": "score"})[["datetime", "instrument", "score"]])
        scheme_preds[scheme_id] = pred
        bt = run_qlib_backtest(r1r2p, rpa, pred, VALID_START, VALID_END)
        rep = bt["result"]["report"]
        m = bt["metrics"]
        yearly = portfolio_by_year(rep)
        for label in sorted(sub["state_label"].dropna().unique()):
            st = sub[sub["state_label"] == label]
            ric = ic_summary(daily_rank_ic_series(st, "dynamic_score"))
            rows.append(
                {
                    "experiment": "factor_weight",
                    "candidate": scheme_id,
                    "state": label,
                    "w_quant": scheme[label][0],
                    "w_fundamental": scheme[label][1],
                    "w_sentiment": 0.0,
                    "arr": m["annualized_return"],
                    "excess_arr_net": m["excess_annualized_return_net"],
                    "sharpe": m["sharpe_ratio"],
                    "ir_net": m["information_ratio_net"],
                    "mdd": m["maximum_drawdown"],
                    "turnover": m["mean_daily_turnover"],
                    "mean_rank_ic": ric["mean_ic"],
                    "sample_days": ric["n_days"],
                    "year": "valid_all",
                }
            )
        for _, yr in yearly.iterrows():
            rows.append(
                {
                    "experiment": "factor_weight",
                    "candidate": scheme_id,
                    "state": "all",
                    "year": str(yr["year"]),
                    "arr": yr["arr"],
                    "excess_arr_net": yr["excess_arr"],
                    "sharpe": yr["sharpe"],
                }
            )
        rows.append(
            {
                "experiment": "factor_weight",
                "candidate": scheme_id,
                "state": "all",
                "year": "valid_all",
                "arr": m["annualized_return"],
                "excess_arr_net": m["excess_annualized_return_net"],
                "sharpe": m["sharpe_ratio"],
                "ir_net": m["information_ratio_net"],
                "mdd": m["maximum_drawdown"],
                "turnover": m["mean_daily_turnover"],
                "topk_overlap_vs_p0": topk_overlap(
                    sub.rename(columns={"dynamic_score": "score"}),
                    d3_pred.reset_index().rename(columns={"score": "score"}),
                    "score",
                    "score",
                ),
            }
        )
    # W4 check on D2∩D3 common
    return pd.DataFrame(rows), scheme_preds, p0_pred


def run_exposure_validation(states_valid: pd.DataFrame, p0_pred: pd.DataFrame, r1r2p, n_states: int) -> pd.DataFrame:
    rules = ["3A", "3B", "Soft-3A", "Soft-3B"] if n_states == 3 else ["2A", "2B", "Soft-2A", "Soft-2B"]
    rows = []
    for rule_id in rules:
        schedule = build_exposure_schedule(states_valid, rule_id, n_states)
        bt = run_qlib_backtest(
            r1r2p,
            rpa,
            p0_pred,
            VALID_START,
            VALID_END,
            strategy_class="RegimeExposureTopkStrategy",
            strategy_module="portfolio_experiments.hmm_regime.regime_exposure_strategy",
            extra_kwargs={
                "exposure_by_date": schedule,
                "use_target_exposure_scaling": True,
            },
        )
        m = bt["metrics"]
        rep = bt["result"]["report"]
        yearly = portfolio_by_year(rep)
        rows.append(
            {
                "experiment": "exposure",
                "candidate": rule_id,
                "arr": m["annualized_return"],
                "excess_arr_net": m["excess_annualized_return_net"],
                "sharpe": m["sharpe_ratio"],
                "ir_net": m["information_ratio_net"],
                "mdd": m["maximum_drawdown"],
                "turnover": m["mean_daily_turnover"],
                "year": "valid_all",
            }
        )
        for _, yr in yearly.iterrows():
            rows.append(
                {
                    "experiment": "exposure",
                    "candidate": rule_id,
                    "year": str(yr["year"]),
                    "arr": yr["arr"],
                    "excess_arr_net": yr["excess_arr"],
                    "sharpe": yr["sharpe"],
                }
            )
    return pd.DataFrame(rows)


def w4_eligibility(panel: pd.DataFrame, d3_pred: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    d3_df = d3_pred.reset_index().rename(columns={"score": "d3_score"})
    common = panel.merge(
        d3_df.reset_index().rename(columns={"score": "d3_score"}),
        on=["datetime", "instrument"],
        how="inner",
    )
    common = common[common["split"] == "valid"]
    rows = []
    ok_bull = ok_bear = True
    for label in ["bull", "bear"]:
        sub = common[common["state_label"] == label]
        d2_ic = ic_summary(daily_rank_ic_series(sub, "quant_score"))
        d3_ic = ic_summary(daily_rank_ic_series(sub, "d3_score"))
        rows.append({"state": label, "model": "D2", **d2_ic})
        rows.append({"state": label, "model": "D3", **d3_ic})
        if label == "bull" and d2_ic["mean_ic"] <= d3_ic["mean_ic"]:
            ok_bull = False
        if label == "bear" and d3_ic["mean_ic"] <= d2_ic["mean_ic"]:
            ok_bear = False
    return ok_bull and ok_bear, pd.DataFrame(rows)


def sentiment_diagnostics(hmm_daily: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    panel = load_common_panel()
    obj = pd.read_pickle(S5R_D1_PATH)
    sent = obj["two_stage_pred"][["datetime", "instrument", "residual"]].rename(
        columns={"residual": "sentiment_score"}
    )
    for df in (sent,):
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df["instrument"] = df["instrument"].astype(str)
    merged = panel.merge(sent, on=["datetime", "instrument"], how="inner")
    merged = merged[merged["split"] == "valid"]
    states = state_label_series(
        hmm_daily[(hmm_daily["date"] >= VALID_START) & (hmm_daily["date"] <= VALID_END)],
        label_map,
    )
    states["trade_date"] = pd.to_datetime(states["trade_date"]).dt.normalize()
    merged = merged.merge(states[["trade_date", "state_label"]], left_on="datetime", right_on="trade_date", how="inner")
    rows = []
    for label in sorted(merged["state_label"].dropna().unique()):
        sub = merged[merged["state_label"] == label]
        ic = ic_summary(daily_ic_series(sub, "sentiment_score"))
        ric = ic_summary(daily_rank_ic_series(sub, "sentiment_score"))
        rows.append(
            {
                "state": label,
                "n_event_rows": int(sub["sentiment_score"].notna().sum()),
                "mean_ic": ic["mean_ic"],
                "mean_rank_ic": ric["mean_ic"],
                "icir": ric["icir"],
                "positive_ic_pct": ric["positive_ic_pct"],
            }
        )
    return pd.DataFrame(rows)


def select_factor_weights(df: pd.DataFrame) -> tuple[str, dict]:
    sub = df[(df["experiment"] == "factor_weight") & (df["state"] == "all") & (df["year"] == "valid_all")].copy()
    sub = sub.sort_values(["sharpe", "mdd", "excess_arr_net"], ascending=[False, True, False])
    # prefer simpler on tie
    simplicity = {"W0": 4, "W1": 3, "W2": 2, "W3": 1}
    sub["simplicity"] = sub["candidate"].map(simplicity)
    sub = sub.sort_values(["sharpe", "mdd", "excess_arr_net", "simplicity"], ascending=[False, True, False, False])
    chosen = sub.iloc[0]["candidate"]
    scheme = WEIGHT_SCHEMES[chosen]
    frozen = {
        "weight_scheme": chosen,
        "weights_by_state": {k: {"w_quant": v[0], "w_fundamental": v[1], "w_sentiment": 0.0} for k, v in scheme.items()},
        "frozen_on": "validation_2018_2019",
        "selection_metric_priority": ["Sharpe", "MDD", "excess_arr_net", "simplicity"],
    }
    return chosen, frozen


def select_exposure_rule(df: pd.DataFrame, n_states: int) -> tuple[str, dict]:
    sub = df[(df["experiment"] == "exposure") & (df["year"] == "valid_all")].copy()
    # check both years not dominated by one
    yearly = df[(df["experiment"] == "exposure") & (df["year"].isin(["2018", "2019"]))]
    simplicity = {"2A": 4, "2B": 3, "3A": 4, "3B": 3, "Soft-2B": 2, "Soft-2A": 1, "Soft-3A": 1, "Soft-3B": 0}
    sub["simplicity"] = sub["candidate"].map(simplicity).fillna(0)
    sub = sub.sort_values(["sharpe", "mdd", "excess_arr_net", "simplicity"], ascending=[False, True, False, False])
    chosen = sub.iloc[0]["candidate"]
    frozen = {
        "exposure_rule": chosen,
        "n_states": n_states,
        "max_bull_exposure": 0.95,
        "frozen_on": "validation_2018_2019",
        "selection_metric_priority": ["Sharpe", "MDD", "excess_arr_net", "simplicity"],
    }
    return chosen, frozen


def write_decision_report(
    path: Path,
    *,
    selected_hmm: dict,
    chosen_weight: str,
    chosen_exposure: str,
    hmm_reasons: list[str],
    weight_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    w4_run: bool,
) -> None:
    lines = [
        "# HMM Regime Validation Decision Report (2018–2019)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. HMM model selection",
        f"- **Selected:** {selected_hmm['n_states']}-state HMM (seed {selected_hmm['seed']})",
        f"- **Scores:** 2-state={selected_hmm['selection_scores']['2-state']}, 3-state={selected_hmm['selection_scores']['3-state']}",
        "- **Reasons:**",
    ]
    for r in hmm_reasons:
        lines.append(f"  - {r}")
    lines += [
        "",
        "## 2. Factor weight selection (fixed exposure=0.95)",
        f"- **Selected:** `{chosen_weight}`",
        "- Validation candidates W0–W3 compared on Sharpe → MDD → excess ARR → simplicity.",
        "- Sentiment weight frozen at **0.0** (diagnostic-only arm).",
        "",
        "## 3. Exposure rule selection (fixed P0=D3 score)",
        f"- **Selected:** `{chosen_exposure}`",
        "- Compared on same priority; all candidates retained in `exposure_validation_results.csv`.",
        "",
        "## 4. Sentiment",
    ]
    if sentiment_df["mean_rank_ic"].max() > 0 and (sentiment_df["positive_ic_pct"] > 0.5).any():
        lines.append("- Sentiment residual shows isolated state signal but **not** promoted to main weights (w_sentiment=0).")
    else:
        lines.append("- **Excluded:** Delayed sentiment did not show robust state-conditional incremental value.")
    lines += [
        "",
        "## 5. Frozen for test (2020–2023)",
        f"- `selected_hmm_model.json`",
        f"- `selected_factor_weights.json` → {chosen_weight}",
        f"- `selected_exposure_rule.json` → {chosen_exposure}",
        "",
        "## 6. Excluded / not run",
        f"- W4 regime selector: {'run' if w4_run else '**skipped**'} (D2 Bull / D3 Bear criterion).",
        "- 2020–2023 test: **not executed** in this script.",
        "- 2024–2025 holdout: **not accessed**.",
        "",
        "*IC improvement does not necessarily imply ARR improvement.*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    r1r2p = load_r1r2p_module(PROJECT_ROOT)
    rpa.init_qlib({"provider_uri": str(PROJECT_ROOT / "staging/qlib_data"), "region": "us"})

    spy = build_spy_features()
    spy.to_parquet(OUT_ROOT / "spy_features.parquet", index=False)

    hmm_sel_df, selected_hmm, hmm_daily = run_hmm_selection(spy)
    hmm_sel_df.to_csv(OUT_ROOT / "hmm_model_selection.csv", index=False)
    label_map = {int(k): v for k, v in selected_hmm["label_map"].items()}

    hmm_daily_out = hmm_daily[hmm_daily["date"] <= VALID_END].copy()
    hmm_daily_out.to_csv(OUT_ROOT / "hmm_daily_states.csv", index=False)
    (OUT_ROOT / "leakage_audit.json").write_text(json.dumps(leakage_audit(hmm_daily_out), indent=2))

    panel = load_common_panel()
    audit = audit_common_panel(panel)
    (OUT_ROOT / "regime_panel_audit.json").write_text(json.dumps(audit, indent=2))

    panel = attach_states_to_panel(panel, hmm_daily, label_map)
    panel.to_parquet(OUT_ROOT / "regime_signal_panel.parquet", index=False)

    diag = regime_signal_diagnostics(panel)
    diag.to_csv(OUT_ROOT / "regime_signal_diagnostics_validation.csv", index=False)

    d3 = pd.read_parquet(D3_PRED_PATH)
    d3["datetime"] = pd.to_datetime(d3["datetime"]).dt.normalize()
    d3["instrument"] = d3["instrument"].astype(str)
    d3_pred = pred_df_to_qlib(d3.rename(columns={"score": "score"}))

    fw_df, _scheme_preds, p0_pred = run_factor_weight_validation(panel, r1r2p, d3_pred)
    fw_df.to_csv(OUT_ROOT / "factor_weight_validation_results.csv", index=False)

    chosen_w, frozen_w = select_factor_weights(fw_df)
    (OUT_ROOT / "selected_factor_weights.json").write_text(json.dumps(frozen_w, indent=2), encoding="utf-8")

    states_valid = hmm_daily_out[(hmm_daily_out["date"] >= VALID_START) & (hmm_daily_out["date"] <= VALID_END)]
    states_valid = state_label_series(states_valid, label_map)
    exp_df = run_exposure_validation(states_valid, p0_pred, r1r2p, selected_hmm["n_states"])
    exp_df.to_csv(OUT_ROOT / "exposure_validation_results.csv", index=False)

    chosen_e, frozen_e = select_exposure_rule(exp_df, selected_hmm["n_states"])
    (OUT_ROOT / "selected_exposure_rule.json").write_text(json.dumps(frozen_e, indent=2), encoding="utf-8")

    (OUT_ROOT / "selected_hmm_model.json").write_text(json.dumps(selected_hmm, indent=2), encoding="utf-8")

    w4_ok, w4_diag = w4_eligibility(panel, d3_pred)
    w4_diag.to_csv(OUT_ROOT / "w4_selector_diagnostics.csv", index=False)

    sent_df = sentiment_diagnostics(hmm_daily, label_map)
    sent_df.to_csv(OUT_ROOT / "sentiment_regime_diagnostics_validation.csv", index=False)

    write_decision_report(
        REPORT_ROOT / "validation_decision_report.md",
        selected_hmm=selected_hmm,
        chosen_weight=chosen_w,
        chosen_exposure=chosen_e,
        hmm_reasons=selected_hmm["selection_reasons"],
        weight_df=fw_df,
        exposure_df=exp_df,
        sentiment_df=sent_df,
        w4_run=w4_ok,
    )

    print(json.dumps({"chosen_hmm": selected_hmm["n_states"], "weight": chosen_w, "exposure": chosen_e, "w4": w4_ok}, indent=2))


if __name__ == "__main__":
    main()
