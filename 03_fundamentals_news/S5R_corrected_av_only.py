#!/usr/bin/env python3
"""S5R corrected AV-only experiments: D0/D1 (LABEL0) and E0/E1 (event-time v2),
including full-panel and two-stage event-residual setups. Sample ends 2023."""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from qlib.contrib.data.handler import DataHandlerLP
from qlib.data.dataset.processor import zscore
from qlib.utils import init_instance_by_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/av_only"
S3_RAW = PROJECT_ROOT / "data/sentiment_experiments/S3_daily_sentiment_panel/S3_daily_sentiment_raw.parquet"
V2_MAP = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/event_time_v2/S5R_event_time_v2_mapping.parquet"
CONSERVATIVE_FLAGS = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/preflight/S5R_event_timing_flags.parquet"
WORKFLOW_YAML = PROJECT_ROOT / "experiments/conf_alpha20_sp500_transfer.yaml"
QLIB_DATA = PROJECT_ROOT / "staging/qlib_data"
LABEL_EXPR = "Ref($close, -2)/Ref($close, -1) - 1"

PERIODS = {
    "train": ("2010-01-01", "2017-12-31"),
    "valid": ("2018-01-01", "2019-12-31"),
    "test": ("2020-01-01", "2023-12-31"),
    "covid": ("2020-01-01", "2021-12-31"),
    "post_covid": ("2022-01-01", "2023-12-31"),
}

ALPHA20 = [
    "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", "ROC60",
    "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW",
]
SENTIMENT = [
    "news_count", "log_news_count", "ticker_mean_sentiment", "relevance_weighted_sentiment",
    "positive_news_share", "negative_news_share", "net_positive_share", "sentiment_dispersion",
    "relevance_mean", "sentiment_change_1d", "sentiment_surprise_20d", "sentiment_3d_decay",
    "sentiment_5d_decay", "abnormal_news_count_20d", "no_news_flag",
]
DIRECTIONAL = [
    "ticker_mean_sentiment", "relevance_weighted_sentiment", "positive_news_share",
    "negative_news_share", "net_positive_share", "sentiment_dispersion",
    "relevance_mean", "sentiment_change_1d", "sentiment_surprise_20d",
]

OFFICIAL_LGB = {
    "objective": "regression", "metric": "mse", "verbosity": -1,
    "colsample_bytree": 0.8879, "learning_rate": 0.2, "subsample": 0.8789,
    "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8, "num_leaves": 210, "num_threads": 20,
}
NUM_BOOST = 1000
EARLY_STOP = 50
SEED = 42
IC_THRESHOLDS = [20, 25, 50]
NW_LAGS = 5
TOP_K = 20


def assign_split_ts(ts: pd.Timestamp) -> str | None:
    d = ts.date()
    for name in ("train", "valid", "test"):
        s, e = PERIODS[name]
        if date.fromisoformat(s) <= d <= date.fromisoformat(e):
            return name
    return None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("S5R-AV")


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is None:
        return None
    return str(obj)


def newey_west_tstat(x: pd.Series, lags: int = NW_LAGS) -> float:
    arr = x.dropna().to_numpy(dtype=float)
    n = len(arr)
    if n <= 1:
        return float("nan")
    u = arr - arr.mean()
    lr = np.dot(u, u) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1)
        lr += 2 * w * (np.dot(u[lag:], u[:-lag]) / n)
    if lr <= 0:
        return float("nan")
    return float(arr.mean() / np.sqrt(lr / n))


def init_qlib() -> None:
    import qlib
    from qlib.constant import REG_US
    qlib.init(provider_uri=str(QLIB_DATA), region=REG_US, kernels=1)


def load_labels() -> pd.DataFrame:
    from qlib.data import D
    init_qlib()
    inst = D.list_instruments(D.instruments("sp500"), start_time="2010-01-01", end_time="2023-12-31", as_list=True)
    raw = D.features(inst, [LABEL_EXPR], "2010-01-01", "2023-12-31", freq="day")
    raw.columns = ["label"]
    df = raw.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_alpha20() -> pd.DataFrame:
    init_qlib()
    conf = yaml.safe_load(WORKFLOW_YAML.read_text(encoding="utf-8"))
    handler = init_instance_by_config({
        "class": "DataHandlerLP", "module_path": "qlib.contrib.data.handler",
        "kwargs": conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"],
    })
    feat = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).reset_index()
    feat["datetime"] = pd.to_datetime(feat["datetime"])
    return feat[["datetime", "instrument"] + ALPHA20]


def build_delayed_panel() -> pd.DataFrame:
    sent = pd.read_parquet(S3_RAW)
    sent["datetime"] = pd.to_datetime(sent["datetime"])
    for col in DIRECTIONAL:
        sent[col] = sent[col].where(sent["no_news_flag"] == 0)
    df = sent.merge(load_labels(), on=["datetime", "instrument"], how="inner")
    df = df.merge(load_alpha20(), on=["datetime", "instrument"], how="inner")
    df = df[df["datetime"] <= pd.Timestamp("2023-12-31")].copy()
    if (df["datetime"] >= pd.Timestamp("2024-01-01")).any():
        raise RuntimeError("2024 rows in delayed panel")
    df["news_event_flag"] = (df["no_news_flag"] == 0).astype(np.int8)
    return df.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def build_event_panel() -> pd.DataFrame:
    ev = pd.read_parquet(V2_MAP)
    ev = ev[ev["tradable_intraday_v2"] == 1].copy()
    ev = ev[ev["event_trade_date_v2"].notna()].copy()
    ev["datetime"] = pd.to_datetime(ev["event_trade_date_v2"])
    ev["instrument"] = "P" + ev["permno"].astype(str)
    ev["split"] = ev["split"].astype(str)

    alpha = load_alpha20()
    sent = pd.read_parquet(S3_RAW)
    sent["datetime"] = pd.to_datetime(sent["datetime"])
    for col in DIRECTIONAL:
        sent[col] = sent[col].where(sent["no_news_flag"] == 0)

    df = ev.merge(alpha, on=["datetime", "instrument"], how="left")
    df = df.merge(
        sent[["datetime", "instrument"] + SENTIMENT],
        on=["datetime", "instrument"],
        how="left",
    )
    df = df[df["datetime"] <= pd.Timestamp("2023-12-31")].copy()
    df["split"] = df["datetime"].map(assign_split_ts)

    # permno-day dedupe for IC (one row per stock-day)
    df = df.sort_values(["permno", "datetime", "time_published_utc"])
    daily = df.drop_duplicates(["permno", "datetime"], keep="first").copy()
    daily["event_row_weight"] = 1.0 / df.groupby(["datetime"]).size().reindex(daily["datetime"]).values
    return daily.reset_index(drop=True)


def build_conservative_event_panel() -> pd.DataFrame:
    ev = pd.read_parquet(V2_MAP)
    flags = pd.read_parquet(CONSERVATIVE_FLAGS)
    m = ev.merge(
        flags[["permno", "canonical_url_hash", "h_intraday_tradable_open_to_close", "eligible_signal_date"]],
        on=["permno", "canonical_url_hash"],
        how="inner",
    )
    m = m[m["h_intraday_tradable_open_to_close"] == True].copy()  # noqa: E712
    m["datetime"] = pd.to_datetime(m["eligible_signal_date"])
    m["instrument"] = "P" + m["permno"].astype(str)
    m = m.rename(columns={
        "H_INTRADAY_T_v2": "H_INTRADAY_T_conservative_anchor",
    })
    init_qlib()
    from qlib.data import D
    inst = sorted(m["instrument"].unique())
    px = D.features(inst, ["$open", "$close"], "2010-01-01", "2023-12-31", freq="day").reset_index()
    px["datetime"] = pd.to_datetime(px["datetime"])
    px = px.rename(columns={"$open": "open", "$close": "close"})
    px = px.sort_values(["instrument", "datetime"])
    px["H_INTRADAY_T"] = px["close"] / px["open"] - 1
    px["H_SIGNAL_DAY"] = px["close"] / px.groupby("instrument")["close"].shift(1) - 1
    px["H_OVERNIGHT"] = px["open"] / px.groupby("instrument")["close"].shift(1) - 1
    m = m.merge(
        px[["datetime", "instrument", "H_INTRADAY_T", "H_SIGNAL_DAY", "H_OVERNIGHT"]],
        on=["datetime", "instrument"],
        how="left",
    )
    alpha = load_alpha20()
    m = m.merge(alpha, on=["datetime", "instrument"], how="left")
    sent = pd.read_parquet(S3_RAW)
    sent["datetime"] = pd.to_datetime(sent["datetime"])
    for col in DIRECTIONAL:
        sent[col] = sent[col].where(sent["no_news_flag"] == 0)
    m = m.merge(sent[["datetime", "instrument"] + SENTIMENT], on=["datetime", "instrument"], how="left")
    return m.drop_duplicates(["permno", "datetime"]).reset_index(drop=True)


def process_train_label(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    out = df.dropna(subset=[label_col]).copy()
    out["label_processed"] = out.groupby("datetime", group_keys=False)[label_col].apply(zscore)
    return out


def lgb_train(train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], weights: pd.Series | None = None) -> lgb.Booster:
    params = OFFICIAL_LGB.copy()
    params.update({"seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED, "data_random_seed": SEED})
    w = weights.loc[train_df.index].to_numpy() if weights is not None else None
    train_set = lgb.Dataset(train_df[features], label=train_df["label_processed"], weight=w, free_raw_data=False)
    valid_set = lgb.Dataset(valid_df[features], label=valid_df["label_processed"], reference=train_set, free_raw_data=False)
    return lgb.train(
        params, train_set, num_boost_round=NUM_BOOST, valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(period=0)],
    )


def daily_ic(
    pred: pd.DataFrame,
    label_col: str,
    min_n: int,
    date_col: str = "datetime",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    eligible_days = 0
    for dt, g in pred.groupby(date_col):
        eligible_days += 1
        sub = g.dropna(subset=["score", label_col])
        if len(sub) < min_n or sub["score"].nunique() < 2 or sub[label_col].nunique() < 2:
            continue
        rows.append({
            date_col: dt,
            "ic": sub["score"].corr(sub[label_col]),
            "rank_ic": sub["score"].corr(sub[label_col], method="spearman"),
            "n_obs": len(sub),
        })
    daily = pd.DataFrame(rows)
    meta = {
        "eligible_calendar_days": eligible_days,
        "valid_ic_days": int(len(daily)),
        "pct_valid_days": float(len(daily) / eligible_days) if eligible_days else 0.0,
        "min_cross_section": min_n,
    }
    return daily, meta


def summarize_ic(daily: pd.DataFrame, meta: dict[str, Any], period: str, spec: str, **extra) -> dict[str, Any]:
    if daily.empty:
        return {
            "spec": spec,
            "period": period,
            "mean_daily_ic": np.nan,
            "mean_daily_rank_ic": np.nan,
            "icir": np.nan,
            "rank_icir": np.nan,
            "rank_ic_newey_west_tstat": np.nan,
            "valid_ic_days": 0,
            "eligible_calendar_days": meta.get("eligible_calendar_days", 0),
            "pct_valid_ic_days": 0.0,
            "min_cross_section": meta.get("min_cross_section", np.nan),
            **extra,
        }
    ic, ric = daily["ic"].dropna(), daily["rank_ic"].dropna()
    ic_m = float(ic.mean()) if len(ic) else np.nan
    ric_m = float(ric.mean()) if len(ric) else np.nan
    ic_s = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    ric_s = float(ric.std(ddof=1)) if len(ric) > 1 else np.nan
    return {
        "spec": spec,
        "period": period,
        "mean_daily_ic": ic_m,
        "mean_daily_rank_ic": ric_m,
        "icir": ic_m / ic_s if ic_s and ic_s > 0 else np.nan,
        "rank_icir": ric_m / ric_s if ric_s and ric_s > 0 else np.nan,
        "rank_ic_newey_west_tstat": newey_west_tstat(ric),
        "valid_ic_days": meta.get("valid_ic_days", int(len(ic))),
        "eligible_calendar_days": meta.get("eligible_calendar_days", np.nan),
        "pct_valid_ic_days": meta.get("pct_valid_days", np.nan),
        "min_cross_section": meta.get("min_cross_section", np.nan),
        **extra,
    }


def eval_periods(
    pred: pd.DataFrame,
    label_col: str,
    spec: str,
    family: str,
    arch: str,
    target: str,
    sample_tag: str,
) -> list[dict[str, Any]]:
    rows = []
    for period, (s, e) in PERIODS.items():
        sub = pred[(pred["datetime"] >= pd.Timestamp(s)) & (pred["datetime"] <= pd.Timestamp(e))].copy()
        if sub.empty:
            continue
        for thr in IC_THRESHOLDS:
            daily, meta = daily_ic(sub, label_col, thr)
            rows.append(summarize_ic(
                daily, meta, period, spec,
                family=family, architecture=arch, target=target, sample=sample_tag,
                ic_threshold=thr,
            ))
    return rows


def event_pooled_metrics(pred: pd.DataFrame, label_col: str) -> dict[str, float]:
    sub = pred.dropna(subset=["score", label_col])
    if len(sub) < 10:
        return {"pooled_pearson": np.nan, "pooled_spearman": np.nan, "directional_accuracy": np.nan, "q5_q1_spread": np.nan}
    q = sub[label_col].quantile([0.2, 0.8])
    low, high = sub[sub[label_col] <= q.iloc[0]]["score"].mean(), sub[sub[label_col] >= q.iloc[1]]["score"].mean()
    return {
        "pooled_pearson": float(sub["score"].corr(sub[label_col])),
        "pooled_spearman": float(sub["score"].corr(sub[label_col], method="spearman")),
        "directional_accuracy": float((np.sign(sub["score"]) == np.sign(sub[label_col])).mean()),
        "q5_q1_spread": float(high - low),
    }


def portfolio_eval(pred: pd.DataFrame, label_col: str, period: str) -> dict[str, Any]:
    s, e = PERIODS[period]
    sub = pred[(pred["datetime"] >= pd.Timestamp(s)) & (pred["datetime"] <= pd.Timestamp(e))].dropna(subset=["score", label_col])
    traded, skipped, holdings, returns = 0, 0, [], []
    for dt, g in sub.groupby("datetime"):
        n = len(g)
        if n < TOP_K:
            skipped += 1
            continue
        k = min(TOP_K, n)
        top = g.nlargest(k, "score")
        holdings.append(k)
        traded += 1
        returns.append(float(top[label_col].mean()))
    eligible = sub.groupby("datetime").size().shape[0]
    return {
        "period": period,
        "eligible_days": eligible,
        "traded_days": traded,
        "skipped_days": skipped,
        "participation_rate": traded / eligible if eligible else np.nan,
        "avg_holdings": float(np.mean(holdings)) if holdings else np.nan,
        "capital_deployment": float(np.mean(holdings) / TOP_K) if holdings else 0.0,
        "mean_long_return": float(np.mean(returns)) if returns else np.nan,
    }


def run_delayed(spec_id: str, use_sentiment: bool) -> tuple[list[dict], dict]:
    df = build_delayed_panel()
    train = process_train_label(df[df["split"] == "train"])
    valid = process_train_label(df[df["split"] == "valid"])
    feats_fp = ALPHA20 + (SENTIMENT if use_sentiment else [])
    metrics = []
    results = {}

    # Full-panel
    log.info("Delayed %s full-panel", spec_id)
    m_fp = lgb_train(train, valid, feats_fp)
    pred_fp = df.copy()
    pred_fp["score"] = m_fp.predict(pred_fp[feats_fp], num_iteration=m_fp.best_iteration)
    metrics.extend(eval_periods(pred_fp, "label", spec_id, "delayed", "full_panel", "LABEL0", "full_panel"))
    results["full_panel_pred"] = pred_fp

    # Two-stage
    log.info("Delayed %s two-stage", spec_id)
    m_s1 = lgb_train(train, valid, ALPHA20)
    pred_s1 = df.copy()
    pred_s1["base"] = m_s1.predict(pred_s1[ALPHA20], num_iteration=m_s1.best_iteration)
    event_mask = df["news_event_flag"] == 1
    ev_train = train[event_mask.loc[train.index]].copy()
    ev_valid = valid[event_mask.loc[valid.index]].copy()
    ev_train["base"] = m_s1.predict(ev_train[ALPHA20], num_iteration=m_s1.best_iteration)
    ev_valid["base"] = m_s1.predict(ev_valid[ALPHA20], num_iteration=m_s1.best_iteration)
    ev_train["label_processed"] = ev_train["label"] - ev_train["base"]
    ev_valid["label_processed"] = ev_valid["label"] - ev_valid["base"]
    day_w = 1.0 / ev_train.groupby("datetime").size()
    ev_train = ev_train.join(day_w.rename("w"), on="datetime")
    stage2_feats = SENTIMENT if use_sentiment else []
    pred_ts = df.copy()
    pred_ts["base"] = m_s1.predict(pred_ts[ALPHA20], num_iteration=m_s1.best_iteration)
    pred_ts["residual"] = 0.0
    if stage2_feats:
        m_s2 = lgb_train(ev_train, ev_valid, stage2_feats, weights=ev_train["w"])
        pred_ts.loc[event_mask, "residual"] = m_s2.predict(df.loc[event_mask, stage2_feats], num_iteration=m_s2.best_iteration)
    pred_ts["score"] = pred_ts["base"] + pred_ts["residual"]
    metrics.extend(eval_periods(pred_ts, "label", spec_id, "delayed", "two_stage", "LABEL0", "full_panel"))
    results["two_stage_pred"] = pred_ts

    # Equivalence probe for D1
    if use_sentiment:
        feats_equiv = ALPHA20 + SENTIMENT
        m_eq = lgb_train(train, valid, feats_equiv)
        pred_eq = df.copy()
        pred_eq["score"] = m_eq.predict(pred_eq[feats_equiv], num_iteration=m_eq.best_iteration)
        v_fp = next(x for x in metrics if x["architecture"] == "full_panel" and x["period"] == "valid" and x["ic_threshold"] == 25)
        v_ts = next(x for x in metrics if x["architecture"] == "two_stage" and x["period"] == "valid" and x["ic_threshold"] == 25)
        results["equivalence"] = {
            "valid_rank_ic_full_panel_thr25": v_fp["mean_daily_rank_ic"],
            "valid_rank_ic_two_stage_thr25": v_ts["mean_daily_rank_ic"],
            "rank_ic_gap": v_fp["mean_daily_rank_ic"] - v_ts["mean_daily_rank_ic"],
            "not_equivalent_to_no_news_flag_only": abs(v_fp["mean_daily_rank_ic"] - v_ts["mean_daily_rank_ic"]) > 1e-6,
        }
    return metrics, results


def run_immediate(spec_id: str, use_sentiment: bool) -> tuple[list[dict], dict]:
    ev = build_event_panel()
    cons = build_conservative_event_panel()
    targets = {
        "H_INTRADAY_T_v2": "primary",
        "H_SIGNAL_DAY_v2": "robustness",
        "H_OVERNIGHT_v2": "robustness",
    }
    metrics = []
    results = {}

    for target, role in targets.items():
        sub = ev.dropna(subset=[target]).copy()
        sub["label"] = sub[target]
        train = process_train_label(sub[sub["split"] == "train"])
        valid = process_train_label(sub[sub["split"] == "valid"])
        w = 1.0 / train.groupby("datetime").size()
        train = train.join(w.rename("w"), on="datetime")

        feats_fp = ALPHA20 + (SENTIMENT if use_sentiment else [])
        log.info("Immediate %s %s full-panel-on-events", spec_id, target)
        m = lgb_train(train, valid, feats_fp, weights=train["w"])
        pred = sub.copy()
        pred["score"] = m.predict(pred[feats_fp], num_iteration=m.best_iteration)
        sample_tag = "v2_tradable"
        metrics.extend(eval_periods(pred, target, spec_id, "immediate", "full_panel", target, sample_tag))
        pm = event_pooled_metrics(pred, target)
        port_rows = []
        for period in PERIODS:
            pf = portfolio_eval(pred, target, period)
            port_rows.append({"spec": spec_id, "family": "immediate", "architecture": "full_panel", "target": target, "sample": sample_tag, **pm, **pf})
        results[f"{target}_portfolio"] = port_rows

        # Two-stage on events
        log.info("Immediate %s %s two-stage", spec_id, target)
        full = build_delayed_panel()
        tr = process_train_label(full[full["split"] == "train"])
        va = process_train_label(full[full["split"] == "valid"])
        m1 = lgb_train(tr, va, ALPHA20)
        pred_ts = sub.copy()
        pred_ts["base"] = m1.predict(pred_ts[ALPHA20], num_iteration=m1.best_iteration)
        tr_ev = train.copy()
        va_ev = valid.copy()
        tr_ev["base"] = m1.predict(tr_ev[ALPHA20], num_iteration=m1.best_iteration)
        va_ev["base"] = m1.predict(va_ev[ALPHA20], num_iteration=m1.best_iteration)
        tr_ev["label_processed"] = tr_ev["label"] - tr_ev["base"]
        va_ev["label_processed"] = va_ev["label"] - va_ev["base"]
        s2_feats = SENTIMENT if use_sentiment else []
        if s2_feats:
            m2 = lgb_train(tr_ev, va_ev, s2_feats, weights=tr_ev["w"])
            pred_ts["residual"] = m2.predict(pred_ts[s2_feats], num_iteration=m2.best_iteration)
            pred_ts["score"] = pred_ts["base"] + pred_ts["residual"]
        else:
            pred_ts["score"] = pred_ts["base"]
        metrics.extend(eval_periods(pred_ts, target, spec_id, "immediate", "two_stage", target, sample_tag))

        if role == "primary":
            results[f"{target}_pred"] = pred_ts

    # S3 conservative robustness (primary target only)
    target = "H_INTRADAY_T"
    if target in cons.columns:
        log.info("Immediate %s conservative robustness", spec_id)
        cons = cons.copy()
        cons["split"] = cons["datetime"].map(assign_split_ts)
        subc = cons.dropna(subset=[target]).copy()
        subc["label"] = subc[target]
        tr = process_train_label(subc[subc["split"] == "train"])
        va = process_train_label(subc[subc["split"] == "valid"])
        if len(tr) and len(va):
            feats = ALPHA20 + (SENTIMENT if use_sentiment else [])
            m = lgb_train(tr, va, feats)
            pred_c = subc.copy()
            pred_c["score"] = m.predict(pred_c[feats], num_iteration=m.best_iteration)
            metrics.extend(eval_periods(pred_c, target, spec_id, "immediate", "full_panel", "H_INTRADAY_T_s3", "s3_conservative_tradable"))

    return metrics, results


def write_report(all_metrics: pd.DataFrame, equiv: dict, out: Path) -> None:
    lines = [
        "# S5R AV-Only Corrected Experiments",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Experiments run",
        "",
        "- **D0** Alpha20 + LABEL0 (full-panel + two-stage)",
        "- **D1** Alpha20 + AV + LABEL0 (full-panel + two-stage)",
        "- **E0** Event v2 controls (full-panel-on-events + two-stage)",
        "- **E1** Event v2 + AV (full-panel-on-events + two-stage)",
        "",
        "## Primary immediate target",
        "",
        "H_INTRADAY_T_v2 on event-time v2 tradable sample (excludes regular-hours + 24 null trade dates).",
        "",
        "## D1 architecture equivalence",
        "",
        json.dumps(equiv.get("D1", {}), indent=2),
        "",
        "## Validation Rank IC (threshold=25)",
        "",
        "| Spec | Family | Arch | Target | Period | Rank IC | Valid days | Pct valid |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    sub = all_metrics[(all_metrics["ic_threshold"] == 25) & (all_metrics["period"] == "valid")]
    for _, r in sub.iterrows():
        lines.append(
            f"| {r.get('spec','')} | {r.get('family','')} | {r.get('architecture','')} | "
            f"{r.get('target','')} | {r.get('period','')} | {r.get('mean_daily_rank_ic', float('nan')):.4f} | "
            f"{int(r.get('valid_ic_days',0))} | {r.get('pct_valid_ic_days', float('nan')):.1%} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "reports").mkdir(exist_ok=True)
    (OUT_ROOT / "manifests").mkdir(exist_ok=True)
    (OUT_ROOT / "delayed").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "immediate").mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    equiv = {}

    for spec_id, sent in [("D0", False), ("D1", True)]:
        m, res = run_delayed(spec_id, sent)
        all_metrics.extend(m)
        if sent and "equivalence" in res:
            equiv["D1"] = res["equivalence"]
        with (OUT_ROOT / f"delayed/{spec_id}_results.pkl").open("wb") as f:
            pickle.dump(res, f)

    for spec_id, sent in [("E0", False), ("E1", True)]:
        m, res = run_immediate(spec_id, sent)
        all_metrics.extend(m)
        with (OUT_ROOT / f"immediate/{spec_id}_results.pkl").open("wb") as f:
            pickle.dump(res, f)

    df = pd.DataFrame(all_metrics)
    df.to_csv(OUT_ROOT / "S5R_av_only_metrics.csv", index=False)
    port_rows = []
    for spec_id in ["E0", "E1"]:
        pkl = OUT_ROOT / f"immediate/{spec_id}_results.pkl"
        if pkl.exists():
            res = pickle.loads(pkl.read_bytes())
            for k, v in res.items():
                if k.endswith("_portfolio"):
                    port_rows.extend(v)
    if port_rows:
        pd.DataFrame(port_rows).to_csv(OUT_ROOT / "S5R_av_only_portfolio_metrics.csv", index=False)
    write_report(df, equiv, OUT_ROOT / "reports/S5R_av_only_results.md")
    (OUT_ROOT / "reports/S5R_architecture_equivalence_audit.md").write_text(
        "# S5R Architecture Equivalence Audit\n\n"
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}\n\n"
        "## D1 full-panel vs two-stage (validation, IC threshold 25)\n\n"
        f"```json\n{json.dumps(json_safe(equiv.get('D1', {})), indent=2)}\n```\n",
        encoding="utf-8",
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "experiments": ["D0", "D1", "E0", "E1"],
        "architectures": ["full_panel", "two_stage"],
        "primary_immediate_target": "H_INTRADAY_T_v2",
        "robustness_targets": ["H_SIGNAL_DAY_v2", "H_OVERNIGHT_v2", "H_INTRADAY_T_s3_conservative"],
        "ic_thresholds": IC_THRESHOLDS,
        "equivalence_audit": equiv,
        "holdout_2024_accessed": False,
    }
    (OUT_ROOT / "manifests/S5R_av_only_manifest.json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    log.info("S5R AV-only complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
