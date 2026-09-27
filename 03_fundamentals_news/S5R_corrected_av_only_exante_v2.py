#!/usr/bin/env python3
"""S5R corrected ex-ante immediate experiments: E0_exante / E1_exante.

Alpha20 is lagged to the prior CRSP trading day; AV sentiment uses event-time
v2 articles. Leaves quarantined av_only/ and finbert_full_history/ untouched.
Sample ends 2023."""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import date, datetime, time, timezone
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
OUT_ROOT = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/av_only_exante_v2"
V2_MAP = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/event_time_v2/S5R_event_time_v2_mapping.parquet"
V2_MANIFEST = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/event_time_v2/S5R_event_time_v2_manifest.json"
ARTICLES_PATH = PROJECT_ROOT / "data/sentiment_raw/alpha_vantage_full/normalized/articles.parquet"
CONSERVATIVE_FLAGS = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/preflight/S5R_event_timing_flags.parquet"
TRADING_DAYS = PROJECT_ROOT / "data/sentiment_raw/alpha_vantage_full/_cache/trading_days.parquet"
WORKFLOW_YAML = PROJECT_ROOT / "experiments/conf_alpha20_sp500_transfer.yaml"
QLIB_DATA = PROJECT_ROOT / "staging/qlib_data"
LABEL_EXPR = "Ref($close, -2)/Ref($close, -1) - 1"

MARKET_OPEN = time(9, 30)
POSITIVE_LABELS = {"Bullish", "Somewhat-Bullish"}
NEGATIVE_LABELS = {"Bearish", "Somewhat-Bearish"}

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
TARGET_COLS = ["H_INTRADAY_T_v2", "H_SIGNAL_DAY_v2", "H_OVERNIGHT_v2", "LABEL0_v2"]
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
ROLL_MIN = 10
ROLL_WIN = 20
DECAY_3D = np.array([np.exp(-np.log(2) * k / 3.0) for k in range(3)])
DECAY_5D = np.array([np.exp(-np.log(2) * k / 5.0) for k in range(5)])

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
BOOTSTRAP_N = 500
SHUFFLE_IC_TOL = 0.02

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("S5R-EXANTE")


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
    if isinstance(obj, date):
        return obj.isoformat()
    if obj is None:
        return None
    return str(obj)


def assign_split_ts(ts: pd.Timestamp) -> str | None:
    d = ts.date()
    for name in ("train", "valid", "test"):
        s, e = PERIODS[name]
        if date.fromisoformat(s) <= d <= date.fromisoformat(e):
            return name
    return None


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


def bootstrap_mean_ci(values: pd.Series, n: int = BOOTSTRAP_N, seed: int = SEED) -> dict[str, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) < 5:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)])
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }


def init_qlib() -> None:
    import qlib
    from qlib.constant import REG_US
    qlib.init(provider_uri=str(QLIB_DATA), region=REG_US, kernels=1)


def load_alpha20_raw() -> pd.DataFrame:
    init_qlib()
    conf = yaml.safe_load(WORKFLOW_YAML.read_text(encoding="utf-8"))
    handler = init_instance_by_config({
        "class": "DataHandlerLP", "module_path": "qlib.contrib.data.handler",
        "kwargs": conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"],
    })
    feat = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).reset_index()
    feat["datetime"] = pd.to_datetime(feat["datetime"])
    return feat[["datetime", "instrument"] + ALPHA20]


def load_trading_calendar() -> pd.DataFrame:
    td = pd.read_parquet(TRADING_DAYS)
    td["date"] = pd.to_datetime(td["date"]).dt.normalize()
    td = td.sort_values(["permno", "date"])
    td["prev_trading_date"] = td.groupby("permno")["date"].shift(1)
    td["next_trading_date"] = td.groupby("permno")["date"].shift(-1)
    return td


def attach_prev_trading_date(df: pd.DataFrame, event_col: str = "event_trade_date_v2") -> pd.DataFrame:
    cal = load_trading_calendar()
    out = df.copy()
    out["_event_dt"] = pd.to_datetime(out[event_col]).dt.normalize()
    out["permno"] = out["permno"].astype(int)
    cal = cal.copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    cal["prev_trading_date"] = pd.to_datetime(cal["prev_trading_date"]).dt.normalize()
    merged = out.merge(
        cal[["permno", "date", "prev_trading_date"]],
        left_on=["permno", "_event_dt"],
        right_on=["permno", "date"],
        how="left",
    )
    merged = merged.rename(columns={"prev_trading_date": "feature_date"})
    merged = merged.drop(columns=["date", "_event_dt"], errors="ignore")
    return merged


def merge_lagged_alpha(panel: pd.DataFrame, event_col: str = "event_trade_date_v2") -> pd.DataFrame:
    alpha = load_alpha20_raw()
    panel = attach_prev_trading_date(panel, event_col=event_col)
    before = len(panel)
    panel = panel.dropna(subset=["feature_date"]).copy()
    dropped_no_prev = before - len(panel)
    if dropped_no_prev:
        log.warning("Dropped %s rows without prior CRSP session for Alpha20 lag", dropped_no_prev)
    panel["instrument"] = "P" + panel["permno"].astype(str)
    alpha = alpha.rename(columns={"datetime": "feature_date"})
    alpha["feature_date"] = pd.to_datetime(alpha["feature_date"]).dt.normalize()
    panel["feature_date"] = pd.to_datetime(panel["feature_date"]).dt.normalize()
    out = panel.merge(alpha, on=["feature_date", "instrument"], how="left")
    missing_alpha = out[ALPHA20].isna().all(axis=1).sum()
    if missing_alpha:
        log.warning("Dropping %s rows with no Alpha20 on feature_date", int(missing_alpha))
        out = out[~out[ALPHA20].isna().all(axis=1)].copy()
    assert_no_target_in_features(out)
    return out


def assert_no_target_in_features(df: pd.DataFrame) -> None:
    forbidden = set(TARGET_COLS) | {"label", "label_processed", "score", "base", "residual"}
    overlap = forbidden & set(df.columns) & set(ALPHA20 + SENTIMENT)
    if overlap:
        raise RuntimeError(f"Target-derived columns in feature set: {overlap}")


def assert_exante_timing(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("Empty panel after ex-ante feature join")
    evt = pd.to_datetime(df["event_trade_date_v2"]).dt.normalize()
    fd = pd.to_datetime(df["feature_date"]).dt.normalize()
    if not (fd < evt).all():
        bad = int((fd >= evt).sum())
        raise RuntimeError(f"feature_date >= event_trade_date_v2 on {bad} rows")
    cal = load_trading_calendar()
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    cal["prev_trading_date"] = pd.to_datetime(cal["prev_trading_date"]).dt.normalize()
    chk = df[["permno", "event_trade_date_v2", "feature_date"]].copy()
    chk["event_trade_date_v2"] = pd.to_datetime(chk["event_trade_date_v2"]).dt.normalize()
    chk = chk.merge(
        cal[["permno", "date", "prev_trading_date"]],
        left_on=["permno", "event_trade_date_v2"],
        right_on=["permno", "date"],
        how="left",
    )
    chk["feature_date"] = pd.to_datetime(chk["feature_date"]).dt.normalize()
    mismatch = chk["feature_date"].ne(chk["prev_trading_date"]) | chk["prev_trading_date"].isna()
    if mismatch.any():
        raise RuntimeError(f"feature_date is not prior CRSP session on {int(mismatch.sum())} rows")


def filter_eligible_articles(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["tradable_intraday_v2"] == 1].copy()
    out = out[out["event_trade_date_v2"].notna()]
    out = out[out["publication_bucket_v2"] != "timestamp_ambiguous"]
    out = out[out["publication_bucket_v2"] != "same_day_regular_hours"]
    pub_d = pd.to_datetime(out["publication_date_et"])
    evt_d = pd.to_datetime(out["event_trade_date_v2"])
    same_day_am = (out["publication_bucket_v2"] == "same_day_aftermarket") & (pub_d == evt_d)
    out = out[~same_day_am]
    if "publication_ts_et" in out.columns:
        ts = pd.to_datetime(out["publication_ts_et"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
        pre = out["publication_bucket_v2"] == "same_day_premarket"
        bad_pre = pre & ts.notna() & (ts.dt.time >= MARKET_OPEN)
        out = out[~bad_pre]
    return out


def weighted_sentiment(g: pd.DataFrame) -> float:
    rel = g["relevance_score_num"]
    score = g["ticker_sentiment_score_num"]
    denom = rel.sum()
    if denom <= 0:
        return np.nan
    return float((score * rel).sum() / denom)


def aggregate_event_sentiment(articles_enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (permno, evt), g in articles_enriched.groupby(["permno", "event_trade_date_v2"]):
        n = len(g)
        labels = g["ticker_sentiment_label"].astype(str)
        pos = labels.isin(POSITIVE_LABELS).sum()
        neg = labels.isin(NEGATIVE_LABELS).sum()
        scores = g["ticker_sentiment_score_num"]
        rows.append({
            "permno": int(permno),
            "event_trade_date_v2": pd.Timestamp(evt).normalize(),
            "news_count": n,
            "log_news_count": float(np.log1p(n)),
            "ticker_mean_sentiment": float(scores.mean()),
            "relevance_weighted_sentiment": weighted_sentiment(g),
            "positive_news_share": float(pos / n),
            "negative_news_share": float(neg / n),
            "net_positive_share": float((pos - neg) / n),
            "sentiment_dispersion": float(scores.std(ddof=0)) if n >= 2 else np.nan,
            "relevance_mean": float(g["relevance_score_num"].mean()),
            "no_news_flag": 0,
        })
    return pd.DataFrame(rows)


def decay_feature(values: pd.Series, weights: np.ndarray) -> pd.Series:
    n = len(weights)

    def _one(window: np.ndarray) -> float:
        valid_mask = ~np.isnan(window)
        if not valid_mask.any():
            return np.nan
        w = weights[: len(window)][valid_mask]
        v = window[valid_mask]
        return float(np.dot(w, v) / w.sum())

    return values.rolling(n, min_periods=1).apply(_one, raw=True)


def add_event_rolling_sentiment(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["permno", "event_trade_date_v2"]).copy()
    parts = []
    for _, grp in panel.groupby("permno", sort=False):
        g = grp.copy()
        w = g["relevance_weighted_sentiment"]
        g["sentiment_change_1d"] = w - w.shift(1)
        prior_mean = w.shift(1).rolling(ROLL_WIN, min_periods=ROLL_MIN).mean()
        g["sentiment_surprise_20d"] = w - prior_mean
        g["sentiment_3d_decay"] = decay_feature(w, DECAY_3D)
        g["sentiment_5d_decay"] = decay_feature(w, DECAY_5D)
        nc = g["news_count"].astype(float)
        roll_mean = nc.rolling(ROLL_WIN, min_periods=ROLL_MIN).mean()
        roll_std = nc.rolling(ROLL_WIN, min_periods=ROLL_MIN).std(ddof=0)
        g["abnormal_news_count_20d"] = (nc - roll_mean) / roll_std.replace(0, np.nan)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    for col in DIRECTIONAL:
        out[col] = out[col].where(out["no_news_flag"] == 0)
    return out


def build_exante_event_panel(include_sentiment: bool) -> pd.DataFrame:
    v2 = pd.read_parquet(V2_MAP)
    arts = pd.read_parquet(
        ARTICLES_PATH,
        columns=[
            "permno", "canonical_url_hash", "relevance_score", "ticker_sentiment_score",
            "ticker_sentiment_label",
        ],
    )
    arts["relevance_score_num"] = pd.to_numeric(arts["relevance_score"], errors="coerce")
    arts["ticker_sentiment_score_num"] = pd.to_numeric(arts["ticker_sentiment_score"], errors="coerce")
    arts = arts.drop_duplicates(["permno", "canonical_url_hash"], keep="first")

    eligible = filter_eligible_articles(v2)
    eligible = eligible.merge(
        arts,
        on=["permno", "canonical_url_hash"],
        how="left",
        validate="many_to_one",
    )
    eligible = eligible[eligible["relevance_score_num"] > 0]

    sent_daily = aggregate_event_sentiment(eligible)
    targets = (
        eligible.sort_values(["permno", "event_trade_date_v2", "time_published_utc"])
        .drop_duplicates(["permno", "event_trade_date_v2"], keep="first")[
            [
                "permno", "event_trade_date_v2", "split", "covid_post",
                "H_INTRADAY_T_v2", "H_SIGNAL_DAY_v2", "H_OVERNIGHT_v2",
            ]
        ]
        .copy()
    )
    targets["event_trade_date_v2"] = pd.to_datetime(targets["event_trade_date_v2"]).dt.normalize()
    panel = targets.merge(sent_daily, on=["permno", "event_trade_date_v2"], how="left")
    panel = merge_lagged_alpha(panel)
    assert_exante_timing(panel)

    if include_sentiment:
        panel = add_event_rolling_sentiment(panel)
    else:
        for c in SENTIMENT:
            if c not in panel.columns:
                panel[c] = np.nan
        panel["no_news_flag"] = 1
        panel["news_count"] = 0
        panel["log_news_count"] = 0.0

    panel["datetime"] = pd.to_datetime(panel["event_trade_date_v2"])
    panel["instrument"] = "P" + panel["permno"].astype(str)
    panel["split"] = panel["datetime"].map(assign_split_ts)
    panel = panel[panel["datetime"] <= pd.Timestamp("2023-12-31")].copy()
    return panel.reset_index(drop=True)


def build_exante_conservative_panel(include_sentiment: bool) -> pd.DataFrame:
    v2 = pd.read_parquet(V2_MAP)
    flags = pd.read_parquet(CONSERVATIVE_FLAGS)
    m = v2.merge(
        flags[["permno", "canonical_url_hash", "h_intraday_tradable_open_to_close", "eligible_signal_date"]],
        on=["permno", "canonical_url_hash"],
        how="inner",
    )
    m = m[m["h_intraday_tradable_open_to_close"] == True].copy()  # noqa: E712
    m["event_trade_date_v2"] = pd.to_datetime(m["eligible_signal_date"]).dt.date

    arts = pd.read_parquet(
        ARTICLES_PATH,
        columns=["permno", "canonical_url_hash", "relevance_score", "ticker_sentiment_score", "ticker_sentiment_label"],
    )
    arts["relevance_score_num"] = pd.to_numeric(arts["relevance_score"], errors="coerce")
    arts["ticker_sentiment_score_num"] = pd.to_numeric(arts["ticker_sentiment_score"], errors="coerce")
    arts = arts.drop_duplicates(["permno", "canonical_url_hash"], keep="first")
    m = m.merge(arts, on=["permno", "canonical_url_hash"], how="left")
    m = m[m["relevance_score_num"] > 0]

    m["event_trade_date_v2"] = pd.to_datetime(m["eligible_signal_date"]).dt.normalize()
    sent_daily = aggregate_event_sentiment(m)
    init_qlib()
    from qlib.data import D
    m["instrument"] = "P" + m["permno"].astype(str)
    inst = sorted(m["instrument"].unique())
    px = D.features(inst, ["$open", "$close"], "2010-01-01", "2023-12-31", freq="day").reset_index()
    px["datetime"] = pd.to_datetime(px["datetime"])
    px = px.rename(columns={"$open": "open", "$close": "close", "datetime": "event_trade_date_v2"})
    px = px.sort_values(["instrument", "event_trade_date_v2"])
    px["H_INTRADAY_T"] = px["close"] / px["open"] - 1

    targets = m.drop_duplicates(["permno", "eligible_signal_date"], keep="first")[
        ["permno", "eligible_signal_date"]
    ].copy()
    targets["event_trade_date_v2"] = pd.to_datetime(targets["eligible_signal_date"]).dt.normalize()
    targets["instrument"] = "P" + targets["permno"].astype(str)
    targets = targets.merge(
        px[["instrument", "event_trade_date_v2", "H_INTRADAY_T"]],
        on=["instrument", "event_trade_date_v2"],
        how="left",
    )
    panel = targets.merge(sent_daily, on=["permno", "event_trade_date_v2"], how="left")
    panel = merge_lagged_alpha(panel)
    assert_exante_timing(panel)
    if include_sentiment:
        panel = add_event_rolling_sentiment(panel)
    panel["datetime"] = pd.to_datetime(panel["event_trade_date_v2"])
    panel["split"] = panel["datetime"].map(assign_split_ts)
    return panel.drop_duplicates(["permno", "datetime"]).reset_index(drop=True)


def build_delayed_lagged_panel() -> pd.DataFrame:
    from qlib.data import D
    init_qlib()
    s3 = pd.read_parquet(PROJECT_ROOT / "data/sentiment_experiments/S3_daily_sentiment_panel/S3_daily_sentiment_raw.parquet")
    s3["datetime"] = pd.to_datetime(s3["datetime"])
    inst = D.list_instruments(D.instruments("sp500"), start_time="2010-01-01", end_time="2023-12-31", as_list=True)
    raw = D.features(inst, [LABEL_EXPR], "2010-01-01", "2023-12-31", freq="day")
    raw.columns = ["label"]
    labels = raw.reset_index()
    labels["datetime"] = pd.to_datetime(labels["datetime"])
    cal = load_trading_calendar()
    cal_map = cal[["permno", "date", "prev_trading_date"]].rename(columns={"date": "datetime"})
    cal_map["permno"] = cal_map["permno"].astype(int)
    s3 = s3.copy()
    s3["permno"] = s3["instrument"].str.lstrip("P").astype(int)
    df = s3.merge(labels, on=["datetime", "instrument"], how="inner")
    df = df.merge(cal_map, on=["permno", "datetime"], how="left")
    df = df.rename(columns={"prev_trading_date": "feature_date", "datetime": "event_trade_date_v2"})
    alpha = load_alpha20_raw().rename(columns={"datetime": "feature_date"})
    df = df.merge(alpha, on=["feature_date", "instrument"], how="inner")
    df["datetime"] = pd.to_datetime(df["event_trade_date_v2"])
    df["split"] = df["datetime"].map(assign_split_ts)
    return df


def process_train_label(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    out = df.dropna(subset=[label_col]).copy()
    out["label_processed"] = out.groupby("datetime", group_keys=False)[label_col].apply(zscore)
    return out


EVENT_LGB = {**OFFICIAL_LGB, "min_data_in_leaf": 5, "min_sum_hessian_in_leaf": 1e-3}


def lgb_train(train_df, valid_df, features, weights=None, params=None):
    p = (params or EVENT_LGB).copy()
    p.update({"seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED, "data_random_seed": SEED})
    w = weights.loc[train_df.index].to_numpy() if weights is not None else None
    train_set = lgb.Dataset(train_df[features], label=train_df["label_processed"], weight=w, free_raw_data=False)
    valid_set = lgb.Dataset(valid_df[features], label=valid_df["label_processed"], reference=train_set, free_raw_data=False)
    return lgb.train(
        p, train_set, num_boost_round=NUM_BOOST,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(period=0)],
    )


def daily_ic(pred, label_col, min_n, date_col="datetime"):
    rows = []
    eligible = 0
    for dt, g in pred.groupby(date_col):
        eligible += 1
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
        "eligible_calendar_days": eligible,
        "valid_ic_days": int(len(daily)),
        "pct_valid_days": float(len(daily) / eligible) if eligible else 0.0,
        "min_cross_section": min_n,
    }
    return daily, meta


def summarize_ic(daily, meta, period, spec, **extra):
    if daily.empty:
        return {
            "spec": spec, "period": period,
            "mean_daily_ic": np.nan, "mean_daily_rank_ic": np.nan,
            "icir": np.nan, "rank_icir": np.nan, "rank_ic_newey_west_tstat": np.nan,
            "valid_ic_days": 0, "eligible_calendar_days": meta.get("eligible_calendar_days", 0),
            "pct_valid_ic_days": 0.0, "min_cross_section": meta.get("min_cross_section", np.nan),
            **extra,
        }
    ic, ric = daily["ic"].dropna(), daily["rank_ic"].dropna()
    ic_m = float(ic.mean()) if len(ic) else np.nan
    ric_m = float(ric.mean()) if len(ric) else np.nan
    ic_s = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    ric_s = float(ric.std(ddof=1)) if len(ric) > 1 else np.nan
    boot = bootstrap_mean_ci(ric)
    return {
        "spec": spec, "period": period,
        "mean_daily_ic": ic_m, "mean_daily_rank_ic": ric_m,
        "icir": ic_m / ic_s if ic_s and ic_s > 0 else np.nan,
        "rank_icir": ric_m / ric_s if ric_s and ric_s > 0 else np.nan,
        "rank_ic_newey_west_tstat": newey_west_tstat(ric),
        "rank_ic_bootstrap_mean": boot["mean"],
        "rank_ic_bootstrap_ci_low": boot["ci_low"],
        "rank_ic_bootstrap_ci_high": boot["ci_high"],
        "valid_ic_days": meta.get("valid_ic_days", int(len(ic))),
        "eligible_calendar_days": meta.get("eligible_calendar_days", np.nan),
        "pct_valid_ic_days": meta.get("pct_valid_days", np.nan),
        "min_cross_section": meta.get("min_cross_section", np.nan),
        **extra,
    }


def eval_periods(pred, label_col, spec, arch, target, sample_tag):
    rows = []
    for period, (s, e) in PERIODS.items():
        sub = pred[(pred["datetime"] >= pd.Timestamp(s)) & (pred["datetime"] <= pd.Timestamp(e))].copy()
        if sub.empty:
            continue
        for thr in IC_THRESHOLDS:
            daily, meta = daily_ic(sub, label_col, thr)
            rows.append(summarize_ic(
                daily, meta, period, spec,
                family="immediate_exante", architecture=arch, target=target,
                sample=sample_tag, ic_threshold=thr,
            ))
    return rows


def event_pooled_metrics(pred, label_col):
    sub = pred.dropna(subset=["score", label_col])
    if len(sub) < 10:
        return {"pooled_pearson": np.nan, "pooled_spearman": np.nan, "directional_accuracy": np.nan, "q5_q1_spread": np.nan}
    q = sub[label_col].quantile([0.2, 0.8])
    low = sub[sub[label_col] <= q.iloc[0]]["score"].mean()
    high = sub[sub[label_col] >= q.iloc[1]]["score"].mean()
    return {
        "pooled_pearson": float(sub["score"].corr(sub[label_col])),
        "pooled_spearman": float(sub["score"].corr(sub[label_col], method="spearman")),
        "directional_accuracy": float((np.sign(sub["score"]) == np.sign(sub[label_col])).mean()),
        "q5_q1_spread": float(high - low),
    }


def portfolio_eval(pred, label_col, period):
    s, e = PERIODS[period]
    sub = pred[(pred["datetime"] >= pd.Timestamp(s)) & (pred["datetime"] <= pd.Timestamp(e))].dropna(subset=["score", label_col])
    traded, skipped, holdings, returns = 0, 0, [], []
    for _, g in sub.groupby("datetime"):
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


def assert_prediction_variance(pred, spec, target) -> bool:
    scores = pred["score"].dropna()
    if scores.nunique() < 2:
        log.error("Constant predictions for %s %s", spec, target)
        return False
    daily_nuniq = pred.groupby("datetime")["score"].nunique()
    if (daily_nuniq <= 1).mean() > 0.5:
        log.error("Majority constant daily predictions for %s %s", spec, target)
        return False
    return True


def run_preacceptance_diagnostics(panel, pred_primary, target="H_INTRADAY_T_v2") -> dict[str, Any]:
    valid = pred_primary[
        (pred_primary["datetime"] >= pd.Timestamp(PERIODS["valid"][0]))
        & (pred_primary["datetime"] <= pd.Timestamp(PERIODS["valid"][1]))
    ].copy()
    shuf = valid.copy()
    shuf[target] = shuf.groupby("datetime")[target].transform(
        lambda s: s.sample(frac=1.0, random_state=SEED).values
    )
    _, meta = daily_ic(shuf, target, 25)
    daily, _ = daily_ic(valid, target, 25)
    shuf_daily, _ = daily_ic(shuf, target, 25)
    shuf_ic = float(shuf_daily["rank_ic"].mean()) if not shuf_daily.empty else float("nan")

    feat_ic = {}
    for c in ["RESI5", "ROC60", "KLEN"]:
        tmp = valid.dropna(subset=[c, target]).copy()
        tmp["score"] = tmp[c]
        daily, _ = daily_ic(tmp, target, 25)
        feat_ic[c] = float(daily["rank_ic"].mean()) if not daily.empty else float("nan")

    overnight = panel["H_OVERNIGHT_v2"].dropna()
    diag = {
        "shuffle_valid_rank_ic_thr25": shuf_ic,
        "shuffle_pass": abs(shuf_ic) <= SHUFFLE_IC_TOL if shuf_ic == shuf_ic else False,
        "feature_date_lt_event_all": bool((pd.to_datetime(panel["feature_date"]) < pd.to_datetime(panel["event_trade_date_v2"])).all()),
        "single_factor_valid_rank_ic": feat_ic,
        "prediction_score_nunique": int(valid["score"].nunique()),
        "H_OVERNIGHT_nunique": int(overnight.nunique()),
        "H_OVERNIGHT_daily_nunique_ge2_pct": float(panel.groupby("datetime")["H_OVERNIGHT_v2"].nunique().ge(2).mean()),
        "valid_rank_ic_thr25": float(daily["rank_ic"].mean()) if not daily.empty else float("nan"),
    }
    return diag


def verify_sample_counts(panel: pd.DataFrame) -> dict[str, Any]:
    manifest = json.loads(V2_MANIFEST.read_text())
    eligible = filter_eligible_articles(pd.read_parquet(V2_MAP))
    return {
        "manifest_tradable_article_rows": manifest["mapping_summary"]["tradable_intraday_v2_rows"],
        "eligible_article_rows_after_filter": int(len(eligible)),
        "model_panel_permno_event_days": int(len(panel)),
        "split_counts": panel.groupby("split").size().to_dict(),
    }


def run_immediate_exante(spec_id: str, use_sentiment: bool) -> tuple[list[dict], dict, dict]:
    panel = build_exante_event_panel(include_sentiment=use_sentiment)
    cons = build_exante_conservative_panel(include_sentiment=use_sentiment)
    targets = {
        "H_INTRADAY_T_v2": "primary",
        "H_SIGNAL_DAY_v2": "robustness",
        "H_OVERNIGHT_v2": "robustness",
    }
    metrics: list[dict] = []
    results: dict = {}
    diagnostics: dict = {}
    constant_flags: list[str] = []

    feats_fp = ALPHA20 + (SENTIMENT if use_sentiment else [])

    for target, role in targets.items():
        sub = panel.dropna(subset=[target]).copy()
        sub["label"] = sub[target]
        train = process_train_label(sub[sub["split"] == "train"])
        valid_df = process_train_label(sub[sub["split"] == "valid"])

        log.info("%s %s full-panel ex-ante", spec_id, target)
        model = lgb_train(train, valid_df, feats_fp, weights=None)
        pred = sub.copy()
        pred["score"] = model.predict(pred[feats_fp], num_iteration=model.best_iteration)
        if not assert_prediction_variance(pred, spec_id, target):
            constant_flags.append(f"{spec_id}:{target}:full_panel")
        metrics.extend(eval_periods(pred, target, spec_id, "full_panel", target, "v2_tradable_exante"))
        pm = event_pooled_metrics(pred, target)
        port_rows = []
        for period in PERIODS:
            pf = portfolio_eval(pred, target, period)
            port_rows.append({
                "spec": spec_id, "family": "immediate_exante", "architecture": "full_panel",
                "target": target, "sample": "v2_tradable_exante", **pm, **pf,
            })
        results[f"{target}_portfolio"] = port_rows
        if role == "primary":
            results[f"{target}_pred"] = pred

        log.info("%s %s two-stage ex-ante", spec_id, target)
        delayed = build_delayed_lagged_panel()
        tr = process_train_label(delayed[delayed["split"] == "train"], "label")
        va = process_train_label(delayed[delayed["split"] == "valid"], "label")
        m1 = lgb_train(tr, va, ALPHA20)
        pred_ts = sub.copy()
        pred_ts["base"] = m1.predict(pred_ts[ALPHA20], num_iteration=m1.best_iteration)
        if use_sentiment:
            tr_ev = train.copy()
            va_ev = valid_df.copy()
            tr_ev["base"] = m1.predict(tr_ev[ALPHA20], num_iteration=m1.best_iteration)
            va_ev["base"] = m1.predict(va_ev[ALPHA20], num_iteration=m1.best_iteration)
            tr_ev["label_processed"] = tr_ev["label"] - tr_ev["base"]
            va_ev["label_processed"] = va_ev["label"] - va_ev["base"]
            m2 = lgb_train(tr_ev, va_ev, SENTIMENT, weights=None)
            pred_ts["residual"] = m2.predict(pred_ts[SENTIMENT], num_iteration=m2.best_iteration)
            pred_ts["score"] = pred_ts["base"] + pred_ts["residual"]
        else:
            pred_ts["score"] = pred_ts["base"]
        if not assert_prediction_variance(pred_ts, spec_id + "_twostage", target):
            constant_flags.append(f"{spec_id}:{target}:two_stage")
        metrics.extend(eval_periods(pred_ts, target, spec_id, "two_stage", target, "v2_tradable_exante"))

    target_c = "H_INTRADAY_T"
    subc = cons.dropna(subset=[target_c]).copy()
    if len(subc):
        log.info("%s conservative ex-ante", spec_id)
        tr = process_train_label(subc[subc["split"] == "train"], target_c)
        va = process_train_label(subc[subc["split"] == "valid"], target_c)
        if len(tr) and len(va):
            m = lgb_train(tr, va, feats_fp)
            pred_c = subc.copy()
            pred_c["score"] = m.predict(pred_c[feats_fp], num_iteration=m.best_iteration)
            if not assert_prediction_variance(pred_c, spec_id + "_s3", target_c):
                constant_flags.append(f"{spec_id}:{target_c}:s3_conservative")
            metrics.extend(eval_periods(
                pred_c, target_c, spec_id, "full_panel", "H_INTRADAY_T_s3", "s3_conservative_tradable_exante",
            ))

    if "H_INTRADAY_T_v2_pred" in results:
        diagnostics = run_preacceptance_diagnostics(panel, results["H_INTRADAY_T_v2_pred"])
        diagnostics["sample_counts"] = verify_sample_counts(panel)
        diagnostics["constant_prediction_flags"] = constant_flags

    return metrics, results, diagnostics


def compute_incremental(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in metrics_df["architecture"].unique():
        for target in metrics_df["target"].unique():
            for period in metrics_df["period"].unique():
                for thr in metrics_df["ic_threshold"].unique():
                    e0 = metrics_df[
                        (metrics_df["spec"] == "E0_exante") & (metrics_df["architecture"] == arch)
                        & (metrics_df["target"] == target) & (metrics_df["period"] == period)
                        & (metrics_df["ic_threshold"] == thr)
                    ]
                    e1 = metrics_df[
                        (metrics_df["spec"] == "E1_exante") & (metrics_df["architecture"] == arch)
                        & (metrics_df["target"] == target) & (metrics_df["period"] == period)
                        & (metrics_df["ic_threshold"] == thr)
                    ]
                    if e0.empty or e1.empty:
                        continue
                    r0, r1 = e0.iloc[0], e1.iloc[0]
                    rows.append({
                        "architecture": arch, "target": target, "period": period, "ic_threshold": thr,
                        "E0_rank_ic": r0["mean_daily_rank_ic"], "E1_rank_ic": r1["mean_daily_rank_ic"],
                        "delta_rank_ic": r1["mean_daily_rank_ic"] - r0["mean_daily_rank_ic"],
                        "E0_ic": r0["mean_daily_ic"], "E1_ic": r1["mean_daily_ic"],
                        "delta_ic": r1["mean_daily_ic"] - r0["mean_daily_ic"],
                    })
    return pd.DataFrame(rows)


def determine_stage_gate(diag_e0: dict, diag_e1: dict) -> tuple[str, list[str]]:
    notes = []
    if not diag_e0.get("shuffle_pass", False):
        notes.append("E0 shuffle-within-date Rank IC not near zero")
    if not diag_e0.get("feature_date_lt_event_all", False):
        notes.append("feature_date >= event_trade_date_v2 detected")
    if diag_e0.get("prediction_score_nunique", 0) < 2:
        notes.append("constant predictions on E0 primary")
    if diag_e0.get("H_OVERNIGHT_nunique", 0) < 2:
        notes.append("H_OVERNIGHT target variance check failed")
    for spec, d in [("E0", diag_e0), ("E1", diag_e1)]:
        if not d:
            notes.append(f"missing diagnostics for {spec}")
        flags = d.get("constant_prediction_flags") or []
        if flags:
            notes.append(f"{spec} constant prediction models: {', '.join(flags)}")
    if notes:
        if any("constant" in n.lower() for n in notes):
            return "BLOCKED_BY_CONSTANT_PREDICTION", notes
        if any("shuffle" in n for n in notes):
            return "BLOCKED_BY_OTHER_TIMING_ERROR", notes
        if any("feature_date" in n for n in notes):
            return "BLOCKED_BY_FEATURE_SHIFT_ALIGNMENT", notes
        return "BLOCKED_BY_OTHER_TIMING_ERROR", notes
    return "EXANTE_E0_E1_READY", notes


def write_report(metrics_df, incremental_df, diag_e0, diag_e1, gate, notes, out: Path) -> None:
    lines = [
        "# S5R Ex-Ante E0/E1 Corrected Results",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Stage gate:** `{gate}`",
        "",
        "## Timing corrections",
        "",
        "- Alpha20 joined on **prior CRSP trading day** (`feature_date = t-1`) for event_trade_date_v2 = t.",
        "- E1 sentiment aggregated from **event-time v2 eligible articles** (not delayed S3 eligible_signal_date).",
        "",
        "## H_SIGNAL_DAY_v2 note",
        "",
        "`H_SIGNAL_DAY_v2 = close/prior_close - 1` spans **overnight gap plus intraday** return; "
        "it is **not** a pure open-to-close tradable target.",
        "",
        "## Pre-acceptance diagnostics (E0 primary)",
        "",
        "```json",
        json.dumps(json_safe(diag_e0), indent=2),
        "```",
        "",
        "## Validation Rank IC (threshold=25, full_panel)",
        "",
        "| Spec | Target | Period | Rank IC | Bootstrap CI | Valid days |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    sub = metrics_df[
        (metrics_df["ic_threshold"] == 25) & (metrics_df["architecture"] == "full_panel") & (metrics_df["period"] == "valid")
    ]
    for _, r in sub.iterrows():
        ci = f"[{r.get('rank_ic_bootstrap_ci_low', float('nan')):.4f}, {r.get('rank_ic_bootstrap_ci_high', float('nan')):.4f}]"
        lines.append(
            f"| {r['spec']} | {r['target']} | {r['period']} | {r['mean_daily_rank_ic']:.4f} | {ci} | {int(r['valid_ic_days'])} |"
        )
    if notes:
        lines.extend(["", "## Gate notes", ""] + [f"- {n}" for n in notes])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for d in ("reports", "manifests", "immediate", "diagnostics"):
        (OUT_ROOT / d).mkdir(exist_ok=True)

    all_metrics: list[dict] = []
    all_diag: dict[str, dict] = {}

    for spec_id, sent in [("E0_exante", False), ("E1_exante", True)]:
        m, res, diag = run_immediate_exante(spec_id, sent)
        all_metrics.extend(m)
        all_diag[spec_id] = diag
        with (OUT_ROOT / f"immediate/{spec_id}_results.pkl").open("wb") as f:
            pickle.dump(res, f)
        (OUT_ROOT / f"diagnostics/{spec_id}_preacceptance.json").write_text(
            json.dumps(json_safe(diag), indent=2), encoding="utf-8",
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_ROOT / "S5R_exante_metrics.csv", index=False)

    port_rows = []
    for spec_id in ["E0_exante", "E1_exante"]:
        pkl = OUT_ROOT / f"immediate/{spec_id}_results.pkl"
        res = pickle.loads(pkl.read_bytes())
        for k, v in res.items():
            if k.endswith("_portfolio"):
                port_rows.extend(v)
    pd.DataFrame(port_rows).to_csv(OUT_ROOT / "S5R_exante_portfolio_metrics.csv", index=False)

    inc = compute_incremental(metrics_df)
    inc.to_csv(OUT_ROOT / "S5R_exante_incremental_E1_minus_E0.csv", index=False)

    gate, notes = determine_stage_gate(all_diag.get("E0_exante", {}), all_diag.get("E1_exante", {}))

    write_report(metrics_df, inc, all_diag.get("E0_exante", {}), all_diag.get("E1_exante", {}), gate, notes,
                 OUT_ROOT / "reports/S5R_exante_results.md")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": gate,
        "experiments": ["E0_exante", "E1_exante"],
        "supersedes_quarantined": "data/sentiment_experiments/S5R_corrected/av_only/",
        "alpha20_timing": "feature_date = prior CRSP trading day of event_trade_date_v2",
        "sentiment_timing": "event-time v2 eligible articles aggregated per permno x event_trade_date_v2",
        "primary_target": "H_INTRADAY_T_v2",
        "robustness_targets": ["H_SIGNAL_DAY_v2", "H_OVERNIGHT_v2", "H_INTRADAY_T_s3"],
        "H_SIGNAL_DAY_note": "close/prior_close-1 includes overnight plus intraday; not pure open-to-close",
        "preacceptance": all_diag,
        "gate_notes": notes,
        "holdout_2024_accessed": False,
    }
    (OUT_ROOT / "manifests/S5R_exante_manifest.json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    log.info("S5R ex-ante complete: %s", gate)
    return 0 if gate == "EXANTE_E0_E1_READY" else 1


if __name__ == "__main__":
    sys.exit(main())
