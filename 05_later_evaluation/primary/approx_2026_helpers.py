#!/usr/bin/env python3
"""Helpers for 2026 approximate evaluation of the frozen VT0 / P5 deployment rules
on Massive data. Integrity gate runs before 2026 P5/VT0 PnL is opened; 2026 is
not used for training or retargeting."""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fundamental_experiments.F1_fundamental_lightgbm import (  # noqa: E402
    ALL12,
    MAIN_DATASET,
    SEEDS as F1C_SEEDS,
    process_training_label,
    train_model as train_f1c,
)
from fundamental_experiments.RD13_v2_downstream_replication_pipeline import (  # noqa: E402
    DELAYED_SEEDS,
    lgb_params as d2_lgb_params,
    process_train_label as process_d2_label,
    train_lgb as train_d2_lgb,
)
from portfolio_experiments.hmm_regime.signal_utils import (  # noqa: E402
    winsorize_series,
    zscore_series,
)
from portfolio_experiments.market_risk.config import (  # noqa: E402
    MR4A_ANN_FACTOR,
    MR4A_COST_BP,
    MR4A_EQUITY_CAP,
    MR4A_FORECAST_FLOOR,
    MR4A_TARGET_VOL,
    OUT,
    REPORT,
)
from portfolio_experiments.market_risk.run_mr4b import (  # noqa: E402
    exposure_stats,
    metrics,
    target_accuracy,
)
from portfolio_experiments.p5_downside_risk.run_r2_overlay import mdd_from_returns  # noqa: E402

try:
    from qlib.data._libs.rolling import rolling_resi, rolling_rsquare
except Exception:  # pragma: no cover
    rolling_resi = None
    rolling_rsquare = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vt0-2026")

STATUS = "2026 VT0 DEPLOYMENT-STYLE VALIDATION / APPROXIMATE REPLAY"
RESEARCH_ID = "VT0_2026_DEPLOYMENT_VALIDATION"
EXPERIMENT_ID = "VT0-2026-DEPLOYMENT-STYLE-VALIDATION-V1"
LABEL = "2026 deployment-style pseudo-holdout / approximate deployment replay"
EVAL_START = pd.Timestamp("2026-01-02")
EVAL_END = pd.Timestamp("2026-08-24")
POUNDS = 1_000_000.0
ANN = float(np.sqrt(MR4A_ANN_FACTOR))
TOPK = 5
N_DROP = 1
HOLD_THRESH = 1
RISK_DEGREE = 0.95
W_D2, W_F1C = 0.7, 0.3
CS_Z_MIN_F1C = 50
FEATURE_JSON = PROJECT_ROOT / "data/portfolio_experiments/d2_internal_attribution/frozen_feature_lists.json"
D2_SEED42 = (
    PROJECT_ROOT
    / "data/rd13_v2_downstream_replication/stage_2_delayed_models/model_D2_ALPHA20_RD13_V2_seed42.pkl"
)
D2_PANEL = (
    PROJECT_ROOT
    / "data/rd13_v2_downstream_replication/stage_1_delayed_panel/delayed_common_sample_rd13_v2.parquet"
)
D2_CANONICAL = (
    PROJECT_ROOT / "data/rd13_v2_downstream_replication/stage_2_delayed_models/pred_D2_ALPHA20_RD13_V2.parquet"
)
MASSIVE_PX = PROJECT_ROOT / "data/massive_2026/normalized/daily_ohlcv_split_adjusted.parquet"
MASSIVE_SPY = PROJECT_ROOT / "data/massive_2026/normalized/spy_daily.parquet"
MEMBER = PROJECT_ROOT / "data/massive_2026/universe/sp500_2026_daily_membership.csv"
SNAPSHOT = PROJECT_ROOT / "data/massive_2026/universe/sp500_2025_12_31_snapshot.csv"
INCOME = PROJECT_ROOT / "data/massive_2026/fundamentals/universe_archive/income_quarterly_or_index.csv"
BALANCE = PROJECT_ROOT / "data/massive_2026/fundamentals/universe_archive/balance_quarterly_or_index.csv"
CASHFLOW = PROJECT_ROOT / "data/massive_2026/fundamentals/universe_archive/cashflow_quarterly_or_index.csv"
FILINGS = PROJECT_ROOT / "data/massive_2026/fundamentals/universe_archive/filings_quarterly_or_index.csv"
PRECOMMIT = PROJECT_ROOT / "data/massive_2026/precommit_2026_deployment_replay.json"
QLIB_DIR = PROJECT_ROOT / "staging/qlib_data"
OUT_DIR = OUT / "vt0_2026"
REPORT_MD = REPORT / "vt0_2026_deployment_validation_report.md"
ALIASES = {"BF": "BF.B", "CDAY": "DAY", "FLT": "CPAY", "PEAK": "DOC"}
F1C_FEATURES = [f for f in ALL12 if f != "operating_profitability_z"]
# Frozen before looking at 2026 PnL.
D2_SPEARMAN_NOT_RELIABLE = 0.30
RET_SAC_LOW_WEALTH_KEEP = 0.90
RET_SAC_LOW_YTD_GAP = 0.03
RET_SAC_HIGH_WEALTH_KEEP = 0.80
RET_SAC_HIGH_YTD_GAP = 0.08
MECH_VOL_REL = 0.10
MECH_TARGET_RMSE_REL = 0.10
MECH_MDD = 0.01


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    return obj


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2), encoding="utf-8")


def map_ticker(t: str) -> str:
    t = str(t).strip().upper()
    return ALIASES.get(t, t)


def rolling_ols_r2_resi(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    r2 = np.full(n, np.nan)
    resi = np.full(n, np.nan)
    if rolling_rsquare is not None and rolling_resi is not None:
        xf = np.asarray(x, dtype=float)
        r2 = np.asarray(rolling_rsquare(xf, window), dtype=float)
        resi = np.asarray(rolling_resi(xf, window), dtype=float)
        std = pd.Series(xf).rolling(window, min_periods=1).std()
        r2[np.isclose(std.to_numpy(), 0.0, atol=2e-05)] = np.nan
        return r2, resi
    idx = np.arange(window, dtype=float)
    xbar = idx.mean()
    sxx = float(((idx - xbar) ** 2).sum())
    for i in range(window - 1, n):
        y = x[i - window + 1 : i + 1].astype(float)
        if not np.isfinite(y).all() or sxx == 0:
            continue
        ybar = float(y.mean())
        sxy = float(((idx - xbar) * (y - ybar)).sum())
        syy = float(((y - ybar) ** 2).sum())
        slope = sxy / sxx
        intercept = ybar - slope * xbar
        yhat = intercept + slope * idx
        resid = y - yhat
        resi[i] = resid[-1]
        r2[i] = 1.0 - float((resid ** 2).sum()) / syy if syy > 0 else np.nan
    return r2, resi


def alpha20_one(g: pd.DataFrame) -> pd.DataFrame:
    o = g["open"].to_numpy(dtype=float)
    h = g["high"].to_numpy(dtype=float)
    l = g["low"].to_numpy(dtype=float)
    c = g["close"].to_numpy(dtype=float)
    v = g["volume"].to_numpy(dtype=float)
    s_c = pd.Series(c)
    s_v = pd.Series(v)
    ret = s_c.pct_change()
    logv = np.log(s_v + 1.0)
    logdv = np.log((s_v / s_v.shift(1)).replace(0, np.nan) + 1.0)
    ami_abs = ret.abs() * s_v
    rsq5, resi5 = rolling_ols_r2_resi(c, 5)
    rsq10, resi10 = rolling_ols_r2_resi(c, 10)
    rsq20, _ = rolling_ols_r2_resi(c, 20)
    rsq60, _ = rolling_ols_r2_resi(c, 60)
    out = pd.DataFrame(index=g.index)
    out["alpha20_resi5"] = resi5 / np.where(c == 0, np.nan, c)
    out["alpha20_resi10"] = resi10 / np.where(c == 0, np.nan, c)
    out["alpha20_rsqr5"] = rsq5
    out["alpha20_rsqr10"] = rsq10
    out["alpha20_rsqr20"] = rsq20
    out["alpha20_rsqr60"] = rsq60
    out["alpha20_wvma5"] = ami_abs.rolling(5, min_periods=5).std() / (
        ami_abs.rolling(5, min_periods=5).mean() + 1e-12
    )
    out["alpha20_wvma60"] = ami_abs.rolling(60, min_periods=60).std() / (
        ami_abs.rolling(60, min_periods=60).mean() + 1e-12
    )
    out["alpha20_klen"] = (h - l) / np.where(o == 0, np.nan, o)
    out["alpha20_klow"] = (np.minimum(o, c) - l) / np.where(o == 0, np.nan, o)
    out["alpha20_corr5"] = s_c.rolling(5, min_periods=5).corr(logv)
    out["alpha20_corr10"] = s_c.rolling(10, min_periods=10).corr(logv)
    out["alpha20_corr20"] = s_c.rolling(20, min_periods=20).corr(logv)
    out["alpha20_corr60"] = s_c.rolling(60, min_periods=60).corr(logv)
    out["alpha20_cord5"] = ret.rolling(5, min_periods=5).corr(logdv)
    out["alpha20_cord10"] = ret.rolling(10, min_periods=10).corr(logdv)
    out["alpha20_cord60"] = ret.rolling(60, min_periods=60).corr(logdv)
    out["alpha20_roc60"] = s_c.shift(60) / s_c
    out["alpha20_vstd5"] = s_v.rolling(5, min_periods=5).std() / (s_v + 1e-12)
    out["alpha20_std5"] = s_c.rolling(5, min_periods=5).std() / s_c.replace(0, np.nan)
    return out


def rd13_panel(px: pd.DataFrame) -> pd.DataFrame:
    """RD13_v2 formulas from RD13_provenance_reconstruction.compute_all_v2_factors."""
    df = px.sort_values(["ticker", "date"]).copy()
    parts = []
    tmp = df.copy()
    tmp["log_ret"] = tmp.groupby("ticker", sort=False)["close"].transform(lambda x: np.log(x / x.shift(1)))
    tmp["market_ret"] = tmp.groupby("date")["log_ret"].transform("mean")
    for ticker, g in tmp.groupby("ticker", sort=False):
        g = g.sort_values("date")
        close = g["close"]
        high = g["high"]
        low = g["low"]
        volume = g["volume"].replace(0, np.nan)
        ret_simple = close.pct_change()
        log_ret = np.log(close / close.shift(1))
        market_ret = g["market_ret"]
        out = pd.DataFrame({"date": g["date"].to_numpy(), "ticker": ticker})
        out["rd13_10_day_momentum"] = (close / close.shift(10) - 1.0).to_numpy()
        out["rd13_20_day_realized_volatility"] = ret_simple.rolling(20, min_periods=20).std(ddof=1).to_numpy()
        out["rd13_5_day_volume_deviation"] = (volume / volume.rolling(5, min_periods=5).mean() - 1.0).to_numpy()
        hl = high - low
        out["rd13_10_day_price_range_ratio"] = (hl / hl.rolling(10, min_periods=10).mean()).to_numpy()
        vol20 = log_ret.rolling(20, min_periods=20).std(ddof=1)
        out["rd13_volatility_adjusted_10d_momentum"] = (out["rd13_10_day_momentum"] / vol20.to_numpy())
        out["rd13_5d_short_term_reversal"] = (-(close / close.shift(5) - 1.0)).to_numpy()
        out["rd13_daily_amihud_illiquidity"] = (ret_simple.abs() / volume).to_numpy()
        cov = log_ret.rolling(20, min_periods=20).cov(market_ret)
        var = market_ret.rolling(20, min_periods=20).var()
        beta = cov / var
        out["rd13_rolling_beta_20d"] = beta.to_numpy()
        eps = ret_simple - beta * market_ret
        out["rd13_idiosyncratic_volatility_10d"] = eps.rolling(10, min_periods=10).std(ddof=1).to_numpy()
        mom5 = close / close.shift(5) - 1.0
        out["_mom5"] = mom5.to_numpy()
        win = (ret_simple > 0).astype(float)
        out["rd13_win_rate_5d"] = win.rolling(5, min_periods=5).mean().to_numpy()
        up_vol = volume * (ret_simple > 0).astype(float)
        out["rd13_upday_volume_ratio_5d"] = (
            up_vol.rolling(5, min_periods=5).sum() / volume.rolling(5, min_periods=5).sum()
        ).to_numpy()
        prev = close.shift(1)
        tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        out["rd13_atr_norm_close_14d"] = (atr / close.rolling(14, min_periods=14).mean()).to_numpy()
        parts.append(out)
    out = pd.concat(parts, ignore_index=True)
    out["rd13_cs_momentum_rank_5d"] = out.groupby("date")["_mom5"].rank(method="average", pct=True)
    return out.drop(columns=["_mom5"])


def compute_price_features(px: pd.DataFrame) -> pd.DataFrame:
    a20_parts = []
    for ticker, g in px.groupby("ticker", sort=False):
        g = g.sort_values("date")
        feat = alpha20_one(g)
        feat["date"] = g["date"].to_numpy()
        feat["ticker"] = ticker
        a20_parts.append(feat)
    a20 = pd.concat(a20_parts, ignore_index=True)
    rd = rd13_panel(px)
    feat = a20.merge(rd, on=["date", "ticker"], how="outer")
    return feat


def apply_robust_z(feat: pd.DataFrame, cols: list[str], mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    out = feat.copy()
    x = out[cols].to_numpy(dtype=float)
    x = (x - mean) / std
    x = np.clip(x, -3.0, 3.0)
    x = np.nan_to_num(x, nan=0.0)
    out[cols] = x
    return out


def fit_robust_z_qlib(alpha20_cols: list[str], cache: Path) -> tuple[np.ndarray, np.ndarray]:
    if cache.exists():
        obj = json.loads(cache.read_text(encoding="utf-8"))
        return np.asarray(obj["mean"], dtype=float), np.asarray(obj["std"], dtype=float)
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(QLIB_DIR), region="us")
    raw = D.features(
        D.instruments("sp500"),
        ["$open", "$high", "$low", "$close", "$volume"],
        start_time="2008-01-01",
        end_time="2017-12-31",
    )
    raw = raw.reset_index()
    raw["date"] = pd.to_datetime(raw.get("datetime", raw.get("date"))).dt.normalize()
    if "ticker" not in raw.columns:
        raw["ticker"] = raw.get("instrument", raw.iloc[:, 1]).astype(str)
    raw = raw.rename(columns={"$open": "open", "$high": "high", "$low": "low", "$close": "close", "$volume": "volume"})
    raw = raw.rename(columns={c: str(c).replace("$", "") for c in raw.columns})
    need = ["date", "ticker", "open", "high", "low", "close", "volume"]
    raw = raw[need]
    log.info("Qlib 2008-2017 OHLCV rows=%d names=%d", len(raw), raw["ticker"].nunique())
    a20_parts = []
    for ticker, g in raw.groupby("ticker", sort=False):
        g = g.sort_values("date")
        feat = alpha20_one(g)
        feat["date"] = g["date"].to_numpy()
        feat["ticker"] = ticker
        a20_parts.append(feat)
    feat = pd.concat(a20_parts, ignore_index=True)
    x = feat[alpha20_cols].to_numpy(dtype=float)
    mean = np.nanmedian(x, axis=0)
    std = np.nanmedian(np.abs(x - mean), axis=0)
    std = np.where(std < 1e-12, 1e-12, std) * 1.4826
    dump_json(cache, {"mean": mean.tolist(), "std": std.tolist(), "cols": alpha20_cols, "n": int(np.isfinite(x).any(axis=1).sum())})
    return mean, std


def load_prices() -> pd.DataFrame:
    px = pd.read_parquet(MASSIVE_PX)
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px["ticker"] = px["ticker"].astype(str).str.upper().map(map_ticker)
    px = px[px["ticker"] != "SPY"].copy()
    px = px.sort_values(["ticker", "date"])
    return px


def load_membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    m = pd.read_csv(MEMBER)
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    m["ticker"] = m["ticker"].astype(str).str.upper().map(map_ticker)
    m["in_index"] = m["in_index"].astype(bool)
    last = m[m["date"] == pd.Timestamp("2026-08-18")]
    extra_dates = [d for d in dates if d > pd.Timestamp("2026-08-18") and d <= EVAL_END]
    rows = [m]
    for d in extra_dates:
        add = last.copy()
        add["date"] = d
        rows.append(add)
    out = pd.concat(rows, ignore_index=True)
    out = out[out["in_index"]].drop_duplicates(["date", "ticker"])
    return out[["date", "ticker"]]


def complete_d2_seeds(feature_names: list[str]) -> dict[int, lgb.Booster]:
    models: dict[int, lgb.Booster] = {}
    with D2_SEED42.open("rb") as f:
        models[42] = pickle.load(f)
    feats = list(models[42].feature_name())
    stage = D2_SEED42.parent
    need_train = False
    for seed in DELAYED_SEEDS:
        if seed == 42:
            continue
        path = stage / f"model_D2_ALPHA20_RD13_V2_seed{seed}.pkl"
        if path.exists():
            with path.open("rb") as f:
                models[seed] = pickle.load(f)
            log.info("loaded existing D2 seed %s", seed)
        else:
            need_train = True
    if not need_train:
        return models
    panel_cols = ["datetime", "instrument", "split", "label"] + feats
    panel = pd.read_parquet(D2_PANEL, columns=panel_cols)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    missing = [c for c in feats if c not in panel.columns]
    if missing:
        raise RuntimeError(f"delayed panel missing D2 features: {missing[:8]}")
    train = process_d2_label(panel[panel["split"] == "train"])
    valid = process_d2_label(panel[panel["split"] == "valid"])
    for seed in DELAYED_SEEDS:
        if seed == 42 or seed in models:
            continue
        path = stage / f"model_D2_ALPHA20_RD13_V2_seed{seed}.pkl"
        log.info("retraining D2 seed %s on frozen 2008-2017/2018-2019 only", seed)
        model = train_d2_lgb(train, valid, feats, seed)
        with path.open("wb") as f:
            pickle.dump(model, f)
        models[seed] = model
    return models


def predict_d2(feat: pd.DataFrame, models: dict[int, lgb.Booster]) -> pd.DataFrame:
    feats = list(models[42].feature_name())
    x = feat[feats]
    scores = []
    for seed, model in models.items():
        pred = model.predict(x, num_iteration=model.best_iteration)
        scores.append(pred)
    feat = feat.copy()
    feat["d2"] = np.mean(np.vstack(scores), axis=0)
    return feat[["date", "ticker", "d2"]]


def parse_stmt_ticker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "query_ticker" in out.columns:
        out["ticker"] = out["query_ticker"].astype(str).str.upper().map(map_ticker)
    else:
        out["ticker"] = out["ticker"].astype(str).str.upper().map(map_ticker)
    out["period_end"] = pd.to_datetime(out["period_end"]).dt.normalize()
    return out


def build_f1c_statement_panel() -> pd.DataFrame:
    inc = parse_stmt_ticker(pd.read_csv(INCOME))
    bal = parse_stmt_ticker(pd.read_csv(BALANCE))
    cf = parse_stmt_ticker(pd.read_csv(CASHFLOW))
    fil = pd.read_csv(FILINGS)
    fil["ticker"] = fil["query_ticker"].fillna(fil.get("ticker")).astype(str).str.upper().map(map_ticker)
    fil["filing_date"] = pd.to_datetime(fil["filing_date"]).dt.normalize()
    fil = fil[fil["form_type"].astype(str).str.upper().isin(["10-Q", "10-K"])]
    edgar = (
        fil.sort_values("filing_date")
        .groupby(["ticker", "filing_date"], as_index=False)
        .first()[["ticker", "filing_date"]]
    )
    inc = inc.sort_values(["ticker", "period_end", "filing_date"])
    inc = inc.drop_duplicates(["ticker", "period_end"], keep="first")
    bal = bal.sort_values(["ticker", "period_end"]).drop_duplicates(["ticker", "period_end"], keep="first")
    cf = cf.sort_values(["ticker", "period_end"]).drop_duplicates(["ticker", "period_end"], keep="first")
    stmt = inc.merge(
        bal[["ticker", "period_end", "total_assets", "total_current_assets", "total_current_liabilities",
             "debt_current", "long_term_debt_and_capital_lease_obligations", "total_equity_attributable_to_parent",
             "total_equity", "preferred_stock"]],
        on=["ticker", "period_end"],
        how="left",
    ).merge(
        cf[["ticker", "period_end", "net_cash_from_operating_activities"]],
        on=["ticker", "period_end"],
        how="left",
    )
    # PIT clock: first EDGAR 10-Q/K on or after period_end, else Massive filing_date.
    stmt["massive_filing"] = pd.to_datetime(stmt["filing_date"], errors="coerce").dt.normalize()
    edgar_s = edgar.rename(columns={"filing_date": "edgar_date"})
    stmt = stmt.sort_values(["ticker", "period_end"])
    edgar_s = edgar_s.sort_values(["ticker", "edgar_date"])
    matched = []
    for ticker, g in stmt.groupby("ticker", sort=False):
        e = edgar_s[edgar_s["ticker"] == ticker]
        if e.empty:
            g = g.copy()
            g["avail_date"] = g["massive_filing"]
            matched.append(g)
            continue
        e_dates = e["edgar_date"].to_numpy()
        avails = []
        for pe in g["period_end"].to_numpy():
            later = e_dates[e_dates >= pe]
            later = later[later <= pe + np.timedelta64(150, "D")]
            avails.append(later[0] if len(later) else pd.NaT)
        g = g.copy()
        g["avail_date"] = pd.to_datetime(avails)
        matched.append(g)
    stmt = pd.concat(matched, ignore_index=True)
    stmt["avail_date"] = stmt["avail_date"].fillna(stmt["massive_filing"])
    book = stmt["total_equity_attributable_to_parent"].fillna(stmt["total_equity"])
    debt = stmt["debt_current"].fillna(0) + stmt["long_term_debt_and_capital_lease_obligations"].fillna(0)
    stmt["ni"] = pd.to_numeric(stmt["net_income_loss_attributable_common_shareholders"], errors="coerce")
    stmt["sale"] = pd.to_numeric(stmt["revenue"], errors="coerce")
    stmt["gp"] = pd.to_numeric(stmt["gross_profit"], errors="coerce")
    stmt["at"] = pd.to_numeric(stmt["total_assets"], errors="coerce")
    stmt["book"] = pd.to_numeric(book, errors="coerce")
    stmt["act"] = pd.to_numeric(stmt["total_current_assets"], errors="coerce")
    stmt["lct"] = pd.to_numeric(stmt["total_current_liabilities"], errors="coerce")
    stmt["debt"] = pd.to_numeric(debt, errors="coerce")
    stmt["oancf"] = pd.to_numeric(stmt["net_cash_from_operating_activities"], errors="coerce")
    stmt["shares"] = pd.to_numeric(stmt["diluted_shares_outstanding"], errors="coerce").fillna(
        pd.to_numeric(stmt["basic_shares_outstanding"], errors="coerce")
    )
    stmt = stmt.sort_values(["ticker", "period_end"])
    for col in ("ni", "sale", "gp", "oancf"):
        stmt[f"ttm_{col}"] = stmt.groupby("ticker")[col].transform(lambda s: s.rolling(4, min_periods=4).sum())
    stmt["at_lag4"] = stmt.groupby("ticker")["at"].shift(4)
    stmt["sale_lag4_ttm"] = stmt.groupby("ticker")["ttm_sale"].shift(4)
    stmt["at_avg"] = (stmt["at"] + stmt.groupby("ticker")["at"].shift(1)) / 2.0
    stmt["book_avg"] = (stmt["book"] + stmt.groupby("ticker")["book"].shift(1)) / 2.0
    return stmt


def _safe_div(num, den) -> float:
    if den is None or not np.isfinite(den) or den == 0 or not np.isfinite(num):
        return np.nan
    return float(num) / float(den)


def f1c_raw_from_stmt(stmt: pd.DataFrame, px: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    stmt = stmt.dropna(subset=["avail_date", "ticker"]).copy()
    stmt["avail_date"] = pd.to_datetime(stmt["avail_date"]).dt.normalize()
    date_index = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    px_sub = px[px["date"].isin(date_index)][["date", "ticker", "close"]].copy()
    pieces = []
    for ticker, g in stmt.groupby("ticker", sort=False):
        g = g.sort_values("avail_date")
        d = pd.DataFrame({"date": date_index})
        m = pd.merge_asof(d, g, left_on="date", right_on="avail_date")
        m["ticker"] = ticker
        pieces.append(m)
    asof = pd.concat(pieces, ignore_index=True)
    asof = asof.merge(px_sub, on=["date", "ticker"], how="inner")
    asof = asof[asof["close"] > 0].copy()
    mcap = asof["shares"] * asof["close"]
    at_avg = asof["at_avg"].where(asof["at_avg"].notna() & (asof["at_avg"] != 0), asof["at"])
    book_avg = asof["book_avg"].where(asof["book_avg"].notna() & (asof["book_avg"] != 0), asof["book"])
    raw = pd.DataFrame({
        "date": asof["date"].to_numpy(),
        "ticker": asof["ticker"].to_numpy(),
        "roe": asof["ttm_ni"] / book_avg,
        "roa": asof["ttm_ni"] / at_avg,
        "gross_profitability": asof["ttm_gp"] / at_avg,
        "sales_growth_yoy": asof["ttm_sale"] / asof["sale_lag4_ttm"] - 1.0,
        "asset_growth_yoy": asof["at"] / asof["at_lag4"] - 1.0,
        "accruals": (asof["ttm_ni"] - asof["ttm_oancf"]) / at_avg,
        "leverage": asof["debt"] / asof["at"],
        "current_ratio": asof["act"] / asof["lct"],
        "book_to_market": asof["book"] / mcap,
        "earnings_yield": asof["ttm_ni"] / mcap,
        "sales_to_price": asof["ttm_sale"] / mcap,
    })
    raw = raw.replace([np.inf, -np.inf], np.nan)
    rename = {
        "roe": "roe_z",
        "roa": "roa_z",
        "gross_profitability": "gross_profitability_z",
        "sales_growth_yoy": "sales_growth_yoy_z",
        "asset_growth_yoy": "asset_growth_yoy_z",
        "accruals": "accruals_z",
        "leverage": "leverage_z",
        "current_ratio": "current_ratio_z",
        "book_to_market": "book_to_market_z",
        "earnings_yield": "earnings_yield_z",
        "sales_to_price": "sales_to_price_z",
    }
    raw = raw.rename(columns=rename)
    for col in F1C_FEATURES:
        if col not in raw.columns:
            raw[col] = np.nan
        raw[col] = raw.groupby("date")[col].transform(
            lambda s: zscore_series(winsorize_series(s.astype(float))) if s.notna().sum() >= CS_Z_MIN_F1C else np.nan
        )
    return raw


def train_f1c_seeds() -> list[lgb.Booster]:
    cache_dir = OUT_DIR / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    models = []
    raw = pd.read_parquet(MAIN_DATASET)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    df = raw[raw["split"] != "holdout"].copy()
    train = process_training_label(df[df["split"] == "train"])
    valid = process_training_label(df[df["split"] == "valid"])
    for seed in F1C_SEEDS:
        path = cache_dir / f"f1c_approx_replay_seed{seed}.pkl"
        if path.exists():
            with path.open("rb") as f:
                models.append(pickle.load(f))
            continue
        log.info("retraining F1C seed %s on frozen 2008-2017/2018-2019 only", seed)
        model = train_f1c(train, valid, F1C_FEATURES, seed)
        with path.open("wb") as f:
            pickle.dump(model, f)
        models.append(model)
    return models


def predict_f1c(raw_z: pd.DataFrame, models: list[lgb.Booster]) -> pd.DataFrame:
    x = raw_z[F1C_FEATURES]
    scores = [m.predict(x, num_iteration=m.best_iteration) for m in models]
    out = raw_z[["date", "ticker"]].copy()
    out["f1c"] = np.mean(np.vstack(scores), axis=0)
    n_ok = raw_z[F1C_FEATURES].notna().all(axis=1)
    out.loc[~n_ok, "f1c"] = np.nan
    return out.dropna(subset=["f1c"])


def cs_winsor_z(s: pd.Series) -> pd.Series:
    return zscore_series(winsorize_series(s.astype(float)))


def build_w1(d2: pd.DataFrame, f1c: pd.DataFrame, membership: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    have_px = px[["date", "ticker"]].drop_duplicates()
    panel = membership.merge(have_px, on=["date", "ticker"], how="inner")
    panel = panel.merge(d2, on=["date", "ticker"], how="inner")
    panel = panel.merge(f1c, on=["date", "ticker"], how="inner")
    panel["d2_z"] = panel.groupby("date")["d2"].transform(cs_winsor_z)
    panel["f1c_z"] = panel.groupby("date")["f1c"].transform(cs_winsor_z)
    panel["score"] = W_D2 * panel["d2_z"] + W_F1C * panel["f1c_z"]
    return panel.dropna(subset=["score"])


def topk_dropout_weights(scores: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    by_date = {d: g.set_index("ticker")["score"] for d, g in scores.groupby("date")}
    held: list[str] = []
    age: dict[str, int] = {}
    rows = []
    for dt in dates:
        pred = by_date.get(dt)
        if pred is None or pred.dropna().empty:
            for tkr in held:
                rows.append({"date": dt, "ticker": tkr, "weight": RISK_DEGREE / max(len(held), 1)})
            continue
        pred = pred.dropna().sort_values(ascending=False)
        last = [t for t in pred.reindex(held).sort_values(ascending=False).index.tolist() if t in pred.index]
        n_buy_cap = N_DROP + TOPK - len(last)
        today = [t for t in pred.index if t not in last][: max(n_buy_cap, 0)]
        comb = pred.reindex(list(dict.fromkeys(last + today))).dropna().sort_values(ascending=False)
        bottom = set(comb.index[-N_DROP:]) if len(comb) else set()
        sell = [t for t in last if t in bottom and age.get(t, 0) >= HOLD_THRESH]
        buy = today[: len(sell) + TOPK - len(last)]
        new_held = [t for t in last if t not in sell] + list(buy)
        new_held = new_held[:TOPK]
        age = {t: (age.get(t, 0) + 1 if t in last and t not in sell else 1) for t in new_held}
        held = new_held
        w = RISK_DEGREE / max(len(held), 1)
        for tkr in held:
            rows.append({"date": dt, "ticker": tkr, "weight": w})
    return pd.DataFrame(rows)


def portfolio_from_weights(weights: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    ret = px[["date", "ticker", "close"]].copy()
    ret["ret"] = ret.groupby("ticker")["close"].pct_change()
    w = weights.rename(columns={"date": "decision_date"})
    cal = sorted(px["date"].drop_duplicates())
    loc = {d: i for i, d in enumerate(cal)}
    rows = []
    prev_w: dict[str, float] = {}
    for dec in sorted(w["decision_date"].unique()):
        i = loc.get(pd.Timestamp(dec))
        if i is None or i + 1 >= len(cal):
            continue
        ret_date = cal[i + 1]
        if ret_date < EVAL_START or ret_date > EVAL_END:
            continue
        wd = w[w["decision_date"] == dec]
        cur = dict(zip(wd["ticker"], wd["weight"]))
        day_ret = ret[ret["date"] == ret_date].set_index("ticker")["ret"]
        gross = 0.0
        eq = 0.0
        for tkr, wt in cur.items():
            r = day_ret.get(tkr, np.nan)
            if not np.isfinite(r):
                continue
            gross += wt * float(r)
            eq += wt
        names = set(cur) | set(prev_w)
        turnover = float(sum(abs(cur.get(t, 0.0) - prev_w.get(t, 0.0)) for t in names))
        cost = MR4A_COST_BP * turnover
        rows.append({
            "trade_date": ret_date,
            "decision_date": dec,
            "r_gross": gross,
            "r_net": gross - cost,
            "turnover": turnover,
            "cost": cost,
            "equity_weight": eq,
            "n_hold": int(len(cur)),
        })
        prev_w = cur
    return pd.DataFrame(rows)


def apply_vt0(daily: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    p = daily.sort_values("trade_date").reset_index(drop=True)
    p["P5_VOL20_ANN"] = p["r_net"].rolling(20, min_periods=20).std(ddof=1) * ANN
    raw = p["P5_VOL20_ANN"].to_numpy(dtype=float)
    forecast = np.maximum(raw, MR4A_FORECAST_FLOOR)
    scale_dec = np.ones(len(p), dtype=float)
    ok = np.isfinite(forecast) & (forecast > 0) & np.isfinite(raw)
    scale_dec[ok] = np.minimum(MR4A_EQUITY_CAP, MR4A_TARGET_VOL / forecast[ok])
    p["forecast_vt0"] = forecast
    p["scale_decision_vt0"] = scale_dec
    p["scale_vt0"] = p["scale_decision_vt0"].shift(1)
    p.loc[0, "scale_vt0"] = 1.0
    p["scale_vt0"] = p["scale_vt0"].fillna(1.0)
    wide = (
        weights.rename(columns={"date": "decision_date"})
        .merge(p[["decision_date", "trade_date"]], on="decision_date", how="inner")
        .pivot_table(index="trade_date", columns="ticker", values="weight", aggfunc="sum")
        .fillna(0.0)
        .reindex(p["trade_date"])
        .fillna(0.0)
    )
    scaled = wide.mul(p.set_index("trade_date")["scale_vt0"], axis=0)
    dabs = scaled.diff().abs().sum(axis=1)
    if len(dabs):
        dabs.iloc[0] = float(scaled.iloc[0].abs().sum())
    p["turnover_p1"] = dabs.to_numpy()
    p["cost_p1"] = MR4A_COST_BP * p["turnover_p1"]
    p["r_p0"] = p["r_net"]
    p["r_p1"] = p["scale_vt0"] * p["r_gross"] - p["cost_p1"]
    p["exposure_p0"] = p["equity_weight"]
    p["exposure_p1"] = p["scale_vt0"] * p["equity_weight"]
    p["rv20_p0"] = p["r_p0"].rolling(20, min_periods=20).std(ddof=1) * ANN
    p["rv20_p1"] = p["r_p1"].rolling(20, min_periods=20).std(ddof=1) * ANN
    return p


def classify(p0: dict[str, float], p1: dict[str, float], acc0: dict[str, float], acc1: dict[str, float]) -> tuple[str, str]:
    vol_cut = np.isfinite(p0["vol"]) and np.isfinite(p1["vol"]) and p1["vol"] <= (1.0 - MECH_VOL_REL) * p0["vol"]
    rmse_cut = (
        np.isfinite(acc0.get("rmse", np.nan))
        and np.isfinite(acc1.get("rmse", np.nan))
        and acc0["rmse"] > 0
        and acc1["rmse"] <= (1.0 - MECH_TARGET_RMSE_REL) * acc0["rmse"]
    )
    closer = np.isfinite(acc0.get("rmse", np.nan)) and np.isfinite(acc1.get("rmse", np.nan)) and acc1["rmse"] < acc0["rmse"]
    mdd_ok = np.isfinite(p0["mdd"]) and np.isfinite(p1["mdd"]) and (p1["mdd"] - p0["mdd"]) >= MECH_MDD
    if (vol_cut or rmse_cut) and (mdd_ok or closer):
        mech = "2026_VT0_MECHANISM_SUPPORTED"
    elif vol_cut or closer or mdd_ok:
        mech = "2026_VT0_MECHANISM_MIXED"
    else:
        mech = "2026_VT0_MECHANISM_NOT_SUPPORTED"
    w0 = p0["ending_1m"]
    w1 = p1["ending_1m"]
    keep = w1 / w0 if w0 and w0 > 0 else np.nan
    ytd_gap = p0["compound"] - p1["compound"]
    if np.isfinite(keep) and keep >= RET_SAC_LOW_WEALTH_KEEP and ytd_gap < RET_SAC_LOW_YTD_GAP:
        sac = "RETURN_SACRIFICE_LOW"
    elif (np.isfinite(keep) and keep < RET_SAC_HIGH_WEALTH_KEEP) or ytd_gap >= RET_SAC_HIGH_YTD_GAP:
        sac = "RETURN_SACRIFICE_HIGH"
    else:
        sac = "RETURN_SACRIFICE_MODERATE"
    return mech, sac


def static_integrity() -> dict[str, Any]:
    reasons = []
    quality = "APPROXIMATE"
    d2_ok = D2_SEED42.exists() and FEATURE_JSON.exists() and MASSIVE_PX.exists()
    f1c_exact = False
    pit_spx = False
    if not d2_ok:
        quality = "NOT_RELIABLE"
        reasons.append("Missing D2 seed42 pkl, feature list, or Massive prices.")
    reasons.append("D2 definitions: frozen RD13_v2 + Alpha20 names unchanged; RobustZScoreNorm fit 2008-2017; no 2026 retrain.")
    reasons.append("D2 preprocessing: Massive-native OHLCV/volume (SIP vs CRSP volume = APPROXIMATE). Not Qlib first-day dollar levels.")
    reasons.append("D2 models: seed42 loaded; seeds 2026/3407 completed by deterministic retrain on frozen 2008-2017/2018-2019 delayed panel only.")
    reasons.append("F1C: frozen 11 features used as model inputs, but values are Massive/EDGAR approximations. Compustat unrestated PIT is unavailable. Massive filing_date is restating; EDGAR 10-Q/K used as availability clock. Accruals/B/M/mcap mappings are weak. Historical SEC sample Spearman vs frozen F1C ≈ 0.50.")
    reasons.append("Universe: EXACT_2026_PIT_SP500=NO. Using RECONSTRUCTED_2026_SP500_WRDS_SNAPSHOT_PLUS_PUBLIC_EVENTS; 2026-08-19..24 carry-forward; aliases BF/CDAY/FLT/PEAK. 34 later-mapped names not in the 482 price book.")
    reasons.append("Portfolio: Top5 / n_drop=1 / hold_thresh=1 / W1 0.7 D2 + 0.3 F1C / risk_degree=0.95 / 1bp. Pandas TopkDropout, not Qlib 2026 fill.")
    if quality != "NOT_RELIABLE":
        # F1C + reconstructed universe block EXACT and HIGH_QUALITY_APPROXIMATION.
        quality = "APPROXIMATE"
    return {
        "2026_P5_REPLAY_QUALITY": quality,
        "label_if_not_exact": LABEL,
        "D2": {
            "definitions_unchanged": True,
            "seed42_available": D2_SEED42.exists(),
            "ensemble": "seed42 loaded; 2026/3407 retrained on frozen delayed panel only",
            "preprocessing": "RobustZScoreNorm 2008-2017 + Fillna(0) on Alpha20; RD13 raw",
            "warmup": "Massive 2025-01-02 onward; max window 60d",
            "price_volume": "Massive-native; volume parity APPROXIMATE",
        },
        "F1C": {
            "frozen_11_unchanged_as_model_inputs": True,
            "exact_pit": f1c_exact,
            "filing_clock": "EDGAR 10-Q/K first filing on/after period_end; fallback Massive filing_date",
            "restatement_safety": "NO — statement values may still be restated even with EDGAR clock",
            "uses_latest_ratios_snapshot": False,
        },
        "universe": {
            "exact_pit_sp500": pit_spx,
            "id": "RECONSTRUCTED_2026_SP500_WRDS_SNAPSHOT_PLUS_PUBLIC_EVENTS",
            "mark": "reconstructed/approximate",
        },
        "portfolio": {
            "topk": TOPK,
            "n_drop": N_DROP,
            "w1": "0.7 D2_z + 0.3 F1C_z",
            "strategy_params_changed": False,
        },
        "reasons": reasons,
        "written_before_vt0_pnl": True,
        "generated": _utc(),
    }


def delayed_rerisk_diag(p: pd.DataFrame) -> dict[str, Any]:
    q = p.dropna(subset=["P5_VOL20_ANN"]).copy()
    if q.empty:
        return {}
    i_max = int(q["P5_VOL20_ANN"].idxmax())
    i_min_exp = int(p["exposure_p1"].idxmin())
    p = p.copy()
    p["exp_drop"] = p["exposure_p1"].diff()
    i_drop = int(p["exp_drop"].idxmin()) if p["exp_drop"].notna().any() else i_min_exp
    min_date = p.loc[i_min_exp, "trade_date"]
    after = p[p["trade_date"] > min_date].head(20)
    p0_20 = float((1.0 + after["r_p0"]).prod() - 1.0) if len(after) else np.nan
    p1_20 = float((1.0 + after["r_p1"]).prod() - 1.0) if len(after) else np.nan
    # recovery of exposure over next 20 sessions after min
    rec = p[p["trade_date"] >= min_date].head(21)
    delayed = bool(
        np.isfinite(p0_20)
        and p0_20 > 0.02
        and rec["exposure_p1"].iloc[-1] < 0.80
        and p.loc[i_min_exp, "exposure_p1"] < 0.80
    ) if len(rec) else False
    return {
        "max_risk_up_date": str(pd.Timestamp(p.loc[i_max, "trade_date"]).date()) if i_max in p.index else None,
        "max_p0_vol20": float(p.loc[i_max, "P5_VOL20_ANN"]) if i_max in p.index else None,
        "min_exposure_date": str(pd.Timestamp(min_date).date()),
        "min_exposure": float(p.loc[i_min_exp, "exposure_p1"]),
        "max_exposure_reduction_date": str(pd.Timestamp(p.loc[i_drop, "trade_date"]).date()) if i_drop in p.index else None,
        "max_exposure_reduction": float(p.loc[i_drop, "exp_drop"]) if i_drop in p.index else None,
        "next20_p0_after_min_exp": p0_20,
        "next20_p1_after_min_exp": p1_20,
        "n_next20": int(len(after)),
        "delayed_rerisking_observed": delayed,
        "mean_exp_20d_after_min": float(after["exposure_p1"].mean()) if len(after) else None,
    }


def write_report(payload: dict[str, Any]) -> None:
    q = payload["integrity"]["2026_P5_REPLAY_QUALITY"]
    p0 = payload["metrics_p0"]
    p1 = payload["metrics_p1"]
    acc0 = payload["target_p0"]
    acc1 = payload["target_p1"]
    exp = payload["exposure_p1"]
    diag = payload["exposure_path"]
    a = payload["answers"]
    lines = [
        "# 2026 VT0 deployment-style validation",
        "",
        f"Generated: `{payload['generated']}`",
        "Human review: `2026-08-28`",
        "",
        f"**Status: `{STATUS}`**",
        f"**Research ID: `{RESEARCH_ID}`**",
        f"**Experiment ID: `{EXPERIMENT_ID}`**",
        "",
        "这是已有 frozen VT0 的额外验证。不是新策略。未改 D2 / W1 / P5。未改 TARGET_VOL。",
        "未加 floor / smoothing。未跑 VT1。未用 2026 结果改 target。未重新训练 risk model。",
        "",
        f"**证据标签: `{LABEL}`**",
        "",
        "禁止称：untouched holdout / true OOS holdout / live performance。",
        "",
        f"**`2026_P5_REPLAY_QUALITY` = `{q}`**",
        f"**`{payload['mechanism']}`**",
        f"**`{payload['return_sacrifice']}`**",
        "",
        "本节点 **STOP**。不调 target。不加 floor。不加 smoothing。不启动新策略。",
        "",
        "---",
        "",
        "## Final answers",
        "",
        f"**A.** {a['A']}",
        "",
        f"**B.** {a['B']}",
        "",
        f"**C.** {a['C']}",
        "",
        f"**D.** {a['D']}",
        "",
        f"**E.** {a['E']}",
        "",
        f"**F.** {a['F']}",
        "",
        f"**G.** {a['G']}",
        "",
        f"**H.** {a['H']}",
        "",
        f"**I.** {a['I']}",
        "",
        f"**J.** {a['J']}",
        "",
        "---",
        "",
        "## 0. Integrity gate (before VT0 PnL)",
        "",
        f"Quality written first: `{q}`. Reasons:",
        "",
    ]
    for r in payload["integrity"]["reasons"]:
        lines.append(f"- {r}")
    lines += [
        "",
        f"- D2 2025 overlap Rank Spearman vs canonical ensemble: `{payload.get('d2_2025_spearman')}`",
        f"- D2 2025 overlap n (name-days): `{payload.get('d2_2025_n')}`",
        "",
        "## 1. Window",
        "",
        f"- first valid 2026 P5 return date: `{payload['first_date']}`",
        f"- last date: `{payload['last_date']}` (not extrapolated)",
        f"- n sessions: `{payload['n']}`",
        "",
        "## 2. Frozen VT0",
        "",
        f"- TARGET_VOL = `{MR4A_TARGET_VOL}`",
        "- risk_forecast_t = annualized P5 realized volatility20",
        "- equity_scale_t = min(1.0, TARGET_VOL / max(P5_VOL20_ANN_t, 1e-6))",
        "- t close decision, t+1 effective; no leverage; no floor; no smoothing; cash return = 0",
        "",
        "## 3. P0 vs P1 metrics",
        "",
        "| metric | P0 2026 P5 baseline | P1 2026 P5 + frozen VT0 |",
        "| --- | ---: | ---: |",
        f"| cumulative return | {p0['compound']:.4f} | {p1['compound']:.4f} |",
        f"| annualized return | {p0['arr']:.4f} | {p1['arr']:.4f} |",
        f"| YTD / window compound | {p0['compound']:.4f} | {p1['compound']:.4f} |",
        f"| annualized volatility | {p0['vol']:.4f} | {p1['vol']:.4f} |",
        f"| Sharpe | {p0['sharpe']:.4f} | {p1['sharpe']:.4f} |",
        f"| MDD | {p0['mdd']:.4f} | {p1['mdd']:.4f} |",
        f"| worst rolling20 drawdown | {p0['worst_roll20_mdd']:.4f} | {p1['worst_roll20_mdd']:.4f} |",
        f"| downside semivolatility | {p0['downside_semi']:.4f} | {p1['downside_semi']:.4f} |",
        f"| CVaR5 | {p0['cvar5']:.4f} | {p1['cvar5']:.4f} |",
        f"| ending wealth from £1m | {p0['ending_1m']:.0f} | {p1['ending_1m']:.0f} |",
        "",
        "## 4. VT0 exposure",
        "",
        f"- mean {exp['mean']:.4f}; median {exp['median']:.4f}; min {exp['min']:.4f}",
        f"- p05 {exp['p05']:.4f}; p25 {exp['p25']:.4f}; p75 {exp['p75']:.4f}; p95 {exp['p95']:.4f}",
        f"- % days exposure <80%: {100*exp['pct_lt_80']:.1f}%",
        f"- % days exposure <60%: {100*exp['pct_lt_60']:.1f}%",
        "",
        "## 5. Risk-target validation vs 18.63%",
        "",
        "| | P0 | VT0 |",
        "| --- | ---: | ---: |",
        f"| MAE | {acc0['mad']:.4f} | {acc1['mad']:.4f} |",
        f"| RMSE | {acc0['rmse']:.4f} | {acc1['rmse']:.4f} |",
        f"| % days within ±10% target | {100*acc0['pct_within_10']:.1f}% | {100*acc1['pct_within_10']:.1f}% |",
        f"| % days >1.25x target | {100*acc0['pct_above_125']:.1f}% | {100*acc1['pct_above_125']:.1f}% |",
        f"| % days >1.5x target | {100*acc0['pct_above_150']:.1f}% | {100*acc1['pct_above_150']:.1f}% |",
        "",
        f"Does frozen VT0 continue to stabilize realized P5 risk in 2026? **{payload['stabilize_answer']}**",
        "",
        "## 6. Return sacrifice (£1m at first valid 2026 date)",
        "",
        f"- P0 ending wealth: £{p0['ending_1m']:,.0f}",
        f"- P1 ending wealth: £{p1['ending_1m']:,.0f}",
        f"- wealth difference (P1−P0): £{p1['ending_1m']-p0['ending_1m']:,.0f}",
        f"- return difference (compound P1−P0): {p1['compound']-p0['compound']:.4f}",
        f"- risk reduction (vol P0−P1): {p0['vol']-p1['vol']:.4f}",
        f"- MDD improvement (P1_mdd − P0_mdd): {p1['mdd']-p0['mdd']:.4f}",
        "",
        "No rule was optimized from these numbers.",
        "",
        "## 7. Exposure path (diagnostic only)",
        "",
        f"- max risk-up (P0 vol20): `{diag.get('max_risk_up_date')}` vol={diag.get('max_p0_vol20')}",
        f"- min VT0 exposure: `{diag.get('min_exposure_date')}` exp={diag.get('min_exposure')}",
        f"- max exposure reduction: `{diag.get('max_exposure_reduction_date')}` Δ={diag.get('max_exposure_reduction')}",
        f"- subsequent 20d P0 / P1: {diag.get('next20_p0_after_min_exp')} / {diag.get('next20_p1_after_min_exp')}",
        f"- delayed re-risking observed: `{diag.get('delayed_rerisking_observed')}`",
        "",
        "## 8. Classification",
        "",
        f"- `{payload['mechanism']}`",
        f"- `{payload['return_sacrifice']}`",
        "",
        "STOP.",
        "",
    ]
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    integrity = static_integrity()
    dump_json(OUT_DIR / "2026_p5_replay_quality.json", integrity)
    log.info("WROTE integrity gate quality=%s BEFORE any 2026 VT0 PnL", integrity["2026_P5_REPLAY_QUALITY"])
    if integrity["2026_P5_REPLAY_QUALITY"] == "NOT_RELIABLE":
        write_report({
            "generated": _utc(),
            "integrity": integrity,
            "mechanism": "NOT_RUN",
            "return_sacrifice": "NOT_RUN",
            "metrics_p0": {k: np.nan for k in ("compound", "arr", "vol", "sharpe", "mdd", "worst_roll20_mdd", "downside_semi", "cvar5", "ending_1m")},
            "metrics_p1": {k: np.nan for k in ("compound", "arr", "vol", "sharpe", "mdd", "worst_roll20_mdd", "downside_semi", "cvar5", "ending_1m")},
            "target_p0": {k: np.nan for k in ("mad", "rmse", "pct_within_10", "pct_above_125", "pct_above_150")},
            "target_p1": {k: np.nan for k in ("mad", "rmse", "pct_within_10", "pct_above_125", "pct_above_150")},
            "exposure_p1": {k: np.nan for k in ("mean", "median", "min", "p05", "p25", "p75", "p95", "pct_lt_80", "pct_lt_60")},
            "exposure_path": {},
            "answers": {k: "STOP: 2026 P5 replay NOT_RELIABLE; VT0 not run." for k in "ABCDEFGHIJ"},
            "first_date": None,
            "last_date": None,
            "n": 0,
            "stabilize_answer": "NOT_RUN",
            "d2_2025_spearman": None,
            "d2_2025_n": None,
        })
        return

    feat_lists = json.loads(FEATURE_JSON.read_text(encoding="utf-8"))
    alpha20_cols = feat_lists["alpha20"]
    all_cols = feat_lists["all"]
    px = load_prices()
    log.info("prices %s .. %s n=%d tickers=%d", px["date"].min().date(), px["date"].max().date(), len(px), px["ticker"].nunique())
    log.info("computing Massive-native Alpha20 + RD13")
    feat_cache = OUT_DIR / "massive_d2_raw_features.parquet"
    if feat_cache.exists():
        feat = pd.read_parquet(feat_cache)
        feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
        log.info("loaded cached Massive features %s", feat_cache)
    else:
        feat = compute_price_features(px)
        feat.to_parquet(feat_cache, index=False)
    mean, std = fit_robust_z_qlib(alpha20_cols, OUT_DIR / "alpha20_robust_z_2008_2017.json")
    feat = apply_robust_z(feat, alpha20_cols, mean, std)
    # RD13 remains raw; Alpha20 already filled. RD13 NaN left for LightGBM.
    models = complete_d2_seeds(all_cols)
    d2 = predict_d2(feat, models)
    # 2025 overlap vs canonical (integrity, not 2026 PnL)
    snap = pd.read_csv(SNAPSHOT)
    snap["ticker"] = snap["ticker"].astype(str).str.upper().map(map_ticker)
    snap["instrument"] = snap["instrument"].astype(str)
    t2i = snap.dropna(subset=["ticker"]).drop_duplicates("ticker").set_index("ticker")["instrument"].to_dict()
    canon = pd.read_parquet(D2_CANONICAL)
    canon["datetime"] = pd.to_datetime(canon["datetime"]).dt.normalize()
    d2_2025 = d2[(d2["date"] >= "2025-01-02") & (d2["date"] <= "2025-12-31")].copy()
    d2_2025["instrument"] = d2_2025["ticker"].map(t2i)
    merged = d2_2025.dropna(subset=["instrument"]).merge(
        canon.rename(columns={"datetime": "date", "score": "d2_canon"}),
        on=["date", "instrument"],
        how="inner",
    )
    spearmans = []
    for _, g in merged.groupby("date"):
        if len(g) < 30 or g["d2"].nunique() < 2 or g["d2_canon"].nunique() < 2:
            continue
        spearmans.append(float(g["d2"].corr(g["d2_canon"], method="spearman")))
    d2_sp = float(np.nanmean(spearmans)) if spearmans else np.nan
    d2_n = int(len(merged))
    integrity["d2_2025_daily_rank_spearman_mean"] = d2_sp
    integrity["d2_2025_overlap_n"] = d2_n
    if np.isfinite(d2_sp) and d2_sp < D2_SPEARMAN_NOT_RELIABLE:
        integrity["2026_P5_REPLAY_QUALITY"] = "NOT_RELIABLE"
        integrity["reasons"].append(f"2025 D2 Rank Spearman vs canonical {d2_sp:.3f} < {D2_SPEARMAN_NOT_RELIABLE}; frozen D2 not recoverable.")
        dump_json(OUT_DIR / "2026_p5_replay_quality.json", integrity)
        log.error("NOT_RELIABLE after 2025 D2 overlap; STOP without 2026 VT0")
        write_report({
            "generated": _utc(),
            "integrity": integrity,
            "mechanism": "NOT_RUN",
            "return_sacrifice": "NOT_RUN",
            "metrics_p0": {k: np.nan for k in ("compound", "arr", "vol", "sharpe", "mdd", "worst_roll20_mdd", "downside_semi", "cvar5", "ending_1m")},
            "metrics_p1": {k: np.nan for k in ("compound", "arr", "vol", "sharpe", "mdd", "worst_roll20_mdd", "downside_semi", "cvar5", "ending_1m")},
            "target_p0": {k: np.nan for k in ("mad", "rmse", "pct_within_10", "pct_above_125", "pct_above_150")},
            "target_p1": {k: np.nan for k in ("mad", "rmse", "pct_within_10", "pct_above_125", "pct_above_150")},
            "exposure_p1": {k: np.nan for k in ("mean", "median", "min", "p05", "p25", "p75", "p95", "pct_lt_80", "pct_lt_60")},
            "exposure_path": {},
            "answers": {k: "STOP: 2026 P5 replay NOT_RELIABLE after 2025 D2 overlap; VT0 not run." for k in "ABCDEFGHIJ"},
            "first_date": None,
            "last_date": None,
            "n": 0,
            "stabilize_answer": "NOT_RUN",
            "d2_2025_spearman": d2_sp,
            "d2_2025_n": d2_n,
        })
        return
    dump_json(OUT_DIR / "2026_p5_replay_quality.json", integrity)
    log.info("2025 D2 Rank Spearman=%.4f n=%d; proceeding to 2026 P5 (still no VT0 look)", d2_sp, d2_n)

    dates_2026 = sorted(d for d in px["date"].drop_duplicates() if EVAL_START <= d <= EVAL_END)
    membership = load_membership(pd.DatetimeIndex(dates_2026))
    log.info("building approximate F1C statement panel")
    stmt = build_f1c_statement_panel()
    f1c_raw = f1c_raw_from_stmt(stmt, px, dates_2026)
    f1c_models = train_f1c_seeds()
    f1c = predict_f1c(f1c_raw, f1c_models)
    w1 = build_w1(d2, f1c, membership, px)
    w1_2026 = w1[w1["date"].isin(dates_2026)].copy()
    log.info("W1 2026 rows=%d days=%d", len(w1_2026), w1_2026["date"].nunique())
    decision_dates = sorted(w1_2026["date"].unique())
    weights = topk_dropout_weights(w1_2026[["date", "ticker", "score"]], decision_dates)
    daily = portfolio_from_weights(weights, px)
    if daily.empty:
        raise RuntimeError("no 2026 P5 daily returns; STOP")
    daily = apply_vt0(daily, weights)
    spy = pd.read_parquet(MASSIVE_SPY)
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
    spy = spy.sort_values("date")
    spy["r_spy"] = spy["close"].pct_change()
    daily = daily.merge(spy[["date", "r_spy"]].rename(columns={"date": "trade_date"}), on="trade_date", how="left")
    p0 = metrics(daily["r_p0"].to_numpy(), daily["r_spy"].to_numpy())
    p1 = metrics(daily["r_p1"].to_numpy(), daily["r_spy"].to_numpy())
    acc0 = target_accuracy(daily["rv20_p0"].to_numpy(), MR4A_TARGET_VOL)
    acc1 = target_accuracy(daily["rv20_p1"].to_numpy(), MR4A_TARGET_VOL)
    exp = exposure_stats(daily["exposure_p1"].to_numpy())
    mech, sac = classify(p0, p1, acc0, acc1)
    diag = delayed_rerisk_diag(daily)
    stabilize = "YES" if mech == "2026_VT0_MECHANISM_SUPPORTED" else ("PARTIAL" if mech.endswith("MIXED") else "NO")
    first = str(pd.Timestamp(daily["trade_date"].min()).date())
    last = str(pd.Timestamp(daily["trade_date"].max()).date())
    answers = {
        "A": (
            f"不能 exact 复现。质量={integrity['2026_P5_REPLAY_QUALITY']}。"
            f"2025 D2 vs canonical Rank Spearman={d2_sp:.3f}。"
            "F1C 非 Compustat PIT；宇宙为重建指数。结果只能叫 2026 approximate deployment replay。"
        ),
        "B": (
            f"年化 vol P0 {p0['vol']:.1%} → P1 {p1['vol']:.1%}。"
            f"{'是，继续下降' if p1['vol'] < p0['vol'] else '否，未下降'}。"
        ),
        "C": (
            f"MDD P0 {p0['mdd']:.1%} → P1 {p1['mdd']:.1%}。"
            f"{'改善' if p1['mdd'] > p0['mdd'] else '未改善'}。"
        ),
        "D": (
            f"CVaR5 {p0['cvar5']:.4f} → {p1['cvar5']:.4f}；semi {p0['downside_semi']:.1%} → {p1['downside_semi']:.1%}。"
            f"{'改善' if abs(p1['cvar5']) < abs(p0['cvar5']) else '未改善'}。"
        ),
        "E": (
            f"roll20 RMSE vs 18.63%：P0 {acc0['rmse']:.1%} → P1 {acc1['rmse']:.1%}；"
            f"within±10% {100*acc0['pct_within_10']:.1f}% → {100*acc1['pct_within_10']:.1f}%。"
            f"{'更接近' if acc1['rmse'] < acc0['rmse'] else '没有更接近'}。"
        ),
        "F": f"mean exposure {exp['mean']:.1%}；median {exp['median']:.1%}；min {exp['min']:.1%}。",
        "G": (
            f"£1m → P0 £{p0['ending_1m']:,.0f} vs P1 £{p1['ending_1m']:,.0f}；"
            f"compound {p0['compound']:.1%} vs {p1['compound']:.1%}；分类 {sac}。"
        ),
        "H": (
            f"最低仓 {diag.get('min_exposure_date')} exp={diag.get('min_exposure'):.1%}；"
            f"随后20日 P0 {diag.get('next20_p0_after_min_exp')} / P1 {diag.get('next20_p1_after_min_exp')}。"
            f"delayed re-risking = {diag.get('delayed_rerisking_observed')}。"
        ),
        "I": (
            f"{mech}。与 2020–2025 相比："
            f"{'方向一致：VT0 仍压缩实现波动' if p1['vol'] < p0['vol'] else '方向不一致'}。"
            "这不是 exact validation，不能单独改写冻结结论。"
        ),
        "J": "approximate deployment evidence（不是 exact validation）。",
    }
    payload = {
        "generated": _utc(),
        "integrity": integrity,
        "mechanism": mech,
        "return_sacrifice": sac,
        "metrics_p0": p0,
        "metrics_p1": p1,
        "target_p0": acc0,
        "target_p1": acc1,
        "exposure_p1": exp,
        "exposure_path": diag,
        "answers": answers,
        "first_date": first,
        "last_date": last,
        "n": int(len(daily)),
        "stabilize_answer": stabilize,
        "d2_2025_spearman": d2_sp,
        "d2_2025_n": d2_n,
        "label": LABEL,
        "TARGET_VOL": MR4A_TARGET_VOL,
    }
    daily.to_csv(OUT_DIR / "vt0_2026_daily.csv", index=False)
    weights.to_csv(OUT_DIR / "vt0_2026_p5_weights.csv", index=False)
    dump_json(OUT_DIR / "vt0_2026_answers.json", payload)
    write_report(payload)
    log.info("DONE quality=%s mech=%s sac=%s n=%d", integrity["2026_P5_REPLAY_QUALITY"], mech, sac, len(daily))


if __name__ == "__main__":
    main()
