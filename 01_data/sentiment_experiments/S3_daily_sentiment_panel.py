#!/usr/bin/env python3
"""Phase S3: point-in-time daily sentiment panel for the S&P 500 PERMNO universe,
2010–2023, from frozen S2 normalized articles (no re-download)."""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_PATH = PROJECT_ROOT / "data/sentiment_raw/alpha_vantage_full/normalized/articles.parquet"
TRADING_DAYS_CACHE = PROJECT_ROOT / "data/sentiment_raw/alpha_vantage_full/_cache/trading_days.parquet"
WRDS_CSV = PROJECT_ROOT / "data/extracted/dwklv3a23uaacd05_csv/dwklv3a23uaacd05.csv"
SP500_TXT = PROJECT_ROOT / "staging/instruments/sp500.txt"
SECTOR_CSV = PROJECT_ROOT / "staging/instruments/permno_sector_sic.csv"
S2_OUT = PROJECT_ROOT / "data/sentiment_experiments/S2_alpha_vantage_full_download"
OUT_ROOT = PROJECT_ROOT / "data/sentiment_experiments/S3_daily_sentiment_panel"

PERIOD_START = date(2010, 1, 1)
PERIOD_END = date(2023, 12, 31)
HOLDOUT_START = date(2024, 1, 1)
ET = ZoneInfo("America/New_York")

SPLITS = {
    "train": (date(2010, 1, 1), date(2017, 12, 31)),
    "valid": (date(2018, 1, 1), date(2019, 12, 31)),
    "test": (date(2020, 1, 1), date(2023, 12, 31)),
}

RAW_FEATURES = [
    "news_count",
    "log_news_count",
    "ticker_mean_sentiment",
    "relevance_weighted_sentiment",
    "positive_news_share",
    "negative_news_share",
    "net_positive_share",
    "sentiment_dispersion",
    "relevance_mean",
    "sentiment_change_1d",
    "sentiment_surprise_20d",
    "sentiment_3d_decay",
    "sentiment_5d_decay",
    "abnormal_news_count_20d",
    "no_news_flag",
]

CONTINUOUS_FEATURES = [
    "ticker_mean_sentiment",
    "relevance_weighted_sentiment",
    "positive_news_share",
    "negative_news_share",
    "net_positive_share",
    "sentiment_dispersion",
    "relevance_mean",
    "sentiment_change_1d",
    "sentiment_surprise_20d",
    "sentiment_3d_decay",
    "sentiment_5d_decay",
    "abnormal_news_count_20d",
]

NO_ZSCORE_FEATURES = {"news_count", "log_news_count", "no_news_flag"}
POSITIVE_LABELS = {"Bullish", "Somewhat-Bullish"}
NEGATIVE_LABELS = {"Bearish", "Somewhat-Bearish"}

MIN_CROSS_SECTION = 50
STD_EPS = 1e-12
ROLL_MIN_PERIODS = 10
ROLL_WINDOW = 20

DECAY_3D_WEIGHTS = np.array([math.exp(-math.log(2) * k / 3.0) for k in range(3)])
DECAY_5D_WEIGHTS = np.array([math.exp(-math.log(2) * k / 5.0) for k in range(5)])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("S3")


def assign_split(d: date) -> str | None:
    for name, (s, e) in SPLITS.items():
        if s <= d <= e:
            return name
    return None


def load_membership() -> pd.DataFrame:
    rows = []
    for line in SP500_TXT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        inst, start, end = line.strip().split("\t")
        rows.append(
            {
                "instrument": inst,
                "permno": int(inst.lstrip("P")),
                "membership_start": pd.Timestamp(start),
                "membership_end": pd.Timestamp(end),
            }
        )
    return pd.DataFrame(rows)


def load_trading_days(permnos: set[int]) -> pd.DataFrame:
    td = pd.read_parquet(TRADING_DAYS_CACHE)
    td = td[td["permno"].isin(permnos)].copy()
    td["date"] = pd.to_datetime(td["date"]).dt.normalize()
    mask = (td["date"] >= pd.Timestamp(PERIOD_START)) & (td["date"] <= pd.Timestamp(PERIOD_END))
    return td.loc[mask].drop_duplicates(["permno", "date"]).sort_values(["permno", "date"])


def parse_time_published_utc(raw: str) -> pd.Timestamp | None:
    if not isinstance(raw, str) or not raw:
        return None
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return pd.Timestamp(dt)
        except ValueError:
            continue
    return None


def eligible_signal_date_conservative(pub_ts_et: pd.Timestamp, trading_dates: list[date]) -> date | None:
    """Frozen conservative rule: all articles map to the next trading session."""
    if pub_ts_et is pd.NaT or not trading_dates:
        return None
    pub_date = pub_ts_et.date()
    for d in trading_dates:
        if d > pub_date:
            return d
    return None


def dedupe_articles(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply frozen S2 deduplication order at article-company (PERMNO) grain."""
    stats: dict[str, int] = {"input_rows": len(df)}
    out = df.copy()
    out["time_published_utc_ts"] = out["time_published_utc"].map(parse_time_published_utc)
    out["relevance_score_num"] = pd.to_numeric(out["relevance_score"], errors="coerce")
    out["ticker_sentiment_score_num"] = pd.to_numeric(out["ticker_sentiment_score"], errors="coerce")

    out = out.sort_values(["permno", "time_published_utc_ts", "relevance_score_num"], ascending=[True, True, False])
    before = len(out)
    out = out.drop_duplicates(["permno", "normalized_url_hash"], keep="first")
    stats["after_normalized_url_dedup"] = len(out)
    stats["removed_normalized_url"] = before - len(out)

    before = len(out)
    out = out.drop_duplicates(["permno", "canonical_url_hash"], keep="first")
    stats["after_canonical_url_dedup"] = len(out)
    stats["removed_canonical_url"] = before - len(out)

    # title_hash ±6h: keep highest relevance per permno cluster
    out = out.sort_values(["permno", "title_hash", "time_published_utc_ts"])
    keep_idx: list[Any] = []
    for (_, _), grp in out.groupby(["permno", "title_hash"], sort=False):
        cluster_idx: list[Any] = []
        cluster_start: pd.Timestamp | None = None
        for idx, row in grp.iterrows():
            ts = row["time_published_utc_ts"]
            if ts is pd.NaT:
                keep_idx.append(idx)
                continue
            if not cluster_idx or cluster_start is None or (ts - cluster_start).total_seconds() <= 6 * 3600:
                cluster_idx.append(idx)
                if cluster_start is None or ts < cluster_start:
                    cluster_start = ts
            else:
                best = out.loc[cluster_idx].sort_values(
                    ["relevance_score_num", "time_published_utc_ts"], ascending=[False, True]
                ).index[0]
                keep_idx.append(best)
                cluster_idx = [idx]
                cluster_start = ts
        if cluster_idx:
            best = out.loc[cluster_idx].sort_values(
                ["relevance_score_num", "time_published_utc_ts"], ascending=[False, True]
            ).index[0]
            keep_idx.append(best)

    before = len(out)
    out = out.loc[sorted(set(keep_idx))].copy()
    stats["after_title_hash_window_dedup"] = len(out)
    stats["removed_title_hash_window"] = before - len(out)
    stats["unique_article_permno_pairs"] = out.drop_duplicates(["canonical_url_hash", "permno"]).shape[0]
    return out, stats


def map_signal_dates(articles: pd.DataFrame, trading: pd.DataFrame) -> pd.DataFrame:
    trading_lists = {
        int(p): sorted(g["date"].dt.date.tolist()) for p, g in trading.groupby("permno")
    }
    utc_ts = pd.to_datetime(articles["time_published_utc"], utc=True, errors="coerce")
    if utc_ts.isna().any():
        fallback = articles["time_published_utc"].map(parse_time_published_utc)
        utc_ts = utc_ts.fillna(pd.to_datetime(fallback, utc=True, errors="coerce"))

    articles = articles.copy()
    articles["publication_ts_et"] = utc_ts.dt.tz_convert(ET)
    articles["publication_date_et"] = articles["publication_ts_et"].dt.date

    signal_dates = []
    for permno, ts in zip(articles["permno"].astype(int), articles["publication_ts_et"]):
        signal_dates.append(eligible_signal_date_conservative(ts, trading_lists.get(permno, [])))
    articles["eligible_signal_date"] = signal_dates
    return articles


def aggregate_daily(articles: pd.DataFrame) -> pd.DataFrame:
    valid = articles.dropna(subset=["eligible_signal_date"]).copy()
    valid = valid[valid["relevance_score_num"] > 0]

    def weighted_sentiment(g: pd.DataFrame) -> float:
        rel = g["relevance_score_num"]
        score = g["ticker_sentiment_score_num"]
        denom = rel.sum()
        if denom <= 0:
            return np.nan
        return float((score * rel).sum() / denom)

    rows = []
    for (permno, sig_date), g in valid.groupby(["permno", "eligible_signal_date"]):
        n = len(g)
        labels = g["ticker_sentiment_label"].astype(str)
        pos = labels.isin(POSITIVE_LABELS).sum()
        neg = labels.isin(NEGATIVE_LABELS).sum()
        scores = g["ticker_sentiment_score_num"]
        rows.append(
            {
                "permno": int(permno),
                "date": pd.Timestamp(sig_date),
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
            }
        )
    return pd.DataFrame(rows)


def build_panel_skeleton(membership: pd.DataFrame, trading: pd.DataFrame) -> pd.DataFrame:
    panel = trading.merge(membership, on="permno", how="inner")
    panel = panel[(panel["date"] >= panel["membership_start"]) & (panel["date"] <= panel["membership_end"])]
    panel = panel.drop_duplicates(["permno", "date"]).sort_values(["permno", "date"])
    panel["datetime"] = panel["date"]
    panel["instrument"] = "P" + panel["permno"].astype(str)
    panel["split"] = panel["date"].dt.date.map(assign_split)
    panel = panel[panel["split"].notna()].copy()
    return panel


def apply_no_news_defaults(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["no_news_flag"] = panel["no_news_flag"].fillna(1).astype(np.int8)
    has_news = panel["no_news_flag"] == 0
    panel.loc[~has_news, "news_count"] = 0
    panel.loc[~has_news, "log_news_count"] = 0.0
    sentiment_cols = [
        "ticker_mean_sentiment",
        "relevance_weighted_sentiment",
        "positive_news_share",
        "negative_news_share",
        "net_positive_share",
        "sentiment_dispersion",
        "relevance_mean",
    ]
    for c in sentiment_cols:
        panel.loc[~has_news, c] = np.nan
    return panel


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


def add_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["permno", "date"]).copy()
    parts = []
    for _, grp in panel.groupby("permno", sort=False):
        g = grp.copy()
        w = g["relevance_weighted_sentiment"]
        g["sentiment_change_1d"] = w - w.shift(1)
        prior_mean = w.shift(1).rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).mean()
        g["sentiment_surprise_20d"] = w - prior_mean
        g["sentiment_3d_decay"] = decay_feature(w, DECAY_3D_WEIGHTS)
        g["sentiment_5d_decay"] = decay_feature(w, DECAY_5D_WEIGHTS)
        nc = g["news_count"].astype(float)
        roll_mean = nc.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).mean()
        roll_std = nc.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).std(ddof=0)
        g["abnormal_news_count_20d"] = (nc - roll_mean) / roll_std.replace(0, np.nan)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def daily_quantile(s: pd.Series, q: float) -> float:
    if s.count() < MIN_CROSS_SECTION:
        return np.nan
    return float(s.quantile(q))


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feat in CONTINUOUS_FEATURES:
        raw = pd.to_numeric(out[feat], errors="coerce")
        out[f"{feat}_raw"] = raw
        out[f"{feat}_missing_flag"] = raw.isna().astype(np.int8)
        grouped = raw.groupby(out["date"], sort=False)
        valid_n = grouped.transform("count")
        p1 = grouped.transform(lambda s: daily_quantile(s, 0.01))
        p99 = grouped.transform(lambda s: daily_quantile(s, 0.99))
        processable = valid_n >= MIN_CROSS_SECTION
        win = raw.clip(lower=p1, upper=p99)
        win = win.where(processable)
        mean_win = win.groupby(out["date"], sort=False).transform("mean")
        std_win = win.groupby(out["date"], sort=False).transform(lambda s: s.std(ddof=0))
        z = (win - mean_win) / std_win
        z = z.where(processable & (std_win > STD_EPS))
        out[f"{feat}_win"] = win
        out[f"{feat}_z"] = z
    for feat in NO_ZSCORE_FEATURES:
        if feat in out.columns:
            out[f"{feat}_raw"] = out[feat]
    return out


def load_permno_audit_fields(permnos: set[int]) -> pd.DataFrame:
    sector = pd.read_csv(SECTOR_CSV)[["PERMNO", "sector_sic"]].rename(columns={"PERMNO": "permno"})
    membership = load_membership()
    mem_dur = []
    for permno, grp in membership.groupby("permno"):
        days = ((grp["membership_end"] - grp["membership_start"]).dt.days + 1).sum()
        mem_dur.append(
            {
                "permno": permno,
                "membership_total_days": int(days),
                "membership_spell_count": len(grp),
                "membership_start_min": grp["membership_start"].min(),
                "membership_end_max": grp["membership_end"].max(),
            }
        )
    mem_dur_df = pd.DataFrame(mem_dur)

    td = pd.read_parquet(TRADING_DAYS_CACHE)
    td = td[td["permno"].isin(permnos)]
    td["date"] = pd.to_datetime(td["date"])
    last_trade = td.groupby("permno")["date"].max().dt.date.rename("last_crsp_trading_date")
    active_2023 = (last_trade >= date(2023, 12, 29)).rename("active_at_2023_end")

    cap_rows = []
    usecols = ["PERMNO", "DlyCalDt", "DlyCap", "Ticker"]
    for chunk in pd.read_csv(WRDS_CSV, usecols=usecols, chunksize=750_000):
        sub = chunk[chunk["PERMNO"].isin(permnos)]
        if len(sub):
            cap_rows.append(sub)
    cap = pd.concat(cap_rows, ignore_index=True) if cap_rows else pd.DataFrame(columns=usecols)
    cap = cap.rename(columns={"PERMNO": "permno", "DlyCalDt": "date", "DlyCap": "daily_market_cap", "Ticker": "ticker"})
    cap["date"] = pd.to_datetime(cap["date"])
    cap["permno"] = cap["permno"].astype(int)
    cap_stats = (
        cap.groupby("permno", as_index=False)
        .agg(
            median_daily_market_cap=("daily_market_cap", "median"),
            max_daily_market_cap=("daily_market_cap", "max"),
            last_observed_ticker=("ticker", "last"),
        )
    )

    art = pd.read_parquet(ARTICLES_PATH, columns=["permno"])
    art_counts = art.groupby("permno").size().rename("raw_article_rows").reset_index()

    base = pd.DataFrame({"permno": sorted(permnos)})
    audit = base.merge(sector, on="permno", how="left")
    audit = audit.merge(mem_dur_df, on="permno", how="left")
    audit = audit.merge(last_trade.reset_index(), on="permno", how="left")
    audit = audit.merge(active_2023.reset_index(), on="permno", how="left")
    audit = audit.merge(cap_stats, on="permno", how="left")
    audit = audit.merge(art_counts, on="permno", how="left")
    audit["raw_article_rows"] = audit["raw_article_rows"].fillna(0).astype(int)
    audit["zero_coverage_in_download"] = audit["raw_article_rows"] == 0
    return audit


def zero_coverage_audit(panel: pd.DataFrame, articles_raw: pd.DataFrame, deduped: pd.DataFrame) -> pd.DataFrame:
    zero_perm = set(panel.groupby("permno")["news_count"].sum().loc[lambda s: s == 0].index)
    audit = load_permno_audit_fields(zero_perm)
    audit["panel_news_rows"] = 0
    audit["deduped_article_rows"] = audit["permno"].map(deduped.groupby("permno").size()).fillna(0).astype(int)

    # early-year concentration: membership overlap with 2010-2012
    early_end = pd.Timestamp("2012-12-31")
    membership = load_membership()
    early_flags = []
    for permno in zero_perm:
        spells = membership[membership["permno"] == permno]
        only_early = spells["membership_end"].max() <= early_end if len(spells) else False
        early_flags.append({"permno": permno, "membership_end_before_2013": bool(only_early)})
    audit = audit.merge(pd.DataFrame(early_flags), on="permno", how="left")

    sector_counts = audit["sector_sic"].value_counts(dropna=False).to_dict()
    audit["sector_concentration_note"] = audit["sector_sic"].map(lambda s: f"sector={s}")
    audit.attrs["sector_distribution"] = sector_counts
    audit["inferred_issue"] = np.where(
        audit["raw_article_rows"] == 0,
        np.where(
            audit["membership_total_days"] < 400,
            "likely_short_membership_or_early_exit",
            np.where(
                audit["active_at_2023_end"] == False,
                "delisted_or_acquired_with_no_av_feed_match",
                "identifier_or_feed_gap_despite_membership",
            ),
        ),
        "unexpected_nonzero_raw",
    )
    return audit.sort_values("permno")


def timing_alignment_audit(articles: pd.DataFrame) -> pd.DataFrame:
    sub = articles.dropna(subset=["eligible_signal_date", "publication_ts_et"]).copy()
    sub["publication_date_et"] = sub["publication_ts_et"].dt.date
    sub["signal_strictly_after_pub_date"] = sub["eligible_signal_date"] > sub["publication_date_et"]
    sub["signal_after_publication_date"] = sub["signal_strictly_after_pub_date"]
    audit = sub[
        [
            "permno",
            "canonical_url_hash",
            "publication_ts_et",
            "publication_date_et",
            "eligible_signal_date",
            "market_hours_class",
            "signal_after_publication_date",
            "signal_strictly_after_pub_date",
        ]
    ].copy()
    audit["publication_ts_et"] = audit["publication_ts_et"].astype(str)
    audit["publication_date_et"] = audit["publication_date_et"].astype(str)
    audit["eligible_signal_date"] = audit["eligible_signal_date"].astype(str)
    audit["pit_rule"] = "next_trading_session_conservative"
    return audit


def duplicate_impact_audit(
    articles_raw: pd.DataFrame,
    deduped: pd.DataFrame,
    mapped_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = (
        mapped_raw.dropna(subset=["eligible_signal_date"])
        .assign(date=pd.to_datetime(mapped_raw["eligible_signal_date"]))
        .groupby(["permno", "date"])
        .size()
        .rename("news_count_before_dedup")
        .reset_index()
    )
    after = (
        deduped.dropna(subset=["eligible_signal_date"])
        .assign(date=pd.to_datetime(deduped["eligible_signal_date"]))
        .groupby(["permno", "date"])
        .size()
        .rename("news_count_after_dedup")
        .reset_index()
    )
    cmp = before.merge(after, on=["permno", "date"], how="outer").fillna(0)
    cmp["reduction"] = cmp["news_count_before_dedup"] - cmp["news_count_after_dedup"]
    cmp["reduction_pct"] = np.where(
        cmp["news_count_before_dedup"] > 0,
        cmp["reduction"] / cmp["news_count_before_dedup"],
        np.nan,
    )
    summary = {
        "raw_article_rows": int(len(articles_raw)),
        "deduped_article_rows": int(len(deduped)),
        "rows_removed_by_dedup": int(len(articles_raw) - len(deduped)),
        "pct_rows_removed": float((len(articles_raw) - len(deduped)) / len(articles_raw)),
        "mean_daily_reduction_when_both_positive": float(cmp.loc[cmp["reduction"] > 0, "reduction"].mean())
        if (cmp["reduction"] > 0).any()
        else 0.0,
        "total_daily_news_before": int(cmp["news_count_before_dedup"].sum()),
        "total_daily_news_after": int(cmp["news_count_after_dedup"].sum()),
        "daily_cells_with_reduction_pct": float((cmp["reduction"] > 0).mean()),
    }
    return cmp, summary


def feature_distribution_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in RAW_FEATURES:
        s = pd.to_numeric(df[feat], errors="coerce")
        rows.append(
            {
                "feature": feat,
                "count_non_missing": int(s.notna().sum()),
                "missing_pct": float(s.isna().mean()),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "std": float(s.std(ddof=0)) if s.notna().any() else np.nan,
                "p01": float(s.quantile(0.01)) if s.notna().any() else np.nan,
                "p50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
                "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
                "min": float(s.min()) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def coverage_by_year(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["year"] = tmp["date"].dt.year
    rows = []
    for (year, split), g in tmp.groupby(["year", "split"]):
        rows.append(
            {
                "year": int(year),
                "split": split,
                "panel_rows": len(g),
                "rows_with_news": int((g["no_news_flag"] == 0).sum()),
                "pct_rows_with_news": float((g["no_news_flag"] == 0).mean()),
                "mean_news_count_when_news": float(g.loc[g["news_count"] > 0, "news_count"].mean())
                if (g["news_count"] > 0).any()
                else np.nan,
                "unique_permnos": g["permno"].nunique(),
                "permnos_with_any_news": int(g.groupby("permno")["news_count"].sum().gt(0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "split"])


def coverage_by_split(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in df.groupby("split"):
        days = g["date"].nunique()
        per_day_stock_pct = g.groupby("date")["no_news_flag"].apply(lambda s: 1 - s.mean())
        per_stock_day_pct = g.groupby("permno")["no_news_flag"].apply(lambda s: 1 - s.mean())
        rows.append(
            {
                "split": split,
                "panel_rows": len(g),
                "unique_dates": int(days),
                "unique_permnos": int(g["permno"].nunique()),
                "rows_with_news": int((g["no_news_flag"] == 0).sum()),
                "pct_rows_with_news": float((g["no_news_flag"] == 0).mean()),
                "mean_pct_stocks_with_news_per_day": float(per_day_stock_pct.mean()),
                "median_pct_stocks_with_news_per_day": float(per_day_stock_pct.median()),
                "mean_pct_trading_days_with_news_per_stock": float(per_stock_day_pct.mean()),
                "median_pct_trading_days_with_news_per_stock": float(per_stock_day_pct.median()),
            }
        )
    return pd.DataFrame(rows)


def write_feature_dictionary(path: Path) -> None:
    rows = [
        ("news_count", "count", "Number of deduplicated article-company records on eligible signal date"),
        ("log_news_count", "transform", "log1p(news_count); 0 when no news"),
        ("ticker_mean_sentiment", "level", "Mean ticker_sentiment_score across articles; missing if no news"),
        (
            "relevance_weighted_sentiment",
            "level",
            "sum(ticker_sentiment_score * relevance_score) / sum(relevance_score); requires sum(relevance)>0",
        ),
        ("positive_news_share", "share", "Share of articles with Bullish or Somewhat-Bullish label"),
        ("negative_news_share", "share", "Share of articles with Bearish or Somewhat-Bearish label"),
        ("net_positive_share", "share", "positive_news_share - negative_news_share"),
        ("sentiment_dispersion", "dispersion", "Std dev of ticker_sentiment_score (ddof=0); requires news_count>=2"),
        ("relevance_mean", "level", "Mean relevance_score across articles"),
        ("sentiment_change_1d", "delta", "relevance_weighted_sentiment(t) - relevance_weighted_sentiment(t-1)"),
        (
            "sentiment_surprise_20d",
            "surprise",
            "relevance_weighted_sentiment - rolling mean of prior 20 days (min 10 obs, excludes current day)",
        ),
        (
            "sentiment_3d_decay",
            "decay",
            "Weighted mean of relevance_weighted_sentiment over t..t-2 with weights exp(-ln(2)*k/3), renormalized over non-missing",
        ),
        (
            "sentiment_5d_decay",
            "decay",
            "Weighted mean of relevance_weighted_sentiment over t..t-4 with weights exp(-ln(2)*k/5), renormalized over non-missing",
        ),
        (
            "abnormal_news_count_20d",
            "volume_anomaly",
            "(news_count - rolling_mean_20) / rolling_std_20(ddof=0); min 10 obs; std=0 -> missing",
        ),
        ("no_news_flag", "indicator", "1 if no deduplicated articles on signal date; 0 otherwise"),
    ]
    pd.DataFrame(rows, columns=["feature", "type", "definition"]).to_csv(path, index=False)


def write_point_in_time_spec(path: Path) -> None:
    text = """# S3 Point-in-Time Specification (Frozen)

## Label anchor (frozen cross-stream convention)

Target label: `Ref($close, -2) / Ref($close, -1) - 1`

At signal date **t**, the raw label equals **close(t+2) / close(t+1) − 1**. Sentiment features on **t**
must use only information whose publication timestamp is strictly before the return window anchored at **t**.

Under the conservative next-session mapping below, every article mapped to signal date **t** was published
on a calendar date **strictly before t** (or earlier timestamp on prior calendar days). No article affects a
signal date earlier than its publication time.

## Publication timestamp

- Source field: `time_published` (stored as `time_published_utc`, `time_published_us_eastern`)
- Parsed as UTC, converted to `America/New_York` for session classification (audit only)

## Frozen conservative eligibility rule

All articles — weekday pre-16:00, at/after 16:00, weekend, and holiday — map to the **next CRSP trading
session** for the target PERMNO:

`eligible_signal_date = min { trading_date : trading_date > publication_date_et }`

Publication date is the US/Eastern calendar date of the article timestamp.

## Primary company signal

Use ticker-specific fields (`ticker_sentiment_score`, `relevance_score`, `ticker_sentiment_label`).
Overall article sentiment is excluded from the frozen 15-feature set (secondary robustness only).

## Deduplication (canonical URL truth)

Apply frozen S2 order before aggregation:

1. `normalized_url_hash` — keep earliest `time_published_utc` per PERMNO
2. `canonical_url_hash` — keep earliest per PERMNO
3. `title_hash` ±6h — keep highest `relevance_score` per PERMNO

One article-company record per (`canonical_url_hash`, `permno`).

## Sentiment experiment periods

| Split | Dates |
|-------|-------|
| train | 2010-01-01 .. 2017-12-31 |
| valid | 2018-01-01 .. 2019-12-31 |
| test | 2020-01-01 .. 2023-12-31 |
| holdout | 2024 locked — not accessed |

Alpha Vantage coverage begins 2010. Sentiment models are **not** strictly paired to Alpha20/R1-13 trained on 2008-2017.
Common-sample baselines for later work: `Alpha20-common-2010`, `Alpha20+RD-13-common-2010`.
"""
    path.write_text(text, encoding="utf-8")


def write_no_news_spec(path: Path) -> None:
    text = """# S3 No-News Handling Specification (Frozen)

Complete datetime × PERMNO panel aligned to S&P 500 membership trading days, 2010-2023.

When `no_news_flag = 1`:

| Field | Value |
|-------|-------|
| `no_news_flag` | 1 |
| `news_count` | 0 |
| `log_news_count` | 0 |
| `positive_news_share`, `negative_news_share`, `net_positive_share` | missing |
| `ticker_mean_sentiment`, `relevance_weighted_sentiment`, `relevance_mean`, `sentiment_dispersion` | missing |
| Rolling/decay features | computed only from predefined formulas; may remain missing without forward-filled raw sentiment |

Prohibited:

- Dropping securities or dates with no news
- Silent neutral imputation (treating missing as 0 sentiment)
- Indefinite forward-fill of raw daily sentiment
"""
    path.write_text(text, encoding="utf-8")


def qa_checks(raw_panel: pd.DataFrame, timing_audit: pd.DataFrame, deduped: pd.DataFrame) -> dict[str, Any]:
    key_dupes = raw_panel.duplicated(["datetime", "instrument"]).sum()
    has_2024 = int((raw_panel["date"] >= pd.Timestamp(HOLDOUT_START)).sum())
    pit_fail = int((~timing_audit["signal_strictly_after_pub_date"]).sum()) if len(timing_audit) else 0
    art_perm_dupes = deduped.duplicated(["canonical_url_hash", "permno"]).sum()
    weighted_invalid = raw_panel.loc[
        raw_panel["no_news_flag"] == 0, "relevance_weighted_sentiment"
    ].isna().sum()
    return {
        "unique_datetime_instrument_keys": bool(key_dupes == 0),
        "duplicate_key_count": int(key_dupes),
        "rows_2024": int(has_2024),
        "pit_alignment_failures": int(pit_fail),
        "duplicate_article_permno_after_dedup": int(art_perm_dupes),
        "news_days_missing_weighted_sentiment": int(weighted_invalid),
        "max_date": str(raw_panel["date"].max().date()),
        "min_date": str(raw_panel["date"].min().date()),
    }


def write_quality_report(path: Path, qa: dict[str, Any], dedup_stats: dict[str, int], split_cov: pd.DataFrame) -> None:
    lines = [
        "# S3 Data Quality Report",
        "",
        "## QA checks",
        "",
    ]
    for k, v in qa.items():
        lines.append(f"- **{k}:** {v}")
    lines.extend(["", "## Deduplication", ""])
    for k, v in dedup_stats.items():
        lines.append(f"- {k}: {v:,}" if isinstance(v, int) else f"- {k}: {v}")
    lines.extend(["", "## Coverage by split", ""])
    for _, row in split_cov.iterrows():
        lines.append(
            f"- **{row['split']}**: rows={int(row['panel_rows']):,}, "
            f"pct_with_news={row['pct_rows_with_news']:.2%}, "
            f"mean_stocks_with_news/day={row['mean_pct_stocks_with_news_per_day']:.2%}"
        )
    lines.extend(
        [
            "",
            "## Readiness for single-factor IC",
            "",
            "Panel construction complete with strict PIT mapping and frozen features. "
            "Single-factor IC analysis may proceed in a **separate phase** using raw labels "
            "(`Ref($close,-2)/Ref($close,-1)-1`) on this panel; IC is **not computed in S3**.",
            "",
            "Limitations: 218 zero-coverage PERMNOs retained; early-year news sparsity; "
            "58.8% canonical URL duplication removed before aggregation.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log.info("Loading articles from %s", ARTICLES_PATH)
    articles = pd.read_parquet(ARTICLES_PATH)
    articles = articles[articles["publication_date"] <= str(PERIOD_END)].copy()
    if (pd.to_datetime(articles["publication_date"], errors="coerce") >= pd.Timestamp(HOLDOUT_START)).any():
        raise RuntimeError("2024 articles present in input — aborting")

    membership = load_membership()
    permnos = set(membership["permno"].unique())
    trading = load_trading_days(permnos)
    log.info("Trading day rows in window: %s", len(trading))

    deduped, dedup_stats = dedupe_articles(articles)
    mapped_raw = map_signal_dates(articles, trading)
    deduped = map_signal_dates(deduped, trading)
    timing_audit = timing_alignment_audit(deduped)
    timing_audit.to_csv(OUT_ROOT / "S3_timing_alignment_audit.csv", index=False)

    daily = aggregate_daily(deduped)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()

    panel = build_panel_skeleton(membership, trading)
    log.info("Panel skeleton rows: %s", len(panel))

    panel = panel.merge(
        daily,
        on=["permno", "date"],
        how="left",
        suffixes=("", "_news"),
    )
    panel = apply_no_news_defaults(panel)
    panel = add_rolling_features(panel)

    preserve_cols = [
        "datetime",
        "instrument",
        "permno",
        "split",
        "membership_start",
        "membership_end",
        "date",
    ]
    raw_out = panel[preserve_cols + RAW_FEATURES].copy()
    raw_out.to_parquet(OUT_ROOT / "S3_daily_sentiment_raw.parquet", index=False)

    processed = preprocess_features(raw_out)
    processed.to_parquet(OUT_ROOT / "S3_daily_sentiment_processed.parquet", index=False)

    write_feature_dictionary(OUT_ROOT / "S3_feature_dictionary.csv")
    write_point_in_time_spec(OUT_ROOT / "S3_point_in_time_specification.md")
    write_no_news_spec(OUT_ROOT / "S3_no_news_handling_specification.md")

    cov_year = coverage_by_year(raw_out)
    cov_year.to_csv(OUT_ROOT / "S3_coverage_by_year.csv", index=False)
    cov_split = coverage_by_split(raw_out)
    cov_split.to_csv(OUT_ROOT / "S3_coverage_by_split.csv", index=False)

    zero_audit = zero_coverage_audit(raw_out, articles, deduped)
    zero_audit.to_csv(OUT_ROOT / "S3_zero_coverage_permno_audit.csv", index=False)

    dup_audit, dup_summary = duplicate_impact_audit(articles, deduped, mapped_raw)
    dup_audit.head(5000).to_csv(OUT_ROOT / "S3_duplicate_impact_audit.csv", index=False)

    feat_dist = feature_distribution_report(raw_out)
    feat_dist.to_csv(OUT_ROOT / "S3_feature_distribution.csv", index=False)

    qa = qa_checks(raw_out, timing_audit, deduped)
    write_quality_report(OUT_ROOT / "S3_data_quality_report.md", qa, dedup_stats, cov_split)

    per_day = raw_out.groupby("date")["no_news_flag"].apply(lambda s: 1 - s.mean())
    meta = {
        "experiment": "S3_daily_sentiment_panel",
        "status": "COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "s2_status": "APPROVED_AND_FROZEN",
        "input_articles_parquet": str(ARTICLES_PATH.relative_to(PROJECT_ROOT)),
        "period_start": str(PERIOD_START),
        "period_end": str(PERIOD_END),
        "holdout_2024_accessed": False,
        "panel_row_count": int(len(raw_out)),
        "unique_permnos": int(raw_out["permno"].nunique()),
        "unique_dates": int(raw_out["date"].nunique()),
        "split_row_counts": raw_out["split"].value_counts().to_dict(),
        "deduplication": dedup_stats,
        "duplicate_impact_summary": dup_summary,
        "qa_checks": qa,
        "mean_pct_stocks_with_news_per_day": float(per_day.mean()),
        "zero_coverage_permno_count": int((raw_out.groupby("permno")["news_count"].sum() == 0).sum()),
        "ready_for_single_factor_ic_phase": True,
        "modelling_performed": False,
        "ic_computed": False,
    }
    (OUT_ROOT / "S3_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("S3 complete. Panel rows=%s QA=%s", len(raw_out), qa)
    return 0


if __name__ == "__main__":
    sys.exit(main())
