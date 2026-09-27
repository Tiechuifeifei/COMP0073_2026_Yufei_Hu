#!/usr/bin/env python3
"""12_expand_quarterly_features_to_daily.py

Expand Phase 3A quarterly signal-event features to a daily point-in-time panel
via backward as-of joins on PERMNO."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MAIN_QUARTERLY = (
    PROJECT_ROOT
    / "data"
    / "quarterly_fundamental_features"
    / "main"
    / "quarterly_fundamental_features.parquet"
)
NEXTDAY_QUARTERLY = (
    PROJECT_ROOT
    / "data"
    / "quarterly_fundamental_features"
    / "nextday"
    / "quarterly_fundamental_features.parquet"
)
CRSP_PATH = PROJECT_ROOT / "data" / "crsp_daily" / "crsp_daily_market.parquet"
SP500_CANDIDATES = [
    PROJECT_ROOT / "staging" / "qlib_data" / "instruments" / "sp500.txt",
    PROJECT_ROOT / "staging" / "instruments" / "sp500.txt",
]
OUT_ROOT = PROJECT_ROOT / "data" / "daily_fundamental_features"

FEATURES = [
    "roe",
    "roa",
    "gross_profitability",
    "operating_profitability",
    "sales_growth_yoy",
    "asset_growth_yoy",
    "accruals",
    "leverage",
    "current_ratio",
    "book_to_market",
    "earnings_yield",
    "sales_to_price",
]

VALUE_FEATURES = ["book_to_market", "earnings_yield", "sales_to_price"]

SIGNAL_AND_ID_COLS = [
    "qlib_instrument",
    "gvkey",
    "datadate",
    "fqtr",
    "fyrq",
    "qtryr",
    "datacqtr",
    "accounting_available_date",
    "signal_start_date",
    "membership_start",
    "membership_end",
    "signal_usable_flag",
    "signal_market_cap",
    "history_only_flag",
]

AUDIT_COLS = [
    "accruals_old_raw_oancfq",
    "oancfq",
    "standalone_oancf_q",
    "oancf_ytd_conversion_flag",
    "oancf_missing_previous_quarter_flag",
    "oancf_nonconsecutive_quarter_flag",
    "ttm_oancf_valid_4q_flag",
    "ttm_niq",
    "ttm_saleq",
    "ttm_saleq_lag4",
    "ttm_gpq",
    "ttm_operating_profit",
    "ttm_oancfq",
    "ttm_oancf",
    "gpq",
    "operating_profit_q",
    "atq",
    "atq_lag1",
    "avg_atq",
    "atq_lag4",
    "book_equity",
    "book_equity_raw_ceqq",
    "book_equity_definition",
    "book_equity_missing_flag",
    "book_equity_nonpositive_flag",
    "avg_book_equity",
    "total_debt",
    "actq",
    "lctq",
    "actq_lctq_missing_flag",
    "accruals_numerator",
    "accruals_denominator",
    "accruals_definition",
]

CRSP_DAILY_COLS = ["permno", "date", "prc", "shrout", "ret", "retx", "vol", "market_cap"]

INTERVAL_COLS = [
    "signal_effective_start_date",
    "signal_effective_end_date",
    "next_signal_start_date",
    "signal_age_calendar_days",
    "signal_age_trading_days",
]

DAILY_MARKET_RENAME = {
    "market_cap": "daily_market_cap",
    "prc": "daily_prc",
    "ret": "daily_ret",
    "retx": "daily_retx",
    "vol": "daily_vol",
    "shrout": "daily_shrout",
}

STALE_CALENDAR_DAYS = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    raise TypeError(type(obj))


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.2f}%"


def resolve_sp500_path() -> Path:
    for path in SP500_CANDIDATES:
        if path.exists():
            log.info("S&P 500 membership source: %s", path)
            return path
    raise FileNotFoundError(
        "Missing sp500.txt. Tried:\n" + "\n".join(f"  - {p}" for p in SP500_CANDIDATES)
    )


def parse_sp500(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        instrument, start, end = line.split("\t")
        rows.append(
            {
                "qlib_instrument": instrument,
                "permno": int(instrument.lstrip("P")),
                "membership_start": pd.Timestamp(start),
                "membership_end": pd.Timestamp(end),
            }
        )
    return pd.DataFrame(rows)


def load_crsp() -> pd.DataFrame:
    df = pd.read_parquet(CRSP_PATH, columns=CRSP_DAILY_COLS)
    df["permno"] = df["permno"].astype(int)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ["prc", "shrout", "ret", "retx", "vol", "market_cap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["permno", "date"]).reset_index(drop=True)


def load_quarterly(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["permno"] = df["permno"].astype(int)
    df["gvkey"] = df["gvkey"].astype(str).str.strip().str.zfill(6)
    for col in ["datadate", "accounting_available_date", "signal_start_date", "membership_start", "membership_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.normalize()
    return df.sort_values(["permno", "signal_start_date", "datadate", "fqtr"]).reset_index(drop=True)


def filter_crsp_to_membership(crsp: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Keep CRSP rows whose date falls in at least one membership spell for the PERMNO."""
    parts: list[pd.DataFrame] = []
    crsp_groups = {p: g for p, g in crsp.groupby("permno", sort=False)}
    for permno, spells in membership.groupby("permno", sort=False):
        cgrp = crsp_groups.get(int(permno))
        if cgrp is None or cgrp.empty:
            continue
        for spell in spells.itertuples(index=False):
            hit = cgrp[(cgrp["date"] >= spell.membership_start) & (cgrp["date"] <= spell.membership_end)].copy()
            if hit.empty:
                continue
            hit["qlib_instrument"] = spell.qlib_instrument
            hit["membership_start"] = spell.membership_start
            hit["membership_end"] = spell.membership_end
            parts.append(hit)
    if not parts:
        return pd.DataFrame(columns=list(crsp.columns) + ["qlib_instrument", "membership_start", "membership_end"])
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["permno", "date", "membership_start"]).drop_duplicates(["permno", "date"], keep="last")
    return out.reset_index(drop=True)


def add_trading_day_index(trading: pd.DataFrame) -> pd.DataFrame:
    out = trading[["permno", "date"]].drop_duplicates().sort_values(["permno", "date"]).copy()
    out["trading_day_idx"] = out.groupby("permno").cumcount()
    return out


def prepare_signal_intervals(
    signals: pd.DataFrame,
    trading_idx: pd.DataFrame,
    crsp_end_by_permno: pd.Series,
) -> pd.DataFrame:
    """Attach next-signal and effective-end dates on the quarterly signal table."""
    out = signals.copy()
    out = out.sort_values(["permno", "signal_start_date", "datadate", "fqtr"]).drop_duplicates(
        ["permno", "signal_start_date"], keep="last"
    )
    out["signal_effective_start_date"] = out["signal_start_date"]
    out["next_signal_start_date"] = out.groupby("permno")["signal_start_date"].shift(-1)

    out = out.merge(
        trading_idx.rename(columns={"date": "signal_start_date", "trading_day_idx": "signal_start_trading_idx"}),
        on=["permno", "signal_start_date"],
        how="left",
    )

    with_next = out["next_signal_start_date"].notna()
    if with_next.any():
        tmp = out.loc[with_next, ["permno", "next_signal_start_date"]].copy()
        tmp = tmp.merge(
            trading_idx.rename(
                columns={"date": "next_signal_start_date", "trading_day_idx": "next_start_trading_idx"}
            ),
            on=["permno", "next_signal_start_date"],
            how="left",
        )
        tmp["end_idx"] = tmp["next_start_trading_idx"] - 1
        tmp = tmp.merge(
            trading_idx.rename(columns={"trading_day_idx": "end_idx", "date": "signal_effective_end_date"}),
            on=["permno", "end_idx"],
            how="left",
        )
        out.loc[with_next, "signal_effective_end_date"] = tmp["signal_effective_end_date"].to_numpy()
    out.loc[~with_next, "signal_effective_end_date"] = pd.NaT

    for permno, idx in out.groupby("permno").groups.items():
        mask_last = out.index.isin(idx) & out["next_signal_start_date"].isna()
        if not mask_last.any():
            continue
        end_candidates = [
            out.loc[mask_last, "membership_end"].max(),
            crsp_end_by_permno.get(int(permno), pd.NaT),
        ]
        end_candidates = [x for x in end_candidates if pd.notna(x)]
        if end_candidates:
            out.loc[mask_last, "signal_effective_end_date"] = min(end_candidates)

    cap_mask = out["signal_effective_end_date"].notna() & out["membership_end"].notna()
    out.loc[cap_mask, "signal_effective_end_date"] = out.loc[cap_mask, ["signal_effective_end_date", "membership_end"]].min(
        axis=1
    )
    crsp_cap = out["permno"].map(crsp_end_by_permno)
    cap2 = out["signal_effective_end_date"].notna() & crsp_cap.notna()
    out.loc[cap2, "signal_effective_end_date"] = np.minimum(
        out.loc[cap2, "signal_effective_end_date"],
        crsp_cap.loc[cap2],
    )

    return out


def merge_daily_to_signals(daily: pd.DataFrame, signal_payload: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of join strictly within each PERMNO."""
    parts: list[pd.DataFrame] = []
    signal_groups = {p: g for p, g in signal_payload.groupby("permno", sort=False)}
    drop_from_right = {
        "permno",
        "qlib_instrument",
        "membership_start",
        "membership_end",
    }
    for permno, dgrp in daily.groupby("permno", sort=False):
        sgrp = signal_groups.get(int(permno))
        if sgrp is None or sgrp.empty:
            parts.append(dgrp.copy())
            continue
        left = dgrp.sort_values("date").reset_index(drop=True)
        right = (
            sgrp.sort_values("signal_start_date")
            .drop(columns=[c for c in drop_from_right if c in sgrp.columns])
            .reset_index(drop=True)
        )
        parts.append(
            pd.merge_asof(
                left,
                right,
                left_on="date",
                right_on="signal_start_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    return pd.concat(parts, ignore_index=True)


def expand_variant(
    quarterly: pd.DataFrame,
    membership_crsp: pd.DataFrame,
    trading_idx: pd.DataFrame,
    crsp_end_by_permno: pd.Series,
    variant: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usable_signals = quarterly[quarterly["signal_usable_flag"].astype(bool)].copy()
    signals = prepare_signal_intervals(usable_signals, trading_idx, crsp_end_by_permno)
    if "membership_start" in signals.columns:
        signals = signals.rename(
            columns={
                "membership_start": "signal_membership_start",
                "membership_end": "signal_membership_end",
            }
        )
    signal_cols = [
        c
        for c in SIGNAL_AND_ID_COLS + FEATURES + AUDIT_COLS + ["feature_variant"]
        if c in signals.columns and c not in {"membership_start", "membership_end"}
    ]
    signal_payload = signals[
        [
            "permno",
            "signal_start_date",
            "next_signal_start_date",
            "signal_effective_end_date",
            "signal_start_trading_idx",
        ]
        + [c for c in signal_cols if c not in {"signal_start_date"}]
    ].copy()

    daily = membership_crsp.sort_values(["permno", "date"]).reset_index(drop=True)
    merged = merge_daily_to_signals(daily, signal_payload)

    pre_match = merged["signal_start_date"].notna()
    active = pre_match.copy()
    active &= merged["date"] >= merged["signal_start_date"]
    next_start = merged["next_signal_start_date"]
    active &= next_start.isna() | (merged["date"] < next_start)
    active &= merged["date"] <= merged["membership_end"]
    effective_end = merged["signal_effective_end_date"].fillna(merged["membership_end"])
    active &= merged["date"] <= effective_end

    matched = merged.loc[active].copy()
    matched["signal_effective_start_date"] = matched["signal_start_date"]
    if "signal_membership_start" in matched.columns:
        matched = matched.rename(
            columns={
                "signal_membership_start": "membership_start_at_signal",
                "signal_membership_end": "membership_end_at_signal",
            }
        )
    matched = matched.merge(trading_idx, on=["permno", "date"], how="left")
    matched["signal_age_calendar_days"] = (matched["date"] - matched["signal_start_date"]).dt.days
    matched["signal_age_trading_days"] = matched["trading_day_idx"] - matched["signal_start_trading_idx"]

    matched = matched.rename(columns=DAILY_MARKET_RENAME)
    matched["feature_variant"] = variant

    stats = {
        "variant": variant,
        "total_crsp_rows": int(len(membership_crsp)),
        "membership_crsp_rows": int(len(membership_crsp)),
        "merge_asof_hits": int(pre_match.sum()),
        "active_signal_rows": int(len(matched)),
        "excluded_no_historical_signal": int((~pre_match).sum()),
        "excluded_before_signal_start": int((pre_match & (merged["date"] < merged["signal_start_date"])).sum()),
        "excluded_after_effective_end": int((pre_match & ~active).sum()),
        "excluded_unusable_quarterly_events": int((~quarterly["signal_usable_flag"].astype(bool)).sum()),
        "integrity_date_before_signal_start": int((matched["date"] < matched["signal_start_date"]).sum()),
        "integrity_future_datadate": int((matched["date"] < matched["datadate"]).sum()),
        "integrity_duplicate_date_permno": int(matched.duplicated(["date", "permno"]).sum()),
    }
    return matched, stats


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        ["date"]
        + [c for c in SIGNAL_AND_ID_COLS if c in df.columns]
        + ["permno"]
        + INTERVAL_COLS
        + [c for c in DAILY_MARKET_RENAME.values() if c in df.columns]
        + FEATURES
        + [c for c in AUDIT_COLS if c in df.columns]
        + [c for c in ["membership_start_at_signal", "membership_end_at_signal"] if c in df.columns]
        + ["feature_variant"]
    )
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def build_daily_coverage_by_year(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["year"] = tmp["date"].dt.year
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        s = pd.to_numeric(tmp[feature], errors="coerce")
        for year, grp in tmp.groupby("year", sort=True):
            ss = s.loc[grp.index]
            rows.append(
                {
                    "feature": feature,
                    "year": int(year),
                    "n_rows": int(len(grp)),
                    "non_missing": int(ss.notna().sum()),
                    "coverage_rate": float(ss.notna().mean()),
                }
            )
    return pd.DataFrame(rows)


def build_signal_age_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, col in [
        ("calendar_days", "signal_age_calendar_days"),
        ("trading_days", "signal_age_trading_days"),
    ]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q = s.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        rows.append(
            {
                "metric": label,
                "count": int(len(s)),
                "mean": float(s.mean()),
                "median": float(q.loc[0.5]),
                "p1": float(q.loc[0.01]),
                "p25": float(q.loc[0.25]),
                "p75": float(q.loc[0.75]),
                "p99": float(q.loc[0.99]),
                "max": float(s.max()),
                "stale_gt_150_calendar_days": int((s > STALE_CALENDAR_DAYS).sum()) if label == "calendar_days" else 0,
            }
        )
    return pd.DataFrame(rows)


def build_unmatched_audit(
    merged_all: pd.DataFrame,
    membership_crsp: pd.DataFrame,
    quarterly: pd.DataFrame,
) -> pd.DataFrame:
    unmatched = merged_all.loc[merged_all["signal_start_date"].isna()].copy()
    if unmatched.empty:
        return pd.DataFrame(
            columns=["permno", "date", "qlib_instrument", "membership_start", "membership_end", "reason"]
        )
    first_signal = (
        quarterly[quarterly["signal_usable_flag"].astype(bool)]
        .groupby("permno")["signal_start_date"]
        .min()
        .rename("first_usable_signal_start")
    )
    unmatched = unmatched.merge(first_signal, on="permno", how="left")
    unmatched["reason"] = np.where(
        unmatched["first_usable_signal_start"].isna(),
        "no_usable_quarterly_signal_history",
        np.where(
            unmatched["date"] < unmatched["first_usable_signal_start"],
            "date_before_first_usable_signal",
            "no_prior_signal_on_date",
        ),
    )
    keep = ["permno", "date", "qlib_instrument", "membership_start", "membership_end", "reason", "first_usable_signal_start"]
    return unmatched[[c for c in keep if c in unmatched.columns]].head(5000)


def build_stale_signal_audit(df: pd.DataFrame) -> pd.DataFrame:
    stale = df[df["signal_age_calendar_days"] > STALE_CALENDAR_DAYS].copy()
    cols = [
        "date",
        "permno",
        "gvkey",
        "qlib_instrument",
        "datadate",
        "signal_start_date",
        "signal_effective_end_date",
        "next_signal_start_date",
        "signal_age_calendar_days",
        "signal_age_trading_days",
    ] + FEATURES[:3]
    return stale[[c for c in cols if c in stale.columns]].sort_values(
        "signal_age_calendar_days", ascending=False
    ).head(5000)


def securities_per_day_summary(df: pd.DataFrame) -> dict[str, float]:
    counts = df.groupby("date")["permno"].nunique()
    if counts.empty:
        return {"median": np.nan, "p1": np.nan, "p99": np.nan}
    q = counts.quantile([0.01, 0.5, 0.99])
    return {"median": float(q.loc[0.5]), "p1": float(q.loc[0.01]), "p99": float(q.loc[0.99])}


def compare_daily_variants(main_df: pd.DataFrame, next_df: pd.DataFrame, path: Path) -> dict[str, Any]:
    key = ["date", "permno"]
    m = main_df.copy()
    n = next_df.copy()
    merged = m.merge(n, on=key, suffixes=("_main", "_nextday"), how="outer", indicator=True)

    shifted = merged[
        (merged["_merge"] == "both")
        & merged["signal_start_date_main"].notna()
        & merged["signal_start_date_nextday"].notna()
        & (merged["signal_start_date_main"] != merged["signal_start_date_nextday"])
    ]

    lines = [
        "# Daily Main vs Next-Day Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"| Metric | Count |",
        f"| --- | ---: |",
        f"| Main daily rows | {len(main_df):,} |",
        f"| Next-day daily rows | {len(next_df):,} |",
        f"| Keys matched | {int((merged['_merge'] == 'both').sum()):,} |",
        f"| Rows with different active `signal_start_date` | {len(shifted):,} |",
        "",
        "## Value-feature differences on matched keys",
        "",
        "| Feature | Both non-missing | Pearson | Spearman | Mean abs diff | Median abs diff | p99 abs diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    both = merged[merged["_merge"] == "both"]
    value_stats: dict[str, Any] = {}
    for feat in VALUE_FEATURES:
        fm = pd.to_numeric(both[f"{feat}_main"], errors="coerce")
        fn = pd.to_numeric(both[f"{feat}_nextday"], errors="coerce")
        nn = fm.notna() & fn.notna()
        if nn.any():
            diff = (fm[nn] - fn[nn]).abs()
            value_stats[feat] = {
                "both_non_missing": int(nn.sum()),
                "pearson": float(fm[nn].corr(fn[nn])),
                "spearman": float(fm[nn].corr(fn[nn], method="spearman")),
                "mean_abs_diff": float(diff.mean()),
                "median_abs_diff": float(diff.median()),
                "p99_abs_diff": float(diff.quantile(0.99)),
            }
            lines.append(
                f"| `{feat}` | {value_stats[feat]['both_non_missing']:,} | "
                f"{value_stats[feat]['pearson']:.6f} | {value_stats[feat]['spearman']:.6f} | "
                f"{value_stats[feat]['mean_abs_diff']:.6g} | {value_stats[feat]['median_abs_diff']:.6g} | "
                f"{value_stats[feat]['p99_abs_diff']:.6g} |"
            )
        else:
            lines.append(f"| `{feat}` | 0 | — | — | — | — | — |")

    acct = [f for f in FEATURES if f not in VALUE_FEATURES]
    acct_mismatch = 0
    acct_same_sig_mismatch = 0
    same_signal_rows = 0
    for feat in acct:
        fm = pd.to_numeric(both[f"{feat}_main"], errors="coerce")
        fn = pd.to_numeric(both[f"{feat}_nextday"], errors="coerce")
        nn = fm.notna() & fn.notna()
        same_sig = both["signal_start_date_main"] == both["signal_start_date_nextday"]
        same_signal_rows = int(same_sig.sum()) if feat == acct[0] else same_signal_rows
        bad = nn & ~np.isclose(fm, fn, rtol=0, atol=1e-12, equal_nan=False)
        acct_mismatch += int(bad.sum())
        acct_same_sig_mismatch += int((bad & same_sig).sum())

    lines.extend(
        [
            "",
            "## Accounting features",
            "",
            f"Non-identical accounting-feature pairs (both non-missing): **{acct_mismatch:,}**",
            f"Non-identical pairs when the same `signal_start_date` is active: **{acct_same_sig_mismatch:,}**",
            f"Rows sharing the same active `signal_start_date`: **{same_signal_rows:,}**",
            "",
            "Observations shifted by one-day timing convention equal rows where the active ",
            f"`signal_start_date` differs across variants: **{len(shifted):,}**.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "main_rows": int(len(main_df)),
        "nextday_rows": int(len(next_df)),
        "matched_keys": int((merged["_merge"] == "both").sum()),
        "different_active_signal_rows": int(len(shifted)),
        "accounting_non_identical_pairs": acct_mismatch,
        "accounting_non_identical_pairs_same_signal_start": acct_same_sig_mismatch,
        "rows_with_same_active_signal_start": same_signal_rows,
        "value_feature_stats": value_stats,
    }


def write_quality_report(
    path: Path,
    crsp_rows_total: int,
    main_stats: dict[str, Any],
    main_df: pd.DataFrame,
    coverage_year: pd.DataFrame,
    age_summary: pd.DataFrame,
    sec_per_day: dict[str, float],
    compare_stats: dict[str, Any],
) -> None:
    feature_cov = {
        f: float(pd.to_numeric(main_df[f], errors="coerce").notna().mean()) for f in FEATURES if f in main_df.columns
    }
    lines = [
        "# Daily Fundamental Feature Expansion — Quality Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. Row counts",
        "",
        "| Metric | Main |",
        "| --- | ---: |",
        f"| Total CRSP daily rows considered | {crsp_rows_total:,} |",
        f"| CRSP rows inside S&P 500 membership | {main_stats['membership_crsp_rows']:,} |",
        f"| Backward as-of matches (`signal_start_date` present) | {main_stats['merge_asof_hits']:,} |",
        f"| Active fundamental signal rows (final panel) | {main_stats['active_signal_rows']:,} |",
        f"| Excluded: no historical signal | {main_stats['excluded_no_historical_signal']:,} |",
        f"| Excluded: before / after active interval | {main_stats['excluded_after_effective_end']:,} |",
        f"| Quarterly events with `signal_usable_flag=False` | {main_stats['excluded_unusable_quarterly_events']:,} |",
        "",
        "## 2. Temporal integrity (main)",
        "",
        "| Check | Count | Expected |",
        "| --- | ---: | ---: |",
        f"| `date < signal_start_date` | {main_stats['integrity_date_before_signal_start']:,} | 0 |",
        f"| `date < datadate` (future accounting) | {main_stats['integrity_future_datadate']:,} | 0 |",
        f"| Duplicate `(date, permno)` | {main_stats['integrity_duplicate_date_permno']:,} | 0 |",
        "",
        "Overlapping active quarterly signals per `(date, permno)` are prevented by the ",
        "backward as-of rule plus `date < next_signal_start_date` filter.",
        "",
        "## 3. Coverage",
        "",
        "| Feature | Daily coverage |",
        "| --- | ---: |",
    ]
    for feat, cov in feature_cov.items():
        lines.append(f"| `{feat}` | {pct(cov)} |")
    lines.extend(
        [
            "",
            f"Securities with active features per day — median: **{sec_per_day['median']:.0f}**, "
            f"p1: **{sec_per_day['p1']:.0f}**, p99: **{sec_per_day['p99']:.0f}**.",
            "",
            "Annual daily coverage: `daily_feature_coverage_by_year.csv`.",
            "",
            "## 4. Signal age",
            "",
            age_summary.to_markdown(index=False),
            "",
            f"Stale signals (> {STALE_CALENDAR_DAYS} calendar days): see `daily_stale_signal_audit.csv`.",
            "",
            "## 5. Membership integrity",
            "",
            "Daily rows are constructed only from CRSP dates inside S&P 500 membership spells ",
            "(`staging/.../sp500.txt`). Rows outside membership or beyond spell end are excluded.",
            "",
            "## 6. Main vs next-day",
            "",
            f"| Metric | Value |",
            f"| --- | ---: |",
            f"| Main daily rows | {compare_stats['main_rows']:,} |",
            f"| Next-day daily rows | {compare_stats['nextday_rows']:,} |",
            f"| Matched `(date, permno)` | {compare_stats['matched_keys']:,} |",
            f"| Different active quarterly event | {compare_stats['different_active_signal_rows']:,} |",
            f"| Accounting-feature non-identical pairs | {compare_stats['accounting_non_identical_pairs']:,} |",
            "",
            "Details: `daily_main_vs_nextday_comparison.md`.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_variant(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = select_output_columns(df)
    out.to_parquet(out_dir / "daily_fundamental_features.parquet", index=False)
    out.to_pickle(out_dir / "daily_fundamental_features.pkl")
    log.info("Wrote %s (%d rows)", out_dir, len(out))


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in [MAIN_QUARTERLY, NEXTDAY_QUARTERLY, CRSP_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    membership = parse_sp500(resolve_sp500_path())
    crsp = load_crsp()
    log.info("CRSP panel: %d rows, %d PERMNOs", len(crsp), crsp["permno"].nunique())

    membership_crsp = filter_crsp_to_membership(crsp, membership)
    log.info("Membership-filtered CRSP: %d rows", len(membership_crsp))

    trading_idx = add_trading_day_index(membership_crsp)
    crsp_end_by_permno = membership_crsp.groupby("permno")["date"].max()

    main_q = load_quarterly(MAIN_QUARTERLY)
    next_q = load_quarterly(NEXTDAY_QUARTERLY)

    main_raw, main_stats = expand_variant(main_q, membership_crsp, trading_idx, crsp_end_by_permno, "main")
    next_raw, next_stats = expand_variant(next_q, membership_crsp, trading_idx, crsp_end_by_permno, "nextday")

    save_variant(main_raw, OUT_ROOT / "main")
    save_variant(next_raw, OUT_ROOT / "nextday")

    main_df = select_output_columns(main_raw)
    coverage_year = build_daily_coverage_by_year(main_df)
    coverage_year.to_csv(OUT_ROOT / "daily_feature_coverage_by_year.csv", index=False)

    age_summary = build_signal_age_summary(main_df)
    age_summary.to_csv(OUT_ROOT / "daily_signal_age_summary.csv", index=False)

    merged_probe = merge_daily_to_signals(
        membership_crsp.sort_values(["permno", "date"]).reset_index(drop=True),
        main_q[main_q["signal_usable_flag"].astype(bool)]
        .sort_values(["permno", "signal_start_date"])
        .drop_duplicates(["permno", "signal_start_date"], keep="last"),
    )
    build_unmatched_audit(merged_probe, membership_crsp, main_q).to_csv(
        OUT_ROOT / "daily_unmatched_rows_audit.csv", index=False
    )
    build_stale_signal_audit(main_df).to_csv(OUT_ROOT / "daily_stale_signal_audit.csv", index=False)

    compare_stats = compare_daily_variants(main_df, select_output_columns(next_raw), OUT_ROOT / "daily_main_vs_nextday_comparison.md")
    sec_per_day = securities_per_day_summary(main_df)

    write_quality_report(
        OUT_ROOT / "daily_expansion_quality_report.md",
        int(len(crsp)),
        main_stats,
        main_df,
        coverage_year,
        age_summary,
        sec_per_day,
        compare_stats,
    )

    meta = {
        "phase": "4A",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "main_quarterly": str(MAIN_QUARTERLY),
            "nextday_quarterly": str(NEXTDAY_QUARTERLY),
            "crsp_daily": str(CRSP_PATH),
            "sp500_membership": str(resolve_sp500_path()),
        },
        "crsp_rows_total": int(len(crsp)),
        "main_stats": main_stats,
        "nextday_stats": next_stats,
        "main_output_rows": int(len(main_df)),
        "nextday_output_rows": int(len(next_raw)),
        "date_range": {
            "min": main_df["date"].min().isoformat() if len(main_df) else None,
            "max": main_df["date"].max().isoformat() if len(main_df) else None,
        },
        "unique_permno": int(main_df["permno"].nunique()) if len(main_df) else 0,
        "unique_instruments": int(main_df["qlib_instrument"].nunique()) if len(main_df) else 0,
        "feature_coverage": {f: float(pd.to_numeric(main_df[f], errors="coerce").notna().mean()) for f in FEATURES},
        "securities_per_day": sec_per_day,
        "compare_stats": compare_stats,
        "methodology": {
            "winsorize_before_expansion": False,
            "zscore_before_expansion": False,
            "join_rule": "backward_asof_by_permno_on_signal_start_date",
            "active_interval_end": "min(trading_day_before_next_signal, membership_end, crsp_end)",
        },
    }
    (OUT_ROOT / "daily_expansion_meta.json").write_text(json.dumps(meta, indent=2, default=json_safe), encoding="utf-8")

    log.info(
        "Phase 4A complete: main rows=%d, dates=%s..%s, permnos=%d",
        len(main_df),
        meta["date_range"]["min"],
        meta["date_range"]["max"],
        meta["unique_permno"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
