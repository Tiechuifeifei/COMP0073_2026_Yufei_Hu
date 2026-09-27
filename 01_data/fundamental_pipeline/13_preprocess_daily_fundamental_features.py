#!/usr/bin/env python3
"""13_preprocess_daily_fundamental_features.py

Daily cross-sectional winsorisation and standardisation of the 12 fundamental
features. Parameters for each (date, feature) use only that day's eligible stocks."""

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

MAIN_INPUT = PROJECT_ROOT / "data" / "daily_fundamental_features" / "main" / "daily_fundamental_features.parquet"
NEXTDAY_INPUT = (
    PROJECT_ROOT / "data" / "daily_fundamental_features" / "nextday" / "daily_fundamental_features.parquet"
)
OUT_ROOT = PROJECT_ROOT / "data" / "daily_fundamental_features_processed"

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

MIN_CROSS_SECTION = 50
STD_EPS = 1e-12
Z_MEAN_TOL = 0.01
Z_STD_TOL = 0.01

PRESERVE_COLS = [
    "date",
    "qlib_instrument",
    "permno",
    "gvkey",
    "datadate",
    "fqtr",
    "fyrq",
    "qtryr",
    "datacqtr",
    "accounting_available_date",
    "signal_start_date",
    "signal_effective_start_date",
    "signal_effective_end_date",
    "next_signal_start_date",
    "membership_start",
    "membership_end",
    "membership_start_at_signal",
    "membership_end_at_signal",
    "signal_usable_flag",
    "signal_market_cap",
    "signal_age_calendar_days",
    "signal_age_trading_days",
    "daily_market_cap",
    "daily_prc",
    "daily_ret",
    "daily_retx",
    "daily_vol",
    "daily_shrout",
    "history_only_flag",
    "feature_variant",
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


def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["permno"] = df["permno"].astype(int)
    return df.sort_values(["date", "permno"]).reset_index(drop=True)


def preprocess_feature(
    df: pd.DataFrame,
    feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return updated df columns plus audit frames for one feature."""
    raw = pd.to_numeric(df[feature], errors="coerce")
    out_raw = f"{feature}_raw"
    out_win = f"{feature}_win"
    out_z = f"{feature}_z"
    out_miss = f"{feature}_missing_flag"

    df[out_raw] = raw
    df[out_miss] = raw.isna().astype(np.int8)

    grouped = raw.groupby(df["date"], sort=False)
    valid_n = grouped.transform("count")

    def daily_quantile(s: pd.Series, q: float) -> float:
        if s.count() < MIN_CROSS_SECTION:
            return np.nan
        return float(s.quantile(q))

    p1 = grouped.transform(lambda s: daily_quantile(s, 0.01))
    p99 = grouped.transform(lambda s: daily_quantile(s, 0.99))

    processable = valid_n >= MIN_CROSS_SECTION
    win = raw.clip(lower=p1, upper=p99)
    win = win.where(processable)

    win_grouped = win.groupby(df["date"], sort=False)
    mean_win = win_grouped.transform("mean")
    std_win = win_grouped.transform(lambda s: s.std(ddof=0))

    z = (win - mean_win) / std_win
    zero_var = processable & ((std_win <= STD_EPS) | std_win.isna())
    z = z.where(~zero_var)
    z = z.where(processable)

    df[out_win] = win
    df[out_z] = z

    finite_raw = raw.notna() & processable
    clipped_low = int((finite_raw & (raw < p1)).sum())
    clipped_high = int((finite_raw & (raw > p99)).sum())
    clipped_total = clipped_low + clipped_high
    n_processable = int(processable.sum())

    threshold = (
        pd.DataFrame({"date": df["date"], "feature": feature, "p1": p1, "p99": p99, "valid_n": valid_n})
        .drop_duplicates(["date"])
        .sort_values("date")
    )
    threshold["processed"] = threshold["valid_n"] >= MIN_CROSS_SECTION

    cross_section = threshold[["date", "feature", "valid_n"]].copy()

    std_audit = (
        pd.DataFrame(
            {
                "date": df["date"],
                "feature": feature,
                "z": z,
                "processable": processable,
            }
        )
        .groupby(["date", "feature"], as_index=False)
        .agg(
            valid_z=("z", lambda s: int(s.notna().sum())),
            z_mean=("z", "mean"),
            z_std=("z", lambda s: s.std(ddof=0)),
            processable=("processable", "max"),
        )
    )
    std_audit["mean_abs"] = std_audit["z_mean"].abs()
    std_audit["std_deviation_from_one"] = (std_audit["z_std"] - 1.0).abs()
    std_audit["mean_fail"] = std_audit["processable"] & (std_audit["mean_abs"] > Z_MEAN_TOL)
    std_audit["std_fail"] = std_audit["processable"] & (std_audit["std_deviation_from_one"] > Z_STD_TOL)

    failures: list[dict[str, Any]] = []
    insuf = threshold[threshold["valid_n"] < MIN_CROSS_SECTION]
    for row in insuf.itertuples(index=False):
        failures.append(
            {
                "date": row.date,
                "feature": feature,
                "valid_n": int(row.valid_n),
                "failure_reason": "insufficient_cross_section",
            }
        )
    zero_var_dates = (
        pd.DataFrame({"date": df["date"], "zero_var": zero_var})
        .groupby("date")["zero_var"]
        .any()
        .reset_index()
    )
    zero_var_dates = zero_var_dates[zero_var_dates["zero_var"]]
    for row in zero_var_dates.itertuples(index=False):
        failures.append(
            {
                "date": row.date,
                "feature": feature,
                "valid_n": int(threshold.loc[threshold["date"] == row.date, "valid_n"].iloc[0]),
                "failure_reason": "zero_variance",
            }
        )

    summary_row = {
        "feature": feature,
        "raw_coverage": float(raw.notna().mean()),
        "win_coverage": float(win.notna().mean()),
        "z_coverage": float(z.notna().mean()),
        "missing_flag_rate": float(raw.isna().mean()),
        "clipped_lower_total": clipped_low,
        "clipped_upper_total": clipped_high,
        "clipped_total": clipped_total,
        "clipped_pct_of_processable": float(clipped_total / n_processable) if n_processable else np.nan,
        "avg_daily_p1": float(threshold.loc[threshold["processed"], "p1"].mean()) if threshold["processed"].any() else np.nan,
        "avg_daily_p99": float(threshold.loc[threshold["processed"], "p99"].mean()) if threshold["processed"].any() else np.nan,
        "min_daily_p1": float(threshold.loc[threshold["processed"], "p1"].min()) if threshold["processed"].any() else np.nan,
        "max_daily_p99": float(threshold.loc[threshold["processed"], "p99"].max()) if threshold["processed"].any() else np.nan,
        "median_valid_stocks_per_day": float(cross_section["valid_n"].median()),
        "p1_valid_stocks_per_day": float(cross_section["valid_n"].quantile(0.01)),
        "p99_valid_stocks_per_day": float(cross_section["valid_n"].quantile(0.99)),
        "dates_insufficient_cross_section": int((cross_section["valid_n"] < MIN_CROSS_SECTION).sum()),
        "dates_zero_variance": int(len(zero_var_dates)),
        "median_abs_daily_z_mean": float(std_audit.loc[std_audit["processable"], "mean_abs"].median())
        if std_audit["processable"].any()
        else np.nan,
        "p99_abs_daily_z_mean": float(std_audit.loc[std_audit["processable"], "mean_abs"].quantile(0.99))
        if std_audit["processable"].any()
        else np.nan,
        "median_daily_z_std": float(std_audit.loc[std_audit["processable"], "z_std"].median())
        if std_audit["processable"].any()
        else np.nan,
        "p1_daily_z_std": float(std_audit.loc[std_audit["processable"], "z_std"].quantile(0.01))
        if std_audit["processable"].any()
        else np.nan,
        "p99_daily_z_std": float(std_audit.loc[std_audit["processable"], "z_std"].quantile(0.99))
        if std_audit["processable"].any()
        else np.nan,
        "dates_mean_tolerance_fail": int(std_audit["mean_fail"].sum()),
        "dates_std_tolerance_fail": int(std_audit["std_fail"].sum()),
        "raw_mean": float(raw.mean()),
        "raw_median": float(raw.median()),
        "raw_std": float(raw.std()),
        "raw_p1": float(raw.quantile(0.01)),
        "raw_p99": float(raw.quantile(0.99)),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "win_mean": float(win.mean()),
        "win_median": float(win.median()),
        "win_std": float(win.std()),
        "win_p1": float(win.quantile(0.01)),
        "win_p99": float(win.quantile(0.99)),
        "win_min": float(win.min()),
        "win_max": float(win.max()),
    }

    failure_df = pd.DataFrame(failures)
    return df, threshold, cross_section, std_audit, pd.DataFrame([summary_row]), failure_df


def coverage_by_year(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["year"] = tmp["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, grp in tmp.groupby("year", sort=True):
        rows.append(
            {
                "feature": feature,
                "year": int(year),
                "n_rows": int(len(grp)),
                "raw_coverage": float(grp[f"{feature}_raw"].notna().mean()),
                "win_coverage": float(grp[f"{feature}_win"].notna().mean()),
                "z_coverage": float(grp[f"{feature}_z"].notna().mean()),
                "missing_flag_rate": float(grp[f"{feature}_missing_flag"].mean()),
            }
        )
    return pd.DataFrame(rows)


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    processed = []
    for f in FEATURES:
        processed.extend([f"{f}_raw", f"{f}_win", f"{f}_z", f"{f}_missing_flag"])
    cols = [c for c in PRESERVE_COLS if c in df.columns]
    cols += processed
    cols += [c for c in AUDIT_COLS if c in df.columns]
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def process_variant(path: Path, variant: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    log.info("Processing variant=%s from %s", variant, path)
    df = load_daily(path)
    input_rows = len(df)

    threshold_parts: list[pd.DataFrame] = []
    cross_parts: list[pd.DataFrame] = []
    std_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    failure_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []

    for feature in FEATURES:
        if feature not in df.columns:
            raise KeyError(f"Missing feature column: {feature}")
        df, threshold, cross_section, std_audit, summary, failures = preprocess_feature(df, feature)
        threshold_parts.append(threshold)
        cross_parts.append(cross_section)
        std_parts.append(std_audit)
        summary_parts.append(summary)
        if len(failures):
            failure_parts.append(failures)
        coverage_parts.append(coverage_by_year(df, feature))
        df = df.drop(columns=[feature], errors="ignore")

    audits = {
        "summary": pd.concat(summary_parts, ignore_index=True),
        "thresholds": pd.concat(threshold_parts, ignore_index=True),
        "cross_section": pd.concat(cross_parts, ignore_index=True),
        "standardisation": pd.concat(std_parts, ignore_index=True),
        "failures": pd.concat(failure_parts, ignore_index=True) if failure_parts else pd.DataFrame(),
        "coverage_by_year": pd.concat(coverage_parts, ignore_index=True),
    }

    out = select_output_columns(df)
    stats = {
        "variant": variant,
        "input_rows": input_rows,
        "output_rows": int(len(out)),
        "unique_dates": int(out["date"].nunique()),
        "unique_permno": int(out["permno"].nunique()),
        "duplicate_date_permno": int(out.duplicated(["date", "permno"]).sum()),
    }
    return out, audits, stats


def compare_processed(main: pd.DataFrame, next_df: pd.DataFrame, path: Path) -> dict[str, Any]:
    key = ["date", "permno"]
    m = main.sort_values(key).reset_index(drop=True)
    n = next_df.sort_values(key).reset_index(drop=True)
    merged = m.merge(n, on=key, suffixes=("_main", "_nextday"), how="outer", indicator=True)

    lines = [
        "# Main vs Next-Day Processed Feature Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"| Metric | Count |",
        f"| --- | ---: |",
        f"| Main rows | {len(main):,} |",
        f"| Next-day rows | {len(next_df):,} |",
        f"| Matched `(date, permno)` | {int((merged['_merge'] == 'both').sum()):,} |",
        "",
        "## Processed feature comparison (matched keys)",
        "",
        "| Feature | Level | Both non-missing | Pearson | Mean abs diff | Median abs diff |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    both = merged[merged["_merge"] == "both"]
    comparison: dict[str, Any] = {"matched_keys": int(len(both))}
    for feat in FEATURES:
        for level in ["win", "z"]:
            col = f"{feat}_{level}"
            fm = pd.to_numeric(both[f"{col}_main"], errors="coerce")
            fn = pd.to_numeric(both[f"{col}_nextday"], errors="coerce")
            nn = fm.notna() & fn.notna()
            if nn.any():
                diff = (fm[nn] - fn[nn]).abs()
                pearson = float(fm[nn].corr(fn[nn]))
                lines.append(
                    f"| `{feat}` | `{level}` | {int(nn.sum()):,} | {pearson:.6f} | "
                    f"{float(diff.mean()):.6g} | {float(diff.median()):.6g} |"
                )
            else:
                lines.append(f"| `{feat}` | `{level}` | 0 | — | — | — |")

    lines.extend(
        [
            "",
            "## Raw accounting identity check",
            "",
        ]
    )
    same_raw_identical_z = 0
    same_raw_diff_z = 0
    for feat in FEATURES:
        rr_m = pd.to_numeric(both[f"{feat}_raw_main"], errors="coerce")
        rr_n = pd.to_numeric(both[f"{feat}_raw_nextday"], errors="coerce")
        same_raw = np.isclose(rr_m, rr_n, rtol=0, atol=1e-12, equal_nan=False)
        if not same_raw.any():
            continue
        zw_m = pd.to_numeric(both.loc[same_raw, f"{feat}_z_main"], errors="coerce")
        zw_n = pd.to_numeric(both.loc[same_raw, f"{feat}_z_nextday"], errors="coerce")
        nn = zw_m.notna() & zw_n.notna()
        identical = nn & np.isclose(zw_m, zw_n, rtol=0, atol=1e-9, equal_nan=False)
        same_raw_identical_z += int(identical.sum())
        same_raw_diff_z += int((nn & ~identical).sum())

    lines.append(
        f"Across all features on matched keys: identical raw with both z non-missing → "
        f"identical z **{same_raw_identical_z:,}**; different z **{same_raw_diff_z:,}**."
    )
    lines.append("")
    lines.append(
        "Identical raw values produce identical processed values **only when the "
        "contemporaneous cross-section (eligible stocks and thresholds) is identical**. "
        "Main and next-day panels differ in active signals and row counts, so z-scores "
        "may diverge even when raw accounting values match."
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    comparison["same_raw_identical_z"] = same_raw_identical_z
    comparison["same_raw_different_z"] = same_raw_diff_z
    return comparison


def write_quality_report(
    path: Path,
    main_stats: dict[str, Any],
    summary: pd.DataFrame,
    failures: pd.DataFrame,
    compare_stats: dict[str, Any],
) -> None:
    lines = [
        "# Daily Feature Preprocessing Quality Report (Phase 4B)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Transformation: independent daily cross-sectional p1/p99 winsorisation (ddof=0 quantiles) ",
        "followed by population standard deviation (ddof=0) z-scoring within each date.",
        "",
        f"Minimum cross-section size: **{MIN_CROSS_SECTION}** stocks.",
        "",
        "## 1. Row integrity",
        "",
        "| Metric | Main |",
        "| --- | ---: |",
        f"| Input rows | {main_stats['input_rows']:,} |",
        f"| Output rows | {main_stats['output_rows']:,} |",
        f"| Rows added/removed | {main_stats['output_rows'] - main_stats['input_rows']:,} |",
        f"| Unique dates | {main_stats['unique_dates']:,} |",
        f"| Unique PERMNO | {main_stats['unique_permno']:,} |",
        f"| Duplicate `(date, permno)` | {main_stats['duplicate_date_permno']:,} |",
        "",
        "## 2. Missingness",
        "",
        summary[
            ["feature", "raw_coverage", "win_coverage", "z_coverage", "missing_flag_rate"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Annual coverage: `daily_processed_coverage_by_year.csv`.",
        "",
        "## 3. Winsorisation",
        "",
        summary[
            [
                "feature",
                "clipped_lower_total",
                "clipped_upper_total",
                "clipped_pct_of_processable",
                "avg_daily_p1",
                "avg_daily_p99",
                "min_daily_p1",
                "max_daily_p99",
            ]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        "Daily thresholds: `daily_winsorisation_thresholds.csv`.",
        "",
        "## 4. Standardisation checks",
        "",
        summary[
            [
                "feature",
                "median_abs_daily_z_mean",
                "p99_abs_daily_z_mean",
                "median_daily_z_std",
                "p1_daily_z_std",
                "p99_daily_z_std",
                "dates_mean_tolerance_fail",
                "dates_std_tolerance_fail",
            ]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        f"Tolerance: |daily mean| > {Z_MEAN_TOL} or |daily std − 1| > {Z_STD_TOL}.",
        "",
        "Full daily audit: `daily_standardisation_audit.csv`.",
        "",
        "## 5. Cross-section size",
        "",
        summary[
            [
                "feature",
                "median_valid_stocks_per_day",
                "p1_valid_stocks_per_day",
                "p99_valid_stocks_per_day",
                "dates_insufficient_cross_section",
            ]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## 6. Outlier effect (raw vs winsorized, full sample)",
        "",
        summary[
            [
                "feature",
                "raw_mean",
                "win_mean",
                "raw_std",
                "win_std",
                "raw_p1",
                "win_p1",
                "raw_p99",
                "win_p99",
                "raw_min",
                "win_min",
                "raw_max",
                "win_max",
            ]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## 7. Temporal integrity",
        "",
        "- Parameters estimated **independently per `(date, feature)`** using only that date's ",
        "non-missing cross-section.",
        "- **No** full-sample quantiles, multi-date pooling, forward-fill, or future-date information.",
        "",
        "## 8. Main vs next-day",
        "",
        f"| Metric | Value |",
        f"| --- | ---: |",
        f"| Main rows | {compare_stats.get('main_rows', 'NA')} |",
        f"| Next-day rows | {compare_stats.get('nextday_rows', 'NA')} |",
        f"| Matched keys | {compare_stats.get('matched_keys', 'NA')} |",
        f"| Same raw, identical z | {compare_stats.get('same_raw_identical_z', 'NA')} |",
        f"| Same raw, different z | {compare_stats.get('same_raw_different_z', 'NA')} |",
        "",
        "Details: `daily_main_vs_nextday_processed_comparison.md`.",
        "",
        "## Preprocessing failures",
        "",
    ]
    if len(failures):
        lines.append(
            f"Total `(date, feature)` failure records: **{len(failures):,}** "
            f"(see `daily_preprocessing_failure_audit.csv`)."
        )
        lines.append("")
        lines.append(failures["failure_reason"].value_counts().reset_index().to_markdown(index=False))
    else:
        lines.append("_No failure records._")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_variant(df: pd.DataFrame, audits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "daily_fundamental_features_processed.parquet", index=False)
    df.to_pickle(out_dir / "daily_fundamental_features_processed.pkl")
    log.info("Wrote %s (%d rows, %d cols)", out_dir, len(df), len(df.columns))


def enhanced_compare(
    main: pd.DataFrame,
    next_df: pd.DataFrame,
    main_th: pd.DataFrame,
    next_th: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    base = compare_processed(main, next_df, path)
    base["main_rows"] = len(main)
    base["nextday_rows"] = len(next_df)

    key = ["date", "permno"]
    merged = main.sort_values(key).merge(next_df.sort_values(key), on=key, suffixes=("_m", "_n"), how="inner")
    identical_cs_identical_z = 0
    identical_cs_diff_z = 0
    for feat in FEATURES:
        same_raw = np.isclose(
            pd.to_numeric(merged[f"{feat}_raw_m"], errors="coerce"),
            pd.to_numeric(merged[f"{feat}_raw_n"], errors="coerce"),
            rtol=0,
            atol=1e-12,
            equal_nan=False,
        )
        if not same_raw.any():
            continue
        mt = main_th[main_th["feature"] == feat][["date", "p1", "p99"]].rename(columns={"p1": "p1_m", "p99": "p99_m"})
        nt = next_th[next_th["feature"] == feat][["date", "p1", "p99"]].rename(columns={"p1": "p1_n", "p99": "p99_n"})
        sub = merged.loc[same_raw, ["date", "permno", f"{feat}_z_m", f"{feat}_z_n"]].merge(mt, on="date").merge(nt, on="date")
        same_cs = np.isclose(sub["p1_m"], sub["p1_n"], rtol=0, atol=1e-12, equal_nan=False) & np.isclose(
            sub["p99_m"], sub["p99_n"], rtol=0, atol=1e-12, equal_nan=False
        )
        if not same_cs.any():
            continue
        sub = sub.loc[same_cs]
        zm = pd.to_numeric(sub[f"{feat}_z_m"], errors="coerce")
        zn = pd.to_numeric(sub[f"{feat}_z_n"], errors="coerce")
        nn = zm.notna() & zn.notna()
        identical_cs_identical_z += int((nn & np.isclose(zm, zn, rtol=0, atol=1e-9, equal_nan=False)).sum())
        identical_cs_diff_z += int((nn & ~np.isclose(zm, zn, rtol=0, atol=1e-9, equal_nan=False)).sum())

    append = [
        "",
        "## Identical cross-section check",
        "",
        f"When raw values match **and** daily p1/p99 thresholds match, identical z-scores: "
        f"**{identical_cs_identical_z:,}**; different z-scores: **{identical_cs_diff_z:,}**.",
        "",
    ]
    path.write_text(path.read_text(encoding="utf-8") + "\n".join(append), encoding="utf-8")
    base["identical_cross_section_identical_z"] = identical_cs_identical_z
    base["identical_cross_section_different_z"] = identical_cs_diff_z
    return base


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in [MAIN_INPUT, NEXTDAY_INPUT]:
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    main_df, main_audits, main_stats = process_variant(MAIN_INPUT, "main")
    next_df, next_audits, next_stats = process_variant(NEXTDAY_INPUT, "nextday")

    save_variant(main_df, main_audits, OUT_ROOT / "main")
    save_variant(next_df, next_audits, OUT_ROOT / "nextday")

    main_audits["summary"].to_csv(OUT_ROOT / "daily_feature_preprocessing_summary.csv", index=False)
    main_audits["thresholds"].to_csv(OUT_ROOT / "daily_winsorisation_thresholds.csv", index=False)
    main_audits["cross_section"].to_csv(OUT_ROOT / "daily_cross_section_size.csv", index=False)
    main_audits["standardisation"].to_csv(OUT_ROOT / "daily_standardisation_audit.csv", index=False)
    main_audits["failures"].to_csv(OUT_ROOT / "daily_preprocessing_failure_audit.csv", index=False)
    main_audits["coverage_by_year"].to_csv(OUT_ROOT / "daily_processed_coverage_by_year.csv", index=False)

    compare_stats = enhanced_compare(
        main_df,
        next_df,
        main_audits["thresholds"],
        next_audits["thresholds"],
        OUT_ROOT / "daily_main_vs_nextday_processed_comparison.md",
    )

    write_quality_report(
        OUT_ROOT / "daily_preprocessing_quality_report.md",
        main_stats,
        main_audits["summary"],
        main_audits["failures"],
        compare_stats,
    )

    meta = {
        "phase": "4B",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"main": str(MAIN_INPUT), "nextday": str(NEXTDAY_INPUT)},
        "main_stats": main_stats,
        "nextday_stats": next_stats,
        "features": FEATURES,
        "min_cross_section": MIN_CROSS_SECTION,
        "std_ddof": 0,
        "winsor_quantiles": [0.01, 0.99],
        "z_mean_tolerance": Z_MEAN_TOL,
        "z_std_tolerance": Z_STD_TOL,
        "feature_summary": main_audits["summary"].to_dict(orient="records"),
        "compare_stats": compare_stats,
        "failure_record_count": int(len(main_audits["failures"])),
    }
    (OUT_ROOT / "daily_preprocessing_meta.json").write_text(
        json.dumps(meta, indent=2, default=json_safe),
        encoding="utf-8",
    )

    log.info(
        "Phase 4B complete: main %d rows x %d cols; z coverage roe=%.2f%%",
        len(main_df),
        len(main_df.columns),
        100 * main_df["roe_z"].notna().mean(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
