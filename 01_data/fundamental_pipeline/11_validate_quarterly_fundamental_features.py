#!/usr/bin/env python3
"""11_validate_quarterly_fundamental_features.py

Validate Phase 3A quarterly fundamental features before daily expansion.
Reads Phase 2 / 3A outputs only."""

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

MAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "quarterly_fundamental_features"
    / "main"
    / "quarterly_fundamental_features.parquet"
)
NEXTDAY_PATH = (
    PROJECT_ROOT
    / "data"
    / "quarterly_fundamental_features"
    / "nextday"
    / "quarterly_fundamental_features.parquet"
)
OUT_DIR = PROJECT_ROOT / "data" / "quarterly_feature_validation"

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

ACCOUNTING_FEATURES = [
    "roe",
    "roa",
    "gross_profitability",
    "operating_profitability",
    "sales_growth_yoy",
    "asset_growth_yoy",
    "accruals",
    "leverage",
    "current_ratio",
]

VALUE_FEATURES = ["book_to_market", "earnings_yield", "sales_to_price"]

RESEARCH_KEY = ["permno", "gvkey", "datadate", "fqtr"]

FEATURE_META: dict[str, dict[str, Any]] = {
    "roe": {
        "numerator": "ttm_niq",
        "denominator": "avg_book_equity",
        "flags": ["book_equity_missing_flag", "book_equity_nonpositive_flag"],
        "ratio": True,
    },
    "roa": {
        "numerator": "ttm_niq",
        "denominator": "avg_atq",
        "flags": [],
        "ratio": True,
    },
    "gross_profitability": {
        "numerator": "ttm_gpq",
        "denominator": "avg_atq",
        "flags": [],
        "ratio": True,
    },
    "operating_profitability": {
        "numerator": "ttm_operating_profit",
        "denominator": "avg_book_equity",
        "flags": ["book_equity_missing_flag", "book_equity_nonpositive_flag"],
        "ratio": True,
    },
    "sales_growth_yoy": {
        "numerator": "ttm_saleq",
        "denominator": "ttm_saleq_lag4",
        "flags": [],
        "ratio": True,
    },
    "asset_growth_yoy": {
        "numerator": "atq",
        "denominator": "atq_lag4",
        "flags": [],
        "ratio": True,
    },
    "accruals": {
        "numerator": "accruals_numerator",
        "denominator": "accruals_denominator",
        "flags": [
            "ttm_oancf_valid_4q_flag",
            "oancf_ytd_conversion_flag",
            "oancf_missing_previous_quarter_flag",
            "oancf_nonconsecutive_quarter_flag",
        ],
        "ratio": True,
    },
    "leverage": {
        "numerator": "total_debt",
        "denominator": "atq",
        "flags": [],
        "ratio": True,
    },
    "current_ratio": {
        "numerator": "actq",
        "denominator": "lctq",
        "flags": ["actq_lctq_missing_flag"],
        "ratio": True,
    },
    "book_to_market": {
        "numerator": "book_equity",
        "denominator": "signal_market_cap",
        "flags": ["book_equity_missing_flag", "book_equity_nonpositive_flag"],
        "ratio": True,
    },
    "earnings_yield": {
        "numerator": "ttm_niq",
        "denominator": "signal_market_cap",
        "flags": [],
        "ratio": True,
    },
    "sales_to_price": {
        "numerator": "ttm_saleq",
        "denominator": "signal_market_cap",
        "flags": [],
        "ratio": True,
    },
}

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


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["signal_start_date"] = pd.to_datetime(df["signal_start_date"])
    df["signal_year"] = df["signal_start_date"].dt.year
    return df


def longest_missing_run(values: pd.Series) -> int:
    missing = values.isna().to_numpy()
    if not missing.any():
        return 0
    max_run = 0
    run = 0
    for is_missing in missing:
        if is_missing:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def coverage_summary(df: pd.DataFrame, feature: str) -> dict[str, Any]:
    s = pd.to_numeric(df[feature], errors="coerce")
    usable = df["signal_usable_flag"].astype(bool)
    valid_mask = s.notna()
    permno_cov = df.loc[valid_mask].groupby("permno").size()
    gvkey_cov = df.loc[valid_mask].groupby("gvkey").size()

    permno_stats = (
        df.groupby("permno")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "coverage": float(pd.to_numeric(g[feature], errors="coerce").notna().mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    permno_stats = permno_stats[permno_stats["n"] >= 8]
    poor_permno = permno_stats[permno_stats["coverage"] < 0.50]

    permno_runs = (
        df.sort_values(["permno", "datadate", "fqtr"])
        .groupby("permno")[feature]
        .apply(longest_missing_run)
        .rename("longest_missing_run")
        .reset_index()
    )
    max_run = int(permno_runs["longest_missing_run"].max()) if len(permno_runs) else 0
    worst_permno = permno_runs.nlargest(10, "longest_missing_run")

    by_fqtr = (
        df.groupby("fqtr", dropna=False)[feature]
        .apply(lambda x: float(pd.to_numeric(x, errors="coerce").notna().mean()))
        .rename("coverage_rate")
        .reset_index()
    )

    return {
        "feature": feature,
        "overall_coverage": float(valid_mask.mean()),
        "usable_coverage": float(s[usable].notna().mean()) if usable.any() else np.nan,
        "non_missing_count": int(valid_mask.sum()),
        "usable_non_missing_count": int(s[usable].notna().sum()),
        "unique_permno_with_value": int(df.loc[valid_mask, "permno"].nunique()),
        "unique_gvkey_with_value": int(df.loc[valid_mask, "gvkey"].nunique()),
        "max_longest_missing_run_by_permno": max_run,
        "permno_ge8_with_coverage_lt50pct": int(len(poor_permno)),
        "coverage_by_fqtr": by_fqtr,
        "worst_missing_run_permno": worst_permno,
        "permno_coverage_stats": permno_stats,
    }


def distribution_summary(df: pd.DataFrame, feature: str) -> dict[str, Any]:
    s = pd.to_numeric(df[feature], errors="coerce")
    finite = s[np.isfinite(s)]
    inf_count = int(np.isinf(s).sum())
    if finite.empty:
        return {"feature": feature, "infinite_count": inf_count}

    q = finite.quantile([0.001, 0.01, 0.05, 0.25, 0.75, 0.95, 0.99, 0.999])
    return {
        "feature": feature,
        "count_non_missing": int(finite.count()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(finite.median()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p0_1": float(q.loc[0.001]),
        "p1": float(q.loc[0.01]),
        "p5": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "p99_9": float(q.loc[0.999]),
        "skewness": float(finite.skew()),
        "excess_kurtosis": float(finite.kurtosis()),
        "pct_equal_zero": float((finite == 0).mean()),
        "pct_negative": float((finite < 0).mean()),
        "infinite_count": inf_count,
    }


def build_outlier_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        s = pd.to_numeric(df[feature], errors="coerce")
        finite = s[np.isfinite(s)]
        if finite.empty:
            continue
        p001, p01, p99, p999 = finite.quantile([0.001, 0.01, 0.99, 0.999])
        meta = FEATURE_META[feature]
        num_col = meta["numerator"]
        den_col = meta["denominator"]
        flag_cols = meta["flags"]

        for label, mask in [
            ("outside_p1_p99", (s < p01) | (s > p99)),
            ("outside_p0_1_p99_9", (s < p001) | (s > p999)),
        ]:
            hit = df.loc[mask.fillna(False)].copy()
            if hit.empty:
                continue
            for rec in hit.itertuples(index=False):
                flags = []
                for fc in flag_cols:
                    val = getattr(rec, fc, np.nan)
                    if pd.notna(val) and bool(val):
                        flags.append(fc)
                rows.append(
                    {
                        "permno": int(rec.permno),
                        "gvkey": rec.gvkey,
                        "datadate": rec.datadate,
                        "signal_start_date": rec.signal_start_date,
                        "feature": feature,
                        "outlier_rule": label,
                        "feature_value": float(getattr(rec, feature)) if pd.notna(getattr(rec, feature)) else np.nan,
                        "numerator": getattr(rec, num_col, np.nan),
                        "denominator": getattr(rec, den_col, np.nan),
                        "relevant_flags": "|".join(flags) if flags else "",
                    }
                )
    out = pd.DataFrame(rows)
    if len(out):
        out["datadate"] = pd.to_datetime(out["datadate"]).dt.date
        out["signal_start_date"] = pd.to_datetime(out["signal_start_date"]).dt.date
    return out


def build_denominator_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for feature, meta in FEATURE_META.items():
        if not meta["ratio"]:
            continue
        den = pd.to_numeric(df[meta["denominator"]], errors="coerce")
        feat = pd.to_numeric(df[feature], errors="coerce")
        missing_den = den.isna()
        zero_den = den == 0
        neg_den = den < 0
        small_den = den.notna() & (den.abs() < 1e-6)
        nonpositive_be = pd.Series(False, index=df.index)
        if "book_equity_nonpositive_flag" in meta["flags"]:
            nonpositive_be = df["book_equity_nonpositive_flag"].astype(bool)

        excluded = feat.isna()
        rows.append(
            {
                "feature": feature,
                "numerator_field": meta["numerator"],
                "denominator_field": meta["denominator"],
                "missing_denominator_count": int(missing_den.sum()),
                "zero_denominator_count": int(zero_den.sum()),
                "negative_denominator_count": int(neg_den.sum()),
                "nonpositive_book_equity_count": int(nonpositive_be.sum()),
                "unusually_small_denominator_count": int(small_den.sum()),
                "feature_missing_count": int(excluded.sum()),
                "pct_excluded_by_denominator_rules": float(
                    ((missing_den | zero_den | nonpositive_be) & excluded).sum() / n
                ),
                "pct_feature_missing": float(excluded.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_temporal_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    for feature in FEATURES:
        annual_rows: list[dict[str, Any]] = []
        for year, grp in df.groupby("signal_year", dropna=False):
            s = pd.to_numeric(grp[feature], errors="coerce")
            finite = s[np.isfinite(s)]
            annual_rows.append(
                {
                    "feature": feature,
                    "signal_year": year,
                    "mean": float(finite.mean()) if len(finite) else np.nan,
                    "median": float(finite.median()) if len(finite) else np.nan,
                    "std": float(finite.std()) if len(finite) else np.nan,
                    "p1": float(finite.quantile(0.01)) if len(finite) else np.nan,
                    "p99": float(finite.quantile(0.99)) if len(finite) else np.nan,
                    "coverage": float(s.notna().mean()),
                    "n_rows": int(len(grp)),
                }
            )
        annual = pd.DataFrame(annual_rows).sort_values("signal_year")
        rows.extend(annual_rows)

        valid = annual.dropna(subset=["median", "coverage"])
        valid = valid[valid["coverage"] >= 0.50]
        for i in range(1, len(valid)):
            prev = valid.iloc[i - 1]
            cur = valid.iloc[i]
            if pd.isna(prev["median"]) or prev["median"] == 0:
                continue
            rel_change = abs(cur["median"] - prev["median"]) / abs(prev["median"])
            std_jump = abs(cur["median"] - prev["median"]) > 3 * (
                prev["std"] if pd.notna(prev["std"]) and prev["std"] > 0 else np.inf
            )
            cov_shift = abs(cur["coverage"] - prev["coverage"])
            if (rel_change > 0.75 and cov_shift > 0.10) or std_jump:
                flags.append(
                    {
                        "feature": feature,
                        "signal_year": cur["signal_year"],
                        "prev_year": prev["signal_year"],
                        "prev_mean": prev["mean"],
                        "cur_mean": cur["mean"],
                        "prev_median": prev["median"],
                        "cur_median": cur["median"],
                        "relative_median_change": rel_change,
                        "prev_coverage": prev["coverage"],
                        "cur_coverage": cur["coverage"],
                        "note": "Review structural break; not automatically treated as error.",
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(flags)


def pairwise_correlation(df: pd.DataFrame, method: str) -> pd.DataFrame:
    mat = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    return mat.corr(method=method, min_periods=100)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, f1 in enumerate(FEATURES):
        for f2 in FEATURES[i + 1 :]:
            val = corr.loc[f1, f2]
            if pd.notna(val) and abs(val) >= threshold:
                rows.append({"feature_1": f1, "feature_2": f2, "correlation": float(val)})
    if not rows:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
    return pd.DataFrame(rows).sort_values("correlation", key=abs, ascending=False)


def annual_correlation_summary(df: pd.DataFrame, method: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for year, grp in df.groupby("signal_year", dropna=False):
        mat = grp[FEATURES].apply(pd.to_numeric, errors="coerce")
        if mat.notna().all(axis=1).sum() < 50:
            continue
        corr = mat.corr(method=method, min_periods=50)
        pairs = high_correlation_pairs(corr, threshold=0.80)
        off_diag = corr.where(~np.eye(len(FEATURES), dtype=bool))
        max_abs = float(off_diag.abs().max().max()) if off_diag.notna().any().any() else np.nan
        summaries.append(
            {
                "signal_year": year,
                "method": method,
                "pairs_abs_ge_0_80": int(len(pairs)),
                "max_abs_correlation": max_abs,
            }
        )
    return summaries


def recommend_treatment(
    feature: str,
    coverage: dict[str, Any],
    dist: dict[str, Any],
    denom: pd.Series | None,
    temporal_flags: pd.DataFrame,
) -> dict[str, Any]:
    cov = coverage["overall_coverage"]
    usable_cov = coverage["usable_coverage"]
    p99 = dist.get("p99", np.nan)
    p1 = dist.get("p1", np.nan)
    std = dist.get("std", np.nan)
    tail_ratio = np.nan
    if pd.notna(p1) and pd.notna(p99) and abs(p1) > 1e-12:
        tail_ratio = abs(p99 / p1)

    denom_pct = float(denom["pct_excluded_by_denominator_rules"]) if denom is not None else np.nan
    break_count = int((temporal_flags["feature"] == feature).sum()) if len(temporal_flags) else 0
    break_note = (
        f" {break_count} annual mean shifts flagged for review."
        if break_count
        else ""
    )

    heavy_tails = (
        (pd.notna(tail_ratio) and tail_ratio > 20)
        or (pd.notna(std) and std > 5)
        or feature in VALUE_FEATURES
    )

    if cov < 0.85 or usable_cov < 0.85:
        classification = "Requires missing-value policy"
        rationale = f"Coverage below 85% (overall={pct(cov)}, usable={pct(usable_cov)}).{break_note}"
    elif pd.notna(denom_pct) and denom_pct > 0.10:
        classification = "Requires denominator filter"
        rationale = f"Denominator-related exclusions affect {pct(denom_pct)} of rows.{break_note}"
    elif heavy_tails:
        classification = "Ready after winsorisation"
        rationale = (
            "Heavy tails or market-cap ratio; winsorisation recommended before modelling."
            + break_note
        )
    else:
        classification = "Ready as-is"
        rationale = "Coverage and denominator diagnostics acceptable; tails moderate." + break_note

    return {
        "feature": feature,
        "recommended_classification": classification,
        "rationale": rationale,
        "overall_coverage": cov,
        "usable_coverage": usable_cov,
        "pct_excluded_by_denominator_rules": denom_pct,
        "temporal_break_flags": break_count,
    }


def compare_accounting_main_nextday(main: pd.DataFrame, nextday: pd.DataFrame) -> dict[str, Any]:
    key = RESEARCH_KEY
    merged = main[key + ACCOUNTING_FEATURES].merge(
        nextday[key + ACCOUNTING_FEATURES],
        on=key,
        suffixes=("_main", "_nextday"),
        how="inner",
    )
    mismatches: dict[str, int] = {}
    for feat in ACCOUNTING_FEATURES:
        fm = pd.to_numeric(merged[f"{feat}_main"], errors="coerce")
        fn = pd.to_numeric(merged[f"{feat}_nextday"], errors="coerce")
        both = fm.notna() & fn.notna()
        bad = both & ~np.isclose(fm, fn, rtol=0, atol=1e-12, equal_nan=False)
        mismatches[feat] = int(bad.sum())
    return {"matched_rows": int(len(merged)), "mismatches_by_feature": mismatches}


def validate_value_features(main: pd.DataFrame, nextday: pd.DataFrame, path: Path) -> pd.DataFrame:
    key = RESEARCH_KEY
    merged = main[key + VALUE_FEATURES + ["signal_start_date", "signal_market_cap"]].merge(
        nextday[key + VALUE_FEATURES + ["signal_start_date", "signal_market_cap"]],
        on=key,
        suffixes=("_main", "_nextday"),
        how="inner",
    )

    lines = [
        "# Main vs Next-Day Value Feature Validation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Accounting features are frozen identical across variants; this report covers ",
        "market-cap-dependent value ratios only.",
        "",
        f"Matched rows: **{len(merged):,}**",
        "",
        "## Pairwise comparison",
        "",
        "| Feature | Both non-missing | Pearson | Spearman | Mean abs diff | Median abs diff | p99 abs diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    annual_rows: list[dict[str, Any]] = []
    for feat in VALUE_FEATURES:
        fm = pd.to_numeric(merged[f"{feat}_main"], errors="coerce")
        fn = pd.to_numeric(merged[f"{feat}_nextday"], errors="coerce")
        both = fm.notna() & fn.notna()
        if both.any():
            diff = (fm[both] - fn[both]).abs()
            pearson = float(fm[both].corr(fn[both]))
            spearman = float(fm[both].corr(fn[both], method="spearman"))
            mean_abs = float(diff.mean())
            med_abs = float(diff.median())
            p99_abs = float(diff.quantile(0.99))
            both_n = int(both.sum())
        else:
            pearson = spearman = mean_abs = med_abs = p99_abs = np.nan
            both_n = 0
        lines.append(
            f"| `{feat}` | {both_n:,} | {pearson:.6f} | {spearman:.6f} | "
            f"{mean_abs:.6g} | {med_abs:.6g} | {p99_abs:.6g} |"
        )

        tmp = merged.loc[both].copy()
        tmp["signal_year"] = pd.to_datetime(tmp["signal_start_date_main"]).dt.year
        for year, grp in tmp.groupby("signal_year", dropna=False):
            a = pd.to_numeric(grp[f"{feat}_main"], errors="coerce")
            b = pd.to_numeric(grp[f"{feat}_nextday"], errors="coerce")
            d = (a - b).abs()
            annual_rows.append(
                {
                    "feature": feat,
                    "signal_year": year,
                    "both_non_missing": int(len(grp)),
                    "pearson": float(a.corr(b)) if len(grp) > 2 else np.nan,
                    "spearman": float(a.corr(b, method="spearman")) if len(grp) > 2 else np.nan,
                    "mean_abs_diff": float(d.mean()),
                    "median_abs_diff": float(d.median()),
                    "p99_abs_diff": float(d.quantile(0.99)) if len(d) else np.nan,
                }
            )

    lines.extend(["", "## Annual comparison", ""])
    annual_df = pd.DataFrame(annual_rows)
    if len(annual_df):
        lines.append(annual_df.to_markdown(index=False))
    else:
        lines.append("_No annual comparisons available._")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return annual_df


def build_coverage_by_year(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    usable = df["signal_usable_flag"].astype(bool)
    for feature in FEATURES:
        s = pd.to_numeric(df[feature], errors="coerce")
        for year, grp in df.groupby("signal_year", dropna=False):
            idx = grp.index
            ss = s.loc[idx]
            uu = usable.loc[idx]
            rows.append(
                {
                    "feature": feature,
                    "signal_year": year,
                    "n_rows": int(len(grp)),
                    "non_missing": int(ss.notna().sum()),
                    "coverage_rate": float(ss.notna().mean()),
                    "usable_n_rows": int(uu.sum()),
                    "usable_non_missing": int(ss[uu].notna().sum()),
                    "usable_coverage_rate": float(ss[uu].notna().mean()) if uu.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_validation_report(
    path: Path,
    main: pd.DataFrame,
    coverage_stats: list[dict[str, Any]],
    dist_df: pd.DataFrame,
    denom_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    temporal_flags: pd.DataFrame,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    high_pearson: pd.DataFrame,
    high_spearman: pd.DataFrame,
    annual_pearson: list[dict[str, Any]],
    annual_spearman: list[dict[str, Any]],
    accounting_check: dict[str, Any],
    treatment_df: pd.DataFrame,
) -> None:
    lines = [
        "# Quarterly Feature Validation Report (Phase 3B)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Primary dataset: `quarterly_fundamental_features/main/quarterly_fundamental_features.parquet`",
        "",
        f"Rows: **{len(main):,}** | Usable (`signal_usable_flag=True`): **{int(main['signal_usable_flag'].sum()):,}**",
        "",
        "Industry/sector breakdown: **not available** in frozen Phase 3A outputs (no industry ",
        "identifier preserved). Sector analysis deferred to daily-expansion stage if merged later.",
        "",
        "## 1. Coverage analysis",
        "",
        "| Feature | Overall cov | Usable cov | Unique PERMNO | Unique GVKEY | Max missing run | PERMNO cov<50% (≥8Q) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in coverage_stats:
        lines.append(
            f"| `{c['feature']}` | {pct(c['overall_coverage'])} | {pct(c['usable_coverage'])} | "
            f"{c['unique_permno_with_value']:,} | {c['unique_gvkey_with_value']:,} | "
            f"{c['max_longest_missing_run_by_permno']} | {c['permno_ge8_with_coverage_lt50pct']:,} |"
        )

    lines.extend(["", "### Coverage by fiscal quarter (`fqtr`)", ""])
    for c in coverage_stats:
        fq = c["coverage_by_fqtr"]
        fq_txt = ", ".join(f"Q{int(r.fqtr)}={pct(r.coverage_rate)}" for r in fq.itertuples() if pd.notna(r.fqtr))
        lines.append(f"- `{c['feature']}`: {fq_txt}")

    lines.extend(
        [
            "",
            "Detailed annual coverage: see `feature_coverage_by_year.csv`.",
            "",
            "## 2. Distribution analysis",
            "",
            "See `feature_distribution_summary.csv` for full percentiles, skewness, and kurtosis.",
            "",
            dist_df[
                ["feature", "mean", "std", "median", "p1", "p99", "pct_negative", "pct_equal_zero", "infinite_count"]
            ].to_markdown(index=False),
            "",
            "## 3. Outlier diagnostics",
            "",
            "Outlier rows exported to `feature_outlier_audit.csv` (rules: outside p1/p99 and p0.1/p99.9). ",
            "No observations were deleted.",
            "",
            "## 4. Denominator diagnostics",
            "",
            denom_df.to_markdown(index=False),
            "",
            "## 5. Temporal stability",
            "",
            "Annual moments exported via `feature_coverage_by_year.csv` and temporal detail below.",
            "",
        ]
    )
    if len(temporal_flags):
        lines.append("### Flagged year-over-year shifts (review only)")
        lines.append("")
        lines.append(temporal_flags.to_markdown(index=False))
    else:
        lines.append("_No large year-over-year mean shifts flagged under review thresholds._")

    lines.extend(
        [
            "",
            "## 6. Pairwise feature dependence",
            "",
            "### High-correlation pairs (|Pearson| ≥ 0.80)",
            "",
        ]
    )
    if len(high_pearson):
        lines.append(high_pearson.to_markdown(index=False))
    else:
        lines.append("_None._")

    lines.extend(["", "### High-correlation pairs (|Spearman| ≥ 0.80)", ""])
    if len(high_spearman):
        lines.append(high_spearman.to_markdown(index=False))
    else:
        lines.append("_None._")

    lines.extend(["", "### Annual correlation summaries", ""])
    lines.append(pd.DataFrame(annual_pearson).to_markdown(index=False))
    lines.append("")
    lines.append(pd.DataFrame(annual_spearman).to_markdown(index=False))

    lines.extend(
        [
            "",
            "## 7. Main vs next-day (accounting features)",
            "",
            f"Matched keys: {accounting_check['matched_rows']:,}",
            "",
            "| Feature | Non-identical (both non-missing) |",
            "| --- | ---: |",
        ]
    )
    for feat, n in accounting_check["mismatches_by_feature"].items():
        lines.append(f"| `{feat}` | {n:,} |")

    lines.extend(
        [
            "",
            "Value-feature timing comparison: see `main_vs_nextday_value_feature_validation.md`.",
            "",
            "## 8. Recommended downstream treatment (not executed)",
            "",
            treatment_df.to_markdown(index=False),
            "",
            "Pearson matrix: `feature_pearson_correlation.csv`",
            "Spearman matrix: `feature_spearman_correlation.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MAIN_PATH.exists() or not NEXTDAY_PATH.exists():
        raise FileNotFoundError("Frozen Phase 3A feature parquet files are required.")

    log.info("Loading main and next-day feature datasets")
    main = load_features(MAIN_PATH)
    nextday = load_features(NEXTDAY_PATH)

    log.info("Coverage analysis")
    coverage_stats = [coverage_summary(main, f) for f in FEATURES]
    coverage_by_year = build_coverage_by_year(main)
    coverage_by_year.to_csv(OUT_DIR / "feature_coverage_by_year.csv", index=False)

    log.info("Distribution analysis")
    dist_rows = [distribution_summary(main, f) for f in FEATURES]
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUT_DIR / "feature_distribution_summary.csv", index=False)

    log.info("Outlier audit")
    outlier_audit = build_outlier_audit(main)
    outlier_audit.to_csv(OUT_DIR / "feature_outlier_audit.csv", index=False)

    log.info("Denominator audit")
    denom_df = build_denominator_audit(main)
    denom_df.to_csv(OUT_DIR / "feature_denominator_audit.csv", index=False)

    log.info("Temporal stability")
    temporal_df, temporal_flags = build_temporal_summary(main)

    log.info("Pairwise correlations")
    pearson = pairwise_correlation(main, "pearson")
    spearman = pairwise_correlation(main, "spearman")
    pearson.to_csv(OUT_DIR / "feature_pearson_correlation.csv")
    spearman.to_csv(OUT_DIR / "feature_spearman_correlation.csv")
    high_pearson = high_correlation_pairs(pearson)
    high_spearman = high_correlation_pairs(spearman)
    annual_pearson = annual_correlation_summary(main, "pearson")
    annual_spearman = annual_correlation_summary(main, "spearman")

    log.info("Main vs next-day checks")
    accounting_check = compare_accounting_main_nextday(main, nextday)
    validate_value_features(main, nextday, OUT_DIR / "main_vs_nextday_value_feature_validation.md")

    log.info("Treatment recommendations")
    denom_lookup = denom_df.set_index("feature")
    treatment_rows = [
        recommend_treatment(
            f,
            next(c for c in coverage_stats if c["feature"] == f),
            next(d for d in dist_rows if d["feature"] == f),
            denom_lookup.loc[f] if f in denom_lookup.index else None,
            temporal_flags,
        )
        for f in FEATURES
    ]
    treatment_df = pd.DataFrame(treatment_rows)
    treatment_df.to_csv(OUT_DIR / "feature_treatment_recommendations.csv", index=False)

    write_validation_report(
        OUT_DIR / "quarterly_feature_validation_report.md",
        main,
        coverage_stats,
        dist_df,
        denom_df,
        temporal_df,
        temporal_flags,
        pearson,
        spearman,
        high_pearson,
        high_spearman,
        annual_pearson,
        annual_spearman,
        accounting_check,
        treatment_df,
    )

    meta = {
        "phase": "3B",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "main_input": str(MAIN_PATH),
        "nextday_input": str(NEXTDAY_PATH),
        "row_count_main": int(len(main)),
        "row_count_nextday": int(len(nextday)),
        "features": FEATURES,
        "outputs": [
            "quarterly_feature_validation_report.md",
            "feature_coverage_by_year.csv",
            "feature_distribution_summary.csv",
            "feature_outlier_audit.csv",
            "feature_denominator_audit.csv",
            "feature_pearson_correlation.csv",
            "feature_spearman_correlation.csv",
            "feature_treatment_recommendations.csv",
            "main_vs_nextday_value_feature_validation.md",
            "validation_meta.json",
        ],
        "industry_breakdown_available": False,
        "accounting_main_nextday_check": accounting_check,
        "high_correlation_pairs_pearson": high_pearson.to_dict(orient="records"),
        "high_correlation_pairs_spearman": high_spearman.to_dict(orient="records"),
        "temporal_break_flag_count": int(len(temporal_flags)),
    }
    (OUT_DIR / "validation_meta.json").write_text(
        json.dumps(meta, indent=2, default=json_safe),
        encoding="utf-8",
    )

    log.info("Phase 3B validation complete: %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
