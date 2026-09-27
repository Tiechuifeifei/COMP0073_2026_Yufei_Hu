#!/usr/bin/env python3
"""10_build_quarterly_fundamental_features.py

Construct quarterly fundamental characteristics at the signal-event level from
Phase 2 outputs. Same accounting formulas for the main and next-day timing
variants."""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MAIN_INPUT = PROJECT_ROOT / "data" / "quarterly_signal_events" / "quarterly_signal_events.parquet"
NEXTDAY_INPUT = (
    PROJECT_ROOT / "data" / "quarterly_signal_events_nextday" / "quarterly_signal_events.parquet"
)
OUT_ROOT = PROJECT_ROOT / "data" / "quarterly_fundamental_features"
CACHE_DIR = OUT_ROOT / "cache"
MEMBERSHIP_PATH = SCRIPT_DIR / "data" / "sp500_pit_master" / "membership_spells.parquet"
ACTQ_LCTQ_CACHE = CACHE_DIR / "pit_actq_lctq_supplement.parquet"
ACTQ_LCTQ_META = CACHE_DIR / "pit_actq_lctq_supplement.meta.json"

RESEARCH_KEY = ["permno", "gvkey", "datadate", "fqtr"]
COMPUSTAT_TO_USD = 1_000_000.0
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

PRESERVE_COLS = [
    "permno",
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
    "history_only_flag",
    "signal_market_cap",
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
    raise TypeError(type(obj))


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.2f}%"


def zgvkey(s: pd.Series | str) -> pd.Series | str:
    if isinstance(s, str):
        return str(s).strip().zfill(6)
    return s.astype(str).str.strip().str.zfill(6)


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["gvkey"] = zgvkey(df["gvkey"])
    df["permno"] = df["permno"].astype(int)
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["accounting_available_date"] = pd.to_datetime(df["accounting_available_date"])
    df["signal_start_date"] = pd.to_datetime(df["signal_start_date"])
    return df


def attach_membership(df: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    membership = membership.copy()
    membership["membership_start"] = pd.to_datetime(membership["membership_start"])
    membership["membership_end"] = pd.to_datetime(membership["membership_end"])

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for row in df.itertuples(index=False):
        ref = row.signal_start_date if pd.notna(row.signal_start_date) else row.accounting_available_date
        spells = membership[membership["permno"] == int(row.permno)]
        hit = spells[(spells["membership_start"] <= ref) & (spells["membership_end"] >= ref)]
        if len(hit):
            starts.append(hit.iloc[0]["membership_start"])
            ends.append(hit.iloc[0]["membership_end"])
        elif len(spells):
            starts.append(spells.iloc[0]["membership_start"])
            ends.append(spells.iloc[0]["membership_end"])
        else:
            starts.append(pd.NaT)
            ends.append(pd.NaT)
    out = df.copy()
    out["membership_start"] = starts
    out["membership_end"] = ends
    return out


def gvkey_sql_list(gvkeys: list[str]) -> str:
    return ", ".join(f"'{g}'" for g in sorted(gvkeys))


def fetch_actq_lctq_supplement(gvkeys: list[str]) -> pd.DataFrame:
    if ACTQ_LCTQ_CACHE.exists() and ACTQ_LCTQ_META.exists():
        meta = json.loads(ACTQ_LCTQ_META.read_text(encoding="utf-8"))
        if set(meta.get("gvkeys", [])) >= set(gvkeys):
            log.info("actq/lctq supplement cache hit")
            sub = pd.read_parquet(ACTQ_LCTQ_CACHE)
            sub["gvkey"] = zgvkey(sub["gvkey"])
            sub["datadate"] = pd.to_datetime(sub["datadate"])
            return sub

    from wrds_utils import connect_wrds

    sql = f"""
        SELECT gvkey, datadate, fqtr, actq, lctq
        FROM comp_pit.pitqtrdataus
        WHERE gvkey IN ({gvkey_sql_list(gvkeys)})
    """
    log.info("WRDS supplement query for actq/lctq (%d gvkeys)", len(gvkeys))
    db = connect_wrds()
    raw = db.raw_sql(sql, date_cols=["datadate"])
    raw["gvkey"] = zgvkey(raw["gvkey"])
    raw["datadate"] = pd.to_datetime(raw["datadate"])
    raw.to_parquet(ACTQ_LCTQ_CACHE, index=False)
    ACTQ_LCTQ_META.write_text(
        json.dumps(
            {
                "sql": sql.strip(),
                "gvkeys": sorted(gvkeys),
                "rows": int(len(raw)),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw


def merge_actq_lctq(df: pd.DataFrame) -> pd.DataFrame:
    gvkeys = sorted(df["gvkey"].unique().tolist())
    try:
        supplement = fetch_actq_lctq_supplement(gvkeys)
    except Exception as exc:
        log.warning("actq/lctq supplement unavailable (%s); current_ratio will be missing", exc)
        out = df.copy()
        out["actq"] = np.nan
        out["lctq"] = np.nan
        out["actq_lctq_missing_flag"] = True
        return out

    merged = df.merge(
        supplement[["gvkey", "datadate", "fqtr", "actq", "lctq"]],
        on=["gvkey", "datadate", "fqtr"],
        how="left",
    )
    merged["actq_lctq_missing_flag"] = merged["actq"].isna() | merged["lctq"].isna()
    return merged


def compute_book_equity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["book_equity_raw_ceqq"] = pd.to_numeric(out["ceqq"], errors="coerce")
    txdit = pd.to_numeric(out["txditcq"], errors="coerce").fillna(0.0)
    pstk = pd.to_numeric(out["pstkq"], errors="coerce").fillna(0.0)
    seqq = pd.to_numeric(out["seqq"], errors="coerce")

    out["book_equity"] = out["book_equity_raw_ceqq"]
    out["book_equity_definition"] = np.where(
        out["book_equity_raw_ceqq"].notna(), "ceqq", np.nan
    )

    missing_ceq = out["book_equity"].isna()
    ff_be = seqq + txdit - pstk
    out.loc[missing_ceq, "book_equity"] = ff_be[missing_ceq]
    out.loc[missing_ceq, "book_equity_definition"] = "seqq_plus_txditcq_minus_pstk"

    still_missing = out["book_equity"].isna()
    out.loc[still_missing, "book_equity_definition"] = "missing"

    out["book_equity_missing_flag"] = out["book_equity"].isna()
    out["book_equity_nonpositive_flag"] = out["book_equity"].notna() & (out["book_equity"] <= 0)
    return out


def parse_datacqtr(value: Any) -> tuple[int | None, int | None]:
    """Parse Compustat `datacqtr` (e.g. Q3Y06) into fiscal-year code and fiscal quarter."""
    if pd.isna(value):
        return None, None
    match = re.match(r"^Q([1-4])Y(\d+)$", str(value).strip().upper())
    if not match:
        return None, None
    return int(match.group(2)), int(match.group(1))


def _rolling_all_valid(s: pd.Series, window: int = 4) -> pd.Series:
    return (
        s.rolling(window, min_periods=window)
        .apply(lambda x: np.all(~np.isnan(x)), raw=True)
        .fillna(False)
        .astype(bool)
    )


def assign_standalone_oancfq(g: pd.DataFrame) -> pd.DataFrame:
    """Convert fiscal-YTD oancfq to standalone quarterly OCF within each fiscal year."""
    out = g.sort_values(["datadate", "fqtr"]).copy()
    out["standalone_oancf_q"] = np.nan
    out["oancf_ytd_conversion_flag"] = False
    out["oancf_missing_previous_quarter_flag"] = False
    out["oancf_nonconsecutive_quarter_flag"] = False

    out["oancfq"] = pd.to_numeric(out["oancfq"], errors="coerce")
    fy_codes: list[int | None] = []
    fiscal_qs: list[int | None] = []
    for val in out["datacqtr"]:
        fy, fq = parse_datacqtr(val)
        fy_codes.append(fy)
        fiscal_qs.append(fq)
    out["fiscal_year_code"] = fy_codes
    out["fiscal_quarter_num"] = fiscal_qs

    valid_fq = out["fiscal_year_code"].notna() & out["fiscal_quarter_num"].notna()
    dup_mask = out.duplicated(subset=["fiscal_year_code", "fiscal_quarter_num"], keep=False) & valid_fq
    out.loc[dup_mask, "oancf_nonconsecutive_quarter_flag"] = True

    conflict_idx: set[Any] = set()
    for _, grp in out.loc[valid_fq].groupby(["fiscal_year_code", "fiscal_quarter_num"], sort=False):
        if len(grp) > 1 and grp["oancfq"].nunique(dropna=True) > 1:
            conflict_idx.update(grp.index.tolist())
    if conflict_idx:
        out.loc[list(conflict_idx), "oancf_nonconsecutive_quarter_flag"] = True

    canonical = (
        out.loc[valid_fq & out["oancfq"].notna()]
        .drop_duplicates(subset=["fiscal_year_code", "fiscal_quarter_num"], keep="last")
        .sort_values(["fiscal_year_code", "fiscal_quarter_num"])
    )
    ytd_map: dict[tuple[int, int], float] = {}
    for row in canonical.itertuples(index=False):
        ytd_map[(int(row.fiscal_year_code), int(row.fiscal_quarter_num))] = float(row.oancfq)

    standalone_map: dict[tuple[int, int], float] = {}
    for (fy, fq), ytd in sorted(ytd_map.items(), key=lambda item: (item[0][0], item[0][1])):
        if fq == 1:
            standalone_map[(fy, fq)] = ytd
            continue
        prev_key = (fy, fq - 1)
        if prev_key not in ytd_map:
            continue
        standalone_map[(fy, fq)] = ytd - ytd_map[prev_key]

    for idx, row in out.iterrows():
        if pd.isna(row.oancfq) or not valid_fq.loc[idx]:
            continue
        if idx in conflict_idx:
            continue
        key = (int(row.fiscal_year_code), int(row.fiscal_quarter_num))
        if key in standalone_map:
            out.loc[idx, "standalone_oancf_q"] = standalone_map[key]
            out.loc[idx, "oancf_ytd_conversion_flag"] = True
        elif int(row.fiscal_quarter_num) > 1:
            out.loc[idx, "oancf_missing_previous_quarter_flag"] = True

    return out


def add_gvkey_lags_and_ttm(df: pd.DataFrame) -> pd.DataFrame:
    """Chronological gvkey-level lags and TTM sums; no cross-gvkey leakage."""
    parts: list[pd.DataFrame] = []
    for _, grp in df.groupby("gvkey", sort=False):
        g = grp.sort_values(["datadate", "fqtr"]).copy()
        g = assign_standalone_oancfq(g)

        g["gpq"] = pd.to_numeric(g["saleq"], errors="coerce") - pd.to_numeric(
            g["cogsq"], errors="coerce"
        )
        g["operating_profit_q"] = pd.to_numeric(g["oibdpq"], errors="coerce") - pd.to_numeric(
            g["dpq"], errors="coerce"
        )

        g["ttm_niq"] = g["niq"].rolling(4, min_periods=4).sum()
        g["ttm_saleq"] = g["saleq"].rolling(4, min_periods=4).sum()
        g["ttm_gpq"] = g["gpq"].rolling(4, min_periods=4).sum()
        g["ttm_operating_profit"] = g["operating_profit_q"].rolling(4, min_periods=4).sum()

        # Legacy incorrect raw-YTD rolling sum (retained for audit).
        g["ttm_oancfq"] = g["oancfq"].rolling(4, min_periods=4).sum()

        g["ttm_oancf_valid_4q_flag"] = _rolling_all_valid(g["standalone_oancf_q"], 4)
        g["ttm_oancf"] = g["standalone_oancf_q"].rolling(4, min_periods=4).sum()
        g.loc[~g["ttm_oancf_valid_4q_flag"], "ttm_oancf"] = np.nan

        g["atq"] = pd.to_numeric(g["atq"], errors="coerce")
        g["atq_lag1"] = g["atq"].shift(1)
        g["atq_lag4"] = g["atq"].shift(4)
        g["avg_atq"] = (g["atq"] + g["atq_lag1"]) / 2.0
        g["avg_book_equity"] = (g["book_equity"] + g["book_equity"].shift(1)) / 2.0
        g["ttm_saleq_lag4"] = g["ttm_saleq"].shift(4)
        g["total_debt"] = pd.to_numeric(g["dlcq"], errors="coerce") + pd.to_numeric(
            g["dlttq"], errors="coerce"
        )
        parts.append(g)

    return pd.concat(parts, ignore_index=True)


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = pd.to_numeric(den, errors="coerce")
    num = pd.to_numeric(num, errors="coerce")
    out = num / den
    out = out.where(den.notna() & (den != 0))
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    valid_be = out["book_equity"].notna() & (out["book_equity"] > 0)
    valid_avg_be = out["avg_book_equity"].notna() & (out["avg_book_equity"] > 0)
    valid_mcap = out["signal_market_cap"].notna() & (out["signal_market_cap"] > 0)
    valid_lctq = out["lctq"].notna() & (out["lctq"] != 0)

    out["roe"] = np.nan
    out.loc[valid_avg_be, "roe"] = safe_div(
        out.loc[valid_avg_be, "ttm_niq"], out.loc[valid_avg_be, "avg_book_equity"]
    ).astype(float)

    out["roa"] = safe_div(out["ttm_niq"], out["avg_atq"]).astype(float)
    out["gross_profitability"] = safe_div(out["ttm_gpq"], out["avg_atq"]).astype(float)
    out["operating_profitability"] = np.nan
    out.loc[valid_avg_be, "operating_profitability"] = safe_div(
        out.loc[valid_avg_be, "ttm_operating_profit"], out.loc[valid_avg_be, "avg_book_equity"]
    ).astype(float)

    out["sales_growth_yoy"] = (safe_div(out["ttm_saleq"], out["ttm_saleq_lag4"]) - 1.0).astype(float)
    out["asset_growth_yoy"] = (safe_div(out["atq"], out["atq_lag4"]) - 1.0).astype(float)

    out["accruals_old_raw_oancfq"] = safe_div(
        out["ttm_niq"] - out["ttm_oancfq"], out["avg_atq"]
    ).astype(float)
    out["accruals_numerator"] = out["ttm_niq"] - out["ttm_oancf"]
    out["accruals_denominator"] = out["avg_atq"]
    out["accruals_definition"] = "sloan_ttm_ni_minus_standalone_oancf_over_avg_at"
    out["accruals"] = safe_div(out["accruals_numerator"], out["accruals_denominator"]).astype(float)

    out["leverage"] = safe_div(out["total_debt"], out["atq"]).astype(float)
    out["current_ratio"] = np.nan
    cr_mask = ~out["actq_lctq_missing_flag"] & valid_lctq
    out.loc[cr_mask, "current_ratio"] = safe_div(out.loc[cr_mask, "actq"], out.loc[cr_mask, "lctq"]).astype(float)

    out["book_to_market"] = np.nan
    bm_mask = valid_be & valid_mcap
    out.loc[bm_mask, "book_to_market"] = safe_div(
        out.loc[bm_mask, "book_equity"] * COMPUSTAT_TO_USD,
        out.loc[bm_mask, "signal_market_cap"],
    ).astype(float)
    out["earnings_yield"] = safe_div(
        out["ttm_niq"] * COMPUSTAT_TO_USD, out["signal_market_cap"]
    ).astype(float)
    out["sales_to_price"] = safe_div(
        out["ttm_saleq"] * COMPUSTAT_TO_USD, out["signal_market_cap"]
    ).astype(float)

    for feat in FEATURES:
        if feat in out.columns:
            out[feat] = pd.to_numeric(out[feat], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def profile_feature(series: pd.Series, name: str, df: pd.DataFrame) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    usable = df["signal_usable_flag"].astype(bool)
    row: dict[str, Any] = {
        "feature": name,
        "non_missing_count": int(s.notna().sum()),
        "coverage_rate": float(s.notna().mean()),
        "usable_non_missing_count": int(s[usable].notna().sum()),
        "usable_coverage_rate": float(s[usable].notna().mean()) if usable.any() else np.nan,
        "mean": float(finite.mean()) if len(finite) else np.nan,
        "std": float(finite.std()) if len(finite) else np.nan,
        "median": float(finite.median()) if len(finite) else np.nan,
        "min": float(finite.min()) if len(finite) else np.nan,
        "p1": float(finite.quantile(0.01)) if len(finite) else np.nan,
        "p5": float(finite.quantile(0.05)) if len(finite) else np.nan,
        "p95": float(finite.quantile(0.95)) if len(finite) else np.nan,
        "p99": float(finite.quantile(0.99)) if len(finite) else np.nan,
        "max": float(finite.max()) if len(finite) else np.nan,
        "infinite_count": int(np.isinf(s).sum()),
    }
    return row


def build_feature_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = [profile_feature(df[f], f, df) for f in FEATURES]
    return pd.DataFrame(rows)


def build_exclusion_detail(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(feature: str, reason: str, mask: pd.Series) -> None:
        rows.append({"feature": feature, "reason": reason, "count": int(mask.sum())})

    add("roe", "avg_book_equity_missing_or_nonpositive", df["avg_book_equity"].isna() | (df["avg_book_equity"] <= 0))
    add("roa", "avg_atq_missing_or_nonpositive", df["avg_atq"].isna() | (df["avg_atq"] <= 0))
    add("gross_profitability", "avg_atq_missing_or_nonpositive", df["avg_atq"].isna() | (df["avg_atq"] <= 0))
    add(
        "operating_profitability",
        "avg_book_equity_missing_or_nonpositive",
        df["avg_book_equity"].isna() | (df["avg_book_equity"] <= 0),
    )
    add("sales_growth_yoy", "ttm_or_lag4_ttm_sales_missing", df["ttm_saleq"].isna() | df["ttm_saleq_lag4"].isna())
    add("sales_growth_yoy", "ttm_saleq_lag4_zero", df["ttm_saleq_lag4"] == 0)
    add("asset_growth_yoy", "atq_or_lag4_atq_missing", df["atq"].isna() | df["atq_lag4"].isna())
    add("asset_growth_yoy", "atq_lag4_zero", df["atq_lag4"] == 0)
    add("accruals", "ttm_niq_missing", df["ttm_niq"].isna())
    add("accruals", "ttm_oancf_missing", df["ttm_oancf"].isna())
    add("accruals", "ttm_oancf_not_valid_4q", ~df["ttm_oancf_valid_4q_flag"].astype(bool))
    add("accruals", "standalone_oancf_not_converted", ~df["oancf_ytd_conversion_flag"].astype(bool))
    add("accruals", "avg_atq_missing_or_zero", df["avg_atq"].isna() | (df["avg_atq"] == 0))
    add("leverage", "atq_missing_or_zero", df["atq"].isna() | (df["atq"] == 0))
    add("current_ratio", "actq_or_lctq_missing", df["actq_lctq_missing_flag"].astype(bool))
    add("current_ratio", "lctq_zero", df["lctq"] == 0)
    add("book_to_market", "book_equity_missing", df["book_equity_missing_flag"].astype(bool))
    add("book_to_market", "book_equity_nonpositive", df["book_equity_nonpositive_flag"].astype(bool))
    add("book_to_market", "signal_market_cap_missing_or_nonpositive", df["signal_market_cap"].isna() | (df["signal_market_cap"] <= 0))
    add("earnings_yield", "ttm_niq_missing", df["ttm_niq"].isna())
    add("earnings_yield", "signal_market_cap_missing_or_nonpositive", df["signal_market_cap"].isna() | (df["signal_market_cap"] <= 0))
    add("sales_to_price", "ttm_saleq_missing", df["ttm_saleq"].isna())
    add("sales_to_price", "signal_market_cap_missing_or_nonpositive", df["signal_market_cap"].isna() | (df["signal_market_cap"] <= 0))
    return pd.DataFrame(rows)


def coverage_by_year(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["signal_year"] = tmp["signal_start_date"].dt.year
    return (
        tmp.groupby("signal_year")
        .agg(
            n_rows=("permno", "size"),
            non_missing=(feature, lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        )
        .assign(coverage_rate=lambda x: x["non_missing"] / x["n_rows"])
        .reset_index()
    )


def sparse_gvkey_histories(df: pd.DataFrame, feature: str, min_quarters: int = 8) -> pd.DataFrame:
    usable = df[df["signal_usable_flag"].astype(bool)].copy()
    counts = usable.groupby("gvkey")[feature].apply(lambda s: pd.to_numeric(s, errors="coerce").notna().sum())
    sparse = counts[counts <= min_quarters].reset_index(name="non_missing_quarters")
    sparse["feature"] = feature
    return sparse.sort_values("non_missing_quarters")


def write_feature_quality_report(
    path: Path,
    variant: str,
    df: pd.DataFrame,
    profile: pd.DataFrame,
    exclusion: pd.DataFrame,
) -> None:
    lines = [
        f"# Quarterly Fundamental Feature Quality Report — {variant}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Rows: **{len(df):,}**",
        f"- Usable signal events: **{int(df['signal_usable_flag'].sum()):,}**",
        f"- Formula dictionary: `../feature_formula_dictionary.md`",
        "",
        "## Feature profile summary",
        "",
        profile.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Exclusion reasons",
        "",
        exclusion.to_markdown(index=False),
        "",
    ]
    for feat in FEATURES[:4]:
        yr = coverage_by_year(df, feat)
        lines.extend([f"### `{feat}` coverage by signal year", "", yr.to_markdown(index=False), ""])

    sparse = pd.concat([sparse_gvkey_histories(df, f) for f in FEATURES], ignore_index=True)
    if len(sparse):
        lines.extend(
            [
                "## Sparse usable histories (≤8 non-missing quarters per GVKEY)",
                "",
                sparse.head(40).to_markdown(index=False),
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_formula_dictionary(path: Path, oancf_coverage: float, actq_coverage: float) -> None:
    text = f"""# Quarterly Fundamental Feature Formula Dictionary

Generated: {datetime.now(timezone.utc).isoformat()}

Shared across **main** (`date >= accounting_available_date`) and **next-day** (`date > accounting_available_date`) variants. Only `signal_start_date` and market-cap-based value ratios may differ across variants.

## General construction rules

- PIT accounting fields from frozen `quarterly_signal_events` (Compustat `comp_pit.pitqtrdataus` unrestated columns as stored in Phase 2).
- All lags and TTM sums are computed **within `gvkey` chronological order** (`datadate`, `fqtr`); no cross-GVKEY successor lags.
- TTM = sum of the current and prior three quarterly observations (4 quarters, `min_periods=4`).
- Average balance-sheet denominators = `(current quarter + lag-1 quarter) / 2` within the same `gvkey`.
- YoY growth uses **four-quarter lags** on TTM or point-in-time balance-sheet values.
- Missing values are **not imputed**; divide-by-zero and invalid denominators yield missing features with explicit exclusion reasons.
- Value ratios use **`signal_market_cap` at `signal_start_date`**, not `datadate` market cap.
- **Units:** Compustat stock/flow fields are in **millions of USD**; CRSP `signal_market_cap` is in **USD**. Value-ratio numerators multiply Compustat quantities by **1,000,000** before division.

## Book equity

| Step | Definition |
| --- | --- |
| Primary | `book_equity = ceqq` |
| Fallback (only if `ceqq` missing) | `book_equity = seqq + txditcq - pstkq`, with missing `txditcq`/`pstkq` treated as 0 |
| Flags | `book_equity_missing_flag`, `book_equity_nonpositive_flag`, `book_equity_definition` |

Reference: Fama-French book equity components (CRSP/Compustat Merged Database documentation).

**Deviation:** Non-positive book equity is flagged and **not** silently adjusted; BE-dependent ratios are set to missing when `book_equity <= 0`.

## Profitability

### ROE (`roe`)
- **Formula:** `TTM(niq) / avg_book_equity`
- **Fields:** `niq`; book equity as above
- **Type:** TTM flow / average stock
- **Denominator:** missing or `<= 0` → missing
- **Missing:** requires 4 quarters for TTM and lagged BE
- **Sign:** higher = more profitable
- **Reference:** Ball et al.; standard profitability literature

### ROA (`roa`)
- **Formula:** `TTM(niq) / avg_atq`
- **Fields:** `niq`, `atq`
- **Type:** TTM / average balance sheet
- **Denominator:** missing or zero → missing
- **Sign:** higher = more profitable

### Gross Profitability (`gross_profitability`)
- **Formula:** `TTM(saleq - cogsq) / avg_atq`
- **Fields:** `saleq`, `cogsq`, `atq`
- **Type:** TTM gross profit / average assets
- **Sign:** higher = more profitable
- **Reference:** Novy-Marx (2013, JFE)
- **Deviation:** uses `saleq - cogsq` because `revtq` is absent from `pitqtrdataus`; TTM/average-asset scaling follows project Table 1 rather than Novy-Marx’s single-quarter GP/AT.

### Operating Profitability (`operating_profitability`)
- **Formula:** `TTM(oibdpq - dpq) / avg_book_equity`
- **Fields:** `oibdpq`, `dpq`, book equity
- **Type:** TTM operating profit / average book equity
- **Sign:** higher = more profitable
- **Reference:** project Table 1 (营业盈利能力)

## Growth / Investment

### Sales Growth YoY (`sales_growth_yoy`)
- **Formula:** `TTM(saleq) / lag4(TTM(saleq)) - 1`
- **Fields:** `saleq`
- **Type:** YoY on TTM sales
- **Denominator:** zero lagged TTM → missing
- **Sign:** higher = faster growth

### Asset Growth YoY (`asset_growth_yoy`)
- **Formula:** `atq / lag4(atq) - 1`
- **Fields:** `atq`
- **Type:** YoY point-in-time total assets
- **Sign:** higher = faster asset growth
- **Reference:** Cooper, Gulen & Schill (2008) asset growth anomaly literature

## Financial quality

### Accruals (`accruals`)

**Important:** In `comp_pit.pitqtrdataus`, `oancfq` is a **fiscal-year-to-date (YTD) cumulative**
operating-cash-flow variable, **not** a standalone calendar-quarter flow. It must be converted
to standalone quarterly OCF before TTM aggregation.

#### Step 1 — Standalone quarterly OCF (`standalone_oancf_q`)

`oancfq` is fiscal-YTD cumulative within each Compustat fiscal year. Continuity is
checked with **`datacqtr`** (`QxYyy`, e.g. `Q3Y06`), which encodes fiscal quarters
1–4 within a fiscal year. (`fyrq` is the fiscal year-end **month**, not a fiscal year ID;
`fqtr`/`qtryr` alone do not always form a clean Q1→Q4 chain for non-December filers.)

| Case | Rule |
|------|------|
| Fiscal Q1 (`QxYyy` where x=1) | `standalone_oancf_q = oancfq` |
| Fiscal Q2–Q4 | `standalone_oancf_q = oancfq - oancfq(previous fiscal quarter in same Yyy)` |
| New fiscal year (Q1 after prior-year Q4) | **Do not** subtract prior-year Q4 YTD |
| Missing / skipped prior fiscal quarter | `standalone_oancf_q = missing` |
| Duplicate or conflicting `(datacqtr)` rows | `standalone_oancf_q = missing` (flagged) |

Audit flags: `oancf_ytd_conversion_flag`, `oancf_missing_previous_quarter_flag`,
`oancf_nonconsecutive_quarter_flag`.

#### Step 2 — TTM operating cash flow

```
ttm_oancf = rolling sum of four valid standalone_oancf_q observations
             (chronological within gvkey; requires all four non-missing)
```

Flag: `ttm_oancf_valid_4q_flag`.

#### Step 3 — Accruals

```
accruals = (TTM(niq) - ttm_oancf) / avg_atq
```

Legacy incorrect version retained as `accruals_old_raw_oancfq` (direct rolling sum of raw YTD `oancfq`).

- **Fields:** `niq`, `oancfq` (YTD, converted), `atq`
- **Coverage:** raw `oancfq` non-missing ≈ **{oancf_coverage:.2%}**
- **Sign:** higher = more accruals (lower cash-flow quality) per Sloan (1996)
- **Reference:** Sloan (1996, The Accounting Review)

### Leverage (`leverage`)
- **Formula:** `(dlcq + dlttq) / atq`
- **Fields:** `dlcq`, `dlttq`, `atq`
- **Type:** point-in-time debt-to-assets
- **Sign:** higher = more levered
- **Reference:** project Table 1 (资产负债率)

### Current Ratio (`current_ratio`)
- **Formula:** `actq / lctq`
- **Fields:** `actq`, `lctq` merged from **`comp_pit.pitqtrdataus` Phase 3A supplement** (not in frozen Phase 2 accounting extract)
- **Type:** point-in-time balance sheet
- **Coverage:** matched `actq/lctq` rate ≈ **{actq_coverage:.2%}**
- **Denominator:** `lctq = 0` → missing
- **Sign:** higher = more liquid
- **Deviation:** requires supplemental WRDS read documented in `cache/pit_actq_lctq_supplement.parquet`.

## Value (signal-date market cap)

### Book-to-Market (`book_to_market`)
- **Formula:** `(book_equity × 1,000,000) / signal_market_cap`
- **Fields:** book equity; `signal_market_cap`
- **Type:** PIT stock / signal-date ME
- **Sign:** higher = cheaper (value)
- **Reference:** Fama & French (1992)

### Earnings Yield (`earnings_yield`)
- **Formula:** `(TTM(niq) × 1,000,000) / signal_market_cap`
- **Fields:** `niq`, `signal_market_cap`
- **Type:** TTM flow / signal-date ME
- **Sign:** higher = cheaper / higher yield

### Sales-to-Price (`sales_to_price`)
- **Formula:** `(TTM(saleq) × 1,000,000) / signal_market_cap`
- **Fields:** `saleq`, `signal_market_cap`
- **Type:** TTM flow / signal-date ME
- **Sign:** higher = more sales per dollar of market value
"""
    path.write_text(text, encoding="utf-8")


def build_features(input_path: Path, variant: str) -> pd.DataFrame:
    log.info("Building features for variant=%s from %s", variant, input_path)
    membership = pd.read_parquet(MEMBERSHIP_PATH)
    df = load_events(input_path)
    df = attach_membership(df, membership)
    df = merge_actq_lctq(df)
    df = compute_book_equity(df)
    df = add_gvkey_lags_and_ttm(df)
    df = compute_features(df)
    df["feature_variant"] = variant
    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = PRESERVE_COLS + FEATURES + AUDIT_COLS + ["feature_variant"]
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def save_variant(df: pd.DataFrame, out_dir: Path, variant: str, input_path: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = select_output_columns(df)

    out_df.to_parquet(out_dir / "quarterly_fundamental_features.parquet", index=False)
    out_df.to_pickle(out_dir / "quarterly_fundamental_features.pkl")

    profile = build_feature_profile(out_df)
    exclusion = build_exclusion_detail(out_df)
    profile.to_csv(out_dir / "quarterly_feature_profile.csv", index=False)
    exclusion.to_csv(out_dir / "feature_exclusion_detail.csv", index=False)
    write_feature_quality_report(
        out_dir / "quarterly_feature_quality_report.md",
        variant,
        out_df,
        profile,
        exclusion,
    )

    meta = {
        "phase": "3A",
        "variant": variant,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "row_count": int(len(out_df)),
        "usable_count": int(out_df["signal_usable_flag"].sum()),
        "features": FEATURES,
    }
    (out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2, default=json_safe), encoding="utf-8")
    log.info("Wrote %s (%d rows)", out_dir, len(out_df))
    return out_df


def compare_main_nextday(main_df: pd.DataFrame, next_df: pd.DataFrame, path: Path) -> None:
    key = RESEARCH_KEY
    m = main_df[key + FEATURES + ["signal_start_date", "signal_market_cap"]].copy()
    n = next_df[key + FEATURES + ["signal_start_date", "signal_market_cap"]].copy()
    merged = m.merge(n, on=key, suffixes=("_main", "_nextday"), how="outer", indicator=True)

    accounting_feats = [
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
    value_feats = ["book_to_market", "earnings_yield", "sales_to_price"]

    lines = [
        "# Main vs Next-Day Feature Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Accounting feature construction is identical; only signal timing and ",
        "market-cap-based value ratios may differ.",
        "",
        "## Row alignment",
        "",
        f"| Check | Count |",
        f"| --- | ---: |",
        f"| Rows in main | {len(main_df):,} |",
        f"| Rows in next-day | {len(next_df):,} |",
        f"| Keys matched | {int((merged['_merge'] == 'both').sum()):,} |",
        f"| Signal dates differ | {int((merged['signal_start_date_main'] != merged['signal_start_date_nextday']).sum()):,} |",
        f"| Signal market cap differs | {int((merged['signal_market_cap_main'] != merged['signal_market_cap_nextday']).sum()):,} |",
        "",
        "## Feature equality checks (matched keys)",
        "",
        "| Feature | Both non-missing | Identical values | Max abs diff | Mean abs diff |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    both = merged[merged["_merge"] == "both"]
    for feat in FEATURES:
        fm = pd.to_numeric(both[f"{feat}_main"], errors="coerce")
        fn = pd.to_numeric(both[f"{feat}_nextday"], errors="coerce")
        nn = fm.notna() & fn.notna()
        if nn.any():
            diff = (fm[nn] - fn[nn]).abs()
            identical = int(np.isclose(fm[nn], fn[nn], rtol=0, atol=1e-12, equal_nan=False).sum())
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())
            both_nn = int(nn.sum())
        else:
            identical = 0
            max_diff = np.nan
            mean_diff = np.nan
            both_nn = 0
        lines.append(
            f"| `{feat}` | {both_nn:,} | {identical:,} | {max_diff:.6g} | {mean_diff:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Expected differences",
            "",
            f"- Accounting features ({', '.join(f'`{f}`' for f in accounting_feats)}): should be **identical** when both non-missing.",
            f"- Value features ({', '.join(f'`{f}`' for f in value_feats)}): may differ with `signal_market_cap`.",
            "",
            "## Accounting feature mismatch detail",
            "",
        ]
    )

    mismatch_rows = []
    for feat in accounting_feats:
        fm = pd.to_numeric(both[f"{feat}_main"], errors="coerce")
        fn = pd.to_numeric(both[f"{feat}_nextday"], errors="coerce")
        bad = both[fm.notna() & fn.notna() & ~np.isclose(fm, fn, rtol=0, atol=1e-12, equal_nan=False)]
        if len(bad):
            mismatch_rows.append({"feature": feat, "mismatch_count": len(bad)})
    if mismatch_rows:
        lines.append(pd.DataFrame(mismatch_rows).to_markdown(index=False))
    else:
        lines.append("_No accounting-feature mismatches beyond floating-point identity._")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def distribution_stats(s: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {k: np.nan for k in ["mean", "median", "p1", "p99"]}
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p1": float(x.quantile(0.01)),
        "p99": float(x.quantile(0.99)),
    }


def write_accruals_correction_report(path: Path, df: pd.DataFrame) -> None:
    raw = df["oancfq"].notna() if "oancfq" in df.columns else pd.Series(False, index=df.index)
    converted = df["oancf_ytd_conversion_flag"].astype(bool) & df["standalone_oancf_q"].notna()
    fail_missing = df["oancf_missing_previous_quarter_flag"].astype(bool)
    fail_nonconsec = df["oancf_nonconsecutive_quarter_flag"].astype(bool)

    old = df["accruals_old_raw_oancfq"]
    new = df["accruals"]
    both = old.notna() & new.notna()
    changed = both & ~np.isclose(old, new, rtol=0, atol=1e-9, equal_nan=False)

    old_stats = distribution_stats(old)
    new_stats = distribution_stats(new)
    pearson = float(old[both].corr(new[both])) if both.any() else np.nan
    spearman = float(old[both].corr(new[both], method="spearman")) if both.any() else np.nan

    annual_df = df.copy()
    annual_df["signal_year"] = pd.to_datetime(annual_df["signal_start_date"]).dt.year
    annual_rows: list[dict[str, Any]] = []
    for year, g in annual_df.groupby("signal_year", dropna=False):
        annual_rows.append(
            {
                "signal_year": year,
                "n_rows": len(g),
                "raw_oancfq": int(g["oancfq"].notna().sum()) if "oancfq" in g else 0,
                "converted_standalone": int(
                    (g["oancf_ytd_conversion_flag"].astype(bool) & g["standalone_oancf_q"].notna()).sum()
                ),
                "corrected_accruals_cov": float(g["accruals"].notna().mean()),
                "old_accruals_cov": float(g["accruals_old_raw_oancfq"].notna().mean()),
                "changed_accruals": int(
                    (
                        g["accruals_old_raw_oancfq"].notna()
                        & g["accruals"].notna()
                        & ~np.isclose(
                            g["accruals_old_raw_oancfq"],
                            g["accruals"],
                            rtol=0,
                            atol=1e-9,
                            equal_nan=False,
                        )
                    ).sum()
                ),
            }
        )
    annual = pd.DataFrame(annual_rows)

    audit = df[both].copy()
    audit["abs_change"] = (audit["accruals"] - audit["accruals_old_raw_oancfq"]).abs()
    top_cols = [
        "permno",
        "gvkey",
        "datadate",
        "datacqtr",
        "fqtr",
        "fyrq",
        "oancfq",
        "standalone_oancf_q",
        "ttm_oancfq",
        "ttm_oancf",
        "accruals_old_raw_oancfq",
        "accruals",
        "abs_change",
    ]
    top = audit.nlargest(20, "abs_change")[[c for c in top_cols if c in audit.columns]].copy()
    if len(top):
        top["datadate"] = pd.to_datetime(top["datadate"]).dt.date

    lines = [
        "# Accruals Correction Report (YTD `oancfq` Fix)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Conversion summary",
        "",
        "| Metric | Count | Share |",
        "| --- | ---: | ---: |",
        f"| Total rows | {len(df):,} | 100% |",
        f"| Raw `oancfq` present | {int(raw.sum()):,} | {pct(float(raw.mean()))} |",
        f"| Standalone OCF successfully converted | {int(converted.sum()):,} | {pct(float(converted.mean()))} |",
        f"| Failed: missing previous fiscal quarter | {int(fail_missing.sum()):,} | {pct(float(fail_missing.mean()))} |",
        f"| Failed: duplicate / non-consecutive fiscal quarter | {int(fail_nonconsec.sum()):,} | {pct(float(fail_nonconsec.mean()))} |",
        f"| Valid 4Q standalone TTM (`ttm_oancf_valid_4q_flag`) | {int(df['ttm_oancf_valid_4q_flag'].astype(bool).sum()):,} | {pct(float(df['ttm_oancf_valid_4q_flag'].astype(bool).mean()))} |",
        f"| Corrected `accruals` coverage | {int(new.notna().sum()):,} | {pct(float(new.notna().mean()))} |",
        "",
        "## Old vs corrected accruals (both non-missing)",
        "",
        "| Statistic | Old (`accruals_old_raw_oancfq`) | Corrected (`accruals`) |",
        "| --- | ---: | ---: |",
        f"| Mean | {old_stats['mean']:.6f} | {new_stats['mean']:.6f} |",
        f"| Median | {old_stats['median']:.6f} | {new_stats['median']:.6f} |",
        f"| p1 | {old_stats['p1']:.6f} | {new_stats['p1']:.6f} |",
        f"| p99 | {old_stats['p99']:.6f} | {new_stats['p99']:.6f} |",
        f"| Pearson correlation | {pearson:.6f} | — |",
        f"| Spearman correlation | {spearman:.6f} | — |",
        f"| Changed observations | {int(changed.sum()):,} | {pct(float(changed.sum() / both.sum() if both.any() else np.nan))} of both-non-missing |",
        f"| Changed (all rows) | {int(changed.sum()):,} | {pct(float(changed.sum() / len(df)))} |",
        "",
        "## Annual coverage",
        "",
        annual.to_markdown(index=False),
        "",
        "## Largest absolute changes (manual audit sample)",
        "",
    ]
    if len(top):
        lines.append(top.to_markdown(index=False))
    else:
        lines.append("_No comparable old/new accruals pairs._")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    if not MAIN_INPUT.exists() or not NEXTDAY_INPUT.exists():
        raise FileNotFoundError("Frozen Phase 2 signal-event inputs are required.")

    # Build formula dictionary from main variant probe
    probe = load_events(MAIN_INPUT)
    probe = merge_actq_lctq(probe)
    write_formula_dictionary(
        OUT_ROOT / "feature_formula_dictionary.md",
        oancf_coverage=float(probe["oancfq"].notna().mean()),
        actq_coverage=float(probe["actq"].notna().mean()) if "actq" in probe.columns else 0.0,
    )

    main_full = build_features(MAIN_INPUT, "main")
    write_accruals_correction_report(OUT_ROOT / "accruals_correction_report.md", main_full)
    main_df = save_variant(main_full, OUT_ROOT / "main", "main", MAIN_INPUT)

    next_full = build_features(NEXTDAY_INPUT, "nextday")
    next_df = save_variant(next_full, OUT_ROOT / "nextday", "nextday", NEXTDAY_INPUT)

    compare_main_nextday(main_df, next_df, OUT_ROOT / "main_vs_nextday_feature_comparison.md")
    log.info("Phase 3A complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
