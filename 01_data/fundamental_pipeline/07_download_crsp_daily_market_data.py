#!/usr/bin/env python3
"""07_download_crsp_daily_market_data.py

Download CRSP daily market data (crsp.dsf) for PERMNOs in the final quarterly
PIT master and compute market_cap = abs(prc) * shrout * 1000."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wrds_utils import connect_wrds

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FINAL_MASTER_CANDIDATES = [
    PROJECT_ROOT / "data" / "sp500_pit_master" / "final_quarterly_fundamentals.parquet",
    SCRIPT_DIR / "data" / "sp500_pit_master" / "final_quarterly_fundamentals.parquet",
]

OUT_DIR = PROJECT_ROOT / "data" / "crsp_daily"
CACHE_DIR = OUT_DIR / "cache"

DATE_START = "2008-01-01"
QLIB_RESEARCH_END = "2025-12-31"
CRSP_SOURCE_TABLE = "crsp.dsf"

OUTPUT_COLUMNS = [
    "permno",
    "date",
    "prc",
    "shrout",
    "ret",
    "retx",
    "vol",
    "bid",
    "ask",
    "openprc",
    "numtrd",
    "market_cap",
]

EXTREME_MCAP_PCT_THRESHOLD = 1.0
EXTREME_MCAP_RATIO_THRESHOLD = 10.0
GAP_CALENDAR_DAYS_THRESHOLD = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def resolve_final_master_path() -> Path:
    for path in FINAL_MASTER_CANDIDATES:
        if path.exists():
            log.info("Final master input: %s", path)
            return path
    raise FileNotFoundError(
        "Missing final_quarterly_fundamentals.parquet. Tried:\n"
        + "\n".join(f"  - {p}" for p in FINAL_MASTER_CANDIDATES)
    )


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(type(obj))


def cache_meta_path(data_path: Path) -> Path:
    return data_path.with_suffix(".meta.json")


def permno_sql_list(permnos: list[int]) -> str:
    return ", ".join(str(int(p)) for p in sorted(permnos))


def compute_date_end(final_master: pd.DataFrame) -> str:
    avail_max = pd.to_datetime(final_master["accounting_available_date"]).max()
    end = max(pd.Timestamp(QLIB_RESEARCH_END), avail_max)
    log.info(
        "Research date end: max(qlib=%s, accounting_available_date=%s) -> %s",
        QLIB_RESEARCH_END,
        avail_max.date(),
        end.date(),
    )
    return end.strftime("%Y-%m-%d")


def build_sql(permnos: list[int], date_start: str, date_end: str) -> str:
    perm_sql = permno_sql_list(permnos)
    return f"""
SELECT
    s.permno,
    s.date,
    s.prc,
    s.shrout,
    s.ret,
    s.retx,
    s.vol,
    s.bid,
    s.ask,
    s.openprc,
    s.numtrd
FROM {CRSP_SOURCE_TABLE} AS s
WHERE s.permno IN ({perm_sql})
  AND s.date >= '{date_start}'
  AND s.date <= '{date_end}'
ORDER BY s.permno, s.date
""".strip()


def cache_fingerprint(permnos: list[int], date_start: str, date_end: str) -> str:
    payload = {
        "table": CRSP_SOURCE_TABLE,
        "permnos": sorted(int(p) for p in permnos),
        "date_start": date_start,
        "date_end": date_end,
        "columns": [
            "permno",
            "date",
            "prc",
            "shrout",
            "ret",
            "retx",
            "vol",
            "bid",
            "ask",
            "openprc",
            "numtrd",
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_wrds_cache(cache_path: Path, fingerprint: str) -> pd.DataFrame | None:
    meta_path = cache_meta_path(cache_path)
    if not cache_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("fingerprint") != fingerprint:
        log.info("Cache stale (fingerprint mismatch): %s", cache_path.name)
        return None
    log.info("WRDS cache hit: %s (%d rows)", cache_path.name, meta.get("row_count", "?"))
    return pd.read_parquet(cache_path)


def save_wrds_cache(cache_path: Path, df: pd.DataFrame, meta: dict[str, Any]) -> None:
    df.to_parquet(cache_path, index=False)
    cache_meta_path(cache_path).write_text(
        json.dumps(meta, indent=2, default=json_safe),
        encoding="utf-8",
    )
    log.info("Cached WRDS download -> %s (%d rows)", cache_path, len(df))


def fetch_crsp_daily(
    permnos: list[int],
    date_start: str,
    date_end: str,
) -> tuple[pd.DataFrame, str, bool]:
    sql = build_sql(permnos, date_start, date_end)
    fingerprint = cache_fingerprint(permnos, date_start, date_end)
    cache_path = CACHE_DIR / f"crsp_dsf_raw_{fingerprint}.parquet"

    cached = load_wrds_cache(cache_path, fingerprint)
    if cached is not None:
        return cached, sql, True

    log.info("WRDS source table: %s", CRSP_SOURCE_TABLE)
    log.info("SQL query:\n%s", sql)
    db = connect_wrds()
    log.info("Connected to WRDS")
    df = db.raw_sql(sql, date_cols=["date"])
    log.info(
        "WRDS returned %d rows, %d unique PERMNOs, date range [%s, %s]",
        len(df),
        df["permno"].nunique() if len(df) else 0,
        df["date"].min() if len(df) else "NA",
        df["date"].max() if len(df) else "NA",
    )

    meta = {
        "label": "crsp_dsf_raw",
        "source_table": CRSP_SOURCE_TABLE,
        "fingerprint": fingerprint,
        "date_start": date_start,
        "date_end": date_end,
        "n_permnos_requested": len(permnos),
        "row_count": int(len(df)),
        "unique_permno_count": int(df["permno"].nunique()) if len(df) else 0,
        "date_min": str(pd.to_datetime(df["date"]).min().date()) if len(df) else None,
        "date_max": str(pd.to_datetime(df["date"]).max().date()) if len(df) else None,
        "sql": sql,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    save_wrds_cache(cache_path, df, meta)
    return df, sql, False


def normalize_panel(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["permno"] = df["permno"].astype(int)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ["prc", "shrout", "ret", "retx", "vol", "bid", "ask", "openprc", "numtrd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["market_cap"] = df["prc"].abs() * df["shrout"] * 1000.0
    return df[OUTPUT_COLUMNS]


def missingness_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns and c not in ("permno", "date", "market_cap")]
    rows = []
    for col in cols + ["market_cap"]:
        s = df[col]
        rows.append(
            {
                "field": col,
                "missing_count": int(s.isna().sum()),
                "missing_rate": float(s.isna().mean()),
                "non_missing_count": int(s.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    key = ["permno", "date"]
    dup_mask = df.duplicated(subset=key, keep=False)
    return df.loc[dup_mask].sort_values(key)


def find_extreme_market_cap_changes(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values(["permno", "date"]).copy()
    work["prev_mcap"] = work.groupby("permno")["market_cap"].shift(1)
    work["prev_date"] = work.groupby("permno")["date"].shift(1)
    valid = work["prev_mcap"].notna() & work["market_cap"].notna() & (work["prev_mcap"] > 0)
    work["mcap_pct_change"] = np.nan
    work.loc[valid, "mcap_pct_change"] = (
        work.loc[valid, "market_cap"] / work.loc[valid, "prev_mcap"] - 1.0
    )
    work["mcap_ratio"] = np.nan
    work.loc[valid, "mcap_ratio"] = (
        work.loc[valid, "market_cap"] / work.loc[valid, "prev_mcap"]
    )
    extreme = work[
        valid
        & (
            (work["mcap_pct_change"].abs() >= EXTREME_MCAP_PCT_THRESHOLD)
            | (work["mcap_ratio"] >= EXTREME_MCAP_RATIO_THRESHOLD)
            | (work["mcap_ratio"] <= (1.0 / EXTREME_MCAP_RATIO_THRESHOLD))
        )
    ].copy()
    return extreme[
        [
            "permno",
            "date",
            "prev_date",
            "prev_mcap",
            "market_cap",
            "mcap_pct_change",
            "mcap_ratio",
            "prc",
            "shrout",
            "ret",
            "vol",
        ]
    ]


def coverage_by_permno(df: pd.DataFrame, universe: set[int]) -> pd.DataFrame:
    if df.empty:
        base = pd.DataFrame({"permno": sorted(universe)})
        base["n_rows"] = 0
        base["first_date"] = pd.NaT
        base["last_date"] = pd.NaT
        base["has_crsp_daily"] = False
        return base

    agg = (
        df.groupby("permno", as_index=False)
        .agg(
            n_rows=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            missing_prc=("prc", lambda s: int(s.isna().sum())),
            missing_shrout=("shrout", lambda s: int(s.isna().sum())),
            non_positive_mcap=("market_cap", lambda s: int((s <= 0).sum())),
            negative_prc=("prc", lambda s: int((s < 0).sum())),
            zero_vol_days=("vol", lambda s: int((s.fillna(0) == 0).sum())),
        )
        .sort_values("permno")
    )
    agg["has_crsp_daily"] = True
    missing = sorted(universe - set(agg["permno"].astype(int)))
    if missing:
        filler = pd.DataFrame({"permno": missing})
        filler["n_rows"] = 0
        filler["first_date"] = pd.NaT
        filler["last_date"] = pd.NaT
        filler["missing_prc"] = 0
        filler["missing_shrout"] = 0
        filler["non_positive_mcap"] = 0
        filler["negative_prc"] = 0
        filler["zero_vol_days"] = 0
        filler["has_crsp_daily"] = False
        agg = pd.concat([agg, filler], ignore_index=True).sort_values("permno")
    return agg


def trading_gaps(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values(["permno", "date"]).copy()
    work["prev_date"] = work.groupby("permno")["date"].shift(1)
    work["gap_days"] = (work["date"] - work["prev_date"]).dt.days
    gaps = work[work["gap_days"] > GAP_CALENDAR_DAYS_THRESHOLD].copy()
    return gaps[["permno", "prev_date", "date", "gap_days"]]


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.2f}%"


def write_quality_report(
    path: Path,
    *,
    final_master_path: Path,
    universe_permnos: list[int],
    df: pd.DataFrame,
    dupes: pd.DataFrame,
    extreme: pd.DataFrame,
    coverage: pd.DataFrame,
    gaps: pd.DataFrame,
    miss_stats: pd.DataFrame,
    sql: str,
    date_start: str,
    date_end: str,
    cache_hit: bool,
) -> None:
    missing_permnos = coverage[~coverage["has_crsp_daily"]]["permno"].tolist()
    n_dup_keys = dupes.drop_duplicates(subset=["permno", "date"]).shape[0] if len(dupes) else 0

    lines = [
        "# CRSP Daily Market Panel — Quality Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Source",
        "",
        f"- Final master: `{final_master_path}`",
        f"- WRDS table: **`{CRSP_SOURCE_TABLE}`**",
        f"- WRDS cache hit: **{cache_hit}**",
        f"- Universe PERMNOs (from final master): **{len(universe_permnos)}**",
        f"- Date window: **{date_start}** .. **{date_end}**",
        "",
        "## Download summary",
        "",
        f"| Metric | Value |",
        f"| --- | ---: |",
        f"| Rows | {len(df):,} |",
        f"| Unique PERMNO | {df['permno'].nunique() if len(df) else 0} |",
        f"| Date min | {df['date'].min().date() if len(df) else 'NA'} |",
        f"| Date max | {df['date'].max().date() if len(df) else 'NA'} |",
        "",
        "## SQL query",
        "",
        "```sql",
        sql.strip(),
        "```",
        "",
        "## Field missingness",
        "",
        miss_stats.assign(missing_rate=miss_stats["missing_rate"].map(pct)).to_markdown(index=False),
        "",
        "## Data-quality checks",
        "",
        f"- Duplicate `(permno, date)` keys: **{n_dup_keys}** (see `duplicate_rows.csv`)",
        f"- Rows with missing `prc`: **{int(df['prc'].isna().sum()):,}**",
        f"- Rows with missing `shrout`: **{int(df['shrout'].isna().sum()):,}**",
        f"- Rows with non-positive `market_cap`: **{int((df['market_cap'] <= 0).sum()):,}**",
        f"- Rows with negative raw `prc` (CRSP bid/ask convention): **{int((df['prc'] < 0).sum()):,}**",
        f"- Zero-volume days (`vol` missing or 0): **{int((df['vol'].fillna(0) == 0).sum()):,}**",
        f"- PERMNOs in master with **no** CRSP daily rows: **{len(missing_permnos)}** "
        f"(see `missing_permnos.csv`)",
        f"- Trading gaps > {GAP_CALENDAR_DAYS_THRESHOLD} calendar days: **{len(gaps):,}** gap events",
        f"- Extreme daily market-cap changes: **{len(extreme):,}** rows "
        f"(|pct_change|≥{EXTREME_MCAP_PCT_THRESHOLD:.0%} or ratio≥{EXTREME_MCAP_RATIO_THRESHOLD})",
        "",
        "### Negative `prc` handling",
        "",
        "Raw `prc` is **preserved** as returned by CRSP. `market_cap` uses `abs(prc) * shrout * 1000`.",
        "Negative prices typically indicate bid/ask midpoint when no closing trade occurred.",
        "",
        "### Forward-fill policy",
        "",
        "**No forward-fill** of `market_cap` (or any field) across non-trading or missing days.",
        "",
        "## PERMNO coverage (summary)",
        "",
        f"| Segment | PERMNOs |",
        f"| --- | ---: |",
        f"| Requested | {len(universe_permnos)} |",
        f"| With ≥1 CRSP daily row | {int(coverage['has_crsp_daily'].sum())} |",
        f"| Missing entirely | {len(missing_permnos)} |",
        "",
    ]

    if missing_permnos:
        lines.extend(
            [
                "### Missing PERMNO list (first 40)",
                "",
                ", ".join(str(p) for p in missing_permnos[:40])
                + (" …" if len(missing_permnos) > 40 else ""),
                "",
            ]
        )

    sparse = coverage[coverage["has_crsp_daily"]].nsmallest(10, "n_rows")
    if len(sparse):
        lines.extend(
            [
                "### Shortest CRSP histories (top 10 by row count)",
                "",
                sparse[
                    ["permno", "n_rows", "first_date", "last_date", "missing_prc", "missing_shrout"]
                ].to_markdown(index=False),
                "",
            ]
        )

    if len(gaps):
        lines.extend(
            [
                "### Largest trading gaps (sample)",
                "",
                gaps.sort_values("gap_days", ascending=False)
                .head(20)
                .assign(
                    prev_date=lambda x: x["prev_date"].dt.date,
                    date=lambda x: x["date"].dt.date,
                )
                .to_markdown(index=False),
                "",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    final_path = resolve_final_master_path()
    final_master = pd.read_parquet(final_path)
    universe = sorted(final_master["permno"].astype(int).unique().tolist())
    date_end = compute_date_end(final_master)

    log.info("Universe: %d unique PERMNOs from final quarterly master", len(universe))

    raw, sql, cache_hit = fetch_crsp_daily(universe, DATE_START, date_end)
    panel = normalize_panel(raw)

    dupes = find_duplicates(panel)
    extreme = find_extreme_market_cap_changes(panel)
    coverage = coverage_by_permno(panel, set(universe))
    gaps = trading_gaps(panel)
    miss_stats = missingness_stats(panel)

    missing_permnos_df = coverage[~coverage["has_crsp_daily"]][["permno"]].copy()
    missing_permnos_df["source"] = "final_quarterly_fundamentals"
    missing_permnos_df.to_csv(OUT_DIR / "missing_permnos.csv", index=False)
    dupes.to_csv(OUT_DIR / "duplicate_rows.csv", index=False)
    extreme.to_csv(OUT_DIR / "extreme_market_cap_changes.csv", index=False)

    out_pq = OUT_DIR / "crsp_daily_market.parquet"
    out_pkl = OUT_DIR / "crsp_daily_market.pkl"
    panel.to_parquet(out_pq, index=False)
    panel.to_pickle(out_pkl)
    log.info("Saved panel: %s (%d rows)", out_pq, len(panel))

    write_quality_report(
        OUT_DIR / "crsp_daily_quality_report.md",
        final_master_path=final_path,
        universe_permnos=universe,
        df=panel,
        dupes=dupes,
        extreme=extreme,
        coverage=coverage,
        gaps=gaps,
        miss_stats=miss_stats,
        sql=sql,
        date_start=DATE_START,
        date_end=date_end,
        cache_hit=cache_hit,
    )

    build_meta = {
        "phase": "2B",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "final_master_path": str(final_path),
        "source_table": CRSP_SOURCE_TABLE,
        "date_start": DATE_START,
        "date_end": date_end,
        "n_permnos_universe": len(universe),
        "row_count": int(len(panel)),
        "unique_permno_count": int(panel["permno"].nunique()) if len(panel) else 0,
        "wrds_cache_hit": cache_hit,
        "outputs": {
            "panel_parquet": str(out_pq),
            "panel_pkl": str(out_pkl),
            "quality_report": str(OUT_DIR / "crsp_daily_quality_report.md"),
            "missing_permnos": str(OUT_DIR / "missing_permnos.csv"),
            "duplicate_rows": str(OUT_DIR / "duplicate_rows.csv"),
            "extreme_market_cap_changes": str(OUT_DIR / "extreme_market_cap_changes.csv"),
        },
    }
    (OUT_DIR / "build_meta.json").write_text(
        json.dumps(build_meta, indent=2, default=json_safe),
        encoding="utf-8",
    )

    log.info("Missingness summary:")
    for _, row in miss_stats.iterrows():
        log.info(
            "  %s: missing=%d (%.2f%%)",
            row["field"],
            row["missing_count"],
            100 * row["missing_rate"],
        )
    log.info(
        "QA: dup_keys=%d missing_permnos=%d extreme_mcap=%d gap_events=%d",
        dupes.drop_duplicates(subset=["permno", "date"]).shape[0] if len(dupes) else 0,
        len(missing_permnos_df),
        len(extreme),
        len(gaps),
    )
    log.info("Phase 2B complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
