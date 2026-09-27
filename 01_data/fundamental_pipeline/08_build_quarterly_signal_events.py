#!/usr/bin/env python3
"""08_build_quarterly_signal_events.py

Map each quarterly PIT accounting row to its first eligible CRSP trading date
(date >= accounting_available_date) and attach contemporaneous market fields."""

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

FINAL_MASTER_CANDIDATES = [
    PROJECT_ROOT / "data" / "sp500_pit_master" / "final_quarterly_fundamentals.parquet",
    SCRIPT_DIR / "data" / "sp500_pit_master" / "final_quarterly_fundamentals.parquet",
]
CRSP_PANEL_CANDIDATES = [
    PROJECT_ROOT / "data" / "crsp_daily" / "crsp_daily_market.parquet",
]
MISSING_PERMNO_CANDIDATES = [
    PROJECT_ROOT / "data" / "crsp_daily" / "missing_permnos.csv",
]
EXTREME_MCAP_CANDIDATES = [
    PROJECT_ROOT / "data" / "crsp_daily" / "extreme_market_cap_changes.csv",
]

MEMBERSHIP_PATH = SCRIPT_DIR / "data" / "sp500_pit_master" / "membership_spells.parquet"
LINK_INTERVALS_PATH = SCRIPT_DIR / "data" / "sp500_pit_master" / "valid_link_intervals.parquet"

OUT_DIR = PROJECT_ROOT / "data" / "quarterly_signal_events"
AUDIT_CACHE_DIR = OUT_DIR / "cache"

RESEARCH_MARKET_END = pd.Timestamp("2024-12-31")
OPEN_END = pd.Timestamp("2099-12-31")
KNOWN_MISSING_CRSP_PERMNOS = {26181, 27083, 27549, 27598}

CRSP_SIGNAL_COLS = ["prc", "shrout", "market_cap", "ret", "retx", "vol"]
CRSP_RENAME = {
    "prc": "signal_prc",
    "shrout": "signal_shrout",
    "market_cap": "signal_market_cap",
    "ret": "signal_ret",
    "retx": "signal_retx",
    "vol": "signal_volume",
    "date": "signal_start_date",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def resolve_path(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            log.info("%s input: %s", label, path)
            return path
    raise FileNotFoundError(
        f"Missing {label}. Tried:\n" + "\n".join(f"  - {p}" for p in candidates)
    )


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["rdqe"] = pd.to_datetime(df["rdqe"], errors="coerce")
    df["accounting_available_date"] = pd.to_datetime(df["accounting_available_date"])
    df["gvkey"] = zgvkey(df["gvkey"])
    df["permno"] = df["permno"].astype(int)
    # Recompute explicitly per spec (should match Phase 2A).
    df["accounting_available_date"] = df[["rdqe", "datadate"]].max(axis=1)
    return df


def load_crsp(path: Path) -> pd.DataFrame:
    usecols = ["permno", "date", *CRSP_SIGNAL_COLS]
    df = pd.read_parquet(path, columns=usecols)
    df["permno"] = df["permno"].astype(int)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in CRSP_SIGNAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["permno", "date"]).reset_index(drop=True)


def map_signal_start_dates(master: pd.DataFrame, crsp: pd.DataFrame) -> pd.DataFrame:
    """First CRSP trading date per PERMNO with date >= accounting_available_date."""
    crsp_sub = crsp.rename(columns={"date": "signal_start_date"}).sort_values(
        ["permno", "signal_start_date"]
    )

    parts: list[pd.DataFrame] = []
    for permno, grp in master.groupby("permno", sort=True):
        left = grp.sort_values(["accounting_available_date", "datadate", "fqtr"]).reset_index(
            drop=True
        )
        right = crsp_sub[crsp_sub["permno"] == permno].drop(columns=["permno"])
        if right.empty:
            merged = left.copy()
            merged["signal_start_date"] = pd.NaT
            for col in CRSP_SIGNAL_COLS:
                merged[col] = np.nan
        else:
            merged = pd.merge_asof(
                left,
                right,
                left_on="accounting_available_date",
                right_on="signal_start_date",
                direction="forward",
                allow_exact_matches=True,
            )
        parts.append(merged)

    out = pd.concat(parts, ignore_index=True)
    rename_cols = {k: v for k, v in CRSP_RENAME.items() if k in out.columns and k != "date"}
    return out.rename(columns=rename_cols)


def pick_membership_spell(
    permno: int,
    ref_date: pd.Timestamp,
    membership: pd.DataFrame,
) -> pd.Series | None:
    spells = membership[membership["permno"] == permno].sort_values("membership_start")
    if spells.empty:
        return None
    hit = spells[
        (spells["membership_start"] <= ref_date) & (spells["membership_end"] >= ref_date)
    ]
    if len(hit):
        return hit.iloc[0]
    prior = spells[spells["membership_end"] < ref_date]
    if len(prior):
        return prior.iloc[-1]
    return spells.iloc[0]


def membership_on_date(permno: int, dt: pd.Timestamp, membership: pd.DataFrame) -> bool:
    if pd.isna(dt):
        return False
    spells = membership[membership["permno"] == permno]
    if spells.empty:
        return False
    return bool(
        ((spells["membership_start"] <= dt) & (spells["membership_end"] >= dt)).any()
    )


def link_valid_on_date(
    permno: int,
    gvkey: str,
    dt: pd.Timestamp,
    links: pd.DataFrame,
) -> bool:
    if pd.isna(dt):
        return False
    sub = links[(links["permno"] == permno) & (links["gvkey"] == gvkey)]
    if sub.empty:
        return False
    return bool(
        ((sub["effective_link_start"] <= dt) & (sub["effective_link_end"] >= dt)).any()
    )


def add_signal_flags(
    df: pd.DataFrame,
    membership: pd.DataFrame,
    links: pd.DataFrame,
    missing_permnos: set[int],
) -> pd.DataFrame:
    out = df.copy()
    out["trading_day_delay_days"] = (
        out["signal_start_date"] - out["accounting_available_date"]
    ).dt.days

    out["same_day_signal_flag"] = (
        out["signal_start_date"].notna()
        & (out["signal_start_date"] == out["accounting_available_date"])
    )
    out["no_future_trading_date_flag"] = out["signal_start_date"].isna()
    out["crsp_permno_missing_flag"] = out["permno"].isin(missing_permnos)

    before_flags: list[bool] = []
    after_flags: list[bool] = []
    in_sp500_signal: list[bool] = []
    link_valid_signal: list[bool] = []

    for row in out.itertuples(index=False):
        permno = int(row.permno)
        gvkey = str(row.gvkey)
        sig = row.signal_start_date
        avail = row.accounting_available_date

        spell = pick_membership_spell(permno, avail, membership)
        if spell is None or pd.isna(sig):
            before_flags.append(False)
            after_flags.append(False)
        else:
            before_flags.append(bool(sig < spell["membership_start"]))
            after_flags.append(bool(sig > spell["membership_end"]))

        in_sp500_signal.append(membership_on_date(permno, sig, membership))
        link_valid_signal.append(link_valid_on_date(permno, gvkey, sig, links))

    out["signal_before_membership_start_flag"] = before_flags
    out["signal_after_membership_end_flag"] = after_flags
    out["in_sp500_on_signal_date"] = in_sp500_signal
    out["ccm_link_valid_on_signal_date"] = link_valid_signal

    valid_mcap = out["signal_market_cap"].notna() & (out["signal_market_cap"] > 0)
    within_cutoff = out["signal_start_date"].notna() & (
        out["signal_start_date"] <= RESEARCH_MARKET_END
    )

    out["signal_usable_flag"] = (
        out["signal_start_date"].notna()
        & within_cutoff
        & out["in_sp500_on_signal_date"]
        & out["ccm_link_valid_on_signal_date"]
        & (~out["history_only_flag"].astype(bool))
        & valid_mcap
    )
    return out


def audit_missing_crsp_permnos(
    permnos: set[int],
    master: pd.DataFrame,
    membership: pd.DataFrame,
    links: pd.DataFrame,
    wrds_audit: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for permno in sorted(permnos):
        mem_sp = membership[membership["permno"] == permno]
        lk = links[links["permno"] == permno]
        acct = master[master["permno"] == permno]
        wrds = wrds_audit.get(permno, {})

        membership_str = "; ".join(
            f"{r.membership_start.date()}..{r.membership_end.date()}"
            for r in mem_sp.itertuples()
        )
        gvkey_str = "; ".join(
            f"{r.gvkey} [{r.effective_link_start.date()}..{r.effective_link_end.date()}]"
            for r in lk.itertuples()
        )
        acct_cov = (
            f"n={len(acct)} datadate {acct['datadate'].min().date()}..{acct['datadate'].max().date()} "
            f"avail {acct['accounting_available_date'].min().date()}..{acct['accounting_available_date'].max().date()}"
            if len(acct)
            else "none"
        )

        rows.append(
            {
                "permno": permno,
                "qlib_instrument": mem_sp["qlib_instrument"].iloc[0] if len(mem_sp) else "",
                "qlib_membership_dates": membership_str,
                "ccm_gvkey_mapping": gvkey_str,
                "compustat_accounting_coverage": acct_cov,
                "crsp_dsf_search": wrds.get("crsp_dsf_search", "not_run"),
                "crsp_msenames_search": wrds.get("crsp_msenames_search", "not_run"),
                "local_sp500_csv_search": wrds.get("local_sp500_csv_search", "not_run"),
                "likely_reason_for_absence": wrds.get(
                    "likely_reason_for_absence",
                    "No CRSP dsf rows in Phase 2B panel",
                ),
                "recommended_treatment": (
                    "Retain accounting rows for audit/lag history only; "
                    "exclude from tradable signals (signal_usable_flag=False). "
                    "Do not substitute PERMNOs. Revisit when WRDS CRSP dsf includes "
                    "these listings or an approved alternate CRSP daily source is wired in."
                ),
            }
        )
    return pd.DataFrame(rows)


def wrds_missing_permno_audit(permnos: set[int]) -> dict[int, dict[str, Any]]:
    cache_path = AUDIT_CACHE_DIR / "missing_permno_wrds_audit.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if set(map(int, cached.keys())) >= permnos:
            log.info("WRDS missing-PERMNO audit cache hit")
            return {int(k): v for k, v in cached.items()}

    audit: dict[int, dict[str, Any]] = {}
    try:
        from wrds_utils import connect_wrds

        db = connect_wrds()
        for permno in sorted(permnos):
            dsf_q = (
                f"SELECT permno, MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS n "
                f"FROM crsp.dsf WHERE permno={permno} GROUP BY permno"
            )
            dsf = db.raw_sql(dsf_q)
            ms_q = (
                f"SELECT permno, MIN(namedt) AS min_namedt, MAX(nameendt) AS max_nameendt, "
                f"COUNT(*) AS n FROM crsp.msenames WHERE permno={permno} GROUP BY permno"
            )
            ms = db.raw_sql(ms_q)

            dsf_str = (
                f"0 rows in crsp.dsf (full history query {datetime.now(timezone.utc).date()})"
                if dsf.empty
                else dsf.to_dict(orient="records")[0]
            )
            ms_str = (
                "0 rows in crsp.msenames"
                if ms.empty
                else ms.to_dict(orient="records")[0]
            )

            audit[permno] = {
                "crsp_dsf_search": str(dsf_str),
                "crsp_msenames_search": str(ms_str),
                "local_sp500_csv_search": _local_sp500_csv_search(permno),
                "likely_reason_for_absence": _infer_absence_reason(permno, dsf.empty, ms.empty),
            }
        cache_path.write_text(
            json.dumps(audit, indent=2, default=json_safe),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("WRDS audit skipped (%s); using static fallback", exc)
        for permno in permnos:
            audit[permno] = {
                "crsp_dsf_search": "0 rows (Phase 2B WRDS pull through 2024-12-31)",
                "crsp_msenames_search": "not queried",
                "local_sp500_csv_search": _local_sp500_csv_search(permno),
                "likely_reason_for_absence": _infer_absence_reason(permno, True, True),
            }
    return audit


def _local_sp500_csv_search(permno: int) -> str:
    csv_path = PROJECT_ROOT / "data" / "extracted" / "dwklv3a23uaacd05_csv" / "dwklv3a23uaacd05.csv"
    if not csv_path.exists():
        return "local SP500 CSV not found"
    for chunk in pd.read_csv(csv_path, usecols=["PERMNO", "DlyCalDt", "Ticker"], chunksize=500_000):
        sub = chunk[chunk["PERMNO"] == permno]
        if len(sub):
            return (
                f"{len(sub)} rows in dwklv3a23uaacd05.csv "
                f"({sub['DlyCalDt'].min()}..{sub['DlyCalDt'].max()}, ticker={sub['Ticker'].iloc[-1]})"
            )
    return "0 rows in local SP500 CSV"


def _infer_absence_reason(permno: int, dsf_empty: bool, ms_empty: bool) -> str:
    if dsf_empty and ms_empty:
        return (
            "PERMNO absent from WRDS crsp.dsf and crsp.msenames as of audit date. "
            "Qlib universe lists late-2025 S&P 500 membership, but Phase 2B CRSP panel "
            "ends 2024-12-31 and WRDS standard CRSP daily tables contain no history for "
            "this PERMNO. Local S&P 500 export may include post-2024 prices (separate product)."
        )
    if dsf_empty:
        return "PERMNO known in CRSP names but no crsp.dsf daily rows."
    return "Unexpected CRSP presence; review manually."


def build_unmapped(df: pd.DataFrame) -> pd.DataFrame:
    unmapped = df[df["signal_start_date"].isna()].copy()
    reason = []
    for row in unmapped.itertuples(index=False):
        if row.crsp_permno_missing_flag:
            reason.append("crsp_permno_missing")
        elif row.accounting_available_date > RESEARCH_MARKET_END:
            reason.append("accounting_available_after_market_cutoff")
        else:
            reason.append("no_crsp_trading_date_on_or_after_availability")
    unmapped["unmapped_reason"] = reason
    return unmapped


def delay_bucket(days: pd.Series) -> pd.DataFrame:
    d = days.dropna().astype(int)
    return pd.DataFrame(
        {
            "metric": [
                "same_day",
                "delay_1_cal_day",
                "delay_2_3_cal_days",
                "delay_4_5_cal_days",
                "delay_gt_5_cal_days",
            ],
            "count": [
                int((d == 0).sum()),
                int((d == 1).sum()),
                int(d.between(2, 3).sum()),
                int(d.between(4, 5).sum()),
                int((d > 5).sum()),
            ],
        }
    )


def write_quality_report(
    path: Path,
    *,
    master_path: Path,
    crsp_path: Path,
    events: pd.DataFrame,
    unmapped: pd.DataFrame,
    missing_audit: pd.DataFrame,
    extreme_path: Path | None,
) -> None:
    mapped = events["signal_start_date"].notna()
    n = len(events)
    delays = events.loc[mapped, "trading_day_delay_days"]

    by_year = (
        events.assign(signal_year=events["signal_start_date"].dt.year)
        .groupby("signal_year", dropna=False)
        .agg(
            n_rows=("permno", "size"),
            n_mapped=("signal_start_date", lambda s: int(s.notna().sum())),
            n_usable=("signal_usable_flag", lambda s: int(s.astype(bool).sum())),
            n_permno=("permno", "nunique"),
        )
        .reset_index()
    )

    avail_year = (
        events.assign(avail_year=events["accounting_available_date"].dt.year)
        .groupby("avail_year")
        .agg(
            n_rows=("permno", "size"),
            n_mapped=("signal_start_date", lambda s: int(s.notna().sum())),
            n_usable=("signal_usable_flag", lambda s: int(s.astype(bool).sum())),
        )
        .reset_index()
    )

    permno_cov = (
        events.groupby("permno")
        .agg(
            n_rows=("permno", "size"),
            n_mapped=("signal_start_date", lambda s: int(s.notna().sum())),
            n_usable=("signal_usable_flag", lambda s: int(s.astype(bool).sum())),
            median_delay=("trading_day_delay_days", "median"),
        )
        .reset_index()
        .sort_values("n_usable", ascending=False)
    )

    invalid_mcap = events["signal_start_date"].notna() & (
        events["signal_market_cap"].isna() | (events["signal_market_cap"] <= 0)
    )

    lines = [
        "# Quarterly Signal Event Mapping — Quality Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Inputs",
        "",
        f"- Final master: `{master_path}`",
        f"- CRSP daily panel: `{crsp_path}`",
        f"- Research market cutoff: **{RESEARCH_MARKET_END.date()}**",
        "",
        "## Mapping summary",
        "",
        f"| Metric | Count | Share |",
        f"| --- | ---: | ---: |",
        f"| Total quarterly observations | {n:,} | 100% |",
        f"| Mapped to a CRSP `signal_start_date` | {int(mapped.sum()):,} | {pct(mapped.mean())} |",
        f"| Same-day signal (`same_day_signal_flag`) | {int(events['same_day_signal_flag'].sum()):,} | {pct(events['same_day_signal_flag'].mean())} |",
        f"| Unmapped (`no_future_trading_date_flag`) | {int(events['no_future_trading_date_flag'].sum()):,} | {pct(events['no_future_trading_date_flag'].mean())} |",
        f"| **`signal_usable_flag=True`** (tradable) | {int(events['signal_usable_flag'].sum()):,} | {pct(events['signal_usable_flag'].mean())} |",
        "",
        "## Delay distribution (mapped rows)",
        "",
        delay_bucket(delays).to_markdown(index=False),
        "",
        f"- Median delay: **{delays.median():.0f}** calendar days",
        f"- Mean delay: **{delays.mean():.2f}** calendar days",
        f"- Delay > 1 calendar day: **{int((delays > 1).sum()):,}** ({pct((delays > 1).mean())})",
        f"- Delay > 3 calendar days: **{int((delays > 3).sum()):,}** ({pct((delays > 3).mean())})",
        f"- Delay > 5 calendar days: **{int((delays > 5).sum()):,}** ({pct((delays > 5).mean())})",
        "",
        "## Quality flags",
        "",
        f"| Check | Count | Share |",
        f"| --- | ---: | ---: |",
        f"| Signal after membership end | {int(events['signal_after_membership_end_flag'].sum()):,} | {pct(events['signal_after_membership_end_flag'].mean())} |",
        f"| Signal before membership start | {int(events['signal_before_membership_start_flag'].sum()):,} | {pct(events['signal_before_membership_start_flag'].mean())} |",
        f"| Signal date after {RESEARCH_MARKET_END.date()} | {int((events['signal_start_date'] > RESEARCH_MARKET_END).sum()):,} | {pct((events['signal_start_date'] > RESEARCH_MARKET_END).mean())} |",
        f"| Availability after cutoff (unmappable in panel) | {int((events['accounting_available_date'] > RESEARCH_MARKET_END).sum()):,} | {pct((events['accounting_available_date'] > RESEARCH_MARKET_END).mean())} |",
        f"| Missing/invalid signal market cap | {int(invalid_mcap.sum()):,} | {pct(invalid_mcap.mean())} |",
        f"| History-only rows (`history_only_flag`) | {int(events['history_only_flag'].sum()):,} | {pct(events['history_only_flag'].mean())} |",
        f"| CRSP PERMNO missing from dsf panel | {int(events['crsp_permno_missing_flag'].sum()):,} | {pct(events['crsp_permno_missing_flag'].mean())} |",
        "",
        "### `signal_usable_flag` definition",
        "",
        "True only when all hold:",
        "",
        "1. Valid `signal_start_date` (first CRSP date ≥ `accounting_available_date` for the PERMNO)",
        f"2. `signal_start_date` ≤ **{RESEARCH_MARKET_END.date()}**",
        "3. S&P 500 member on `signal_start_date`",
        "4. CCM PERMNO–GVKEY link valid on `signal_start_date`",
        "5. `history_only_flag=False`",
        "6. `signal_market_cap` present and > 0",
        "",
        "Rows failing the rule are retained for audit/lag history but must not generate tradable signals.",
        "",
        "## Coverage by availability year",
        "",
        avail_year.to_markdown(index=False),
        "",
        "## Coverage by signal year",
        "",
        by_year.to_markdown(index=False),
        "",
        "## PERMNO coverage (top 15 by usable signals)",
        "",
        permno_cov.head(15).to_markdown(index=False),
        "",
        "## Missing CRSP PERMNO audit",
        "",
        missing_audit.to_markdown(index=False),
        "",
        "## Unmapped events",
        "",
        f"See `unmapped_signal_events.csv` (**{len(unmapped):,}** rows).",
        "",
    ]

    if extreme_path and extreme_path.exists():
        lines.extend(
            [
                "## Extreme market-cap reference",
                "",
                f"Phase 2B flagged extreme daily market-cap changes in `{extreme_path.name}` "
                "(not merged; retained as upstream QA reference).",
                "",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()

    master_path = resolve_path(FINAL_MASTER_CANDIDATES, "Final master")
    crsp_path = resolve_path(CRSP_PANEL_CANDIDATES, "CRSP panel")
    missing_path = resolve_path(MISSING_PERMNO_CANDIDATES, "Missing PERMNOs")
    extreme_path = EXTREME_MCAP_CANDIDATES[0] if EXTREME_MCAP_CANDIDATES[0].exists() else None

    master = load_master(master_path)
    crsp = load_crsp(crsp_path)
    membership = pd.read_parquet(MEMBERSHIP_PATH)
    membership["membership_start"] = pd.to_datetime(membership["membership_start"])
    membership["membership_end"] = pd.to_datetime(membership["membership_end"])
    links = pd.read_parquet(LINK_INTERVALS_PATH)
    links["gvkey"] = zgvkey(links["gvkey"])
    links["permno"] = links["permno"].astype(int)
    links["effective_link_start"] = pd.to_datetime(links["effective_link_start"])
    links["effective_link_end"] = pd.to_datetime(links["effective_link_end"])

    missing_df = pd.read_csv(missing_path)
    missing_permnos = set(missing_df["permno"].astype(int).tolist()) | KNOWN_MISSING_CRSP_PERMNOS

    log.info(
        "CRSP panel: %d rows, %d PERMNOs, max date %s",
        len(crsp),
        crsp["permno"].nunique(),
        crsp["date"].max().date(),
    )
    log.info("Master rows: %d; research cutoff %s", len(master), RESEARCH_MARKET_END.date())

    events = map_signal_start_dates(master, crsp)
    events = add_signal_flags(events, membership, links, missing_permnos)

    wrds_audit = wrds_missing_permno_audit(missing_permnos)
    missing_audit = audit_missing_crsp_permnos(
        missing_permnos, master, membership, links, wrds_audit
    )

    unmapped = build_unmapped(events)

    out_pq = OUT_DIR / "quarterly_signal_events.parquet"
    out_pkl = OUT_DIR / "quarterly_signal_events.pkl"
    events.to_parquet(out_pq, index=False)
    events.to_pickle(out_pkl)
    unmapped.to_csv(OUT_DIR / "unmapped_signal_events.csv", index=False)
    missing_audit.to_csv(OUT_DIR / "missing_crsp_permno_audit.csv", index=False)

    write_quality_report(
        OUT_DIR / "signal_mapping_quality_report.md",
        master_path=master_path,
        crsp_path=crsp_path,
        events=events,
        unmapped=unmapped,
        missing_audit=missing_audit,
        extreme_path=extreme_path,
    )

    meta = {
        "phase": "2C",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "final_master": str(master_path),
            "crsp_daily_market": str(crsp_path),
            "missing_permnos": str(missing_path),
            "extreme_market_cap_changes": str(extreme_path) if extreme_path else None,
            "membership_spells": str(MEMBERSHIP_PATH),
            "valid_link_intervals": str(LINK_INTERVALS_PATH),
        },
        "research_market_end": RESEARCH_MARKET_END.date().isoformat(),
        "row_count": int(len(events)),
        "mapped_count": int(events["signal_start_date"].notna().sum()),
        "same_day_count": int(events["same_day_signal_flag"].sum()),
        "usable_count": int(events["signal_usable_flag"].sum()),
        "unmapped_count": int(len(unmapped)),
        "missing_crsp_permno_count": int(len(missing_permnos)),
        "outputs": {
            "events_parquet": str(out_pq),
            "events_pkl": str(out_pkl),
            "quality_report": str(OUT_DIR / "signal_mapping_quality_report.md"),
            "unmapped": str(OUT_DIR / "unmapped_signal_events.csv"),
            "missing_audit": str(OUT_DIR / "missing_crsp_permno_audit.csv"),
        },
    }
    (OUT_DIR / "signal_mapping_meta.json").write_text(
        json.dumps(meta, indent=2, default=json_safe),
        encoding="utf-8",
    )

    log.info(
        "Mapped %d / %d; usable %d; unmapped %d",
        meta["mapped_count"],
        meta["row_count"],
        meta["usable_count"],
        meta["unmapped_count"],
    )
    log.info("Phase 2C complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
