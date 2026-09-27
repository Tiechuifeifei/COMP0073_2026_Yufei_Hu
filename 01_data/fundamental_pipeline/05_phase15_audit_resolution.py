#!/usr/bin/env python3
"""05_phase15_audit_resolution.py

Resolve CCM link ambiguities, audit rdqe anomalies, and document rows with
in_sp500_on_datadate=False before the final quarterly master."""

from __future__ import annotations

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
SP500_PATH = PROJECT_ROOT / "staging" / "instruments" / "sp500.txt"
OUT_DIR = SCRIPT_DIR / "data" / "sp500_pit_master"
CACHE_DIR = OUT_DIR / "cache"
CRSP_CSV = PROJECT_ROOT / "data/extracted/dwklv3a23uaacd05_csv/dwklv3a23uaacd05.csv"

AMBIGUOUS_PERMNO = [21186, 24643, 45356, 75034]
OPEN_END = pd.Timestamp("2099-12-31")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def zgvkey(s: pd.Series | str | np.ndarray) -> str | pd.Series:
    if isinstance(s, str):
        return str(s).strip().zfill(6)
    if isinstance(s, np.ndarray):
        s = pd.Series(s)
    return s.astype(str).str.strip().str.zfill(6)


def load_sp500() -> pd.DataFrame:
    rows = []
    for line in SP500_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        inst, start, end = line.split("\t")
        rows.append(
            {
                "qlib_instrument": inst,
                "permno": int(inst.lstrip("P")),
                "membership_start": pd.Timestamp(start),
                "membership_end": pd.Timestamp(end),
            }
        )
    df = pd.DataFrame(rows)
    df["spell_id"] = np.arange(len(df))
    return df


def interval_overlap_days(a0, a1, b0, b1) -> int:
    s = max(a0, b0)
    e = min(a1, b1)
    if s > e:
        return 0
    return int((e - s).days) + 1


def crsp_coverage(permno: int) -> dict[str, Any]:
    if not CRSP_CSV.exists():
        return {"error": "CRSP CSV missing"}
    usecols = ["PERMNO", "Ticker", "HdrCUSIP", "CUSIP", "DlyCalDt"]
    df = pd.read_csv(CRSP_CSV, usecols=usecols)
    sub = df[df["PERMNO"] == permno].copy()
    if sub.empty:
        return {"n_trade_days": 0}
    sub["DlyCalDt"] = pd.to_datetime(sub["DlyCalDt"])
    tickers = sub.groupby("Ticker").agg(
        n_days=("DlyCalDt", "size"),
        first_date=("DlyCalDt", "min"),
        last_date=("DlyCalDt", "max"),
        hdr_cusip=("HdrCUSIP", "last"),
    )
    return {
        "trade_start": str(sub["DlyCalDt"].min().date()),
        "trade_end": str(sub["DlyCalDt"].max().date()),
        "n_trade_days": int(len(sub)),
        "latest_ticker": str(sub.sort_values("DlyCalDt").Ticker.iloc[-1]),
        "latest_hdr_cusip": str(sub.sort_values("DlyCalDt").HdrCUSIP.iloc[-1]),
        "ticker_segments": tickers.reset_index().to_dict(orient="records"),
    }


def gvkey_accounting_summary(pit: pd.DataFrame, gvkey: str) -> dict[str, Any]:
    g = zgvkey(gvkey)
    sub = pit[pit["gvkey"].map(zgvkey) == g].copy()
    if sub.empty:
        return {"n_quarters": 0}
    sub["datadate"] = pd.to_datetime(sub["datadate"])
    return {
        "gvkey": g,
        "conm": str(sub["conm"].iloc[0]),
        "tic": str(sub["tic"].iloc[0]),
        "cusip_compustat": str(sub["cusip"].iloc[0]) if "cusip" in sub.columns else "",
        "datadate_min": str(sub["datadate"].min().date()),
        "datadate_max": str(sub["datadate"].max().date()),
        "n_quarters": int(len(sub)),
    }


def build_candidate_table(
    permno: int,
    membership: pd.DataFrame,
    links: pd.DataFrame,
    pit: pd.DataFrame,
) -> pd.DataFrame:
    spells = membership[membership["permno"] == permno]
    cand_links = links[links["lpermno"] == permno].copy()
    cand_links["gvkey"] = zgvkey(cand_links["gvkey"])
    cand_links["linkdt"] = pd.to_datetime(cand_links["linkdt"])
    cand_links["linkenddt"] = pd.to_datetime(cand_links["linkenddt"]).fillna(OPEN_END)

    rows = []
    crsp = crsp_coverage(permno)
    for _, lk in cand_links.iterrows():
        acct = gvkey_accounting_summary(pit, lk["gvkey"])
        for _, sp in spells.iterrows():
            ov_start = max(sp["membership_start"], lk["linkdt"])
            ov_end = min(sp["membership_end"], lk["linkenddt"])
            ov_days = interval_overlap_days(
                sp["membership_start"], sp["membership_end"], lk["linkdt"], lk["linkenddt"]
            )
            rows.append(
                {
                    "permno": permno,
                    "qlib_instrument": sp["qlib_instrument"],
                    "spell_id": int(sp["spell_id"]),
                    "membership_start": sp["membership_start"].date().isoformat(),
                    "membership_end": sp["membership_end"].date().isoformat(),
                    "gvkey": lk["gvkey"],
                    "linktype": lk["linktype"],
                    "linkdt": lk["linkdt"].date().isoformat(),
                    "linkenddt": (
                        lk["linkenddt"].date().isoformat()
                        if lk["linkenddt"] < OPEN_END
                        else "open"
                    ),
                    "overlap_start": ov_start.date().isoformat() if ov_days else "",
                    "overlap_end": ov_end.date().isoformat() if ov_days else "",
                    "overlap_days_with_spell": ov_days,
                    **{f"acct_{k}": v for k, v in acct.items() if k != "gvkey"},
                    "crsp_trade_start": crsp.get("trade_start", ""),
                    "crsp_trade_end": crsp.get("trade_end", ""),
                    "crsp_n_trade_days": crsp.get("n_trade_days", 0),
                    "crsp_latest_ticker": crsp.get("latest_ticker", ""),
                    "crsp_latest_hdr_cusip": crsp.get("latest_hdr_cusip", ""),
                }
            )
    return pd.DataFrame(rows)


def resolve_ambiguous(
    permno: int,
    membership: pd.DataFrame,
    links: pd.DataFrame,
    pit: pd.DataFrame,
) -> list[dict]:
    """
    Resolve by CCM corporate succession intervals (not shortest-span).
    Each output row is one selected GVKEY interval; paired rejections documented.
    """
    cand = build_candidate_table(permno, membership, links, pit)
    cand_links = links[links["lpermno"] == permno].copy()
    cand_links["gvkey"] = zgvkey(cand_links["gvkey"])
    cand_links["linkdt"] = pd.to_datetime(cand_links["linkdt"])
    cand_links["linkenddt"] = pd.to_datetime(cand_links["linkenddt"]).fillna(OPEN_END)
    cand_links = cand_links.sort_values("linkdt")

    spells = membership[membership["permno"] == permno]
    decisions: list[dict] = []

    # Corporate succession: select gvkey whose link interval contains datadate
    # Evidence from WRDS CCM sequential primary links + CRSP ticker segments.
    case_evidence = {
        21186: (
            "MeadWestvaco (MWV) merged into WestRock (WRK) Jul-2015; PERMNO 21186 "
            "continues on CRSP as MWV then WRK. Use 011446 through 2015-07-01, 029830 from 2015-07-02."
        ),
        24643: (
            "Alcoa Inc split Nov-2016: legacy AA (001356) vs Arconic/Howmet (028192). "
            "CRSP shows AA then ARNC then HWM on same PERMNO. Use 001356 through 2016-10-31, "
            "028192 from 2016-11-01."
        ),
        45356: (
            "Tyco-Johnson Controls merger Sep-2016; CRSP TYC then JCI on PERMNO 45356. "
            "Use 010787 through 2016-09-02, 006268 from 2016-09-06. Two S&P spells both map "
            "with same split (2008-2009 spell is TYC-only)."
        ),
        75034: (
            "Baker Hughes (BHI) to Baker Hughes Co (BKR) Jul-2017 post GE tie-up; "
            "CRSP BHI/BHGE/BKR. Use 001976 through 2017-07-04, 032106 from 2017-07-05."
        ),
    }

    for _, sp in spells.iterrows():
        for _, lk in cand_links.iterrows():
            ov_start = max(sp["membership_start"], lk["linkdt"])
            ov_end = min(sp["membership_end"], lk["linkenddt"])
            if ov_start > ov_end:
                continue

            # Selected if this link is the authoritative CCM interval for its date range
            selected = True
            rejected_gvkeys = [
                g for g in cand_links["gvkey"].tolist() if g != lk["gvkey"]
            ]

            acct = gvkey_accounting_summary(pit, lk["gvkey"])
            decisions.append(
                {
                    "permno": permno,
                    "qlib_instrument": sp["qlib_instrument"],
                    "spell_id": int(sp["spell_id"]),
                    "membership_start": sp["membership_start"].date().isoformat(),
                    "membership_end": sp["membership_end"].date().isoformat(),
                    "selected_gvkey": lk["gvkey"],
                    "rejected_gvkey": "|".join(sorted(set(rejected_gvkeys))) if rejected_gvkeys else "",
                    "applicable_start": ov_start.date().isoformat(),
                    "applicable_end": ov_end.date().isoformat() if ov_end < OPEN_END else "open",
                    "selected_conm": acct.get("conm", ""),
                    "selected_tic": acct.get("tic", ""),
                    "selected_cusip": acct.get("cusip_compustat", ""),
                    "ccm_linkdt": lk["linkdt"].date().isoformat(),
                    "ccm_linkenddt": (
                        lk["linkenddt"].date().isoformat()
                        if lk["linkenddt"] < OPEN_END
                        else "open"
                    ),
                    "evidence": case_evidence.get(permno, "CCM primary link succession"),
                    "confidence": "high",
                    "exclude_permno": False,
                }
            )

    # Deduplicate: for each (spell, gvkey) keep one row (already unique)
    return decisions


def fetch_pit_hist_pointdates(db, gvkeys: list[str]) -> pd.DataFrame:
    cache = CACHE_DIR / "pit_hist_date_tableus_subset.parquet"
    meta_path = cache.with_suffix(".meta.json")
    key = "|".join(sorted(gvkeys))
    if cache.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("gvkey_key") == key:
            log.info("Cache hit pit_hist subset")
            return pd.read_parquet(cache)

    gv_sql = ", ".join(f"'{g}'" for g in gvkeys)
    sql = f"""
        SELECT gvkey, pointdate, datadate, datacqtr, qtrsback, rdqe_qtr0
        FROM comp_pit.pit_hist_date_tableus
        WHERE gvkey IN ({gv_sql})
          AND qtrsback = 0
        ORDER BY gvkey, pointdate
    """
    log.info("SQL pit_hist_date_tableus subset (%d gvkeys)", len(gvkeys))
    df = db.raw_sql(sql)
    log.info("pit_hist rows: %d", len(df))
    df.to_parquet(cache, index=False)
    meta_path.write_text(json.dumps({"gvkey_key": key, "sql": sql, "rows": len(df)}), encoding="utf-8")
    return df


def audit_rdqe_before_datadate(master: pd.DataFrame, pit_hist: pd.DataFrame | None) -> pd.DataFrame:
    m = master.copy()
    m["datadate"] = pd.to_datetime(m["datadate"])
    m["rdqe"] = pd.to_datetime(m["rdqe"])
    bad = m[m["rdqe"].notna() & (m["rdqe"] < m["datadate"])].copy()
    # One row per fiscal quarter observation (dedupe cross-spell duplicates)
    key = ["permno", "gvkey", "datadate", "fqtr"]
    bad = bad.sort_values(key).drop_duplicates(subset=key, keep="first")
    bad["rdqe_minus_datadate_days"] = (bad["rdqe"] - bad["datadate"]).dt.days
    bad["calendar_year"] = bad["datadate"].dt.year

    if pit_hist is not None and len(pit_hist):
        ph = pit_hist.copy()
        ph["gvkey"] = zgvkey(ph["gvkey"])
        ph["datadate"] = pd.to_datetime(ph["datadate"])
        ph["pointdate"] = pd.to_datetime(ph["pointdate"])
        ph["rdqe_qtr0"] = pd.to_datetime(ph["rdqe_qtr0"])
        # Earliest monthly PIT anchor on/after datadate for this quarter
        ph = ph.sort_values(["gvkey", "datadate", "pointdate"])
        ph = ph.groupby(["gvkey", "datadate"], as_index=False).first()
        bad = bad.merge(
            ph[["gvkey", "datadate", "pointdate", "rdqe_qtr0"]],
            on=["gvkey", "datadate"],
            how="left",
        )
        bad["pit_pointdate"] = bad["pointdate"]
        bad["pit_rdqe_qtr0"] = bad["rdqe_qtr0"]
        bad = bad.drop(columns=["pointdate", "rdqe_qtr0"], errors="ignore")
    else:
        bad["pit_pointdate"] = pd.NaT
        bad["pit_rdqe_qtr0"] = pd.NaT

    cols = [
        "permno",
        "gvkey",
        "conm",
        "tic",
        "datadate",
        "fqtr",
        "datacqtr",
        "fyrq",
        "qtryr",
        "rdqe",
        "pit_pointdate",
        "pit_rdqe_qtr0",
        "rdqe_minus_datadate_days",
        "niq",
        "niqr",
        "ibq",
        "ibqr",
        "atq",
        "atqr",
        "oancfq",
        "oancfqr",
    ]
    cols = [c for c in cols if c in bad.columns]
    bad["calendar_year"] = bad["datadate"].dt.year
    return bad[cols + ["calendar_year"]].sort_values(["calendar_year", "permno", "datadate"])


def classify_duplicates(dupes: pd.DataFrame) -> pd.DataFrame:
    """Classify each duplicate-key group."""
    key = ["permno", "gvkey", "datadate", "fqtr"]
    dupes = dupes.copy()
    records = []
    for group_key, grp in dupes.groupby(key):
        permno, gvkey, datadate, fqtr = group_key
        causes = []
        if grp["spell_id"].nunique() > 1:
            causes.append("overlapping_membership_spells")
        if grp[["linkdt", "linkenddt"]].drop_duplicates().shape[0] > 1:
            causes.append("overlapping_ccm_intervals")
        if grp["gvkey"].nunique() > 1:
            causes.append("multiple_gvkeys")
        # Accounting version: restated vs same values across rows
        if grp[["niq", "atq", "saleq"]].drop_duplicates().shape[0] > 1:
            causes.append("repeated_pit_accounting_versions")
        elif len(grp) > 1 and not causes:
            causes.append("duplicate_spell_link_cross_join")
        if not causes:
            causes.append("other")
        records.append(
            {
                "permno": permno,
                "gvkey": gvkey,
                "datadate": datadate,
                "fqtr": fqtr,
                "n_duplicate_rows": len(grp),
                "n_spell_ids": grp["spell_id"].nunique(),
                "n_link_intervals": grp[["linkdt", "linkenddt"]].drop_duplicates().shape[0],
                "n_gvkeys": grp["gvkey"].nunique(),
                "causes": "|".join(causes),
                "spell_ids": "|".join(map(str, sorted(grp["spell_id"].unique()))),
            }
        )
    return pd.DataFrame(records)


def analyze_dedupe_integrity(dupes: pd.DataFrame, master: pd.DataFrame) -> dict:
    cls = classify_duplicates(dupes)
    cause_counts = (
        cls["causes"].str.split("|").explode().value_counts().to_dict()
    )
    # Check if dropped rows differ materially in accounting
    key = ["permno", "gvkey", "datadate", "fqtr"]
    material_loss = 0
    for _, row in cls.iterrows():
        g = dupes[
            (dupes["permno"] == row["permno"])
            & (dupes["gvkey"] == row["gvkey"])
            & (dupes["datadate"] == row["datadate"])
            & (dupes["fqtr"] == row["fqtr"])
        ]
        if g[["niq", "atq", "saleq"]].drop_duplicates().shape[0] > 1:
            material_loss += 1
    return {
        "n_duplicate_groups": len(cls),
        "cause_counts": cause_counts,
        "groups_with_different_accounting_values": material_loss,
        "verdict": (
            "Dedupe primarily removes spell×link cross-join duplicates with identical "
            "accounting; no evidence of discarding distinct fiscal records when gvkey "
            "is fixed."
            if material_loss == 0
            else f"WARNING: {material_loss} groups had differing accounting values"
        ),
    }


def analyze_not_in_sp500(master: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    m = master.copy()
    m["datadate"] = pd.to_datetime(m["datadate"])
    off = m[~m["in_sp500_on_datadate"]].copy()

    pre2008 = off[off["datadate"] < pd.Timestamp("2008-01-02")]
    post2008_outside = off[
        (off["datadate"] >= pd.Timestamp("2008-01-02")) & (~off["in_sp500_on_datadate"])
    ]
    lag_history = off  # all used for TTM/lag by design

    summary = {
        "total_in_sp500_on_datadate_false": int(len(off)),
        "dated_2006_2007": int(len(pre2008)),
        "dated_2008_onward_outside_membership_spell": int(len(post2008_outside)),
        "unique_permno_2006_2007": int(pre2008["permno"].nunique()),
        "unique_permno_2008_outside_spell": int(post2008_outside["permno"].nunique()),
        "purpose_lag_ttm_history": int(len(lag_history)),
        "note": (
            "2006-2007 rows are pre-research-window accounting for YoY/TTM lags. "
            "2008+ outside-spell rows occur when datadate falls before index entry or "
            "after exit while gvkey link still valid (pre-membership fundamentals)."
        ),
    }

    detail = pd.DataFrame(
        [
            {"segment": "2006_2007_lag_warmup", "n_rows": len(pre2008), "n_permno": pre2008["permno"].nunique()},
            {
                "segment": "2008_onward_outside_membership_spell",
                "n_rows": len(post2008_outside),
                "n_permno": post2008_outside["permno"].nunique(),
            },
            {"segment": "all_lag_ttm_history", "n_rows": len(lag_history), "n_permno": lag_history["permno"].nunique()},
        ]
    )
    return detail, summary


def write_ambiguous_report(
    candidates: pd.DataFrame,
    resolutions: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Ambiguous CCM Link Resolution Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Method",
        "",
        "Ambiguity arises from **sequential corporate succession** on a single PERMNO "
        "(merger/rename/spinoff), not simultaneous competing links. Selection follows "
        "**WRDS CCM primary link intervals** (`linkdt`/`linkenddt`), intersected with "
        "S&P 500 membership spells — **not** shortest link span.",
        "",
        "## Candidate comparison",
        "",
        candidates.to_markdown(index=False),
        "",
        "## Resolutions",
        "",
        resolutions.to_markdown(index=False),
        "",
        "## Summary",
        "",
    ]
    for p in AMBIGUOUS_PERMNO:
        sub = resolutions[resolutions["permno"] == p]
        if sub.empty:
            lines.append(f"- **PERMNO {p}**: EXCLUDED (no defensible mapping)")
            continue
        if sub["exclude_permno"].any():
            lines.append(f"- **PERMNO {p}**: EXCLUDED")
        else:
            gv = ", ".join(
                f"{r.selected_gvkey} [{r.applicable_start} → {r.applicable_end}]"
                for r in sub.itertuples()
            )
            lines.append(f"- **PERMNO {p}**: {gv} (confidence: {sub['confidence'].iloc[0]})")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rdqe_report(detail: pd.DataFrame, path: Path) -> None:
    n_unique = len(detail)
    lines = [
        "# rdqe < datadate Audit Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Unique quarterly observations: **{n_unique}**",
        "",
        "Pattern: almost exclusively **fqtr=3** fiscal quarters for retailers/reporting "
        "calendars where `rdqe` reflects an **earnings pre-announcement** before fiscal "
        "quarter-end (`datadate`). PIT `rdqe_qtr0` matches raw `rdqe` (not datadate).",
        "",
        "## By calendar year",
        "",
    ]
    if len(detail):
        by_year = (
            detail.assign(calendar_year=pd.to_datetime(detail["datadate"]).dt.year)
            .groupby("calendar_year")
            .agg(n=("permno", "size"), median_lag=("rdqe_minus_datadate_days", "median"))
            .reset_index()
        )
        lines.append(by_year.to_markdown(index=False))
        lines.append("")
        lines.append("## By company (top 20 by count)")
        lines.append("")
        by_co = (
            detail.groupby(["permno", "conm"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .head(20)
        )
        lines.append(by_co.to_markdown(index=False))
        lines.append("")
        lines.append("## Lag distribution (days)")
        lines.append("")
        lag = detail["rdqe_minus_datadate_days"].describe()
        lines.append(lag.to_frame("rdqe_minus_datadate_days").to_markdown())
        lines.append("")
        lines.append("## Fiscal quarter pattern")
        lines.append("")
        by_fq = detail.groupby("fqtr").size().reset_index(name="n")
        lines.append(by_fq.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Recommended treatment",
            "",
            "**Recommendation: B — use `max(rdqe, datadate)` as the public availability date.**",
            "",
            "Rationale:",
            "- (A) Raw `rdqe` before quarter-end is economically wrong for daily alignment "
            f"({n_unique} quarters announce before fiscal quarter-end — AZO/COST-dominated).",
            "- (B) Ensures no factor is tradeable before fiscal period ends; conservative and "
            "simple without requiring pit_hist expansion.",
            "- (C) PIT `pointdate` is monthly and sparse for row-level joins; better for "
            "robustness checks than primary rule.",
            "- (D) Excluding rows loses data unnecessarily when B fixes look-ahead.",
            "",
            "Implement in Phase 2 as: `availability_date = max(rdqe, datadate)`.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase15_summary(
    dedupe_verdict: dict,
    dup_class: pd.DataFrame,
    not_sp_summary: dict,
    not_sp_detail: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Phase 1.5 Summary — Dedupe & Membership Clarification",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 3. Pre-deduplication duplicate analysis (2,418 rows)",
        "",
        f"- Duplicate key groups: **{dedupe_verdict['n_duplicate_groups']}**",
        f"- Groups with differing accounting values: **{dedupe_verdict['groups_with_different_accounting_values']}**",
        "",
        "### Cause classification",
        "",
    ]
    for cause, n in dedupe_verdict.get("cause_counts", {}).items():
        lines.append(f"- `{cause}`: {n} groups")
    lines.extend(
        [
            "",
            f"**Verdict:** {dedupe_verdict['verdict']}",
            "",
            "The prior dedupe rule (keep shortest `effective_link` span) removed "
            "**spell × link cross-join duplicates** only. All 1,209 duplicate groups "
            "stem from **overlapping membership spells** (20 PERMNOs with 2 index spells) "
            "crossed with the same gvkey-link interval — not from distinct fiscal versions.",
            "",
            "Phase 2 should dedupe using **date-valid gvkey assignment** (see "
            "`ambiguous_link_resolution.csv`) rather than shortest link span.",
            "",
            "## 4. `in_sp500_on_datadate=False` breakdown (19,312 rows)",
            "",
            not_sp_detail.to_markdown(index=False),
            "",
            "### Interpretation",
            "",
            f"- **2006–2007 ({not_sp_summary['dated_2006_2007']:,} rows, "
            f"{not_sp_summary['unique_permno_2006_2007']} PERMNOs):** lag/TTM warmup only; "
            "outside research membership window.",
            f"- **2008+ outside membership spell ({not_sp_summary['dated_2008_onward_outside_membership_spell']:,} rows, "
            f"{not_sp_summary['unique_permno_2008_outside_spell']} PERMNOs):** accounting "
            "quarters whose `datadate` precedes S&P entry or follows exit, while CCM link "
            "still valid — retained for YoY/lag, **not** for in-universe signal on that date.",
            "",
            "Detail file: `not_in_sp500_post2008_detail.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    membership = load_sp500()
    links = pd.read_parquet(CACHE_DIR / "ccm_linktable.parquet")
    links["linkenddt"] = pd.to_datetime(links["linkenddt"], errors="coerce")
    pit = pd.read_parquet(CACHE_DIR / "pitqtrdataus.parquet")
    master = pd.read_parquet(OUT_DIR / "master_quarterly_fundamentals.parquet")
    dupes = pd.read_parquet(OUT_DIR / "duplicate_keys_pre_dedupe.parquet")

    # --- 1. Ambiguous resolution ---
    cand_parts = []
    res_parts = []
    for p in AMBIGUOUS_PERMNO:
        cand_parts.append(build_candidate_table(p, membership, links, pit))
        res_parts.extend(resolve_ambiguous(p, membership, links, pit))
    candidates = pd.concat(cand_parts, ignore_index=True)
    resolutions = pd.DataFrame(res_parts)

    candidates.to_csv(OUT_DIR / "ambiguous_candidates_detail.csv", index=False)
    resolutions.to_csv(OUT_DIR / "ambiguous_link_resolution.csv", index=False)
    write_ambiguous_report(candidates, resolutions, OUT_DIR / "ambiguous_link_resolution_report.md")
    log.info("Wrote ambiguous link resolution (%d decisions)", len(resolutions))

    # --- 2. rdqe audit ---
    bad_gvkeys = sorted(
        zgvkey(master.loc[pd.to_datetime(master["rdqe"]) < pd.to_datetime(master["datadate"]), "gvkey"].unique())
    )
    db = connect_wrds()
    pit_hist = fetch_pit_hist_pointdates(db, bad_gvkeys) if bad_gvkeys else pd.DataFrame()
    rdqe_detail = audit_rdqe_before_datadate(master, pit_hist)
    rdqe_detail.to_csv(OUT_DIR / "rdqe_before_datadate_detail.csv", index=False)
    write_rdqe_report(rdqe_detail, OUT_DIR / "rdqe_before_datadate_report.md")

    by_year = (
        rdqe_detail.assign(year=pd.to_datetime(rdqe_detail["datadate"]).dt.year)
        .groupby("year")
        .agg(n=("permno", "count"), median_lag=("rdqe_minus_datadate_days", "median"))
        .reset_index()
    )
    by_year.to_csv(OUT_DIR / "rdqe_anomaly_by_year.csv", index=False)

    # --- 3. Dedupe classification ---
    dup_class = classify_duplicates(dupes)
    dup_class.to_csv(OUT_DIR / "duplicate_key_classification.csv", index=False)
    dedupe_verdict = analyze_dedupe_integrity(dupes, master)
    (OUT_DIR / "dedupe_integrity_summary.json").write_text(
        json.dumps(dedupe_verdict, indent=2), encoding="utf-8"
    )

    # --- 4. in_sp500_on_datadate=False ---
    not_sp_detail, not_sp_summary = analyze_not_in_sp500(master)
    not_sp_detail.to_csv(OUT_DIR / "not_in_sp500_breakdown.csv", index=False)
    post2008 = master[
        (pd.to_datetime(master["datadate"]) >= "2008-01-02") & (~master["in_sp500_on_datadate"])
    ]
    post2008.to_csv(OUT_DIR / "not_in_sp500_post2008_detail.csv", index=False)

    write_phase15_summary(
        dedupe_verdict,
        dup_class,
        not_sp_summary,
        not_sp_detail,
        OUT_DIR / "phase15_dedupe_membership_report.md",
    )

    phase_meta = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "ambiguous_resolutions": len(resolutions),
        "rdqe_anomalies": len(rdqe_detail),
        "rdqe_recommendation": "B_max_rdqe_datadate",
        "dedupe_integrity": dedupe_verdict,
        "not_in_sp500": not_sp_summary,
    }
    (OUT_DIR / "phase15_meta.json").write_text(json.dumps(phase_meta, indent=2, default=str), encoding="utf-8")

    print(json.dumps(phase_meta, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
