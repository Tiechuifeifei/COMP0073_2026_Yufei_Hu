#!/usr/bin/env python3
"""04_build_sp500_pit_master.py

Build the S&P 500 point-in-time quarterly fundamental master from
staging/instruments/sp500.txt, crsp_a_ccm.ccmxpf_linktable, and
comp_pit.pitqtrdataus (WRDS)."""

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
SP500_PATH = PROJECT_ROOT / "staging" / "instruments" / "sp500.txt"
OUT_DIR = SCRIPT_DIR / "data" / "sp500_pit_master"
CACHE_DIR = OUT_DIR / "cache"

ACCOUNTING_START = "2006-01-01"
MEMBERSHIP_RESEARCH_START = "2008-01-02"
OPEN_END = pd.Timestamp("2099-12-31")

# Identifiers / PIT timing (comp_pit.pitqtrdataus; datafmt/consol/popsrc/indfmt/fyear not in PIT table)
ID_AND_PIT_COLS = [
    "gvkey",
    "conm",
    "tic",
    "datadate",
    "datacqtr",
    "fqtr",
    "fyrq",
    "qtryr",
    "qtrend",
    "rdqe",
    "prelimqprd",
    "finalqprd",
    "compstq",
    "compstqr",
]

# Raw accounting fields for 12 planned characteristics (+ unrestated PIT columns)
ACCOUNTING_COLS = [
    "atq",
    "atqr",
    "ceqq",
    "ceqqr",
    "seqq",
    "seqqr",
    "txditcq",
    "txditcqr",
    "pstkq",
    "pstkqr",
    "niq",
    "niqr",
    "ibq",
    "ibqr",
    "saleq",
    "saleqr",
    "cogsq",
    "cogsqr",
    "oibdpq",
    "oibdpqr",
    "dpq",
    "dpqr",
    "oancfq",
    "oancfqr",
    "dlcq",
    "dlcqr",
    "dlttq",
    "dlttqr",
    "ltq",
    "ltqr",
    "xintq",
    "xintqr",
]

PIT_SELECT_COLS = ID_AND_PIT_COLS + ACCOUNTING_COLS

CCM_COLS = [
    "gvkey",
    "linkprim",
    "linktype",
    "lpermno",
    "lpermco",
    "usedflag",
    "linkdt",
    "linkenddt",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def parse_sp500(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"Bad sp500 line: {line!r}")
        instrument, start, end = parts
        permno = int(instrument.lstrip("P"))
        rows.append(
            {
                "qlib_instrument": instrument,
                "permno": permno,
                "membership_start": pd.Timestamp(start),
                "membership_end": pd.Timestamp(end),
            }
        )
    df = pd.DataFrame(rows)
    df["spell_id"] = np.arange(len(df), dtype=int)
    log.info("Loaded sp500.txt: %d spells, %d unique PERMNOs", len(df), df["permno"].nunique())
    return df


def cache_key(label: str, payload: str) -> Path:
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{label}_{digest}.parquet"


def cache_meta_path(data_path: Path) -> Path:
    return data_path.with_suffix(".meta.json")


def load_cache(path: Path) -> pd.DataFrame | None:
    if path.exists() and cache_meta_path(path).exists():
        log.info("Cache hit: %s", path.name)
        return pd.read_parquet(path)
    return None


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def save_cache(path: Path, df: pd.DataFrame, meta: dict) -> None:
    df.to_parquet(path, index=False)
    cache_meta_path(path).write_text(
        json.dumps(meta, indent=2, default=json_safe), encoding="utf-8"
    )
    log.info("Cached %s (%d rows) -> %s", meta.get("label", path.stem), len(df), path.name)


def gvkey_sql_list(gvkeys: list[str]) -> str:
    return ", ".join(f"'{g}'" for g in gvkeys)


def permno_sql_list(permnos: list[int]) -> str:
    return ", ".join(str(int(p)) for p in sorted(permnos))


def fetch_ccm_links(db, permnos: list[int]) -> tuple[pd.DataFrame, str]:
    perm_sql = permno_sql_list(permnos)
    sql = f"""
        SELECT {", ".join(CCM_COLS)}
        FROM crsp_a_ccm.ccmxpf_linktable
        WHERE linkprim = 'P'
          AND linktype IN ('LU', 'LC')
          AND usedflag = 1
          AND lpermno IN ({perm_sql})
        ORDER BY lpermno, linkdt, gvkey
    """
    log.info("SQL [ccmxpf_linktable]:\n%s", sql.strip())
    df = db.raw_sql(sql)
    log.info("WRDS returned ccmxpf_linktable: %d rows", len(df))
    return df, sql


def fetch_pitqtr(db, gvkeys: list[str]) -> tuple[pd.DataFrame, str]:
    cols = ", ".join(PIT_SELECT_COLS)
    gv_sql = gvkey_sql_list(gvkeys)
    sql = f"""
        SELECT {cols}
        FROM comp_pit.pitqtrdataus
        WHERE gvkey IN ({gv_sql})
          AND datadate >= '{ACCOUNTING_START}'
        ORDER BY gvkey, datadate, fqtr
    """
    log.info("SQL [pitqtrdataus] gvkeys=%d datadate>=%s", len(gvkeys), ACCOUNTING_START)
    log.info("SQL [pitqtrdataus]:\n%s", sql.strip()[:2000])
    df = db.raw_sql(sql)
    log.info("WRDS returned pitqtrdataus: %d rows x %d cols", len(df), len(df.columns))
    return df, sql


def normalize_gvkey(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.zfill(6)


def normalize_links(links: pd.DataFrame) -> pd.DataFrame:
    out = links.copy()
    out["gvkey"] = normalize_gvkey(out["gvkey"])
    out["permno"] = out["lpermno"].astype(int)
    out["linkdt"] = pd.to_datetime(out["linkdt"], errors="coerce")
    out["linkenddt"] = pd.to_datetime(out["linkenddt"], errors="coerce").fillna(OPEN_END)
    return out


def intervals_overlap(a0, a1, b0, b1) -> bool:
    return a0 <= b1 and b0 <= a1


def spell_link_overlaps(spell: pd.Series, link: pd.Series) -> bool:
    return intervals_overlap(
        spell["membership_start"],
        spell["membership_end"],
        link["linkdt"],
        link["linkenddt"],
    )


def audit_ccm_mapping(
    membership: pd.DataFrame, links: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return audit rows, unmatched permnos, ambiguous links, spell-level audit, summary stats."""
    permnos = sorted(membership["permno"].unique())
    spell_audit_rows: list[dict] = []

    for _, spell in membership.iterrows():
        sub = links[links["permno"] == spell["permno"]]
        overlapping = sub[sub.apply(lambda r: spell_link_overlaps(spell, r), axis=1)]
        gvkeys = sorted(overlapping["gvkey"].unique())
        spell_audit_rows.append(
            {
                "spell_id": int(spell["spell_id"]),
                "qlib_instrument": spell["qlib_instrument"],
                "permno": int(spell["permno"]),
                "membership_start": spell["membership_start"].date().isoformat(),
                "membership_end": spell["membership_end"].date().isoformat(),
                "n_overlapping_links": int(len(overlapping)),
                "n_distinct_gvkeys": int(len(gvkeys)),
                "gvkeys": "|".join(gvkeys),
                "has_link": len(overlapping) > 0,
                "ambiguous_multi_gvkey": len(gvkeys) > 1,
            }
        )

    spell_audit = pd.DataFrame(spell_audit_rows)

    permno_summary = (
        spell_audit.groupby("permno", as_index=False)
        .agg(
            n_spells=("spell_id", "count"),
            spells_with_link=("has_link", "sum"),
            spells_without_link=("has_link", lambda s: int((~s).sum())),
            any_ambiguous_spell=("ambiguous_multi_gvkey", "max"),
            gvkeys_seen=("gvkeys", lambda s: "|".join(sorted({g for x in s if x for g in x.split("|")}))),
        )
        .sort_values("permno")
    )
    permno_summary["has_any_link"] = permno_summary["spells_with_link"] > 0

    unmatched_permnos = permno_summary[~permno_summary["has_any_link"]].copy()

    ambiguous_spells = spell_audit[spell_audit["ambiguous_multi_gvkey"]].copy()
    ambiguous_links = ambiguous_spells[
        [
            "spell_id",
            "qlib_instrument",
            "permno",
            "membership_start",
            "membership_end",
            "n_overlapping_links",
            "n_distinct_gvkeys",
            "gvkeys",
        ]
    ]

    # PERMNOs with multiple GVKEYs over time (any point, not necessarily simultaneous)
    multi_gvkey_permnos = permno_summary[
        permno_summary["gvkeys_seen"].str.contains("\\|") & permno_summary["has_any_link"]
    ].copy()

    # Coverage by calendar year: spell active in year AND has link overlapping that year
    year_rows = []
    for year in range(2006, pd.Timestamp.now().year + 2):
        y0 = pd.Timestamp(f"{year}-01-01")
        y1 = pd.Timestamp(f"{year}-12-31")
        active_spells = membership[
            (membership["membership_start"] <= y1) & (membership["membership_end"] >= y0)
        ]
        linked = 0
        unlinked = 0
        for _, sp in active_spells.iterrows():
            sub = links[(links["permno"] == sp["permno"]) & links.apply(lambda r: spell_link_overlaps(sp, r), axis=1)]
            if len(sub):
                linked += 1
            else:
                unlinked += 1
        year_rows.append(
            {
                "calendar_year": year,
                "active_membership_spells": len(active_spells),
                "spells_with_ccm_link": linked,
                "spells_without_ccm_link": unlinked,
                "spell_link_rate": linked / len(active_spells) if len(active_spells) else np.nan,
            }
        )
    coverage_by_year = pd.DataFrame(year_rows)

    summary = {
        "total_permnos": int(membership["permno"].nunique()),
        "total_membership_spells": int(len(membership)),
        "permnos_with_at_least_one_valid_link": int(permno_summary["has_any_link"].sum()),
        "unmatched_permnos": int(len(unmatched_permnos)),
        "membership_spells_without_link": int((~spell_audit["has_link"]).sum()),
        "spells_with_multiple_simultaneous_gvkeys": int(spell_audit["ambiguous_multi_gvkey"].sum()),
        "permnos_with_multiple_gvkeys_over_time": int(len(multi_gvkey_permnos)),
    }

    return spell_audit, unmatched_permnos, ambiguous_links, coverage_by_year, summary


def build_valid_link_intervals(membership: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """PERMNO-GVKEY link intervals that overlap at least one membership spell."""
    rows: list[dict] = []
    for _, link in links.iterrows():
        spells = membership[membership["permno"] == link["permno"]]
        for _, spell in spells.iterrows():
            if not spell_link_overlaps(spell, link):
                continue
            eff_start = max(spell["membership_start"], link["linkdt"])
            eff_end = min(spell["membership_end"], link["linkenddt"])
            if eff_start > eff_end:
                continue
            rows.append(
                {
                    "permno": int(link["permno"]),
                    "qlib_instrument": spell["qlib_instrument"],
                    "gvkey": link["gvkey"],
                    "linkdt": link["linkdt"],
                    "linkenddt": link["linkenddt"],
                    "linktype": link["linktype"],
                    "linkprim": link["linkprim"],
                    "membership_start": spell["membership_start"],
                    "membership_end": spell["membership_end"],
                    "spell_id": int(spell["spell_id"]),
                    "effective_link_start": eff_start,
                    "effective_link_end": eff_end,
                }
            )
    out = pd.DataFrame(rows)
    log.info("Built %d valid PERMNO-GVKEY link intervals (spell x link overlaps)", len(out))
    return out


def in_membership(date: pd.Timestamp, membership: pd.DataFrame, permno: int) -> bool:
    sub = membership[membership["permno"] == permno]
    return bool(((sub["membership_start"] <= date) & (sub["membership_end"] >= date)).any())


def build_master_quarterly(
    pit: pd.DataFrame,
    link_intervals: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    pit = pit.copy()
    pit["gvkey"] = normalize_gvkey(pit["gvkey"])
    pit["datadate"] = pd.to_datetime(pit["datadate"], errors="coerce")
    pit["rdqe"] = pd.to_datetime(pit["rdqe"], errors="coerce")

    # Join pit rows to link intervals on gvkey with datadate within link validity
    merged_parts: list[pd.DataFrame] = []
    for gvkey, pit_sub in pit.groupby("gvkey", sort=False):
        links_sub = link_intervals[link_intervals["gvkey"] == gvkey]
        if links_sub.empty:
            continue
        for _, lk in links_sub.iterrows():
            mask = (pit_sub["datadate"] >= lk["linkdt"]) & (pit_sub["datadate"] <= lk["linkenddt"])
            hit = pit_sub.loc[mask].copy()
            if hit.empty:
                continue
            for col in [
                "permno",
                "qlib_instrument",
                "linkdt",
                "linkenddt",
                "linktype",
                "linkprim",
                "membership_start",
                "membership_end",
                "spell_id",
                "effective_link_start",
                "effective_link_end",
            ]:
                hit[col] = lk[col]
            hit["ccm_valid_on_datadate"] = True
            hit["in_sp500_on_datadate"] = hit["datadate"].apply(
                lambda d: intervals_overlap(
                    d,
                    d,
                    lk["membership_start"],
                    lk["membership_end"],
                )
            )
            hit["in_sp500_research_window"] = hit["datadate"] >= pd.Timestamp(MEMBERSHIP_RESEARCH_START)
            hit["accounting_available_date"] = hit["rdqe"]
            hit["look_ahead_risk_rdqe_before_datadate"] = hit["rdqe"].notna() & (
                hit["rdqe"] < hit["datadate"]
            )
            merged_parts.append(hit)

    if not merged_parts:
        return pd.DataFrame()

    master = pd.concat(merged_parts, ignore_index=True)
    master = master.sort_values(["permno", "gvkey", "datadate", "fqtr", "spell_id"]).reset_index(drop=True)
    log.info("Master quarterly rows before dedupe: %d", len(master))
    return master


def dedupe_master(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["permno", "gvkey", "datadate", "fqtr"]
    dup_mask = master.duplicated(subset=key, keep=False)
    dups = master[dup_mask].sort_values(key)
    # Keep row with narrowest effective link window (most specific spell-link)
    master = master.copy()
    master["_link_span"] = (master["effective_link_end"] - master["effective_link_start"]).dt.days
    master = master.sort_values(key + ["_link_span"]).drop_duplicates(subset=key, keep="first")
    master = master.drop(columns=["_link_span"])
    log.info("Master quarterly rows after dedupe: %d; duplicate-key groups: %d", len(master), len(dups))
    return master, dups


def write_ccm_mapping_report(
    summary: dict,
    coverage_by_year: pd.DataFrame,
    unmatched: pd.DataFrame,
    ambiguous: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# CCM Mapping Audit Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Universe: `{SP500_PATH}`",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"| --- | ---: |",
    ]
    for k, v in summary.items():
        lines.append(f"| {k} | {v} |")

    lines.extend(["", "## Coverage by calendar year", "", coverage_by_year.to_markdown(index=False)])

    lines.extend(
        [
            "",
            "## Unmatched PERMNOs",
            "",
            f"Count: **{len(unmatched)}**",
            "",
        ]
    )
    if len(unmatched):
        lines.append(unmatched.head(50).to_markdown(index=False))
        if len(unmatched) > 50:
            lines.append(f"\n_({len(unmatched) - 50} more in unmatched_permnos.csv)_")

    lines.extend(
        [
            "",
            "## Ambiguous membership spells (multiple GVKEYs overlapping spell)",
            "",
            f"Count: **{len(ambiguous)}**",
            "",
        ]
    )
    if len(ambiguous):
        lines.append(ambiguous.head(30).to_markdown(index=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def write_master_quality_report(
    master: pd.DataFrame,
    membership: pd.DataFrame,
    spell_audit: pd.DataFrame,
    summary: dict,
    dupes: pd.DataFrame,
    path: Path,
) -> None:
    rdqe_cov = float(master["rdqe"].notna().mean()) if len(master) else 0.0
    miss = master[ID_AND_PIT_COLS + ACCOUNTING_COLS].isna().mean().sort_values(ascending=False)

    # Coverage by year on datadate
    m = master.copy()
    m["calendar_year"] = m["datadate"].dt.year
    by_year = (
        m.groupby("calendar_year")
        .agg(
            n_rows=("gvkey", "size"),
            n_permno=("permno", "nunique"),
            n_gvkey=("gvkey", "nunique"),
            rdqe_nonnull=("rdqe", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    by_year["rdqe_rate"] = by_year["rdqe_nonnull"] / by_year["n_rows"]

    by_membership = pd.DataFrame(
        [
            {
                "segment": "in_sp500_on_datadate=True",
                "n_rows": int(m["in_sp500_on_datadate"].sum()),
                "n_permno": int(m.loc[m["in_sp500_on_datadate"], "permno"].nunique()),
            },
            {
                "segment": "in_sp500_on_datadate=False (pre-membership accounting for TTM)",
                "n_rows": int((~m["in_sp500_on_datadate"]).sum()),
                "n_permno": int(m.loc[~m["in_sp500_on_datadate"], "permno"].nunique()),
            },
            {
                "segment": "in_sp500_research_window (datadate>=2008-01-02)",
                "n_rows": int(m["in_sp500_research_window"].sum()),
                "n_permno": int(m.loc[m["in_sp500_research_window"], "permno"].nunique()),
            },
        ]
    )

    # Sparse histories: permnos with few quarters in research window
    research = m[m["in_sp500_research_window"]]
    qcounts = research.groupby("permno").size().sort_values()
    sparse = qcounts[qcounts <= 8].reset_index(name="n_quarters_research_window")

    dup_key = master.duplicated(subset=["permno", "gvkey", "datadate", "fqtr"], keep=False).sum()
    look_ahead = int(m["look_ahead_risk_rdqe_before_datadate"].sum()) if "look_ahead_risk_rdqe_before_datadate" in m else 0

    lines = [
        "# SP500 PIT Master — Data Quality Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        f"| --- | ---: |",
        f"| Unique PERMNO | {master['permno'].nunique()} |",
        f"| Unique GVKEY | {master['gvkey'].nunique()} |",
        f"| Quarterly observations | {len(master):,} |",
        f"| datadate range | {master['datadate'].min().date()} → {master['datadate'].max().date()} |",
        f"| rdqe coverage | {pct(rdqe_cov)} |",
        f"| Duplicate keys (permno,gvkey,datadate,fqtr) | {dup_key} |",
        f"| rdqe before datadate (look-ahead flag) | {look_ahead} |",
        "",
        "## CCM audit (from build)",
        "",
    ]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")

    lines.extend(["", "## Coverage by calendar year (datadate)", "", by_year.to_markdown(index=False)])
    lines.extend(["", "## Coverage by membership status", "", by_membership.to_markdown(index=False)])

    lines.extend(["", "## Field missingness (top 20)", ""])
    top_miss = miss.head(20).reset_index()
    top_miss.columns = ["field", "missing_rate"]
    top_miss["missing_rate"] = top_miss["missing_rate"].map(pct)
    lines.append(top_miss.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## PIT metadata note",
            "",
            "`comp_pit.pitqtrdataus` does not expose `datafmt`, `consol`, `popsrc`, `indfmt`, or `fyear` "
            "(fundq fields). Fiscal year quarter uses `fyrq` / `qtryr` instead.",
            "",
            "## Sparse accounting histories (≤8 quarters in research window)",
            "",
            f"Count: **{len(sparse)}** PERMNOs",
            "",
        ]
    )
    if len(sparse):
        lines.append(sparse.head(30).to_markdown(index=False))

    if len(dupes):
        lines.extend(
            [
                "",
                "## Duplicate-key groups (pre-dedupe sample)",
                "",
                f"Groups: **{dupes['permno'].nunique()}** permnos; see cache for full list",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Look-ahead risk assessment",
            "",
            "- Factor availability must use **`rdqe`** (not `datadate`) when aligning to CRSP daily bars.",
            f"- Rows with `rdqe < datadate`: **{look_ahead}** (should be 0 for clean PIT).",
            "- Unmatched PERMNOs have **no** Compustat rows in this master (no silent gvkey guess).",
            "- CCM links are **date-valid** (`linkdt`–`linkenddt`); no timeless permno→gvkey map.",
            "",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    if not SP500_PATH.exists():
        log.error("Missing universe file: %s", SP500_PATH)
        return 1

    membership = parse_sp500(SP500_PATH)
    membership.to_parquet(OUT_DIR / "membership_spells.parquet", index=False)

    permnos = sorted(membership["permno"].unique())
    ccm_cache = CACHE_DIR / "ccm_linktable.parquet"
    ccm_meta = {
        "label": "ccmxpf_linktable",
        "permnos": [int(p) for p in permnos],
        "filters": "linkprim=P, linktype in LC/LU, usedflag=1",
    }

    links = load_cache(ccm_cache)
    db = None
    if links is None:
        db = connect_wrds()
        log.info("Connected to WRDS")
        links, ccm_sql = fetch_ccm_links(db, permnos)
        save_cache(ccm_cache, links, {**ccm_meta, "sql": ccm_sql})
    else:
        ccm_sql = json.loads(cache_meta_path(ccm_cache).read_text()).get("sql", "")

    links = normalize_links(links)

    spell_audit, unmatched, ambiguous, coverage_by_year, ccm_summary = audit_ccm_mapping(
        membership, links
    )

    spell_audit.to_csv(OUT_DIR / "ccm_mapping_audit.csv", index=False)
    unmatched.to_csv(OUT_DIR / "unmatched_permnos.csv", index=False)
    ambiguous.to_csv(OUT_DIR / "ambiguous_links.csv", index=False)
    coverage_by_year.to_csv(OUT_DIR / "ccm_coverage_by_year.csv", index=False)
    write_ccm_mapping_report(
        ccm_summary,
        coverage_by_year,
        unmatched,
        ambiguous,
        OUT_DIR / "ccm_mapping_report.md",
    )
    log.info("CCM audit complete: %s", ccm_summary)

    mapped_gvkeys = sorted(links["gvkey"].unique())
    pit_cache = CACHE_DIR / "pitqtrdataus.parquet"
    pit_meta = {
        "label": "pitqtrdataus",
        "gvkeys": mapped_gvkeys,
        "datadate_min": ACCOUNTING_START,
        "columns": PIT_SELECT_COLS,
    }

    pit = load_cache(pit_cache)
    if pit is None:
        if db is None:
            db = connect_wrds()
            log.info("Connected to WRDS")
        pit, pit_sql = fetch_pitqtr(db, mapped_gvkeys)
        save_cache(pit_cache, pit, {**pit_meta, "sql": pit_sql})
    else:
        pit_sql = json.loads(cache_meta_path(pit_cache).read_text()).get("sql", "")

    link_intervals = build_valid_link_intervals(membership, links)
    link_intervals.to_parquet(OUT_DIR / "valid_link_intervals.parquet", index=False)

    master = build_master_quarterly(pit, link_intervals, membership)
    master, dupes = dedupe_master(master)
    if len(dupes):
        dupes.to_parquet(OUT_DIR / "duplicate_keys_pre_dedupe.parquet", index=False)

    master_path_pq = OUT_DIR / "master_quarterly_fundamentals.parquet"
    master_path_pkl = OUT_DIR / "master_quarterly_fundamentals.pkl"
    master.to_parquet(master_path_pq, index=False)
    master.to_pickle(master_path_pkl)
    log.info("Wrote master: %s (%d rows)", master_path_pq, len(master))

    write_master_quality_report(
        master,
        membership,
        spell_audit,
        ccm_summary,
        dupes,
        OUT_DIR / "master_data_quality_report.md",
    )

    run_meta = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sp500_path": str(SP500_PATH),
        "ccm_summary": ccm_summary,
        "master_rows": len(master),
        "master_unique_permno": int(master["permno"].nunique()),
        "master_unique_gvkey": int(master["gvkey"].nunique()),
        "sql_logged": {"ccm": ccm_sql[:500], "pit": pit_sql[:500] if pit_sql else ""},
    }
    (OUT_DIR / "build_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(json.dumps(run_meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
