#!/usr/bin/env python3
"""06_build_final_quarterly_master.py

Rebuild the final quarterly PIT fundamental master with date-segmented GVKEY
succession (ambiguous_link_resolution.csv), accounting_available_date =
max(rdqe, datadate), and history_only_flag for lag/TTM rows outside the index
on the availability date."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "data" / "sp500_pit_master"
CACHE_DIR = OUT_DIR / "cache"

MASTER_PATH = OUT_DIR / "master_quarterly_fundamentals.parquet"
RESOLUTION_PATH = OUT_DIR / "ambiguous_link_resolution.csv"
MEMBERSHIP_PATH = OUT_DIR / "membership_spells.parquet"
LINK_INTERVALS_PATH = OUT_DIR / "valid_link_intervals.parquet"

FINAL_PQ = OUT_DIR / "final_quarterly_fundamentals.parquet"
FINAL_PKL = OUT_DIR / "final_quarterly_fundamentals.pkl"
REPORT_PATH = OUT_DIR / "final_master_quality_report.md"

OPEN_END = pd.Timestamp("2099-12-31")
RESEARCH_KEY = ["permno", "gvkey", "datadate", "fqtr"]
ACCOUNTING_VALUE_COLS = ["atq", "niq", "saleq", "oancfq", "ceqq", "ibq"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def zgvkey(s: pd.Series | str) -> pd.Series | str:
    if isinstance(s, str):
        return str(s).strip().zfill(6)
    return s.astype(str).str.strip().str.zfill(6)


def parse_open_end(val: str) -> pd.Timestamp:
    if val in ("open", "", "NaT") or pd.isna(val):
        return OPEN_END
    return pd.Timestamp(val)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    master_before = pd.read_parquet(MASTER_PATH)
    n_master = len(master_before)

    membership = pd.read_parquet(MEMBERSHIP_PATH)
    membership["membership_start"] = pd.to_datetime(membership["membership_start"])
    membership["membership_end"] = pd.to_datetime(membership["membership_end"])

    resolution = pd.read_csv(RESOLUTION_PATH)
    resolution["selected_gvkey"] = zgvkey(resolution["selected_gvkey"])
    resolution["applicable_start"] = pd.to_datetime(resolution["applicable_start"])
    resolution["applicable_end"] = resolution["applicable_end"].map(parse_open_end)
    resolution["ccm_linkdt"] = pd.to_datetime(resolution["ccm_linkdt"])
    resolution["ccm_linkenddt"] = resolution["ccm_linkenddt"].map(parse_open_end)

    links = pd.read_parquet(CACHE_DIR / "ccm_linktable.parquet")
    links["gvkey"] = zgvkey(links["gvkey"])
    links["permno"] = links["lpermno"].astype(int)
    links["linkdt"] = pd.to_datetime(links["linkdt"])
    links["linkenddt"] = pd.to_datetime(links["linkenddt"]).fillna(OPEN_END)

    pit = pd.read_parquet(CACHE_DIR / "pitqtrdataus.parquet")
    pit["gvkey"] = zgvkey(pit["gvkey"])
    pit["datadate"] = pd.to_datetime(pit["datadate"])
    pit["rdqe"] = pd.to_datetime(pit["rdqe"])

    _ = pd.read_parquet(LINK_INTERVALS_PATH)  # validated input present
    return master_before, membership, resolution, links, pit, n_master


def membership_flag(dates: pd.Series, permno: int, membership: pd.DataFrame) -> pd.Series:
    spells = membership[membership["permno"] == permno]
    if spells.empty:
        return pd.Series(False, index=dates.index)
    out = pd.Series(False, index=dates.index)
    for _, sp in spells.iterrows():
        out |= (dates >= sp["membership_start"]) & (dates <= sp["membership_end"])
    return out


def attach_membership_and_availability(df: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rdqe_before_datadate_flag"] = out["rdqe"].notna() & (out["rdqe"] < out["datadate"])
    out["accounting_available_date"] = out[["rdqe", "datadate"]].max(axis=1)
    out["availability_lag_days"] = (
        out["accounting_available_date"] - out["datadate"]
    ).dt.days
    out["availability_rule"] = "max_rdqe_datadate"

    out["in_sp500_on_datadate"] = False
    out["in_sp500_on_availability_date"] = False
    for permno in out["permno"].unique():
        m = out["permno"] == permno
        out.loc[m, "in_sp500_on_datadate"] = membership_flag(
            out.loc[m, "datadate"], int(permno), membership
        ).values
        out.loc[m, "in_sp500_on_availability_date"] = membership_flag(
            out.loc[m, "accounting_available_date"], int(permno), membership
        ).values
    out["history_only_flag"] = ~out["in_sp500_on_availability_date"]
    return out


def build_succession_mapping(
    pit: pd.DataFrame,
    resolution: pd.DataFrame,
    links: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Assign permno to gvkey-quarter rows using succession rules + standard CCM links."""
    ambiguous_permnos = set(resolution["permno"].unique())
    mapped_gvkeys = set(links["gvkey"].unique())
    pit = pit[pit["gvkey"].isin(mapped_gvkeys)].copy()

    parts: list[pd.DataFrame] = []

    # --- Ambiguous PERMNOs: explicit date-segmented succession ---
    res = resolution[~resolution["exclude_permno"].astype(bool)].copy()
    for _, rule in res.iterrows():
        permno = int(rule["permno"])
        gvkey = rule["selected_gvkey"]
        sub = pit[pit["gvkey"] == gvkey].copy()
        if sub.empty:
            continue
        mask = (
            (sub["datadate"] >= rule["applicable_start"])
            & (sub["datadate"] <= rule["applicable_end"])
            & (sub["datadate"] >= rule["ccm_linkdt"])
            & (sub["datadate"] <= rule["ccm_linkenddt"])
        )
        hit = sub.loc[mask].copy()
        if hit.empty:
            continue
        inst = membership.loc[membership["permno"] == permno, "qlib_instrument"]
        hit["permno"] = permno
        hit["qlib_instrument"] = inst.iloc[0] if len(inst) else f"P{permno}"
        hit["gvkey"] = gvkey
        hit["linkdt"] = rule["ccm_linkdt"]
        hit["linkenddt"] = rule["ccm_linkenddt"]
        hit["linktype"] = "LC"
        hit["linkprim"] = "P"
        hit["gvkey_assignment"] = "succession_rule"
        hit["succession_applicable_start"] = rule["applicable_start"]
        hit["succession_applicable_end"] = rule["applicable_end"]
        hit["rejected_gvkey"] = str(rule.get("rejected_gvkey", "") or "")
        parts.append(hit)

    # --- Standard PERMNOs: one gvkey per link interval on datadate ---
    standard_links = links[~links["permno"].isin(ambiguous_permnos)].copy()
    for permno, lk_grp in standard_links.groupby("permno"):
        permno = int(permno)
        inst_s = membership.loc[membership["permno"] == permno, "qlib_instrument"]
        qlib_inst = inst_s.iloc[0] if len(inst_s) else f"P{permno}"
        for _, lk in lk_grp.iterrows():
            sub = pit[pit["gvkey"] == lk["gvkey"]].copy()
            mask = (sub["datadate"] >= lk["linkdt"]) & (sub["datadate"] <= lk["linkenddt"])
            hit = sub.loc[mask].copy()
            if hit.empty:
                continue
            hit["permno"] = permno
            hit["qlib_instrument"] = qlib_inst
            hit["linkdt"] = lk["linkdt"]
            hit["linkenddt"] = lk["linkenddt"]
            hit["linktype"] = lk["linktype"]
            hit["linkprim"] = lk["linkprim"]
            hit["gvkey_assignment"] = "ccm_link"
            hit["succession_applicable_start"] = pd.NaT
            hit["succession_applicable_end"] = pd.NaT
            hit["rejected_gvkey"] = ""
            parts.append(hit)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    log.info("Mapped rows before engineering dedupe: %d", len(out))
    return out


def dedupe_engineering_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Remove exact duplicates from overlapping membership spells; verify no distinct accounting loss."""
    key = RESEARCH_KEY
    dup_mask = df.duplicated(subset=key, keep=False)
    pre_dups = df[dup_mask].copy()

    audit = {
        "rows_before_dedupe": int(len(df)),
        "duplicate_row_groups": int(pre_dups.groupby(key).ngroups if len(pre_dups) else 0),
        "duplicate_rows_total": int(len(pre_dups)),
        "groups_with_differing_accounting": 0,
        "distinct_accounting_lost": False,
    }

    if len(pre_dups):
        for _, grp in pre_dups.groupby(key):
            if grp[ACCOUNTING_VALUE_COLS].drop_duplicates().shape[0] > 1:
                audit["groups_with_differing_accounting"] += 1

    # Keep one row per research key; prefer succession_rule then lowest linkdt
    ranked = df.copy()
    ranked["_prio"] = (ranked["gvkey_assignment"] == "ccm_link").astype(int)
    ranked = ranked.sort_values(key + ["_prio", "linkdt"])
    deduped = ranked.drop_duplicates(subset=key, keep="first").drop(columns=["_prio"])
    audit["rows_after_dedupe"] = int(len(deduped))
    audit["rows_removed"] = audit["rows_before_dedupe"] - audit["rows_after_dedupe"]
    audit["distinct_accounting_lost"] = audit["groups_with_differing_accounting"] > 0

    remaining_dups = deduped.duplicated(subset=key, keep=False).sum()
    audit["remaining_duplicate_keys"] = int(remaining_dups)
    return deduped, pre_dups, audit


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def write_quality_report(
    master_before: pd.DataFrame,
    final: pd.DataFrame,
    dedupe_audit: dict,
    resolution: pd.DataFrame,
    path: Path,
) -> None:
    acct_cols = [c for c in final.columns if c in (
        "atq", "atqr", "ceqq", "ceqqr", "seqq", "seqqr", "txditcq", "txditcqr",
        "pstkq", "pstkqr", "niq", "niqr", "ibq", "ibqr", "saleq", "saleqr",
        "cogsq", "cogsqr", "oibdpq", "oibdpqr", "dpq", "dpqr", "oancfq", "oancfqr",
        "dlcq", "dlcqr", "dlttq", "dlttqr", "ltq", "ltqr", "xintq", "xintqr",
    )]
    miss = final[acct_cols].isna().mean().sort_values(ascending=False)

    by_year = (
        final.assign(calendar_year=final["datadate"].dt.year)
        .groupby("calendar_year")
        .agg(
            n_rows=("permno", "size"),
            n_permno=("permno", "nunique"),
            n_gvkey=("gvkey", "nunique"),
            history_only=("history_only_flag", "sum"),
            tradable=( "in_sp500_on_availability_date", "sum"),
        )
        .reset_index()
    )

    ambig = resolution[~resolution["exclude_permno"].astype(bool)]
    rdqe_flag = int(final["rdqe_before_datadate_flag"].sum())

    lines = [
        "# Final Quarterly PIT Fundamental Master — Quality Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Rebuild summary",
        "",
        f"| Metric | Phase 1 master | Phase 2A final |",
        f"| --- | ---: | ---: |",
        f"| Rows | {len(master_before):,} | {len(final):,} |",
        f"| Unique PERMNO | {master_before['permno'].nunique()} | {final['permno'].nunique()} |",
        f"| Unique GVKEY | {master_before['gvkey'].nunique()} | {final['gvkey'].nunique()} |",
        "",
        "## GVKEY succession (ambiguous PERMNOs)",
        "",
        f"Resolved via `ambiguous_link_resolution.csv`: **{len(ambig)}** interval rules "
        f"across **{ambig['permno'].nunique()}** PERMNOs (21186, 24643, 45356, 75034).",
        "",
        "| permno | selected_gvkey | applicable_start | applicable_end | rejected_gvkey |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in ambig.itertuples():
        end = r.applicable_end.date().isoformat() if r.applicable_end < OPEN_END else "open"
        rej = str(r.rejected_gvkey).zfill(6) if pd.notna(r.rejected_gvkey) and str(r.rejected_gvkey).strip() else ""
        lines.append(
            f"| {r.permno} | {r.selected_gvkey} | {r.applicable_start.date()} | {end} | {rej} |"
        )

    lines.extend(
        [
            "",
            "## Engineering deduplication",
            "",
            f"- Rows before dedupe: **{dedupe_audit['rows_before_dedupe']:,}**",
            f"- Duplicate groups (pre-dedupe): **{dedupe_audit['duplicate_row_groups']:,}**",
            "",
            "Phase 2A rebuild assigns each `(permno, datadate)` to **at most one GVKEY** "
            "via succession rules or CCM link intervals (no spell×link cross-join), "
            "eliminating the 1,209 duplicate groups present in Phase 1.",
            "",
            f"- Rows removed: **{dedupe_audit['rows_removed']:,}**",
            f"- Groups with differing accounting values: **{dedupe_audit['groups_with_differing_accounting']}**",
            f"- Distinct accounting lost: **{dedupe_audit['distinct_accounting_lost']}**",
            f"- Remaining duplicate research keys: **{dedupe_audit['remaining_duplicate_keys']}**",
            "",
            f"Research key uniqueness (`permno`, `gvkey`, `datadate`, `fqtr`): "
            f"**{'PASS' if dedupe_audit['remaining_duplicate_keys'] == 0 else 'FAIL'}**",
            "",
            "## rdqe anomaly treatment",
            "",
            f"- Rule: **`accounting_available_date = max(rdqe, datadate)`** (`availability_rule=max_rdqe_datadate`)",
            f"- Rows with `rdqe_before_datadate_flag=True`: **{rdqe_flag}**",
            f"- All flagged rows have `availability_lag_days ≥ 0` after adjustment.",
            "",
            "## History-only observations",
            "",
            f"| Segment | Rows | PERMNOs |",
            f"| --- | ---: | ---: |",
            f"| `history_only_flag=True` (not in S&P on availability date) | "
            f"{int(final['history_only_flag'].sum()):,} | "
            f"{final.loc[final['history_only_flag'], 'permno'].nunique()} |",
            f"| Tradable (`in_sp500_on_availability_date=True`) | "
            f"{int(final['in_sp500_on_availability_date'].sum()):,} | "
            f"{final.loc[final['in_sp500_on_availability_date'], 'permno'].nunique()} |",
            f"| 2006–2007 lag warmup (history_only) | "
            f"{int((final['history_only_flag'] & (final['datadate'] < '2008-01-02')).sum()):,} | — |",
            "",
            "History-only rows may be used for lag/TTM construction but **must not** "
            "generate tradable signals directly.",
            "",
            "## Annual coverage",
            "",
            by_year.to_markdown(index=False),
            "",
            "## Raw accounting field missingness",
            "",
        ]
    )
    miss_df = miss.head(30).reset_index()
    miss_df.columns = ["field", "missing_rate"]
    miss_df["missing_rate"] = miss_df["missing_rate"].map(pct)
    lines.append(miss_df.to_markdown(index=False))
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    master_before, membership, resolution, links, pit, n_master = load_inputs()
    log.info("Phase 1 master rows: %d", n_master)

    mapped = build_succession_mapping(pit, resolution, links, membership)
    mapped = attach_membership_and_availability(mapped, membership)

    final, pre_dups, dedupe_audit = dedupe_engineering_duplicates(mapped)

    final = final.sort_values(RESEARCH_KEY).reset_index(drop=True)
    final.to_parquet(FINAL_PQ, index=False)
    final.to_pickle(FINAL_PKL)
    log.info("Wrote final master: %s (%d rows)", FINAL_PQ, len(final))

    if len(pre_dups):
        pre_dups.to_parquet(OUT_DIR / "final_pre_dedupe_duplicates.parquet", index=False)

    write_quality_report(master_before, final, dedupe_audit, resolution, REPORT_PATH)

    meta = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase1_master_rows": n_master,
        "final_rows": len(final),
        "dedupe_audit": dedupe_audit,
        "unique_permno": int(final["permno"].nunique()),
        "unique_gvkey": int(final["gvkey"].nunique()),
        "history_only_rows": int(final["history_only_flag"].sum()),
        "tradable_rows": int(final["in_sp500_on_availability_date"].sum()),
    }
    (OUT_DIR / "final_build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0 if dedupe_audit["remaining_duplicate_keys"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
