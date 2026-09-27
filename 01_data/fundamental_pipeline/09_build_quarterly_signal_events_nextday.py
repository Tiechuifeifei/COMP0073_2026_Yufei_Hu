#!/usr/bin/env python3
"""09_build_quarterly_signal_events_nextday.py

Robustness variant of 08: first CRSP trading day strictly after
accounting_available_date. Writes parallel outputs; the main pipeline (08)
keeps date >= availability for Alpha20 / RD-Agent alignment."""

from __future__ import annotations

import importlib.util
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

OUT_DIR = PROJECT_ROOT / "data" / "quarterly_signal_events_nextday"
MAIN_OUT_DIR = PROJECT_ROOT / "data" / "quarterly_signal_events"
AUDIT_CACHE_DIR = MAIN_OUT_DIR / "cache"

TIMING_RULE = "strict_after"
TIMING_RULE_DESC = (
    "signal_start_date = first CRSP trading day with date > accounting_available_date"
)
RESEARCH_KEY = ["permno", "gvkey", "datadate", "fqtr"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_base_module():
    base_path = SCRIPT_DIR / "08_build_quarterly_signal_events.py"
    spec = importlib.util.spec_from_file_location("signal_events_08", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base module from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def map_signal_start_dates_nextday(master: pd.DataFrame, crsp: pd.DataFrame) -> pd.DataFrame:
    """First CRSP trading date per PERMNO with date > accounting_available_date."""
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
            for col in base.CRSP_SIGNAL_COLS:
                merged[col] = np.nan
        else:
            merged = pd.merge_asof(
                left,
                right,
                left_on="accounting_available_date",
                right_on="signal_start_date",
                direction="forward",
                allow_exact_matches=False,
            )
        parts.append(merged)

    out = pd.concat(parts, ignore_index=True)
    rename_cols = {
        k: v for k, v in base.CRSP_RENAME.items() if k in out.columns and k != "date"
    }
    return out.rename(columns=rename_cols)


def build_unmapped_nextday(df: pd.DataFrame) -> pd.DataFrame:
    unmapped = df[df["signal_start_date"].isna()].copy()
    reasons: list[str] = []
    for row in unmapped.itertuples(index=False):
        if row.crsp_permno_missing_flag:
            reasons.append("crsp_permno_missing")
        elif row.accounting_available_date > base.RESEARCH_MARKET_END:
            reasons.append("accounting_available_after_market_cutoff")
        else:
            reasons.append("no_crsp_trading_date_strictly_after_availability")
    unmapped["unmapped_reason"] = reasons
    return unmapped


def write_quality_report_nextday(
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
        "# Quarterly Signal Event Mapping — Quality Report (Next-Day Robustness)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Timing rule",
        "",
        f"**{TIMING_RULE_DESC}**",
        "",
        "Purpose: eliminate possible look-ahead from WRDS `rdqe` carrying an "
        "announcement calendar date without an intraday timestamp (before open vs after close).",
        "",
        "Main pipeline (`08_build_quarterly_signal_events.py`) uses "
        "`date >= accounting_available_date` and is unchanged.",
        "",
        "## Inputs",
        "",
        f"- Final master: `{master_path}`",
        f"- CRSP daily panel: `{crsp_path}`",
        f"- Research market cutoff: **{base.RESEARCH_MARKET_END.date()}**",
        "",
        "## Mapping summary",
        "",
        "| Metric | Count | Share |",
        "| --- | ---: | ---: |",
        f"| Total quarterly observations | {n:,} | 100% |",
        f"| Mapped to a CRSP `signal_start_date` | {int(mapped.sum()):,} | {base.pct(mapped.mean())} |",
        f"| Same-day signal (`same_day_signal_flag`) | {int(events['same_day_signal_flag'].sum()):,} | {base.pct(events['same_day_signal_flag'].mean())} |",
        f"| Unmapped (`no_future_trading_date_flag`) | {int(events['no_future_trading_date_flag'].sum()):,} | {base.pct(events['no_future_trading_date_flag'].mean())} |",
        f"| **`signal_usable_flag=True`** (tradable) | {int(events['signal_usable_flag'].sum()):,} | {base.pct(events['signal_usable_flag'].mean())} |",
        "",
        "## Delay distribution (mapped rows)",
        "",
        base.delay_bucket(delays).to_markdown(index=False),
        "",
        f"- Median delay: **{delays.median():.0f}** calendar days",
        f"- Mean delay: **{delays.mean():.2f}** calendar days",
        f"- Delay > 1 calendar day: **{int((delays > 1).sum()):,}** ({base.pct((delays > 1).mean())})",
        f"- Delay > 3 calendar days: **{int((delays > 3).sum()):,}** ({base.pct((delays > 3).mean())})",
        f"- Delay > 5 calendar days: **{int((delays > 5).sum()):,}** ({base.pct((delays > 5).mean())})",
        "",
        "## Quality flags",
        "",
        "| Check | Count | Share |",
        "| --- | ---: | ---: |",
        f"| Signal after membership end | {int(events['signal_after_membership_end_flag'].sum()):,} | {base.pct(events['signal_after_membership_end_flag'].mean())} |",
        f"| Signal before membership start | {int(events['signal_before_membership_start_flag'].sum()):,} | {base.pct(events['signal_before_membership_start_flag'].mean())} |",
        f"| Signal date after {base.RESEARCH_MARKET_END.date()} | {int((events['signal_start_date'] > base.RESEARCH_MARKET_END).sum()):,} | {base.pct((events['signal_start_date'] > base.RESEARCH_MARKET_END).mean())} |",
        f"| Availability after cutoff (unmappable in panel) | {int((events['accounting_available_date'] > base.RESEARCH_MARKET_END).sum()):,} | {base.pct((events['accounting_available_date'] > base.RESEARCH_MARKET_END).mean())} |",
        f"| Missing/invalid signal market cap | {int(invalid_mcap.sum()):,} | {base.pct(invalid_mcap.mean())} |",
        f"| History-only rows (`history_only_flag`) | {int(events['history_only_flag'].sum()):,} | {base.pct(events['history_only_flag'].mean())} |",
        f"| CRSP PERMNO missing from dsf panel | {int(events['crsp_permno_missing_flag'].sum()):,} | {base.pct(events['crsp_permno_missing_flag'].mean())} |",
        "",
        "### `signal_usable_flag` definition",
        "",
        "True only when all hold:",
        "",
        "1. Valid `signal_start_date` (first CRSP date **>** `accounting_available_date` for the PERMNO)",
        f"2. `signal_start_date` ≤ **{base.RESEARCH_MARKET_END.date()}**",
        "3. S&P 500 member on `signal_start_date`",
        "4. CCM PERMNO–GVKEY link valid on `signal_start_date`",
        "5. `history_only_flag=False`",
        "6. `signal_market_cap` present and > 0",
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
        "## Cross-rule comparison",
        "",
        "See `timing_rule_comparison.md` in this directory.",
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


def load_events_summary(path: Path, label: str) -> dict[str, Any]:
    meta_path = path / "signal_mapping_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "label": label,
            "path": str(path),
            "row_count": meta.get("row_count"),
            "mapped_count": meta.get("mapped_count"),
            "same_day_count": meta.get("same_day_count"),
            "usable_count": meta.get("usable_count"),
            "unmapped_count": meta.get("unmapped_count"),
        }

    df = pd.read_parquet(path / "quarterly_signal_events.parquet")
    mapped = df["signal_start_date"].notna()
    return {
        "label": label,
        "path": str(path),
        "row_count": int(len(df)),
        "mapped_count": int(mapped.sum()),
        "same_day_count": int(df["same_day_signal_flag"].sum()),
        "usable_count": int(df["signal_usable_flag"].sum()),
        "unmapped_count": int((~mapped).sum()),
        "delays": df.loc[mapped, "trading_day_delay_days"],
        "events": df,
    }


def write_timing_rule_comparison(
    path: Path,
    main_events: pd.DataFrame,
    next_events: pd.DataFrame,
    main_meta: dict[str, Any],
    next_meta: dict[str, Any],
) -> None:
    main_mapped = main_events["signal_start_date"].notna()
    next_mapped = next_events["signal_start_date"].notna()
    main_delays = main_events.loc[main_mapped, "trading_day_delay_days"]
    next_delays = next_events.loc[next_mapped, "trading_day_delay_days"]

    merged = main_events[RESEARCH_KEY + ["accounting_available_date", "signal_start_date", "signal_usable_flag"]].merge(
        next_events[RESEARCH_KEY + ["signal_start_date", "signal_usable_flag"]],
        on=RESEARCH_KEY,
        how="outer",
        suffixes=("_on_or_after", "_strict_after"),
        indicator=True,
    )

    both = merged[merged["_merge"] == "both"].copy()
    both_mapped = both["signal_start_date_on_or_after"].notna() & both["signal_start_date_strict_after"].notna()
    date_changed = both_mapped & (
        both["signal_start_date_on_or_after"] != both["signal_start_date_strict_after"]
    )
    main_only_mapped = both["signal_start_date_on_or_after"].notna() & both[
        "signal_start_date_strict_after"
    ].isna()
    usable_lost = both["signal_usable_flag_on_or_after"].astype(bool) & ~both[
        "signal_usable_flag_strict_after"
    ].astype(bool)
    usable_gained = ~both["signal_usable_flag_on_or_after"].astype(bool) & both[
        "signal_usable_flag_strict_after"
    ].astype(bool)
    main_same_day = both_mapped & (
        both["signal_start_date_on_or_after"] == both["accounting_available_date"]
    )
    shifted_by_stricter_rule = both_mapped & date_changed

    delay_delta = (
        both.loc[both_mapped, "signal_start_date_strict_after"]
        - both.loc[both_mapped, "signal_start_date_on_or_after"]
    ).dt.days
    delay_delta = delay_delta[delay_delta.notna()]

    avail_on_crsp_day = both_mapped & main_same_day

    main_avail_year = (
        main_events.assign(avail_year=main_events["accounting_available_date"].dt.year)
        .groupby("avail_year")
        .agg(
            n_rows=("permno", "size"),
            mapped_on_or_after=("signal_start_date", lambda s: int(s.notna().sum())),
            usable_on_or_after=("signal_usable_flag", lambda s: int(s.astype(bool).sum())),
        )
        .reset_index()
    )
    next_avail_year = (
        next_events.assign(avail_year=next_events["accounting_available_date"].dt.year)
        .groupby("avail_year")
        .agg(
            mapped_strict_after=("signal_start_date", lambda s: int(s.notna().sum())),
            usable_strict_after=("signal_usable_flag", lambda s: int(s.astype(bool).sum())),
        )
        .reset_index()
    )
    annual = main_avail_year.merge(next_avail_year, on="avail_year", how="outer").sort_values(
        "avail_year"
    )

    sample_changed = both.loc[date_changed].head(20).copy()
    if len(sample_changed):
        sample_changed["delay_shift_days"] = (
            sample_changed["signal_start_date_strict_after"]
            - sample_changed["signal_start_date_on_or_after"]
        ).dt.days
        sample_cols = [
            "permno",
            "gvkey",
            "datadate",
            "fqtr",
            "accounting_available_date",
            "signal_start_date_on_or_after",
            "signal_start_date_strict_after",
            "delay_shift_days",
            "signal_usable_flag_on_or_after",
            "signal_usable_flag_strict_after",
        ]
        sample_md = sample_changed[sample_cols].assign(
            datadate=lambda x: pd.to_datetime(x["datadate"]).dt.date,
            accounting_available_date=lambda x: pd.to_datetime(
                x["accounting_available_date"]
            ).dt.date,
            signal_start_date_on_or_after=lambda x: pd.to_datetime(
                x["signal_start_date_on_or_after"]
            ).dt.date,
            signal_start_date_strict_after=lambda x: pd.to_datetime(
                x["signal_start_date_strict_after"]
            ).dt.date,
        ).to_markdown(index=False)
    else:
        sample_md = "_No rows with differing signal dates._"

    lines = [
        "# Timing Rule Comparison — On/After vs Strict-After",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Rules compared",
        "",
        "| Variant | Script | Output directory | Rule |",
        "| --- | --- | --- | --- |",
        "| **Main (Alpha20 / RD-Agent)** | `08_build_quarterly_signal_events.py` | `data/quarterly_signal_events/` | "
        "First CRSP day with `date >= accounting_available_date` |",
        "| **Robustness (next-day)** | `09_build_quarterly_signal_events_nextday.py` | "
        "`data/quarterly_signal_events_nextday/` | "
        "First CRSP day with `date > accounting_available_date` |",
        "",
        f"Research market cutoff: **{base.RESEARCH_MARKET_END.date()}**",
        "",
        "## Headline counts",
        "",
        "| Metric | On/after (main) | Strict-after (robustness) | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Total observations | {main_meta['row_count']:,} | {next_meta['row_count']:,} | 0 |",
        f"| Mapped | {main_meta['mapped_count']:,} | {next_meta['mapped_count']:,} | "
        f"{next_meta['mapped_count'] - main_meta['mapped_count']:+,} |",
        f"| Same-day mappings | {main_meta['same_day_count']:,} | {next_meta['same_day_count']:,} | "
        f"{next_meta['same_day_count'] - main_meta['same_day_count']:+,} |",
        f"| Usable (`signal_usable_flag`) | {main_meta['usable_count']:,} | {next_meta['usable_count']:,} | "
        f"{next_meta['usable_count'] - main_meta['usable_count']:+,} |",
        f"| Unmapped | {main_meta['unmapped_count']:,} | {next_meta['unmapped_count']:,} | "
        f"{next_meta['unmapped_count'] - main_meta['unmapped_count']:+,} |",
        "",
        "## Delay distribution",
        "",
        "### Main: on or after",
        "",
        base.delay_bucket(main_delays).to_markdown(index=False),
        "",
        f"- Median / mean delay: **{main_delays.median():.0f}** / **{main_delays.mean():.2f}** calendar days",
        "",
        "### Robustness: strictly after",
        "",
        base.delay_bucket(next_delays).to_markdown(index=False),
        "",
        f"- Median / mean delay: **{next_delays.median():.0f}** / **{next_delays.mean():.2f}** calendar days",
        "",
        "## Observations affected by the stricter rule",
        "",
        f"| Effect | Count | Share of all obs |",
        f"| --- | ---: | ---: |",
        f"| Both rules mapped, different `signal_start_date` | {int(date_changed.sum()):,} | {base.pct(date_changed.sum() / len(main_events))} |",
        f"| Main mapped on availability CRSP day (`same-day` under main rule) | {int(main_same_day.sum()):,} | {base.pct(main_same_day.sum() / len(both))} |",
        f"| Availability date is a CRSP trading day (main same-day candidates) | {int(avail_on_crsp_day.sum()):,} | {base.pct(avail_on_crsp_day.sum() / len(both))} |",
        f"| Mapped under main but unmapped under strict-after | {int(main_only_mapped.sum()):,} | {base.pct(main_only_mapped.sum() / len(both))} |",
        f"| Usable under main, not usable under strict-after | {int(usable_lost.sum()):,} | {base.pct(usable_lost.sum() / len(both))} |",
        f"| Not usable under main, usable under strict-after | {int(usable_gained.sum()):,} | {base.pct(usable_gained.sum() / len(both))} |",
        "",
    ]

    if len(delay_delta):
        lines.extend(
            [
                "### Signal-date shift (strict-after minus on/after), mapped under both rules",
                "",
                f"- Median shift: **{delay_delta.median():.0f}** calendar days",
                f"- Mean shift: **{delay_delta.mean():.2f}** calendar days",
                f"- Shift = 1 day: **{int((delay_delta == 1).sum()):,}**",
                f"- Shift > 1 day: **{int((delay_delta > 1).sum()):,}**",
                "",
            ]
        )

    lines.extend(
        [
            "## Annual coverage (by availability year)",
            "",
            annual.to_markdown(index=False),
            "",
            "## Sample rows with different signal dates (first 20)",
            "",
            sample_md,
            "",
            "## Interpretation",
            "",
            "The strict-after rule shifts signals later when `accounting_available_date` "
            "falls on an actual CRSP trading day for the PERMNO. When availability falls "
            "on a non-trading day, both rules typically select the same next CRSP session.",
            "",
            "Same-day mappings drop to zero under strict-after by construction. Usable-count "
            "changes reflect both the one-day delay and edge cases near membership/link "
            f"boundaries or the **{base.RESEARCH_MARKET_END.date()}** market cutoff.",
            "",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()

    master_path = base.resolve_path(base.FINAL_MASTER_CANDIDATES, "Final master")
    crsp_path = base.resolve_path(base.CRSP_PANEL_CANDIDATES, "CRSP panel")
    missing_path = base.resolve_path(base.MISSING_PERMNO_CANDIDATES, "Missing PERMNOs")
    extreme_path = (
        base.EXTREME_MCAP_CANDIDATES[0] if base.EXTREME_MCAP_CANDIDATES[0].exists() else None
    )

    master = base.load_master(master_path)
    crsp = base.load_crsp(crsp_path)
    membership = pd.read_parquet(base.MEMBERSHIP_PATH)
    membership["membership_start"] = pd.to_datetime(membership["membership_start"])
    membership["membership_end"] = pd.to_datetime(membership["membership_end"])
    links = pd.read_parquet(base.LINK_INTERVALS_PATH)
    links["gvkey"] = base.zgvkey(links["gvkey"])
    links["permno"] = links["permno"].astype(int)
    links["effective_link_start"] = pd.to_datetime(links["effective_link_start"])
    links["effective_link_end"] = pd.to_datetime(links["effective_link_end"])

    missing_df = pd.read_csv(missing_path)
    missing_permnos = set(missing_df["permno"].astype(int).tolist()) | base.KNOWN_MISSING_CRSP_PERMNOS

    log.info("Timing rule: %s", TIMING_RULE_DESC)
    log.info(
        "CRSP panel: %d rows, %d PERMNOs, max date %s",
        len(crsp),
        crsp["permno"].nunique(),
        crsp["date"].max().date(),
    )

    events = map_signal_start_dates_nextday(master, crsp)
    events = base.add_signal_flags(events, membership, links, missing_permnos)

    # Reuse Phase 2C WRDS audit cache (read-only).
    original_audit_dir = base.AUDIT_CACHE_DIR
    base.AUDIT_CACHE_DIR = AUDIT_CACHE_DIR
    wrds_audit = base.wrds_missing_permno_audit(missing_permnos)
    base.AUDIT_CACHE_DIR = original_audit_dir

    missing_audit = base.audit_missing_crsp_permnos(
        missing_permnos, master, membership, links, wrds_audit
    )
    unmapped = build_unmapped_nextday(events)

    out_pq = OUT_DIR / "quarterly_signal_events.parquet"
    out_pkl = OUT_DIR / "quarterly_signal_events.pkl"
    events.to_parquet(out_pq, index=False)
    events.to_pickle(out_pkl)
    unmapped.to_csv(OUT_DIR / "unmapped_signal_events.csv", index=False)
    missing_audit.to_csv(OUT_DIR / "missing_crsp_permno_audit.csv", index=False)

    write_quality_report_nextday(
        OUT_DIR / "signal_mapping_quality_report.md",
        master_path=master_path,
        crsp_path=crsp_path,
        events=events,
        unmapped=unmapped,
        missing_audit=missing_audit,
        extreme_path=extreme_path,
    )

    meta = {
        "phase": "2C_robustness_nextday",
        "timing_rule": TIMING_RULE,
        "timing_rule_description": TIMING_RULE_DESC,
        "main_pipeline_dir": str(MAIN_OUT_DIR),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "final_master": str(master_path),
            "crsp_daily_market": str(crsp_path),
            "missing_permnos": str(missing_path),
            "extreme_market_cap_changes": str(extreme_path) if extreme_path else None,
            "membership_spells": str(base.MEMBERSHIP_PATH),
            "valid_link_intervals": str(base.LINK_INTERVALS_PATH),
        },
        "research_market_end": base.RESEARCH_MARKET_END.date().isoformat(),
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
            "timing_rule_comparison": str(OUT_DIR / "timing_rule_comparison.md"),
        },
    }
    (OUT_DIR / "signal_mapping_meta.json").write_text(
        json.dumps(meta, indent=2, default=base.json_safe),
        encoding="utf-8",
    )

    main_events_path = MAIN_OUT_DIR / "quarterly_signal_events.parquet"
    if not main_events_path.exists():
        log.warning("Main pipeline events not found; skipping timing_rule_comparison.md")
    else:
        main_events = pd.read_parquet(main_events_path)
        main_meta = json.loads((MAIN_OUT_DIR / "signal_mapping_meta.json").read_text())
        write_timing_rule_comparison(
            OUT_DIR / "timing_rule_comparison.md",
            main_events=main_events,
            next_events=events,
            main_meta=main_meta,
            next_meta=meta,
        )
        log.info("Wrote timing rule comparison report")

    log.info(
        "Mapped %d / %d; usable %d; unmapped %d",
        meta["mapped_count"],
        meta["row_count"],
        meta["usable_count"],
        meta["unmapped_count"],
    )
    log.info("Phase 2C next-day robustness complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
