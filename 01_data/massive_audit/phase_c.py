"""Phase C: Massive vs WRDS/CRSP price equivalence on 2023-01-01..2025-12-31."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from massive_audit.client import MassiveClient
from massive_audit.config import (
    AUDIT_DIR,
    CRSP_CSV_DIR,
    PRICE_OVERLAP_END,
    PRICE_OVERLAP_START,
    PRICE_SAMPLE,
    REPORT_DIR,
    SPY_CSV,
    ensure_dirs,
)


def _ms_to_date(ms: int) -> pd.Timestamp:
    # Massive daily bars are ET session; date in UTC-5/UTC-4 can shift. Use UTC date then
    # normalize via pandas; compare on calendar date from timestamp in US/Eastern.
    return pd.to_datetime(ms, unit="ms", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)


def load_crsp(permno: int) -> pd.DataFrame:
    path = CRSP_CSV_DIR / f"p{permno}.csv" if permno != 84398 else SPY_CSV
    if permno == 84398:
        path = SPY_CSV
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= PRICE_OVERLAP_START) & (df["date"] <= PRICE_OVERLAP_END)].copy()
    df["ret"] = df["close"].pct_change()
    return df


def load_massive_daily(client: MassiveClient, ticker: str, adjusted: bool) -> pd.DataFrame:
    rows = client.paginate(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{PRICE_OVERLAP_START}/{PRICE_OVERLAP_END}",
        {"adjusted": "true" if adjusted else "false", "limit": 50000, "sort": "asc"},
        max_pages=10,
    )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["date"] = out["t"].map(_ms_to_date)
    out = out.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    out["ret"] = out["close"].pct_change()
    return out[["date", "open", "high", "low", "close", "volume", "ret"]].drop_duplicates("date")


def _corr(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 10:
        return np.nan
    return float(a[mask].corr(b[mask]))


def _mae(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(a[mask] - b[mask])))


def _med_rel(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna() & (b.abs() > 1e-12)
    if not mask.any():
        return np.nan
    return float(np.median(np.abs(a[mask] / b[mask] - 1.0)))


def run_price_equivalence() -> dict:
    ensure_dirs()
    client = MassiveClient()
    rows = []
    samples = list(PRICE_SAMPLE) + [("SPY", 84398)]
    for ticker, permno in samples:
        crsp = load_crsp(permno)
        adj = load_massive_daily(client, ticker, True)
        raw = load_massive_daily(client, ticker, False)
        if adj.empty:
            rows.append({"ticker": ticker, "permno": permno, "status": "MASSIVE_EMPTY", "splice_ok": "NO"})
            continue
        merged = crsp.merge(adj, on="date", how="outer", suffixes=("_crsp", "_mass"))
        both = merged.dropna(subset=["close_crsp", "close_mass"])
        crsp_only = merged[merged["close_crsp"].notna() & merged["close_mass"].isna()]
        mass_only = merged[merged["close_mass"].notna() & merged["close_crsp"].isna()]
        # Compare split-adjusted CRSP (staging/csv) to Massive adjusted.
        close_corr = _corr(both["close_crsp"], both["close_mass"])
        ret_corr = _corr(both["ret_crsp"], both["ret_mass"])
        rel = _med_rel(both["close_mass"], both["close_crsp"])
        vol_rel = _med_rel(both["volume_mass"], both["volume_crsp"])
        ret_mae = _mae(both["ret_crsp"], both["ret_mass"])
        # Unadjusted Massive vs CRSP raw = close/factor
        crsp_raw_close = crsp["close"] / crsp["factor"] if "factor" in crsp.columns else crsp["close"]
        raw_corr = np.nan
        if not raw.empty and "factor" in crsp.columns:
            tmp = crsp.assign(crsp_raw_close=crsp_raw_close).merge(
                raw[["date", "close"]].rename(columns={"close": "mass_raw_close"}),
                on="date",
                how="inner",
            )
            raw_corr = _corr(tmp["crsp_raw_close"], tmp["mass_raw_close"])
        splice = "YES" if (ret_corr is not np.nan and ret_corr >= 0.999 and both.shape[0] >= 700) else "NO"
        if ret_corr is not np.nan and 0.99 <= ret_corr < 0.999:
            splice = "CAUTION"
        rows.append(
            {
                "ticker": ticker,
                "permno": permno,
                "crsp_n": int(len(crsp)),
                "massive_adj_n": int(len(adj)),
                "overlap_n": int(len(both)),
                "crsp_only_n": int(len(crsp_only)),
                "massive_only_n": int(len(mass_only)),
                "overlap_start": str(both["date"].min().date()) if len(both) else "",
                "overlap_end": str(both["date"].max().date()) if len(both) else "",
                "split_adj_close_corr": close_corr,
                "split_adj_close_median_rel_abs": rel,
                "return_corr": ret_corr,
                "return_mae": ret_mae,
                "volume_median_rel_abs": vol_rel,
                "unadj_close_corr_vs_crsp_dlyclose": raw_corr,
                "definition_note": "CRSP staging/csv close = DlyClose/DlyCumFacPr (split-adj to latest). Massive adjusted=true is split-adjusted, typically not dividend-adjusted. Levels may differ; returns should match if split handling agrees.",
                "splice_ok": splice,
                "status": "OK",
            }
        )
        print(f"C {ticker}: overlap={len(both)} ret_corr={ret_corr} splice={splice}")

    df = pd.DataFrame(rows)
    csv_path = AUDIT_DIR / "price_equivalence_audit.csv"
    df.to_csv(csv_path, index=False)

    ok = df["splice_ok"].eq("YES").all() if len(df) else False
    caution = df["splice_ok"].isin(["YES", "CAUTION"]).all() if len(df) else False
    verdict = "PASS_RETURNS_SPLICABLE" if ok else ("PASS_WITH_CAUTION" if caution else "FAIL")
    cols = [
        "ticker", "overlap_n", "crsp_only_n", "massive_only_n", "return_corr",
        "return_mae", "split_adj_close_corr", "split_adj_close_median_rel_abs",
        "volume_median_rel_abs", "splice_ok",
    ]
    table_df = df[[c for c in cols if c in df.columns]]
    lines = [
        "# Price equivalence audit",
        "",
        f"Window: {PRICE_OVERLAP_START} to {PRICE_OVERLAP_END}.",
        "",
        f"**Splice verdict:** `{verdict}`",
        "",
        "CRSP source: `staging/csv/p{permno}.csv` (split-adjusted via DlyCumFacPr). Massive: `/v2/aggs` `adjusted=true`.",
        "",
        "Bitwise equality is not required. Splice is acceptable if daily return correlation ≥ 0.999 on the overlap and missing-date gaps are small.",
        "",
        "```",
        table_df.to_string(index=False),
        "```",
        "",
        "## Can Massive 2026 prices be concatenated onto D2?",
        "",
        "If return correlation is ≥ 0.999: **yes, via return-splicing** onto the last overlapping CRSP/Qlib close, after applying the same first-day Qlib normalization ratio. Do not paste raw Massive dollar levels into Qlib bins without rebasing — Massive and CRSP levels can differ even when returns match.",
        "",
        "Volume is a known definition gap (CRSP vs consolidated SIP). Alpha20 uses volume in correlations; a 2026 extension should keep Massive volume internally consistent rather than mix CRSP and Massive volume in the same rolling window.",
        "",
    ]
    (REPORT_DIR / "price_equivalence_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "verdict": verdict,
        "n_tickers": int(len(df)),
        "min_return_corr": None if df.empty else float(np.nanmin(pd.to_numeric(df["return_corr"], errors="coerce"))),
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    (AUDIT_DIR / "price_equivalence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {csv_path} verdict={verdict}")
    return summary
