#!/usr/bin/env python3
"""
02_download_mag5_pit_panel.py

Download a small point-in-time quarterly accounting sample for:
  AAPL, MSFT, AMZN, GOOGL (Alphabet), META

Tables:
  - comp_pit.pitqtrdataus
  - comp_pit.pit_hist_date_tableus
  - crsp_a_ccm.ccmxpf_linktable

Outputs under: fundamental_pipeline/data/mag5_pit_sample/
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wrds_utils import connect_wrds

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "data" / "mag5_pit_sample"

TARGET_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
# Known gvkeys as fallback when ticker history is ambiguous.
FALLBACK_GVKEYS = {
    "AAPL": "001690",
    "MSFT": "012141",
    "AMZN": "064768",
    "GOOGL": "160329",
    "META": "166014",
}


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def gvkey_list_sql(gvkeys: list[str]) -> str:
    return ", ".join(f"'{g}'" for g in gvkeys)


def resolve_gvkeys(db) -> pd.DataFrame:
    """Map target tickers to gvkey using pitqtrdataus metadata."""
    tickers_sql = ", ".join(f"'{t}'" for t in TARGET_TICKERS)
    query = f"""
        SELECT gvkey,
               UPPER(TRIM(tic)) AS tic,
               conm,
               MIN(datadate) AS first_datadate,
               MAX(datadate) AS last_datadate,
               COUNT(*) AS n_rows
        FROM comp_pit.pitqtrdataus
        WHERE UPPER(TRIM(tic)) IN ({tickers_sql})
        GROUP BY gvkey, UPPER(TRIM(tic)), conm
        ORDER BY tic, gvkey
    """
    found = db.raw_sql(query)
    if found.empty:
        found = pd.DataFrame(columns=["gvkey", "tic", "conm"])

    rows: list[dict] = []
    for ticker in TARGET_TICKERS:
        subset = found[found["tic"] == ticker].copy()
        if subset.empty:
            gvkey = FALLBACK_GVKEYS[ticker]
            rows.append(
                {
                    "ticker": ticker,
                    "gvkey": gvkey,
                    "conm": None,
                    "source": "fallback",
                    "n_rows": None,
                    "first_datadate": None,
                    "last_datadate": None,
                }
            )
            continue

        # Prefer the gvkey with the most rows (handles rare duplicate tic spellings).
        best = subset.sort_values("n_rows", ascending=False).iloc[0]
        rows.append(
            {
                "ticker": ticker,
                "gvkey": str(best["gvkey"]).zfill(6),
                "conm": best["conm"],
                "source": "pitqtrdataus",
                "n_rows": int(best["n_rows"]),
                "first_datadate": str(best["first_datadate"]),
                "last_datadate": str(best["last_datadate"]),
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest["gvkey"].duplicated().any():
        dupes = manifest[manifest["gvkey"].duplicated(keep=False)]
        raise RuntimeError(f"Duplicate gvkey mapping:\n{dupes}")

    return manifest


def download_table(db, sql: str, label: str) -> pd.DataFrame:
    print(f"Downloading {label} ...")
    df = db.raw_sql(sql)
    print(f"  -> {len(df):,} rows x {len(df.columns)} cols")
    return df


def main() -> int:
    ensure_out_dir()
    db = connect_wrds()
    print("Connected to WRDS.\n")

    manifest = resolve_gvkeys(db)
    gvkeys = manifest["gvkey"].tolist()
    gv_sql = gvkey_list_sql(gvkeys)

    manifest_path = OUT_DIR / "company_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("Company manifest:")
    print(manifest.to_string(index=False))
    print()

    pitq_sql = f"""
        SELECT *
        FROM comp_pit.pitqtrdataus
        WHERE gvkey IN ({gv_sql})
        ORDER BY gvkey, datadate, fqtr
    """
    pitq = download_table(db, pitq_sql, "comp_pit.pitqtrdataus")

    hist_sql = f"""
        SELECT *
        FROM comp_pit.pit_hist_date_tableus
        WHERE gvkey IN ({gv_sql})
        ORDER BY gvkey, pointdate, qtrsback
    """
    hist = download_table(db, hist_sql, "comp_pit.pit_hist_date_tableus")

    link_sql = f"""
        SELECT *
        FROM crsp_a_ccm.ccmxpf_linktable
        WHERE gvkey IN ({gv_sql})
        ORDER BY gvkey, linkdt, lpermno
    """
    link = download_table(db, link_sql, "crsp_a_ccm.ccmxpf_linktable")

    pitq_path = OUT_DIR / "pitqtrdataus.parquet"
    hist_path = OUT_DIR / "pit_hist_date_tableus.parquet"
    link_path = OUT_DIR / "ccmxpf_linktable.parquet"

    pitq.to_parquet(pitq_path, index=False)
    hist.to_parquet(hist_path, index=False)
    link.to_parquet(link_path, index=False)

    meta = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_tickers": TARGET_TICKERS,
        "gvkeys": gvkeys,
        "rows": {
            "pitqtrdataus": len(pitq),
            "pit_hist_date_tableus": len(hist),
            "ccmxpf_linktable": len(link),
        },
        "columns": {
            "pitqtrdataus": len(pitq.columns),
            "pit_hist_date_tableus": len(hist.columns),
            "ccmxpf_linktable": len(link.columns),
        },
        "files": {
            "company_manifest": manifest_path.name,
            "pitqtrdataus": pitq_path.name,
            "pit_hist_date_tableus": hist_path.name,
            "ccmxpf_linktable": link_path.name,
        },
    }
    (OUT_DIR / "download_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print()
    print(f"Saved -> {OUT_DIR.relative_to(SCRIPT_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
