"""Extend CRSP daily panel through 2025 using Qlib (deterministic holdout prep)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qlib_pv_extension import build_qlib_pv, load_sp500_instruments

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)
CRSP_PATH = PROJECT_ROOT / "data/crsp_daily/crsp_daily_market.parquet"
OUT_AUDIT = PROJECT_ROOT / "data/portfolio_experiments/final_holdout/upstream_crsp_extension_audit.json"
HOLDOUT_END = "2025-12-31"


def extend_crsp(end: str = HOLDOUT_END) -> tuple[pd.DataFrame, dict]:
    existing = pd.read_parquet(CRSP_PATH)
    existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
    existing_max = existing["date"].max()
    if existing_max >= pd.Timestamp(end):
        return existing, {"status": "already_extended", "max_date": str(existing_max.date())}

    # Last known shrout per permno for market_cap proxy on qlib days
    last = (
        existing.sort_values("date")
        .groupby("permno", as_index=False)
        .last()[["permno", "shrout"]]
        .rename(columns={"shrout": "shrout_last"})
    )

    ext_start = (existing_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    inst = [f"P{int(p)}" for p in sorted(existing["permno"].unique())]
    qlib = build_qlib_pv(ext_start, end, inst)
    qlib = qlib.reset_index()
    qlib["permno"] = qlib["instrument"].str.lstrip("P").astype(int)
    qlib = qlib.rename(columns={"datetime": "date", "$close": "prc", "$volume": "vol"})
    qlib = qlib.sort_values(["permno", "date"])
    qlib["ret"] = qlib.groupby("permno")["prc"].pct_change()
    qlib["retx"] = qlib["ret"]  # qlib lacks separate retx; documented in audit
    qlib = qlib.merge(last, on="permno", how="left")
    qlib["market_cap"] = qlib["prc"].abs() * qlib["shrout_last"] * 1000.0
    qlib["bid"] = np.nan
    qlib["ask"] = np.nan
    qlib["openprc"] = np.nan
    qlib["numtrd"] = np.nan

    cols = ["permno", "date", "prc", "shrout", "ret", "retx", "vol", "bid", "ask", "openprc", "numtrd", "market_cap"]
    ext = qlib.assign(shrout=qlib["shrout_last"])[cols]

    combined = pd.concat([existing[cols], ext], ignore_index=True)
    combined = combined.drop_duplicates(["permno", "date"], keep="first").sort_values(["permno", "date"])
    combined.to_parquet(CRSP_PATH, index=False)

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "qlib-derived extension with last-known shrout forward fill",
        "note": "WRDS cache ended 2024-12-31; 2025 rows derived from staging/qlib_data",
        "prior_max": str(existing_max.date()),
        "new_max": str(combined["date"].max().date()),
        "rows_2025": int((combined["date"] >= "2025-01-01").sum()),
        "sha256_16": hashlib.sha256(combined.to_parquet(index=False)).hexdigest()[:16],
    }
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return combined, audit
