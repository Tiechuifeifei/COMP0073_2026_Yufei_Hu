"""Extend RD13_v2 panel through holdout using deterministic formula reconstruction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_experiments.final_holdout.qlib_pv_extension import extend_daily_pv_through

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_PANEL = PROJECT_ROOT / "data/fundamental_experiments/RD13_provenance/RD13_v2_CORRECTED/rd13_v2_panel.parquet"
OUT_AUDIT = PROJECT_ROOT / "data/portfolio_experiments/final_holdout/upstream_rd13_extension_audit.json"
HOLDOUT_END = "2025-12-31"


def _load_compute_fn():
    path = PROJECT_ROOT / "fundamental_experiments/RD13_provenance_reconstruction.py"
    spec = importlib.util.spec_from_file_location("rd13_prov", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_all_v2_factors


def sha256_series(s: pd.Series) -> str:
    arr = np.ascontiguousarray(s.to_numpy(dtype=np.float64, na_value=np.nan))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def extend_rd13_panel(end: str = HOLDOUT_END) -> tuple[pd.DataFrame, dict]:
    existing = pd.read_parquet(V2_PANEL)
    existing["datetime"] = pd.to_datetime(existing["datetime"]).dt.normalize()
    existing["instrument"] = existing["instrument"].astype(str)
    existing_max = existing["datetime"].max()

    if existing_max >= pd.Timestamp(end):
        audit = {"status": "already_extended", "max_date": str(existing_max.date())}
        return existing, audit

    pv = extend_daily_pv_through(end)
    compute = _load_compute_fn()
    factors = compute(pv)
    ext_flat = factors.reset_index()
    ext_flat["datetime"] = pd.to_datetime(ext_flat["datetime"]).dt.normalize()
    ext_flat["instrument"] = ext_flat["instrument"].astype(str)
    ext_only = ext_flat[ext_flat["datetime"] > existing_max].copy()

    # Overlap verification on last 60 shared dates
    overlap_dates = sorted(
        set(existing["datetime"].unique()) & set(ext_flat["datetime"].unique())
    )[-60:]
    overlap_checks = []
    rd_cols = [c for c in existing.columns if c.startswith("rd13_")]
    for dt in overlap_dates:
        for col in rd_cols:
            a = existing.loc[existing["datetime"] == dt, ["instrument", col]].set_index("instrument")[col]
            b = ext_flat.loc[ext_flat["datetime"] == dt, ["instrument", col]].set_index("instrument")[col]
            common = a.index.intersection(b.index)
            if len(common) < 10:
                continue
            diff = (a.reindex(common) - b.reindex(common)).abs()
            overlap_checks.append(
                {
                    "date": str(dt.date()),
                    "column": col,
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                }
            )

    bad = [r for r in overlap_checks if r["max_abs_diff"] > 1e-4]
    if bad:
        raise RuntimeError(f"RD13 overlap verification failed: {bad[:5]}")

    combined = pd.concat([existing, ext_only[existing.columns]], ignore_index=True)
    combined = combined.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    combined.to_parquet(V2_PANEL, index=False)

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "deterministic reconstruction of the frozen training procedure",
        "formula_source": "RD13_provenance_reconstruction.compute_all_v2_factors",
        "pv_source": "frozen daily_pv.h5 + qlib extension",
        "prior_max_date": str(existing_max.date()),
        "new_max_date": str(combined["datetime"].max().date()),
        "holdout_rows": int((combined["datetime"] >= "2024-01-01").sum()),
        "overlap_checks_n": len(overlap_checks),
        "overlap_failures": len(bad),
        "panel_sha256_16": hashlib.sha256(combined.to_parquet(index=False)).hexdigest()[:16],
    }
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return combined, audit
