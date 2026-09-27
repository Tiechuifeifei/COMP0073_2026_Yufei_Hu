#!/usr/bin/env python3
"""Correction gate for atr_20d_x_volume_ratio_20d (frozen relative-ATR definition).

Fails closed: any check failure exits non-zero and must block RD-Agent loops.
English-only outputs.
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    """Resolve package root for this curated layout."""
    import os
    env = os.environ.get("PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Prefer git-style layout: this file lives under 0X_*/...
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "README.md").exists() and (parent / "paper").exists():
            return parent
    return here.parents[min(2, len(here.parents)-1)]

def _rdagent_root() -> Path:
    import os
    env = os.environ.get("RDAGENT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    raise RuntimeError("Set RDAGENT_ROOT to your RD-Agent checkout (patched us-market-experiment).")


ROOT = (Path(os.environ["PROJECT_ROOT"]) / "reports/rdagent_corrected_semantic_rerun_20260918") if os.environ.get("PROJECT_ROOT") else (_repo_root() / "reports/rdagent_corrected_semantic_rerun_20260918")
ABLATION = (Path(os.environ["PROJECT_ROOT"]) / "reports/matched_loop2_semantic_ablation_20260918") if os.environ.get("PROJECT_ROOT") else (_repo_root() / "reports/matched_loop2_semantic_ablation_20260918")
DAILY_PV = ROOT / "source_data_clean" / "daily_pv.h5"
ABLATION_SERIES = ABLATION / "factors" / "atr_20d_x_volume_ratio_20d_corrected.parquet"
HOLDOUT = pd.Timestamp("2024-01-01")
OUT = ROOT / "gates"


def classic_tr(high, low, close):
    prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev).abs()
    tr3 = (low - prev).abs()
    tr = pd.DataFrame(
        np.maximum(np.maximum(tr1.values, tr2.values), tr3.values),
        index=tr1.index,
        columns=tr1.columns,
    )
    return tr, prev


def compute_corrected(df: pd.DataFrame) -> pd.DataFrame:
    high = df["$high"].unstack("instrument")
    low = df["$low"].unstack("instrument")
    close = df["$close"].unstack("instrument")
    volume = df["$volume"].unstack("instrument")
    tr, prev = classic_tr(high, low, close)
    rel_atr20 = (tr / prev).rolling(20, min_periods=20).mean()
    vr = volume / volume.rolling(20, min_periods=20).mean()
    return rel_atr20 * vr


def compute_absolute_interaction(df: pd.DataFrame) -> pd.DataFrame:
    high = df["$high"].unstack("instrument")
    low = df["$low"].unstack("instrument")
    close = df["$close"].unstack("instrument")
    volume = df["$volume"].unstack("instrument")
    tr, _ = classic_tr(high, low, close)
    abs_atr20 = tr.rolling(20, min_periods=20).mean()
    vr = volume / volume.rolling(20, min_periods=20).mean()
    return abs_atr20 * vr


def stack(wide: pd.DataFrame, name: str) -> pd.Series:
    s = wide.stack()
    s.index.names = ["datetime", "instrument"]
    s.name = name
    return s


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"ok": True, "checks": []}

    def check(name: str, passed: bool, detail: str) -> None:
        report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            report["ok"] = False
            print(f"FAIL {name}: {detail}", flush=True)
        else:
            print(f"PASS {name}: {detail}", flush=True)

    df = pd.read_hdf(DAILY_PV, key="data")
    if (df.index.get_level_values(0) >= HOLDOUT).any():
        check("no_holdout_in_daily_pv", False, "found 2024+ rows")
        (OUT / "correction_gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 2
    check("no_holdout_in_daily_pv", True, "all dates < 2024-01-01")

    corrected = compute_corrected(df)
    absolute = compute_absolute_interaction(df)
    corr_s = stack(corrected, "corrected")
    abs_s = stack(absolute, "absolute")

    # NaN/inf
    arr = corr_s.to_numpy(dtype=float)
    finite = np.isfinite(arr)
    check(
        "finite_values_present",
        bool(finite.sum() > 1_000_000),
        f"finite={int(finite.sum())} nan={int(np.isnan(arr).sum())} inf={int(np.isinf(arr).sum())}",
    )
    check("no_inf", bool(np.isinf(arr).sum() == 0), f"inf_count={int(np.isinf(arr).sum())}")

    # Scale invariance OHLC x10
    df2 = df.copy()
    for c in ("$open", "$high", "$low", "$close"):
        df2[c] = df2[c] * 10.0
    corrected2 = compute_corrected(df2)
    b = corrected.stack().dropna()
    a = corrected2.stack().reindex(b.index).dropna()
    idx = b.index.intersection(a.index)
    ratio = (a.loc[idx] / b.loc[idx].replace(0, np.nan)).dropna()
    med = float(ratio.median())
    frac1 = float(((ratio - 1).abs() < 1e-6).mean())
    check(
        "scale_invariance_ohlc_x10",
        bool(abs(med - 1.0) < 1e-6 and frac1 > 0.99),
        f"median_ratio={med:.8f} frac_near_1={frac1:.6f}",
    )

    # Absolute original should still be x10 (sanity that test is sensitive)
    absolute2 = compute_absolute_interaction(df2)
    b2 = absolute.stack().dropna()
    a2 = absolute2.stack().reindex(b2.index).dropna()
    idx2 = b2.index.intersection(a2.index)
    ratio2 = (a2.loc[idx2] / b2.loc[idx2].replace(0, np.nan)).dropna()
    med2 = float(ratio2.median())
    check(
        "absolute_interaction_still_x10",
        bool(abs(med2 - 10.0) < 1e-6),
        f"median_ratio={med2:.8f}",
    )

    # Parity vs ablation series
    abl = pd.read_parquet(ABLATION_SERIES).iloc[:, 0]
    both = pd.concat([corr_s, abl], axis=1, join="inner").dropna()
    both.columns = ["gate", "ablation"]
    pearson = float(both["gate"].corr(both["ablation"]))
    max_abs = float((both["gate"] - both["ablation"]).abs().max())
    check(
        "parity_vs_ablation_corrected_series",
        bool(pearson > 0.999999 and max_abs < 1e-10),
        f"pearson={pearson:.12f} max_abs_diff={max_abs:.6g} n={len(both)}",
    )

    # Correlations 2020-2023 confirming correction
    mask = (corr_s.index.get_level_values(0) >= "2020-01-01") & (corr_s.index.get_level_values(0) <= "2023-12-31")
    c = corr_s.loc[mask]
    close = stack(df["$close"].unstack("instrument"), "close").reindex(c.index)
    tr, prev = classic_tr(
        df["$high"].unstack("instrument"),
        df["$low"].unstack("instrument"),
        df["$close"].unstack("instrument"),
    )
    abs_atr = stack(tr.rolling(20, min_periods=20).mean(), "abs_atr").reindex(c.index)
    rel_atr = stack((tr / prev).rolling(20, min_periods=20).mean(), "rel_atr").reindex(c.index)
    corr_price = float(pd.concat([c, close], axis=1).dropna().corr().iloc[0, 1])
    corr_abs = float(pd.concat([c, abs_atr], axis=1).dropna().corr().iloc[0, 1])
    corr_rel = float(pd.concat([c, rel_atr], axis=1).dropna().corr().iloc[0, 1])
    report["correlations_2020_2023"] = {
        "vs_price_level": corr_price,
        "vs_absolute_ATR20": corr_abs,
        "vs_relative_ATR20": corr_rel,
    }
    # Expect: much higher corr with relative ATR than absolute ATR / price
    check(
        "corr_profile_relative_not_price",
        bool(corr_rel > 0.5 and corr_abs < 0.4 and corr_price < 0.4),
        f"price={corr_price:.4f} absATR={corr_abs:.4f} relATR={corr_rel:.4f}",
    )

    # Persist corrected series used by Loop2 injection
    out_series = ROOT / "factors" / "corrected_interaction_gate.parquet"
    corr_s.to_frame().to_parquet(out_series)
    report["corrected_series_path"] = str(out_series)

    (OUT / "correction_gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("GATE", "PASSED" if report["ok"] else "FAILED", flush=True)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
