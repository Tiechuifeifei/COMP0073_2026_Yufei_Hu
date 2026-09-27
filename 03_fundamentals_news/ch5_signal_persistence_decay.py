#!/usr/bin/env python3
"""Rank persistence (Spearman lag) + multi-horizon Rank IC for W1 and F_BASE_W010."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_US
from qlib.data import D
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "reports/ch5_signal_persistence_decay_20260923"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 5, 10, 20]
LAGS = [1, 5, 10, 20]
PERIODS = {
    "valid_2018_2019": ("2018-01-01", "2019-12-31"),
    "test_2020_2023": ("2020-01-01", "2023-12-31"),
}


def period_slice(s: pd.Series, start: str, end: str) -> pd.Series:
    d = s.index.get_level_values(0)
    return s[(d >= start) & (d <= end)]


def daily_rank_ic(score: pd.Series, ret: pd.Series) -> pd.Series:
    df = pd.concat([score.rename("score"), ret.rename("ret")], axis=1, join="inner").dropna()

    def _one(g: pd.DataFrame) -> float:
        if len(g) < 5:
            return np.nan
        return float(spearmanr(g["score"].values, g["ret"].values).correlation)

    return df.groupby(level=0, sort=False).apply(_one)


def daily_rank_autocorr(score: pd.Series, lag: int) -> pd.Series:
    wide = score.unstack("instrument")
    vals = wide.to_numpy(dtype=np.float64, copy=False)
    dates = wide.index
    n_dates, _ = vals.shape
    out = np.full(n_dates, np.nan, dtype=np.float64)
    for i in range(n_dates - lag):
        a = vals[i]
        b = vals[i + lag]
        m = np.isfinite(a) & np.isfinite(b)
        n = int(m.sum())
        if n < 5:
            continue
        out[i] = spearmanr(a[m], b[m]).correlation
    return pd.Series(out, index=dates)


def main() -> None:
    qlib.init(provider_uri=str(PROJECT / "staging/qlib_data"), region=REG_US)

    w1 = pd.read_parquet(
        PROJECT / "reports/legacy_RD13_system_audit_20260921_074429/predictions/pred_legacy_W1.parquet"
    )
    w1["datetime"] = pd.to_datetime(w1["datetime"])
    w1 = w1.set_index(["datetime", "instrument"])["score"].sort_index()
    w1.name = "W1"

    with open(PROJECT / "reports/portfolio_topk_drop_joint_20260919_141851/F_BASE_W010_score.pkl", "rb") as f:
        fb = pickle.load(f)
    if isinstance(fb, pd.DataFrame):
        fb = fb["score"] if "score" in fb.columns else fb.iloc[:, 0]
    fb = fb.sort_index()
    fb.name = "F_BASE_W010"

    # Instruments overlapping analysis window
    mask = (w1.index.get_level_values(0) >= "2018-01-01") & (w1.index.get_level_values(0) <= "2023-12-31")
    insts = sorted(set(w1.loc[mask].index.get_level_values(1).unique()))
    print(f"n instruments={len(insts)}")

    close = D.features(insts, ["$close"], start_time="2017-12-01", end_time="2024-02-15")
    close = close["$close"].unstack("instrument").sort_index()
    print(f"close shape={close.shape}")

    c1 = close.shift(-1)
    fwd_long: dict[int, pd.Series] = {}
    for h in HORIZONS:
        s = (close.shift(-(1 + h)) / c1 - 1).stack()
        s.index.names = ["datetime", "instrument"]
        s.name = f"fwd_{h}"
        fwd_long[h] = s

    # Sanity: h=1 vs LABEL0 on a sample day
    lab = pd.read_parquet(
        PROJECT / "reports/loop9_fund_incremental_20260918_222943/datasets/B3_alpha20_loop9lib8_fund.parquet",
        columns=["datetime", "instrument", "label"],
    )
    lab["datetime"] = pd.to_datetime(lab["datetime"])
    lab = lab.set_index(["datetime", "instrument"])["label"]
    common = pd.concat([fwd_long[1].rename("fwd"), lab.rename("lab")], axis=1, join="inner").dropna()
    common = common.loc[(common.index.get_level_values(0) >= "2018-01-01") & (common.index.get_level_values(0) <= "2023-12-31")]
    recon_err = float((common["fwd"] - common["lab"]).abs().max())
    print(f"LABEL0 recon max abs err (fwd_1 vs B3 label)={recon_err:.3e}")

    rows_ac = []
    rows_ic = []
    signals = {"W1": w1, "F_BASE_W010": fb}

    for sig_name, score in signals.items():
        score = score.sort_index()
        for per_name, (start, end) in PERIODS.items():
            sc = period_slice(score, start, end)
            print(f"AC {sig_name} {per_name}")
            for lag in LAGS:
                ac = daily_rank_autocorr(sc, lag)
                rows_ac.append(
                    {
                        "signal": sig_name,
                        "period": per_name,
                        "lag": lag,
                        "mean_spearman": float(np.nanmean(ac.values)),
                        "median_spearman": float(np.nanmedian(ac.values)),
                        "n_days": int(np.isfinite(ac.values).sum()),
                    }
                )
            print(f"IC {sig_name} {per_name}")
            for h in HORIZONS:
                ret = period_slice(fwd_long[h], start, end)
                ic = daily_rank_ic(sc, ret)
                mu = float(np.nanmean(ic.values))
                sd = float(np.nanstd(ic.values, ddof=1)) if np.isfinite(ic).sum() > 1 else np.nan
                rows_ic.append(
                    {
                        "signal": sig_name,
                        "period": per_name,
                        "horizon": h,
                        "mean_rank_ic": mu,
                        "median_rank_ic": float(np.nanmedian(ic.values)),
                        "rank_icir": (mu / sd) if sd and sd > 0 else np.nan,
                        "n_days": int(np.isfinite(ic.values).sum()),
                    }
                )

    ac_df = pd.DataFrame(rows_ac)
    ic_df = pd.DataFrame(rows_ic)
    ac_df.to_csv(OUT / "rank_autocorr.csv", index=False)
    ic_df.to_csv(OUT / "rank_ic_horizons.csv", index=False)
    (OUT / "label0_recon.json").write_text(
        pd.Series({"fwd1_vs_B3_label_max_abs_err": recon_err}).to_json(),
        encoding="utf-8",
    )
    print(ac_df.to_string(index=False))
    print(ic_df.to_string(index=False))
    print("OUT", OUT)


if __name__ == "__main__":
    main()
