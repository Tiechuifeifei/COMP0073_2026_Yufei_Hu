#!/usr/bin/env python3
"""Chapter 5 follow-up: D-series identity checks, paired ΔRankIC
(block=21, B=2000), and D*_R50 capacity. Paired tests use frozen early-stop
predictions; R50 D1/D3 are controlled retrains. Holdout 2024–2025 is not used
for selection."""

from __future__ import annotations
import os

import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.data.dataset.processor import zscore

PROJECT = Path(os.environ["PROJECT_ROOT"]) if os.environ.get("PROJECT_ROOT") else Path(__file__).resolve().parents[1]
PANEL = PROJECT / "data/rd13_v2_downstream_replication/stage_1_delayed_panel/delayed_common_sample_rd13_v2.parquet"
STAGE2 = PROJECT / "data/rd13_v2_downstream_replication/stage_2_delayed_models"
D2_R50_PRED = PROJECT / "reports/rd13_l20_downstream_cmp_20260923_190325/pred_Alpha20_RD13_v2_R50.parquet"
W1_PRED = PROJECT / "reports/legacy_RD13_system_audit_20260921_074429/predictions/pred_legacy_W1.parquet"
F1C_PRED = PROJECT / "reports/corrected_downstream_loop2_20260917/predictions/pred_F1C.parquet"
BLEND_AUDIT = PROJECT / "reports/pv_fund_blend_r50_20260919_003636/input_audit.md"
A_METRICS = PROJECT / "reports/rd13_l20_unified_ABCD_20260923_193417/A_period_metrics_seed_mean_std.csv"
BLEND_GRID = PROJECT / "reports/pv_fund_blend_r50_20260919_003636/blend_grid_metrics.csv"

ALPHA20 = [
    "resi5", "wvma5", "rsqr5", "klen", "rsqr10", "corr5", "cord5", "corr10", "roc60",
    "resi10", "vstd5", "rsqr60", "corr60", "wvma60", "std5", "rsqr20", "cord60", "cord10", "corr20", "klow",
]
FUND_FEATURES = [
    "fund_roe_z", "fund_roa_z", "fund_gross_profitability_z", "fund_sales_growth_yoy_z",
    "fund_asset_growth_yoy_z", "fund_accruals_z", "fund_leverage_z", "fund_current_ratio_z",
    "fund_book_to_market_z", "fund_earnings_yield_z", "fund_sales_to_price_z",
]
OFFICIAL_LGB = {
    "objective": "regression", "metric": "mse", "verbosity": -1,
    "colsample_bytree": 0.8879, "learning_rate": 0.2, "subsample": 0.8789,
    "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8, "num_leaves": 210, "num_threads": 8,
}
SEEDS = [42, 2026, 3407]
NUM_BOOST = 1000
EARLY_STOP = 50
FIXED_ROUNDS = 50
MIN_CS = 30
BOOT_BLOCK = 21
BOOT_N = 2000
BOOT_SEED = 20260923
AMIHUD = "rd13_daily_amihud_illiquidity"
PERIODS = {
    "valid": ("2018-01-01", "2019-12-31"),
    "test": ("2020-01-01", "2023-12-31"),
}

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = PROJECT / f"reports/ch5_d_series_identity_paired_audit_{STAMP}"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def process_train_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["label"]).copy()
    out["label_processed"] = out.groupby("datetime", group_keys=False)["label"].apply(zscore)
    return out


def lgb_params(seed: int) -> dict[str, Any]:
    p = OFFICIAL_LGB.copy()
    p.update({"seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed, "data_random_seed": seed})
    return p


def train_lgb(train_df, valid_df, features, seed: int, fixed_rounds: int | None = None) -> lgb.Booster:
    p = lgb_params(seed)
    tr = lgb.Dataset(train_df[features], label=train_df["label_processed"], free_raw_data=False)
    va = lgb.Dataset(valid_df[features], label=valid_df["label_processed"], reference=tr, free_raw_data=False)
    if fixed_rounds is not None:
        return lgb.train(
            p, tr, num_boost_round=fixed_rounds, valid_sets=[va], valid_names=["valid"],
            callbacks=[lgb.log_evaluation(period=0)],
        )
    return lgb.train(
        p, tr, num_boost_round=NUM_BOOST, valid_sets=[va], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(period=0)],
    )


def feature_groups(spec_id: str, rd_cols: list[str]) -> list[str]:
    a20 = [f"alpha20_{c}" for c in ALPHA20]
    if spec_id == "D0_ALPHA20":
        return a20
    if spec_id == "D1_ALPHA20_FUND":
        return a20 + FUND_FEATURES
    if spec_id == "D2_ALPHA20_RD13_V2":
        return a20 + rd_cols
    if spec_id == "D3_ALPHA20_RD13_V2_FUND":
        return a20 + rd_cols + FUND_FEATURES
    raise ValueError(spec_id)


def block_bootstrap_ci(x: np.ndarray, block: int, n_boot: int, seed: int) -> tuple[float, float]:
    n = len(x)
    if n < max(block, 10):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    max_start = max(n - block + 1, 1)
    for i in range(n_boot):
        idx: list[int] = []
        while len(idx) < n:
            s = int(rng.integers(0, max_start))
            idx.extend(range(s, min(s + block, n)))
        means[i] = float(x[np.asarray(idx[:n])].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def daily_rank_ic(score: pd.Series, label: pd.Series) -> float:
    return float(score.corr(label, method="spearman"))


def paired_delta_rankic(
    cand: pd.DataFrame,
    base: pd.DataFrame,
    labels: pd.DataFrame,
    period: str,
    seed: int,
) -> dict[str, Any]:
    s, e = PERIODS[period]
    m = cand.merge(base, on=["datetime", "instrument"], suffixes=("_c", "_b"))
    m = m.merge(labels, on=["datetime", "instrument"])
    m = m[(m["datetime"] >= pd.Timestamp(s)) & (m["datetime"] <= pd.Timestamp(e))]
    rows = []
    for dt, g in m.groupby("datetime"):
        if len(g) < MIN_CS:
            continue
        if g["score_c"].nunique() < 2 or g["score_b"].nunique() < 2:
            continue
        ric_c = daily_rank_ic(g["score_c"], g["label"])
        ric_b = daily_rank_ic(g["score_b"], g["label"])
        if np.isnan(ric_c) or np.isnan(ric_b):
            continue
        rows.append({"datetime": dt, "delta": ric_c - ric_b, "ric_c": ric_c, "ric_b": ric_b})
    if not rows:
        return {"period": period, "n_common_dates": 0, "delta_rankic": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    d = pd.DataFrame(rows).sort_values("datetime")
    arr = d["delta"].to_numpy(dtype=float)
    lo, hi = block_bootstrap_ci(arr, BOOT_BLOCK, BOOT_N, seed)
    return {
        "period": period,
        "n_common_dates": int(len(d)),
        "delta_rankic": float(arr.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "mean_ric_cand": float(d["ric_c"].mean()),
        "mean_ric_base": float(d["ric_b"].mean()),
        "bootstrap_B": BOOT_N,
        "bootstrap_block": BOOT_BLOCK,
        "bootstrap_seed": seed,
    }


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pickle.load(path.open("rb"))
    if not isinstance(df, pd.DataFrame):
        raise TypeError(path)
    cols = {c.lower(): c for c in df.columns}
    # normalize
    out = df.copy()
    if "datetime" not in out.columns:
        if isinstance(out.index, pd.MultiIndex):
            out = out.reset_index()
    rename = {}
    for want in ("datetime", "instrument", "score"):
        if want not in out.columns:
            for c in out.columns:
                if c.lower() == want or c.lower() in ("date", "dt") and want == "datetime":
                    rename[c] = want
                elif c.lower() in ("symbol", "asset", "ticker") and want == "instrument":
                    rename[c] = want
                elif c.lower() in ("pred", "prediction") and want == "score":
                    rename[c] = want
    out = out.rename(columns=rename)
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out[["datetime", "instrument", "score"]].drop_duplicates(["datetime", "instrument"])


def inspect_existing_pickles() -> pd.DataFrame:
    rows = []
    for spec in ["D0_ALPHA20", "D1_ALPHA20_FUND", "D2_ALPHA20_RD13_V2", "D3_ALPHA20_RD13_V2_FUND"]:
        for seed in SEEDS:
            p = STAGE2 / f"model_{spec}_seed{seed}.pkl"
            if not p.exists():
                rows.append({
                    "spec": spec, "seed": seed, "source": "missing_pickle",
                    "best_iteration": np.nan, "num_trees": np.nan,
                    "amihud_split": np.nan, "amihud_gain": np.nan,
                })
                continue
            model = pickle.load(p.open("rb"))
            bi = int(model.best_iteration)
            nt = int(model.num_trees())
            names = model.feature_name()
            amihud_split = amihud_gain = np.nan
            if AMIHUD in names:
                i = names.index(AMIHUD)
                amihud_split = int(model.feature_importance("split")[i])
                amihud_gain = float(model.feature_importance("gain")[i])
            rows.append({
                "spec": spec, "seed": seed, "source": "existing_pickle",
                "best_iteration": bi, "num_trees": nt,
                "amihud_split": amihud_split, "amihud_gain": amihud_gain,
            })
    return pd.DataFrame(rows)


def recover_best_iteration(panel: pd.DataFrame, rd_cols: list[str]) -> pd.DataFrame:
    """Retrain early-stop D0/D1/D3 (all seeds) to recover per-seed best_iteration."""
    train = process_train_label(panel[panel["split"] == "train"])
    valid = process_train_label(panel[panel["split"] == "valid"])
    rows = []
    for spec in ["D0_ALPHA20", "D1_ALPHA20_FUND", "D3_ALPHA20_RD13_V2_FUND"]:
        feats = feature_groups(spec, rd_cols)
        for seed in SEEDS:
            t0 = time.perf_counter()
            model = train_lgb(train, valid, feats, seed, fixed_rounds=None)
            elapsed = time.perf_counter() - t0
            amihud_split = amihud_gain = np.nan
            if AMIHUD in feats:
                i = feats.index(AMIHUD)
                amihud_split = int(model.feature_importance("split")[i])
                amihud_gain = float(model.feature_importance("gain")[i])
            rows.append({
                "spec": spec, "seed": seed, "source": "retrain_early_stop_identity",
                "best_iteration": int(model.best_iteration),
                "num_trees": int(model.num_trees()),
                "amihud_split": amihud_split, "amihud_gain": amihud_gain,
                "n_features": len(feats), "runtime_sec": elapsed,
            })
            # persist recovered seed pickles under OUT only (do not overwrite stage_2)
            with (OUT / "models" / f"model_{spec}_seed{seed}_earlystop_recovered.pkl").open("wb") as f:
                pickle.dump(model, f)
            print(f"[identity] {spec} seed={seed} best_iteration={model.best_iteration} ({elapsed:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def train_r50(panel: pd.DataFrame, rd_cols: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    train = process_train_label(panel[panel["split"] == "train"])
    valid = process_train_label(panel[panel["split"] == "valid"])
    preds: dict[str, pd.DataFrame] = {}
    meta_rows = []
    for spec in ["D1_ALPHA20_FUND", "D3_ALPHA20_RD13_V2_FUND"]:
        feats = feature_groups(spec, rd_cols)
        seed_preds = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            model = train_lgb(train, valid, feats, seed, fixed_rounds=FIXED_ROUNDS)
            elapsed = time.perf_counter() - t0
            sub = panel.dropna(subset=feats + ["label"]).copy()
            sub["score"] = model.predict(sub[feats], num_iteration=FIXED_ROUNDS)
            pred = sub[["datetime", "instrument", "score"]]
            seed_preds.append(pred)
            with (OUT / "models" / f"model_{spec}_R50_seed{seed}.pkl").open("wb") as f:
                pickle.dump(model, f)
            meta_rows.append({
                "spec": spec, "seed": seed, "num_boost_round": FIXED_ROUNDS,
                "early_stopping": False, "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
                "num_trees": int(model.num_trees()), "runtime_sec": elapsed, "n_features": len(feats),
            })
            print(f"[R50] {spec} seed={seed} trees={model.num_trees()} ({elapsed:.1f}s)", flush=True)
        ens = pd.concat(seed_preds).groupby(["datetime", "instrument"], as_index=False)["score"].mean()
        alias = "D1_R50" if spec.startswith("D1") else "D3_R50"
        ens.to_parquet(OUT / f"pred_{alias}.parquet", index=False)
        preds[alias] = ens
    return pd.DataFrame(meta_rows), preds


def sample_contrast() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL, columns=["datetime", "instrument", "split", "label"] + FUND_FEATURES[:1])
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    rows = []
    # delayed common by split
    for split, name in [("valid", "delayed_common_valid"), ("test", "delayed_common_test")]:
        sub = panel[panel["split"] == split]
        rows.append({
            "universe": "delayed_common_sample_rd13_v2",
            "period": split,
            "n_rows": int(len(sub)),
            "n_dates": int(sub["datetime"].nunique()),
            "n_instruments": int(sub["instrument"].nunique()),
            "requires_fund_nonnull": False,
            "note": "Ch4-style Alpha20_RD8 / D-series delayed panel",
        })
    # fund-nonnull subset of delayed (approx for fund models)
    fund_col = FUND_FEATURES[0]
    for split in ("valid", "test"):
        sub = panel[(panel["split"] == split) & panel[fund_col].notna()]
        rows.append({
            "universe": "delayed_common ∩ fund_nonnull",
            "period": split,
            "n_rows": int(len(sub)),
            "n_dates": int(sub["datetime"].nunique()),
            "n_instruments": int(sub["instrument"].nunique()),
            "requires_fund_nonnull": True,
            "note": "approximate fund availability on delayed panel",
        })
    # blend / B1_R50 from audit numbers + grid
    grid = pd.read_csv(BLEND_GRID)
    for period, n_rows in [("valid", 241619), ("test", 485670)]:
        g = grid[(grid["signal_pair"] == "PV_F1C_R50") & (grid["w"] == 1.0) & (grid["period"] == period)].iloc[0]
        rows.append({
            "universe": "PV×F1C_R50 common (B1_R50 / Ch5表5.4)",
            "period": period,
            "n_rows": n_rows,
            "n_dates": int(g["n_dates"]),
            "n_instruments": np.nan,
            "requires_fund_nonnull": True,
            "note": f"rank_ic={g['rank_ic_mean']:.6f}; total_rows=1880525",
        })
    # Ch4 Alpha20_RD8 R50
    ap = pd.read_csv(A_METRICS)
    for period_key, period in [("valid_2018_2019", "valid"), ("test_2020_2023", "test")]:
        g = ap[(ap["spec"] == "Alpha20_RD8") & (ap["num_boost_round"] == 50) & (ap["period"] == period_key)].iloc[0]
        rows.append({
            "universe": "delayed_common Alpha20_RD8 R50 (Ch4表4.8口径)",
            "period": period,
            "n_rows": np.nan,
            "n_dates": int(g["n_days_ensemble"]),
            "n_instruments": np.nan,
            "requires_fund_nonnull": False,
            "note": f"RankIC_ens={g['RankIC_ens']:.6f}",
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    t_all = time.perf_counter()

    # --- 1. existing pickle identity + Amihud ---
    existing = inspect_existing_pickles()
    existing.to_csv(OUT / "existing_pickle_identity.csv", index=False)

    print("Loading delayed panel...", flush=True)
    panel = pd.read_parquet(PANEL)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    rd_cols = sorted([c for c in panel.columns if c.startswith("rd13_")])
    labels = panel[["datetime", "instrument", "label"]].dropna()

    # recover missing best_iteration
    recovered = recover_best_iteration(panel, rd_cols)
    recovered.to_csv(OUT / "recovered_best_iteration_D0_D1_D3.csv", index=False)

    # D2 Amihud from existing 3-seed pickles
    d2_amihud = existing[existing["spec"] == "D2_ALPHA20_RD13_V2"].copy()
    d2_amihud.to_csv(OUT / "d2_amihud_split_usage.csv", index=False)

    identity = pd.concat([
        existing[existing["spec"] == "D2_ALPHA20_RD13_V2"],
        recovered,
    ], ignore_index=True)
    identity.to_csv(OUT / "model_identity_best_iteration.csv", index=False)

    # --- 2. paired bootstrap from frozen preds ---
    print("Paired bootstrap from frozen preds...", flush=True)
    preds = {
        "D0": load_pred(STAGE2 / "pred_D0_ALPHA20.parquet"),
        "D1": load_pred(STAGE2 / "pred_D1_ALPHA20_FUND.parquet"),
        "D2": load_pred(STAGE2 / "pred_D2_ALPHA20_RD13_V2.parquet"),
        "D3": load_pred(STAGE2 / "pred_D3_ALPHA20_RD13_V2_FUND.parquet"),
        "W1": load_pred(W1_PRED),
        "F1C": load_pred(F1C_PRED),
    }
    pairs = [
        ("D1_minus_D0", "D1", "D0"),
        ("D3_minus_D2", "D3", "D2"),
        ("D3_minus_D1", "D3", "D1"),
        ("W1_minus_D2", "W1", "D2"),
        ("W1_minus_F1C", "W1", "F1C"),
    ]
    paired_rows = []
    for i, (name, hi, lo) in enumerate(pairs):
        for j, period in enumerate(("valid", "test")):
            seed = BOOT_SEED + i * 10 + j
            row = paired_delta_rankic(preds[hi], preds[lo], labels, period, seed)
            row["comparison"] = name
            row["candidate"] = hi
            row["baseline"] = lo
            paired_rows.append(row)
            print(
                f"[paired] {name} {period}: Δ={row['delta_rankic']:.5f} "
                f"CI=[{row['ci_low']:.5f},{row['ci_high']:.5f}] n={row['n_common_dates']}",
                flush=True,
            )
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(OUT / "paired_delta_rankic_bootstrap.csv", index=False)

    # cross-check vs stage_2 saved increments
    legacy = pd.read_csv(STAGE2 / "delayed_increment_metrics.csv")
    legacy.to_csv(OUT / "stage2_delayed_increment_metrics_reference.csv", index=False)

    # --- 3. R50 D1/D3 + D3_R50 - D2_R50 ---
    print("Training D1_R50 / D3_R50...", flush=True)
    r50_meta, r50_preds = train_r50(panel, rd_cols)
    r50_meta.to_csv(OUT / "r50_train_meta.csv", index=False)
    d2_r50 = load_pred(D2_R50_PRED)
    d2_r50.to_parquet(OUT / "pred_D2_R50_from_downstream_cmp.parquet", index=False)
    r50_preds["D2_R50"] = d2_r50

    def absolute_rankic(pred: pd.DataFrame, period: str) -> dict[str, Any]:
        s, e = PERIODS[period]
        m = pred.merge(labels, on=["datetime", "instrument"])
        m = m[(m["datetime"] >= pd.Timestamp(s)) & (m["datetime"] <= pd.Timestamp(e))]
        daily = []
        for _, g in m.groupby("datetime"):
            if len(g) < MIN_CS or g["score"].nunique() < 2:
                continue
            ric = daily_rank_ic(g["score"], g["label"])
            if not np.isnan(ric):
                daily.append(ric)
        return {
            "period": period,
            "n_dates": int(len(daily)),
            "mean_rank_ic": float(np.mean(daily)) if daily else float("nan"),
        }

    r50_paired = []
    r50_abs = []
    for j, period in enumerate(("valid", "test")):
        seed = BOOT_SEED + 500 + j
        row = paired_delta_rankic(r50_preds["D3_R50"], r50_preds["D2_R50"], labels, period, seed)
        row["comparison"] = "D3_R50_minus_D2_R50"
        row["candidate"] = "D3_R50"
        row["baseline"] = "D2_R50"
        r50_paired.append(row)
        print(
            f"[R50 paired] D3-D2 {period}: Δ={row['delta_rankic']:.5f} "
            f"CI=[{row['ci_low']:.5f},{row['ci_high']:.5f}]",
            flush=True,
        )
        for alias in ("D1_R50", "D3_R50", "D2_R50"):
            abs_row = absolute_rankic(r50_preds[alias], period)
            abs_row["spec"] = alias
            r50_abs.append(abs_row)
    pd.DataFrame(r50_paired).to_csv(OUT / "r50_paired_delta_rankic.csv", index=False)
    pd.DataFrame(r50_abs).to_csv(OUT / "r50_absolute_rankic.csv", index=False)

    # --- 4. sample contrast Ch5 vs Ch4 ---
    sample_df = sample_contrast()
    sample_df.to_csv(OUT / "b1_sample_contrast_ch5_vs_ch4.csv", index=False)

    manifest = {
        "created_utc": utc_now(),
        "out_dir": str(OUT),
        "protocol": {
            "bootstrap_B": BOOT_N,
            "bootstrap_block": BOOT_BLOCK,
            "bootstrap_seed_base": BOOT_SEED,
            "fixed_rounds": FIXED_ROUNDS,
            "early_stop": EARLY_STOP,
            "seeds": SEEDS,
            "min_cs": MIN_CS,
        },
        "sources": {
            "panel": str(PANEL),
            "stage2_preds": str(STAGE2),
            "d2_r50_pred": str(D2_R50_PRED),
            "w1_pred": str(W1_PRED),
            "f1c_pred": str(F1C_PRED),
        },
        "runtime_sec": time.perf_counter() - t_all,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # markdown summary
    lines = [
        "# Ch5 D-series identity + paired audit",
        "",
        f"Generated: `{manifest['created_utc']}`",
        f"Output: `{OUT}`",
        "",
        "## 1. best_iteration (early-stop identity)",
        "",
        identity.to_markdown(index=False),
        "",
        "### D2 Amihud split usage (original pickles)",
        "",
        d2_amihud.to_markdown(index=False),
        "",
        "## 2. Paired ΔRankIC (frozen preds; B=2000, block=21)",
        "",
        paired_df[
            ["comparison", "period", "delta_rankic", "ci_low", "ci_high", "n_common_dates", "mean_ric_cand", "mean_ric_base"]
        ].to_markdown(index=False),
        "",
        "## 3. Fixed-50 capacity: D3_R50 − D2_R50",
        "",
        pd.DataFrame(r50_paired)[
            ["period", "delta_rankic", "ci_low", "ci_high", "n_common_dates", "mean_ric_cand", "mean_ric_base"]
        ].to_markdown(index=False),
        "",
        "### Absolute RankIC (R50)",
        "",
        pd.DataFrame(r50_abs).to_markdown(index=False),
        "",
        "## 4. B1 sample contrast (Ch5表5.4 vs Ch4表4.8)",
        "",
        sample_df.to_markdown(index=False),
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"DONE -> {OUT} ({manifest['runtime_sec']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
