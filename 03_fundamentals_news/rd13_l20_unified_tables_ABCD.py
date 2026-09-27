#!/usr/bin/env python3
"""Unified downstream tables A–D for Alpha20 / RD8 / RD13_v2 / RD4.

Same setting as rd13_l20_downstream_comparison.py:
  fixed LightGBM OFFICIAL params, R∈{10,20,50}, seeds 42/2026/3407, no early stop,
  delayed common-sample panel, train 2008–2017.
"""
from __future__ import annotations
import os

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.data.dataset.processor import zscore

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rd13_l20_downstream_comparison import (  # noqa: E402
    A20_COLS,
    BOOST_ROUNDS,
    BOOT_BLOCK,
    BOOT_N,
    BOOT_SEED,
    DELAYED,
    MIN_CS,
    OFFICIAL_LGB,
    RD13,
    RD4,
    RD8,
    RD8_EXT,
    RD8_PREFIXED,
    SEEDS,
    block_bootstrap_mean,
    daily_ic,
    load_merged_panel,
    process_train_label,
    train_fixed,
    utc_now,
)

PROJECT = Path(os.environ["PROJECT_ROOT"]) if os.environ.get("PROJECT_ROOT") else Path(__file__).resolve().parents[1]
D2_PRED = PROJECT / "data/rd13_v2_downstream_replication/stage_2_delayed_models/pred_D2_ALPHA20_RD13_V2.parquet"
D2_METRICS = PROJECT / "data/rd13_v2_downstream_replication/stage_2_delayed_models/delayed_metrics_by_spec.csv"
D2_MODELS = PROJECT / "data/rd13_v2_downstream_replication/stage_2_delayed_models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("unified_ABCD")

PERIODS = {
    "valid_2018_2019": ("2018-01-01", "2019-12-31"),
    "test_2020_2023": ("2020-01-01", "2023-12-31"),
    "holdout_2024_2025": ("2024-01-01", "2025-12-31"),
}
YEARS_DEV = [2020, 2021, 2022, 2023]
YEARS_HOLD = [2024, 2025]
SPECS = {
    "Alpha20": A20_COLS,
    "Alpha20_RD8": A20_COLS + RD8_PREFIXED,  # B1
    "Alpha20_RD13_v2": A20_COLS + RD13,
    "Alpha20_RD4": A20_COLS + RD4,
}


def summarize_daily(d: pd.DataFrame) -> dict[str, float]:
    if d is None or len(d) == 0:
        return {"n_days": 0, "IC": np.nan, "RankIC": np.nan, "ICIR": np.nan, "RankICIR": np.nan}
    ic_m, ric_m = float(d["ic"].mean()), float(d["rank_ic"].mean())
    ic_s, ric_s = float(d["ic"].std(ddof=1)), float(d["rank_ic"].std(ddof=1))
    return {
        "n_days": int(len(d)),
        "IC": ic_m,
        "RankIC": ric_m,
        "ICIR": ic_m / ic_s if ic_s > 0 else np.nan,
        "RankICIR": ric_m / ric_s if ric_s > 0 else np.nan,
    }


def yearly_rankic(pred: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    rows = []
    for y in years:
        d = daily_ic(pred, f"{y}-01-01", f"{y}-12-31")
        s = summarize_daily(d)
        rows.append({"year": y, **s})
    return pd.DataFrame(rows)


def paired_delta_rankic(
    base_daily: pd.DataFrame, other_daily: pd.DataFrame, seed_offset: int
) -> dict[str, Any]:
    m = base_daily.merge(other_daily, on="datetime", suffixes=("_base", "_x"))
    delta = (m["rank_ic_x"] - m["rank_ic_base"]).to_numpy()
    boot = block_bootstrap_mean(delta, seed=BOOT_SEED + seed_offset)
    return {
        "n_common_days": boot["n"],
        "delta_RankIC": boot["mean"],
        "boot_ci_lo": boot["ci_lo"],
        "boot_ci_hi": boot["ci_hi"],
        "boot_p_two_sided": boot["p_two_sided"],
    }


def train_all_specs(panel: pd.DataFrame) -> dict[str, dict[int, lgb.Booster]]:
    """Train R50 models per (spec, seed); R10/R20 via num_iteration at predict time."""
    train = process_train_label(
        panel[(panel["datetime"] >= "2008-01-01") & (panel["datetime"] <= "2017-12-31")]
    )
    models: dict[str, dict[int, lgb.Booster]] = {}
    for spec, feats in SPECS.items():
        log.info("train %s R50 × %d seeds", spec, len(SEEDS))
        tr = train.dropna(subset=feats + ["label_processed"])
        models[spec] = {seed: train_fixed(tr, feats, seed, 50) for seed in SEEDS}
    return models


def predict_seed(
    panel: pd.DataFrame, model: lgb.Booster, feats: list[str], n_round: int
) -> pd.DataFrame:
    mask = panel[feats].notna().all(axis=1) & panel["label"].notna()
    sub = panel.loc[mask, ["datetime", "instrument", "label"]].copy()
    sub["score"] = model.predict(panel.loc[mask, feats], num_iteration=n_round)
    return sub


def run_part_a(panel: pd.DataFrame, models: dict, out: Path) -> dict:
    period_seed_rows = []
    period_agg_rows = []
    yearly_rows = []
    paired_rows = []
    ens_daily: dict[tuple, pd.DataFrame] = {}

    for n_round in BOOST_ROUNDS:
        seed_dailies: dict[str, dict[int, dict[str, pd.DataFrame]]] = {
            spec: {seed: {} for seed in SEEDS} for spec in SPECS
        }
        ens_preds: dict[str, pd.DataFrame] = {}

        for spec, feats in SPECS.items():
            log.info("predict %s R%d", spec, n_round)
            per_seed_preds = []
            for seed in SEEDS:
                pred = predict_seed(panel, models[spec][seed], feats, n_round)
                per_seed_preds.append(pred.assign(seed=seed))
                for period, (s, e) in PERIODS.items():
                    d = daily_ic(pred, s, e)
                    seed_dailies[spec][seed][period] = d
                    sm = summarize_daily(d)
                    period_seed_rows.append(
                        {
                            "spec": spec,
                            "num_boost_round": n_round,
                            "period": period,
                            "seed": seed,
                            **sm,
                        }
                    )
            # ensemble = mean score across seeds
            cat = pd.concat(per_seed_preds, ignore_index=True)
            ens = (
                cat.groupby(["datetime", "instrument", "label"], as_index=False)["score"]
                .mean()
            )
            ens_preds[spec] = ens
            ens.to_parquet(out / f"pred_{spec}_R{n_round}_ensemble.parquet", index=False)

            for period, (s, e) in PERIODS.items():
                d_ens = daily_ic(ens, s, e)
                ens_daily[(spec, n_round, period)] = d_ens
                # seed mean / std of period-level metrics
                seed_mets = [
                    summarize_daily(seed_dailies[spec][seed][period]) for seed in SEEDS
                ]
                sm_df = pd.DataFrame(seed_mets)
                ens_sm = summarize_daily(d_ens)
                period_agg_rows.append(
                    {
                        "spec": spec,
                        "alias": "B1" if spec == "Alpha20_RD8" else "",
                        "num_boost_round": n_round,
                        "period": period,
                        "n_days_ensemble": ens_sm["n_days"],
                        "IC_ens": ens_sm["IC"],
                        "RankIC_ens": ens_sm["RankIC"],
                        "ICIR_ens": ens_sm["ICIR"],
                        "RankICIR_ens": ens_sm["RankICIR"],
                        "IC_seed_mean": float(sm_df["IC"].mean()),
                        "IC_seed_std": float(sm_df["IC"].std(ddof=1)),
                        "RankIC_seed_mean": float(sm_df["RankIC"].mean()),
                        "RankIC_seed_std": float(sm_df["RankIC"].std(ddof=1)),
                        "ICIR_seed_mean": float(sm_df["ICIR"].mean()),
                        "ICIR_seed_std": float(sm_df["ICIR"].std(ddof=1)),
                        "RankICIR_seed_mean": float(sm_df["RankICIR"].mean()),
                        "RankICIR_seed_std": float(sm_df["RankICIR"].std(ddof=1)),
                    }
                )

            # yearly RankIC (ensemble) for 2020–23 and 2024–25
            for years, tag in ((YEARS_DEV, "dev"), (YEARS_HOLD, "holdout")):
                ydf = yearly_rankic(ens, years)
                for _, r in ydf.iterrows():
                    yearly_rows.append(
                        {
                            "spec": spec,
                            "num_boost_round": n_round,
                            "window": tag,
                            "year": int(r["year"]),
                            "n_days": int(r["n_days"]),
                            "IC": r["IC"],
                            "RankIC": r["RankIC"],
                            "ICIR": r["ICIR"],
                            "RankICIR": r["RankICIR"],
                        }
                    )

        # paired ΔRankIC vs Alpha20: overall periods + yearly
        base_spec = "Alpha20"
        for period in PERIODS:
            base_d = ens_daily[(base_spec, n_round, period)]
            for i, spec in enumerate(["Alpha20_RD8", "Alpha20_RD13_v2", "Alpha20_RD4"]):
                other_d = ens_daily[(spec, n_round, period)]
                boot = paired_delta_rankic(base_d, other_d, seed_offset=n_round * 100 + i)
                paired_rows.append(
                    {
                        "contrast": f"{spec}_minus_Alpha20",
                        "num_boost_round": n_round,
                        "period": period,
                        "year": "",
                        **boot,
                        "block": BOOT_BLOCK,
                        "n_boot": BOOT_N,
                    }
                )
            # RD13 vs RD8
            boot = paired_delta_rankic(
                ens_daily[("Alpha20_RD8", n_round, period)],
                ens_daily[("Alpha20_RD13_v2", n_round, period)],
                seed_offset=n_round * 100 + 50,
            )
            paired_rows.append(
                {
                    "contrast": "Alpha20_RD13_v2_minus_Alpha20_RD8",
                    "num_boost_round": n_round,
                    "period": period,
                    "year": "",
                    **boot,
                    "block": BOOT_BLOCK,
                    "n_boot": BOOT_N,
                }
            )

        for years, tag in ((YEARS_DEV, "dev"), (YEARS_HOLD, "holdout")):
            for y in years:
                base_d = daily_ic(ens_preds[base_spec], f"{y}-01-01", f"{y}-12-31")
                for i, spec in enumerate(["Alpha20_RD8", "Alpha20_RD13_v2", "Alpha20_RD4"]):
                    other_d = daily_ic(ens_preds[spec], f"{y}-01-01", f"{y}-12-31")
                    boot = paired_delta_rankic(
                        base_d, other_d, seed_offset=n_round * 1000 + y * 10 + i
                    )
                    paired_rows.append(
                        {
                            "contrast": f"{spec}_minus_Alpha20",
                            "num_boost_round": n_round,
                            "period": tag,
                            "year": y,
                            **boot,
                            "block": BOOT_BLOCK,
                            "n_boot": BOOT_N,
                        }
                    )

    pd.DataFrame(period_seed_rows).to_csv(out / "A_period_metrics_by_seed.csv", index=False)
    agg = pd.DataFrame(period_agg_rows)
    agg.to_csv(out / "A_period_metrics_seed_mean_std.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(out / "A_yearly_rankic.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "A_paired_delta_rankic_bootstrap.csv", index=False)
    return {"agg": agg, "ens_daily": ens_daily}


def run_part_b(panel: pd.DataFrame, ens_daily: dict, out: Path) -> pd.DataFrame:
    """D2 original (early-stop) vs unified R50 Alpha20+RD13_v2."""
    pred_d2 = pd.read_parquet(D2_PRED)
    pred_d2["datetime"] = pd.to_datetime(pred_d2["datetime"]).dt.normalize()
    pred_d2["instrument"] = pred_d2["instrument"].astype(str)
    lab = panel[["datetime", "instrument", "label"]].dropna(subset=["label"])
    m = pred_d2.merge(lab, on=["datetime", "instrument"], how="inner")

    # model meta
    model_meta = []
    for seed in SEEDS:
        mp = D2_MODELS / f"model_D2_ALPHA20_RD13_V2_seed{seed}.pkl"
        booster = None
        try:
            import pickle

            with mp.open("rb") as f:
                booster = pickle.load(f)
            model_meta.append(
                {
                    "seed": seed,
                    "best_iteration": int(booster.best_iteration or 0),
                    "num_trees": int(booster.num_trees()),
                    "path": str(mp),
                }
            )
        except Exception as e:
            model_meta.append({"seed": seed, "error": str(e), "path": str(mp)})
    pd.DataFrame(model_meta).to_csv(out / "B_D2_original_model_meta.csv", index=False)

    rows = []
    for period, (s, e) in PERIODS.items():
        if period == "holdout_2024_2025":
            # still report if available
            pass
        d_d2 = daily_ic(m, s, e)
        sm = summarize_daily(d_d2)
        rows.append(
            {
                "source": "D2_original_earlystop",
                "spec": "D2_ALPHA20_RD13_V2",
                "num_boost_round": "early_stop(max1000,patience50)",
                "period": period,
                **sm,
            }
        )

        # unified R50 from ens_daily
        key = ("Alpha20_RD13_v2", 50, period)
        if key in ens_daily:
            smu = summarize_daily(ens_daily[key])
            rows.append(
                {
                    "source": "unified_R50_no_earlystop",
                    "spec": "Alpha20_RD13_v2",
                    "num_boost_round": 50,
                    "period": period,
                    **smu,
                }
            )

    # also archived per-seed metrics for reference
    arch = pd.read_csv(D2_METRICS)
    arch = arch[(arch.spec == "D2_ALPHA20_RD13_V2") & (arch.seed == "ensemble")].copy()
    arch.to_csv(out / "B_D2_archived_ensemble_metrics.csv", index=False)

    cmp = pd.DataFrame(rows)
    cmp.to_csv(out / "B_D2_original_vs_unified_R50.csv", index=False)

    note = f"""# B. D2 原始口径 vs 统一 R50

**Generated:** {utc_now()}

## 原始 D2（W1/H0/P5 价量腿）

| 项 | 值 |
|---|---|
| 特征 | Alpha20 + RD13_v2（13） |
| 训练 | train 2008–2017；valid 2018–2019 **用于 early stopping** |
| LightGBM | OFFICIAL 超参；`num_boost_round=1000`；`early_stopping=50` |
| 种子 | 42 / 2026 / 3407；预测为 3-seed 均值 |
| 产物 | `pred_D2_ALPHA20_RD13_V2.parquet`；`model_D2_ALPHA20_RD13_V2_seed{{42,2026,3407}}.pkl` |

实测三份模型的 `best_iteration` / `num_trees` **均为 1**（valid 上 early stop 极早触发）。
因此「原始 D2」实质是 **1 棵树的浅模型**，与统一口径固定 R50 不可直接等同。

## 统一口径（本表 A）

| 项 | 值 |
|---|---|
| 特征 | 同上 Alpha20 + RD13_v2 |
| 训练 | 仅 train 2008–2017；**不用** valid early stop |
| 轮数 | 固定 R∈{{10,20,50}}（本对比取 R50） |
| 种子 | 同 42 / 2026 / 3407 均值 |

## 并列 IC（见 `B_D2_original_vs_unified_R50.csv`）

差异主因：early-stop→1 树 vs 固定 50 轮；非特征集或股票池不同。
归档 `delayed_metrics_by_spec.csv` 中 D2 ensemble valid/test IC 与本脚本用同一 pred 重算一致可核。
"""
    (out / "B_D2_ORIGINAL_VS_UNIFIED.md").write_text(note, encoding="utf-8")
    return cmp


def run_part_c(panel: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Single-factor IC for RD8 + RD13_v2; CS corr vs Alpha20 factors."""
    factor_cols = [(f"rd8_{c}", "RD8", c) for c in RD8] + [(c, "RD13_v2", c) for c in RD13]
    rows = []
    for period, (s, e) in {
        "valid_2018_2019": PERIODS["valid_2018_2019"],
        "test_2020_2023": PERIODS["test_2020_2023"],
    }.items():
        sub = panel[(panel["datetime"] >= s) & (panel["datetime"] <= e)].copy()
        for col, family, name in factor_cols:
            # treat factor as score
            tmp = sub[["datetime", "instrument", "label", col]].dropna()
            tmp = tmp.rename(columns={col: "score"})
            d = daily_ic(tmp, s, e)
            sm = summarize_daily(d)
            rows.append({"family": family, "factor": name, "period": period, **sm})
    ic_df = pd.DataFrame(rows)
    ic_df.to_csv(out / "C_single_factor_ic.csv", index=False)

    # max |ρ| with Alpha20 factors — mean daily CS Pearson on 2020–2023 (also report 2018–19)
    corr_rows = []
    for period, (s, e) in {
        "valid_2018_2019": PERIODS["valid_2018_2019"],
        "test_2020_2023": PERIODS["test_2020_2023"],
    }.items():
        sub = panel[(panel["datetime"] >= s) & (panel["datetime"] <= e)]
        for col, family, name in factor_cols:
            sums = {a: 0.0 for a in A20_COLS}
            counts = {a: 0 for a in A20_COLS}
            for dt, g in sub.groupby("datetime"):
                if len(g) < MIN_CS:
                    continue
                sf = g[col]
                for a in A20_COLS:
                    sa = g[a]
                    m = sf.notna() & sa.notna()
                    if m.sum() < MIN_CS:
                        continue
                    r = float(sf[m].corr(sa[m]))
                    if np.isfinite(r):
                        sums[a] += r
                        counts[a] += 1
            best_a, best_abs, best_r, best_n = None, -1.0, np.nan, 0
            for a in A20_COLS:
                if counts[a] == 0:
                    continue
                mean_r = sums[a] / counts[a]
                if abs(mean_r) > best_abs:
                    best_abs = abs(mean_r)
                    best_r = mean_r
                    best_a = a
                    best_n = counts[a]
            corr_rows.append(
                {
                    "family": family,
                    "factor": name,
                    "period": period,
                    "max_abs_mean_cs_corr_vs_alpha20": best_abs if best_a else np.nan,
                    "mean_cs_corr": best_r,
                    "matched_alpha20_factor": best_a,
                    "n_days": best_n,
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out / "C_rd_vs_alpha20_max_cs_corr.csv", index=False)
    return ic_df


def run_part_d(models: dict, out: Path) -> pd.DataFrame:
    """LightGBM gain importance for R50 Alpha20+RD8 and Alpha20+RD13_v2 (3-seed mean)."""
    rows = []
    summaries = []
    for spec in ("Alpha20_RD8", "Alpha20_RD13_v2"):
        feats = SPECS[spec]
        gains = np.zeros(len(feats), dtype=float)
        for seed in SEEDS:
            g = models[spec][seed].feature_importance(importance_type="gain")
            # align by feature name
            names = models[spec][seed].feature_name()
            gmap = dict(zip(names, g))
            gains += np.array([float(gmap.get(f, 0.0)) for f in feats])
        gains /= len(SEEDS)
        total = gains.sum()
        share = gains / total if total > 0 else gains
        rd_mask = np.array([not f.startswith("alpha20_") for f in feats])
        rd_share = float(share[rd_mask].sum())
        a20_share = float(share[~rd_mask].sum())
        for f, g, sh in zip(feats, gains, share):
            rows.append(
                {
                    "spec": spec,
                    "feature": f,
                    "gain_mean_3seed": float(g),
                    "gain_share": float(sh),
                    "is_rd": bool(not f.startswith("alpha20_")),
                }
            )
        top10 = sorted(zip(feats, share), key=lambda x: -x[1])[:10]
        summaries.append(
            {
                "spec": spec,
                "n_features": len(feats),
                "rd_gain_share": rd_share,
                "alpha20_gain_share": a20_share,
                "top10": "; ".join(f"{n}={s:.4f}" for n, s in top10),
            }
        )
        pd.DataFrame(
            [{"rank": i + 1, "feature": n, "gain_share": s} for i, (n, s) in enumerate(top10)]
        ).to_csv(out / f"D_top10_{spec}_R50.csv", index=False)

    imp = pd.DataFrame(rows)
    imp.to_csv(out / "D_feature_importance_gain_R50.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "D_rd_block_gain_share_R50.csv", index=False)
    return imp


def write_summary(out: Path, a_agg: pd.DataFrame, b_cmp: pd.DataFrame, c_ic: pd.DataFrame) -> None:
    lines = [
        "# 统一下游设定表 A–D",
        "",
        f"**Generated:** {utc_now()}  ",
        f"**Directory:** `{out.name}`",
        "",
        "设定：固定 LightGBM OFFICIAL；R∈{10,20,50}；seeds 42/2026/3407；无 early stopping；",
        "delayed common sample；train 2008–2017。Alpha20_RD8 ≡ B1。",
        "",
        "## A. 价量模型表（验证 2018–19 / 开发 2020–23 / 附 2024–25）",
        "",
        "完整数值见 `A_period_metrics_seed_mean_std.csv`（含种子均值±标准差与 ensemble）、",
        "`A_yearly_rankic.csv`、`A_paired_delta_rankic_bootstrap.csv`。",
        "",
        "### R50 验证期 / 开发期（ensemble + seed mean±std of RankIC）",
        "",
        "| spec | period | RankIC_ens | RankIC seed μ±σ | IC_ens | ICIR_ens |",
        "|---|---|---:|---|---:|---:|",
    ]
    sub = a_agg[a_agg.num_boost_round == 50]
    for period in ("valid_2018_2019", "test_2020_2023", "holdout_2024_2025"):
        for spec in SPECS:
            r = sub[(sub.spec == spec) & (sub.period == period)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                f"| {spec} | {period} | {r.RankIC_ens:.6f} | "
                f"{r.RankIC_seed_mean:.6f}±{r.RankIC_seed_std:.6f} | "
                f"{r.IC_ens:.6f} | {r.ICIR_ens:.4f} |"
            )

    lines += [
        "",
        "## B. D2 原始 vs 统一 R50",
        "",
        "见 `B_D2_ORIGINAL_VS_UNIFIED.md` 与 `B_D2_original_vs_unified_R50.csv`。",
        "原始 D2：early_stopping=50、max 1000、三种子；**实际 best_iteration=1**。",
        "",
    ]
    for _, r in b_cmp.iterrows():
        if r.period in ("valid_2018_2019", "test_2020_2023"):
            lines.append(
                f"- {r.source} / {r.period}: IC={r.IC:.6f}, RankIC={r.RankIC:.6f}"
            )

    # C highlight 2020-23
    c_test = c_ic[c_ic.period == "test_2020_2023"].sort_values("RankIC", ascending=False)
    lines += [
        "",
        "## C. 单因子诊断（2020–2023 RankIC 前 5）",
        "",
        "| family | factor | RankIC | IC | ICIR |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in c_test.head(5).iterrows():
        lines.append(
            f"| {r.family} | {r.factor} | {r.RankIC:.6f} | {r.IC:.6f} | {r.ICIR:.4f} |"
        )
    lines += [
        "",
        "与 Alpha20 最大 |ρ| 见 `C_rd_vs_alpha20_max_cs_corr.csv`。",
        "",
        "## D. 特征重要性（R50，3-seed 均值 gain）",
        "",
        "见 `D_feature_importance_gain_R50.csv`、`D_rd_block_gain_share_R50.csv`、`D_top10_*_R50.csv`。",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT / "reports" / f"rd13_l20_unified_ABCD_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    log.info("out=%s", out)

    panel = load_merged_panel()
    models = train_all_specs(panel)
    a = run_part_a(panel, models, out)
    b = run_part_b(panel, a["ens_daily"], out)
    c = run_part_c(panel, out)
    run_part_d(models, out)
    write_summary(out, a["agg"], b, c)

    manifest = {
        "generated_at": utc_now(),
        "out": str(out),
        "delayed_panel": str(DELAYED),
        "rd8_extended": str(RD8_EXT),
        "seeds": SEEDS,
        "boost_rounds": BOOST_ROUNDS,
        "early_stopping": False,
        "lgb": OFFICIAL_LGB,
        "boot_block": BOOT_BLOCK,
        "boot_n": BOOT_N,
        "runtime_sec": time.perf_counter() - t0,
        "d2_pred": str(D2_PRED),
        "note": "Alpha20_RD8 alias B1; R10/R20 predicted via num_iteration from R50 models",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("done in %.1fs -> %s", time.perf_counter() - t0, out)


if __name__ == "__main__":
    main()
