#!/usr/bin/env python3
"""RD13 L20 trajectory table + Alpha20/RD8/RD13_v2/RD4 downstream IC comparison + redundancy.

Primary window 2020–2023; 2024–2025 reported as appendix.
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

PROJECT = Path(os.environ["PROJECT_ROOT"]) if os.environ.get("PROJECT_ROOT") else Path(__file__).resolve().parents[1]
DELAYED = PROJECT / "data/rd13_v2_downstream_replication/stage_1_delayed_panel/delayed_common_sample_rd13_v2.parquet"
RD8_EXT = PROJECT / "reports/holdout_2024_2025_K5D1_WD_no_retrain_20260921_000350/rd_loop9_panel_extended.pkl"
SPY_CSV = PROJECT / "reports/rd_factor_run_identity_audit_20260923/LEGACY_FIRST20_SPY_REVALUATION.csv"
FACTOR_DETAIL = PROJECT / "docs/experiment_log/0710_log/phase2_experiment_data/rdagent_40loop_factor_detail.csv"
SESSION_FB = (Path(os.environ["RDAGENT_ROOT"]) / "log/2026-07-07_12-00-59-480728")

ALPHA20 = [
    "resi5", "wvma5", "rsqr5", "klen", "rsqr10", "corr5", "cord5", "corr10", "roc60",
    "resi10", "vstd5", "rsqr60", "corr60", "wvma60", "std5", "rsqr20", "cord60", "cord10", "corr20", "klow",
]
A20_COLS = [f"alpha20_{c}" for c in ALPHA20]
RD13 = [
    "rd13_10_day_momentum", "rd13_20_day_realized_volatility", "rd13_5_day_volume_deviation",
    "rd13_10_day_price_range_ratio", "rd13_volatility_adjusted_10d_momentum", "rd13_5d_short_term_reversal",
    "rd13_daily_amihud_illiquidity", "rd13_rolling_beta_20d", "rd13_idiosyncratic_volatility_10d",
    "rd13_cs_momentum_rank_5d", "rd13_win_rate_5d", "rd13_upday_volume_ratio_5d", "rd13_atr_norm_close_14d",
]
RD4 = [
    "rd13_10_day_momentum", "rd13_volatility_adjusted_10d_momentum",
    "rd13_rolling_beta_20d", "rd13_win_rate_5d",
]
RD8 = [
    "reversal_1d", "atr_10d", "volume_ratio_10d", "vol_norm_volume_momentum_20d",
    "vol_norm_price_momentum_10d", "rv_corr_20d", "dynamic_price_momentum", "dynamic_volume_momentum",
]
RD8_PREFIXED = [f"rd8_{c}" for c in RD8]

OFFICIAL_LGB = {
    "objective": "regression", "metric": "mse", "verbosity": -1,
    "colsample_bytree": 0.8879, "learning_rate": 0.2, "subsample": 0.8789,
    "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8, "num_leaves": 210, "num_threads": 8,
}
SEEDS = [42, 2026, 3407]
BOOST_ROUNDS = [10, 20, 50]
MIN_CS = 30
BOOT_BLOCK = 21
BOOT_N = 2000
BOOT_SEED = 20260923

# ≤10 Chinese chars themes for loops 0–19
THEMES = {
    0: "动量波动量价",
    1: "波动动量反转",
    2: "隔夜日内偏度",
    3: "Beta特质波动",
    4: "多期波动动量",
    5: "EWMA波动排名",
    6: "突破RSI下行",
    7: "胜率量ATR",
    8: "十日胜率VWAP",
    9: "动量量能加速",
    10: "美元量流动性",
    11: "收益PCA暴露",
    12: "原factor变换",
    13: "Factor中性化",
    14: "个股PCA潜因子",
    15: "AE+LGBM混合",
    16: "AE三维+LGBM",
    17: "LSTM+网络中心",
    18: "注意力图网络",
    19: "Hurst熵分形",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rd13_downstream_cmp")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def process_train_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["label"]).copy()
    out["label_processed"] = out.groupby("datetime", group_keys=False)["label"].apply(zscore)
    return out


def train_fixed(train_df, features: list[str], seed: int, n_round: int) -> lgb.Booster:
    p = OFFICIAL_LGB.copy()
    p.update({"seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed, "data_random_seed": seed})
    tr = lgb.Dataset(train_df[features], label=train_df["label_processed"], free_raw_data=False)
    return lgb.train(p, tr, num_boost_round=n_round)


def daily_ic(pred: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    sub = pred[(pred["datetime"] >= pd.Timestamp(start)) & (pred["datetime"] <= pd.Timestamp(end))]
    rows = []
    for dt, g in sub.groupby("datetime"):
        if len(g) < MIN_CS or g["score"].nunique() < 2:
            continue
        rows.append(
            {
                "datetime": dt,
                "ic": float(g["score"].corr(g["label"])),
                "rank_ic": float(g["score"].corr(g["label"], method="spearman")),
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def block_bootstrap_mean(x: np.ndarray, block: int = BOOT_BLOCK, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block:
        return {"mean": float(np.mean(x)) if n else np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_two_sided": np.nan, "n": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([x[s : s + block] for s in starts])[:n]
        boots.append(sample.mean())
    boots = np.asarray(boots)
    mean = float(np.mean(x))
    # two-sided p: fraction of bootstrap means with opposite sign or zero vs observed direction
    if mean >= 0:
        p = float((boots <= 0).mean())
    else:
        p = float((boots >= 0).mean())
    p = min(2 * p, 1.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": mean, "ci_lo": float(lo), "ci_hi": float(hi), "p_two_sided": p, "n": n}


def build_trajectory(out: Path) -> pd.DataFrame:
    spy = pd.read_csv(SPY_CSV)
    detail = pd.read_csv(FACTOR_DETAIL)
    hyp = detail.groupby("loop")["hypothesis"].first()
    rows = []
    for loop in range(20):
        import pickle

        fb_files = sorted((SESSION_FB / f"Loop_{loop}/feedback/feedback").rglob("*.pkl"))
        fb = pickle.load(fb_files[-1].open("rb")) if fb_files else None
        r = spy[spy.loop == loop].iloc[0]
        # 有效 = has PortAna metrics in SPY revaluation (all L0–19 do)
        valid = bool(np.isfinite(r["spy_excess_ARR_with_cost"]))
        decision = "Replace=no" if (fb is None or not bool(fb.decision)) else "Replace=yes"
        rows.append(
            {
                "Loop": loop,
                "搜索主题": THEMES[loop],
                "IC": float(r["IC"]),
                "ARR_SPY_with_cost": float(r["spy_excess_ARR_with_cost"]),
                "MDD_SPY_with_cost": float(r["spy_excess_MDD_with_cost"]),
                "ARR_pct": f"{100*float(r['spy_excess_ARR_with_cost']):.2f}%",
                "MDD_pct": f"{100*float(r['spy_excess_MDD_with_cost']):.2f}%",
                "是否有效": "有效" if valid else "invalid",
                "结果": f"{'有效' if valid else 'invalid'} / {decision}",
                "hypothesis_snippet": str(hyp.get(loop, ""))[:120],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "rd13_L20_trajectory_E1_style.csv", index=False)
    return df


def load_merged_panel() -> pd.DataFrame:
    panel = pd.read_parquet(DELAYED)
    panel["datetime"] = pd.to_datetime(panel["datetime"]).dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    r8 = pd.read_pickle(RD8_EXT)
    if isinstance(r8.index, pd.MultiIndex):
        r8 = r8.reset_index()
    if isinstance(r8.columns, pd.MultiIndex):
        r8.columns = [c[-1] if isinstance(c, tuple) else c for c in r8.columns]
    r8["datetime"] = pd.to_datetime(r8["datetime"]).dt.normalize()
    r8["instrument"] = r8["instrument"].astype(str)
    rename = {c: f"rd8_{c}" for c in RD8 if c in r8.columns}
    r8 = r8.rename(columns=rename)
    keep = ["datetime", "instrument"] + [f"rd8_{c}" for c in RD8]
    r8 = r8[keep]
    m = panel.merge(r8, on=["datetime", "instrument"], how="left")
    log.info("merged panel rows=%d rd8_nonnull=%d", len(m), int(m["rd8_reversal_1d"].notna().sum()))
    return m


def run_downstream(panel: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = process_train_label(panel[(panel["datetime"] >= "2008-01-01") & (panel["datetime"] <= "2017-12-31")])
    # valid unused for fixed rounds but kept for protocol note
    specs = {
        "Alpha20": A20_COLS,
        "Alpha20_RD8": A20_COLS + RD8_PREFIXED,
        "Alpha20_RD13_v2": A20_COLS + RD13,
        "Alpha20_RD4": A20_COLS + RD4,
    }
    # require non-null features on train for each spec separately
    summary_rows = []
    daily_store = {}  # (spec, n_round, period) -> daily ic df
    pred_paths = []

    for n_round in BOOST_ROUNDS:
        for spec, feats in specs.items():
            log.info("train %s rounds=%d", spec, n_round)
            tr = train.dropna(subset=feats + ["label_processed"])
            models = [train_fixed(tr, feats, seed, n_round) for seed in SEEDS]
            # predict on full panel where features available
            mask = panel[feats].notna().all(axis=1) & panel["label"].notna()
            sub = panel.loc[mask, ["datetime", "instrument", "label"] + feats].copy()
            scores = np.mean([m.predict(sub[feats]) for m in models], axis=0)
            sub["score"] = scores
            pred = sub[["datetime", "instrument", "label", "score"]]
            pred.to_parquet(out / f"pred_{spec}_R{n_round}.parquet", index=False)
            pred_paths.append(str(out / f"pred_{spec}_R{n_round}.parquet"))

            for period, (s, e) in {
                "test_2020_2023": ("2020-01-01", "2023-12-31"),
                "holdout_2024_2025": ("2024-01-01", "2025-12-31"),
            }.items():
                d = daily_ic(pred, s, e)
                daily_store[(spec, n_round, period)] = d
                summary_rows.append(
                    {
                        "spec": spec,
                        "num_boost_round": n_round,
                        "period": period,
                        "n_days": int(len(d)),
                        "IC": float(d["ic"].mean()) if len(d) else np.nan,
                        "RankIC": float(d["rank_ic"].mean()) if len(d) else np.nan,
                        "ICIR": float(d["ic"].mean() / d["ic"].std()) if len(d) and d["ic"].std() > 0 else np.nan,
                        "RankICIR": float(d["rank_ic"].mean() / d["rank_ic"].std()) if len(d) and d["rank_ic"].std() > 0 else np.nan,
                    }
                )
                d.assign(spec=spec, num_boost_round=n_round, period=period).to_csv(
                    out / f"daily_ic_{spec}_R{n_round}_{period}.csv", index=False
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "ic_summary_by_spec_rounds.csv", index=False)

    # Paired ΔRankIC on common dates
    pair_rows = []
    for n_round in BOOST_ROUNDS:
        for period in ("test_2020_2023", "holdout_2024_2025"):
            base = daily_store[("Alpha20", n_round, period)]
            for spec in ("Alpha20_RD8", "Alpha20_RD13_v2", "Alpha20_RD4"):
                other = daily_store[(spec, n_round, period)]
                m = base.merge(other, on="datetime", suffixes=("_a20", "_x"))
                delta = (m["rank_ic_x"] - m["rank_ic_a20"]).to_numpy()
                boot = block_bootstrap_mean(delta, seed=BOOT_SEED + n_round + hash(spec + period) % 10000)
                pair_rows.append(
                    {
                        "contrast": f"{spec}_minus_Alpha20",
                        "num_boost_round": n_round,
                        "period": period,
                        "n_common_days": boot["n"],
                        "delta_RankIC": boot["mean"],
                        "boot_ci_lo": boot["ci_lo"],
                        "boot_ci_hi": boot["ci_hi"],
                        "boot_p_two_sided": boot["p_two_sided"],
                        "block": BOOT_BLOCK,
                        "n_boot": BOOT_N,
                    }
                )
            # RD8 vs RD13_v2
            a = daily_store[("Alpha20_RD8", n_round, period)]
            b = daily_store[("Alpha20_RD13_v2", n_round, period)]
            m = a.merge(b, on="datetime", suffixes=("_rd8", "_rd13"))
            delta = (m["rank_ic_rd13"] - m["rank_ic_rd8"]).to_numpy()
            boot = block_bootstrap_mean(delta, seed=BOOT_SEED + 99 + n_round)
            pair_rows.append(
                {
                    "contrast": "Alpha20_RD13_v2_minus_Alpha20_RD8",
                    "num_boost_round": n_round,
                    "period": period,
                    "n_common_days": boot["n"],
                    "delta_RankIC": boot["mean"],
                    "boot_ci_lo": boot["ci_lo"],
                    "boot_ci_hi": boot["ci_hi"],
                    "boot_p_two_sided": boot["p_two_sided"],
                    "block": BOOT_BLOCK,
                    "n_boot": BOOT_N,
                }
            )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "paired_delta_rankic_bootstrap.csv", index=False)
    return summary, pairs, panel


def factor_redundancy(panel: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Mean daily CS Pearson corr 2020–2023 between RD8 and RD13_v2 factors."""
    sub = panel[(panel["datetime"] >= "2020-01-01") & (panel["datetime"] <= "2023-12-31")].copy()
    cols8 = RD8_PREFIXED
    cols13 = RD13
    # accumulate corr matrices
    sums = {c8: {c13: 0.0 for c13 in cols13} for c8 in cols8}
    counts = {c8: {c13: 0 for c13 in cols13} for c8 in cols8}
    n_days = 0
    for dt, g in sub.groupby("datetime"):
        if len(g) < MIN_CS:
            continue
        # drop rows with all-nan in either block
        gg = g[cols8 + cols13].astype(float)
        if gg[cols8].notna().any(axis=1).sum() < MIN_CS:
            continue
        n_days += 1
        for c8 in cols8:
            s8 = gg[c8]
            for c13 in cols13:
                s13 = gg[c13]
                m = s8.notna() & s13.notna()
                if m.sum() < MIN_CS:
                    continue
                corr = float(s8[m].corr(s13[m]))
                if np.isfinite(corr):
                    sums[c8][c13] += corr
                    counts[c8][c13] += 1
    # RD8 -> best RD13
    rows = []
    for c8 in cols8:
        best_c, best_v = None, -np.inf
        for c13 in cols13:
            if counts[c8][c13] == 0:
                continue
            v = abs(sums[c8][c13] / counts[c8][c13])
            mean_r = sums[c8][c13] / counts[c8][c13]
            if v > best_v:
                best_v = v
                best_c = c13
                best_mean = mean_r
                best_n = counts[c8][c13]
        rows.append(
            {
                "direction": "RD8_to_RD13_v2",
                "source_factor": c8.replace("rd8_", ""),
                "best_match": best_c,
                "mean_cs_corr": best_mean if best_c else np.nan,
                "abs_mean_cs_corr": abs(best_mean) if best_c else np.nan,
                "n_days": best_n if best_c else 0,
            }
        )
    # RD13 -> best RD8
    for c13 in cols13:
        best_c, best_v, best_mean, best_n = None, -np.inf, np.nan, 0
        for c8 in cols8:
            if counts[c8][c13] == 0:
                continue
            mean_r = sums[c8][c13] / counts[c8][c13]
            if abs(mean_r) > best_v:
                best_v = abs(mean_r)
                best_c = c8.replace("rd8_", "")
                best_mean = mean_r
                best_n = counts[c8][c13]
        rows.append(
            {
                "direction": "RD13_v2_to_RD8",
                "source_factor": c13,
                "best_match": best_c,
                "mean_cs_corr": best_mean,
                "abs_mean_cs_corr": abs(best_mean) if best_c else np.nan,
                "n_days": best_n,
            }
        )
    # full matrix
    mat = []
    for c8 in cols8:
        for c13 in cols13:
            if counts[c8][c13]:
                mat.append(
                    {
                        "rd8": c8.replace("rd8_", ""),
                        "rd13": c13,
                        "mean_cs_corr": sums[c8][c13] / counts[c8][c13],
                        "n_days": counts[c8][c13],
                    }
                )
    pd.DataFrame(mat).to_csv(out / "rd8_rd13_cs_corr_matrix_2020_2023.csv", index=False)
    red = pd.DataFrame(rows)
    red.to_csv(out / "rd8_rd13_redundancy_best_match.csv", index=False)
    log.info("redundancy days used≈%d", n_days)
    return red


def write_summary(out: Path, traj: pd.DataFrame, summary: pd.DataFrame, pairs: pd.DataFrame, red: pd.DataFrame) -> None:
    # trajectory markdown table
    lines = [
        "# RD13 L20 轨迹 + 下游因子集对比（2020–2023 主 / 2024–2025 附）",
        "",
        f"**Generated:** {utc_now()}  ",
        f"**Directory:** `{out.name}`",
        "",
        "## 1. RD13 轨迹 loops 0–19（对齐第4章附录 E.1 格式；ARR/MDD 为 SPY 口径）",
        "",
        "| Loop | 搜索主题 | IC | ARR（含成本，SPY） | MDD（含成本，SPY） | 结果 |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for _, r in traj.iterrows():
        lines.append(
            f"| {int(r.Loop)} | {r.搜索主题} | {r.IC:.6f} | {r.ARR_pct} | {r.MDD_pct} | {r.结果} |"
        )
    lines += [
        "",
        "注：历史 session 全轮 `Replace Best Result=no`；「有效」= 有可评估 PortAna 输出（L0–19 均有效）。"
        " ARR/MDD 来自 `LEGACY_FIRST20_SPY_REVALUATION.csv`（非运行时 Oracle）。",
        "",
        "## 2. 下游因子集对比（固定 LightGBM OFFICIAL 超参；num_boost_round∈{10,20,50}；无 early stopping；seeds 42/2026/3407 均值）",
        "",
        "股票池/标签/切分：delayed common sample panel（train 2008–2017，valid 2018–2019 协议对齐，本实验固定 round 不用 valid early-stop；评测 test 2020–2023 / holdout 2024–2025）。",
        "",
        "### 2.1 IC / RankIC",
        "",
    ]
    # pivot primary
    prim = summary[summary.period == "test_2020_2023"]
    lines.append("**主窗 2020–2023**")
    lines.append("")
    lines.append("| spec | R10 IC | R10 RankIC | R20 IC | R20 RankIC | R50 IC | R50 RankIC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for spec in ["Alpha20", "Alpha20_RD8", "Alpha20_RD13_v2", "Alpha20_RD4"]:
        cells = [spec]
        for nr in [10, 20, 50]:
            row = prim[(prim.spec == spec) & (prim.num_boost_round == nr)]
            if row.empty:
                cells += ["—", "—"]
            else:
                cells += [f"{row.iloc[0].IC:.6f}", f"{row.iloc[0].RankIC:.6f}"]
        lines.append("| " + " | ".join(cells) + " |")
    hold = summary[summary.period == "holdout_2024_2025"]
    lines += ["", "**附窗 2024–2025**", "", "| spec | R10 IC | R10 RankIC | R20 IC | R20 RankIC | R50 IC | R50 RankIC |", "|---|---:|---:|---:|---:|---:|---:|"]
    for spec in ["Alpha20", "Alpha20_RD8", "Alpha20_RD13_v2", "Alpha20_RD4"]:
        cells = [spec]
        for nr in [10, 20, 50]:
            row = hold[(hold.spec == spec) & (hold.num_boost_round == nr)]
            if row.empty:
                cells += ["—", "—"]
            else:
                cells += [f"{row.iloc[0].IC:.6f}", f"{row.iloc[0].RankIC:.6f}"]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "### 2.2 Paired ΔRankIC（共同交易日；block=21，n_boot=2000）",
        "",
        "| contrast | period | R | ΔRankIC | 95% CI | p (two-sided) |",
        "|---|---|---:|---:|---|---:|",
    ]
    for _, r in pairs.iterrows():
        lines.append(
            f"| {r.contrast} | {r.period} | {int(r.num_boost_round)} | {r.delta_RankIC:+.6f} | [{r.boot_ci_lo:+.6f}, {r.boot_ci_hi:+.6f}] | {r.boot_p_two_sided:.4f} |"
        )

    lines += [
        "",
        "### 2.3 Phase 2 中 `A20_RD6` 与本组对比的差异",
        "",
        "- **A20_RD6**（`reports/corrected_downstream_loop2_20260917`）：Alpha20 + matched **20-loop 修正前** Replace 链 Loop1+2 的 **6** 个因子"
        "（`reversal_1d, atr_10d, volume_ratio_10d, atr_20d, volume_ratio_20d, atr_20d_x_volume_ratio_20d`），clean 面板。",
        "- **本组 Alpha20_RD8**：修正语义 20-loop **Loop9 累积 library8**（另含 vol_norm_* / rv_corr / dynamic_*），"
        "历史至 2023 用归档 parquet，2024–2025 用 Qlib 重算拼接（`rd_loop9_panel_extended.pkl`）。",
        "- **本组 Alpha20_RD13_v2**：legacy **40-loop** loops 0/1/3/7 的 13 个独立重建因子（D2 价量腿）。",
        "- **本组 Alpha20_RD4**：RD13_v2 中 4 个 preserved_canonical 列（v1 有效独立向量）。",
        "- 因此 A20_RD6 ≠ RD8 ≠ RD13；不可直接把 Phase2 A20_RD6 数字当作本组 RD8/RD13 结果。",
        "",
        "## 3. RD8 ↔ RD13_v2 因子冗余（2020–2023 日度横截面相关均值）",
        "",
        "### RD8 → 最相关 RD13_v2",
        "",
        "| RD8 | best RD13_v2 | mean CS corr |",
        "|---|---|---:|",
    ]
    for _, r in red[red.direction == "RD8_to_RD13_v2"].iterrows():
        lines.append(f"| {r.source_factor} | {r.best_match} | {r.mean_cs_corr:.4f} |")
    lines += ["", "### RD13_v2 → 最相关 RD8", "", "| RD13_v2 | best RD8 | mean CS corr |", "|---|---|---:|"]
    for _, r in red[red.direction == "RD13_v2_to_RD8"].iterrows():
        lines.append(f"| {r.source_factor} | {r.best_match} | {r.mean_cs_corr:.4f} |")
    lines += ["", "全矩阵：`rd8_rd13_cs_corr_matrix_2020_2023.csv`。", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT / f"reports/rd13_l20_downstream_cmp_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    log.info("output %s", out)

    traj = build_trajectory(out)
    panel = load_merged_panel()
    summary, pairs, panel = run_downstream(panel, out)
    red = factor_redundancy(panel, out)
    write_summary(out, traj, summary, pairs, red)

    manifest = {
        "generated_at": utc_now(),
        "delayed_panel": str(DELAYED),
        "rd8_extended": str(RD8_EXT),
        "seeds": SEEDS,
        "boost_rounds": BOOST_ROUNDS,
        "early_stopping": False,
        "lgb": OFFICIAL_LGB,
        "boot_block": BOOT_BLOCK,
        "boot_n": BOOT_N,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("done → %s", out)
    print(out)


if __name__ == "__main__":
    main()
