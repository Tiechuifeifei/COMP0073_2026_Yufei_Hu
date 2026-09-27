#!/usr/bin/env python3
"""PV × fundamental signal-layer linear blending weights (R50 capacity).

Select w* on validation only; report test metrics. Holdout 2024–2025 is not
used for weight selection."""

from __future__ import annotations

import json
import logging
import pickle
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loop9_fixed_capacity_robustness import (
    enrich_paired,
    predict_fixed,
    train_fixed,
    two_sided_p_from_t,
)
from loop9_fund_incremental_ablation import (
    FUND_F1C,
    NW_LAGS,
    PERIODS,
    SEEDS,
    daily_ic_metrics,
    ensemble_predictions,
    newey_west_tstat,
    paired_delta,
    process_training_label,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PRINCIPAL_ID = "loop9_fund_incremental_20260918_222943"
PRINCIPAL_ROOT = PROJECT_ROOT / "reports" / PRINCIPAL_ID
FIXED_ID = "loop9_fund_incremental_fixed_capacity_20260919_000518"
FIXED_ROOT = PROJECT_ROOT / "reports" / FIXED_ID

B1_R50_DIR = FIXED_ROOT / "models" / "round_50" / "B1"
B3_R50_DIR = FIXED_ROOT / "models" / "round_50" / "B3"
B3_DATASET = PRINCIPAL_ROOT / "datasets" / "B3_alpha20_loop9lib8_fund.parquet"

COMMON_N = 1_880_525
NUM_BOOST_ROUND = 50
HOLDOUT_START = "2024-01-01"
MODELLING_END = "2023-12-31"
W_GRID = [round(x * 0.1, 1) for x in range(0, 11)]
BOOT_B = 2000
BOOT_BLOCK = 20
BOOT_SEED = 20260919
TIE_EPS = 1e-6

F1C_FEATURES = [f"fund_{c}" for c in FUND_F1C]

# Literature-signed EW composite (pre-specified; not fit on this data)
EW_POS = [
    "fund_roe_z",
    "fund_roa_z",
    "fund_gross_profitability_z",
    "fund_book_to_market_z",
    "fund_earnings_yield_z",
    "fund_sales_to_price_z",
    "fund_current_ratio_z",
]
EW_NEG = [
    "fund_asset_growth_yoy_z",
    "fund_accruals_z",
]
EW_EXCLUDED = [
    "fund_sales_growth_yoy_z",
    "fund_leverage_z",
]

REUSED_FUNCTIONS = [
    "fundamental_experiments/loop9_fund_incremental_ablation.py::daily_ic_metrics",
    "fundamental_experiments/loop9_fund_incremental_ablation.py::paired_delta",
    "fundamental_experiments/loop9_fund_incremental_ablation.py::newey_west_tstat",
    "fundamental_experiments/loop9_fund_incremental_ablation.py::ensemble_predictions",
    "fundamental_experiments/loop9_fund_incremental_ablation.py::process_training_label",
    "fundamental_experiments/loop9_fund_incremental_ablation.py::PERIODS (covid/post_covid)",
    "fundamental_experiments/loop9_fixed_capacity_robustness.py::train_fixed",
    "fundamental_experiments/loop9_fixed_capacity_robustness.py::predict_fixed",
    "fundamental_experiments/loop9_fixed_capacity_robustness.py::enrich_paired / evidence labels",
    "fundamental_experiments/loop9_fixed_capacity_robustness.py::two_sided_p_from_t",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pv_fund_blend")


def git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            )
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def assert_no_holdout(dates: pd.Series | pd.Index, label: str) -> None:
    mx = pd.to_datetime(dates).max()
    if pd.isna(mx) or mx > pd.Timestamp(MODELLING_END) or mx >= pd.Timestamp(HOLDOUT_START):
        raise AssertionError(f"{label}: max(date)={mx} violates holdout closed rule")


def load_pred(path: Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.Series):
        obj = obj.to_frame("score")
    if "score" not in obj.columns:
        obj = obj.rename(columns={obj.columns[0]: "score"})
    return obj.sort_index()


def append_preregistration(decision_log: Path, stamp: str) -> None:
    block = f"""
### [{stamp}] PV × Fundamental blending 预注册
- 目的：估计量价信号与基本面信号在 1 日预测期（LABEL0）上的线性组合权重
- 量价信号：B1-R50 ensemble 预测（Alpha20 + Loop9 RD8，num_boost_round=50，无 early stopping）
- 基本面信号（主）：F1C-R50 ensemble 预测（11 个 fund_*_z，同一协议）
- 基本面信号（对照）：文献符号等权 composite（定义见 README，不经数据拟合）
- 组合：每日截面 z-score 后 score = w·z(PV) + (1−w)·z(Fund)，w ∈ {{0.0, 0.1, …, 1.0}}
- 选择规则：w* = valid 期（2018–2019）ensemble 平均 Rank IC 最大的 w；若差距 < 1e-6，取 |w−0.5| 最小者
- test 期仅报告，不参与选择；holdout 关闭
- 选择使用 R50 预测的理由：early-stop 版本的 valid 期参与了模型选择，不能用于二次选择
"""
    if decision_log.exists():
        prev = decision_log.read_text(encoding="utf-8")
    else:
        prev = "# Decision log — pv_fund_blend_r50\n"
    decision_log.write_text(prev.rstrip() + "\n" + block.strip() + "\n", encoding="utf-8")


def audit_b1_r50(out_root: Path, label_df: pd.DataFrame) -> tuple[bool, str]:
    lines: list[str] = ["# Input audit — B1-R50", ""]
    ok = True

    def fail(msg: str) -> None:
        nonlocal ok
        ok = False
        lines.append(f"- **FAIL:** {msg}")

    def pass_(msg: str) -> None:
        lines.append(f"- **PASS:** {msg}")

    # 1. location / seeds / ensemble
    lines.append("## 1. Location, seeds, ensemble")
    if not B1_R50_DIR.exists():
        fail(f"missing directory {B1_R50_DIR}")
    else:
        pass_(f"directory exists: `{B1_R50_DIR}`")
    seed_preds = []
    for seed in SEEDS:
        p = B1_R50_DIR / f"pred_seed{seed}.pkl"
        m = B1_R50_DIR / f"model_seed{seed}.pkl"
        if not p.exists() or not m.exists():
            fail(f"missing seed={seed} pred/model")
        else:
            pass_(f"seed={seed} pred+model present")
            seed_preds.append(load_pred(p))
    ens_path = B1_R50_DIR / "pred_ensemble.pkl"
    if not ens_path.exists():
        fail("missing pred_ensemble.pkl")
    else:
        ens = load_pred(ens_path)
        if seed_preds:
            recon = ensemble_predictions(seed_preds)
            max_abs = float((ens["score"] - recon["score"]).abs().max())
            if max_abs > 1e-10:
                fail(f"ensemble != equal-weight mean of seeds (max_abs={max_abs})")
            else:
                pass_(f"ensemble equals equal-weight mean of seeds (max_abs={max_abs})")
        pass_(f"seeds set = {SEEDS}")

    # 2. capacity / early stopping
    lines.append("")
    lines.append("## 2. num_boost_round=50, no early stopping")
    for seed in SEEDS:
        mp = B1_R50_DIR / f"model_seed{seed}.pkl"
        if not mp.exists():
            continue
        with open(mp, "rb") as f:
            model = pickle.load(f)
        n_trees = int(model.num_trees())
        best = int(model.best_iteration) if model.best_iteration else 0
        if n_trees != NUM_BOOST_ROUND:
            fail(f"seed={seed} num_trees={n_trees} != 50")
        else:
            pass_(f"seed={seed} num_trees={n_trees}")
        # LightGBM: best_iteration==0 when early stopping unused
        if best not in (0, NUM_BOOST_ROUND):
            fail(f"seed={seed} best_iteration={best} suggests early stopping")
        else:
            pass_(f"seed={seed} best_iteration={best} (no early-stop truncation)")
    mm = pd.read_csv(FIXED_ROOT / "fixed_capacity_model_metrics.csv")
    sub = mm[(mm["model_id"] == "B1") & (mm["num_boost_round"] == 50)]
    if sub.empty:
        fail("fixed_capacity_model_metrics missing B1 round=50")
    else:
        if "early_stopping" in sub.columns and bool(sub["early_stopping"].any()):
            fail("metrics flag early_stopping=True for B1-R50")
        else:
            pass_("fixed_capacity metrics record early_stopping=False for B1-R50")

    # 3. coverage periods
    lines.append("")
    lines.append("## 3. Period coverage")
    ens = load_pred(ens_path)
    assert_no_holdout(ens.index.get_level_values("datetime"), "B1-R50 ensemble")
    dates = ens.index.get_level_values("datetime")
    for period, (s, e) in [("train", PERIODS["train"]), ("valid", PERIODS["valid"]), ("test", PERIODS["test"])]:
        mask = (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
        n = int(mask.sum())
        if n == 0:
            fail(f"{period} has 0 prediction rows")
        else:
            pass_(f"{period}: n_rows={n}, date span within [{s},{e}]")
    pass_(f"overall date range [{dates.min().date()}, {dates.max().date()}]")

    # 4. common sample identity
    lines.append("")
    lines.append("## 4. Common sample identity")
    if len(ens) != COMMON_N:
        fail(f"ensemble rows={len(ens)} != {COMMON_N}")
    else:
        pass_(f"ensemble rows={len(ens)}")
    idx = ens.index.to_frame(index=False)
    if idx.duplicated(["datetime", "instrument"]).any():
        fail("duplicate (datetime, instrument) in ensemble")
    else:
        pass_("no duplicate (datetime, instrument)")
    # coverage vs label common sample
    lab = label_df.set_index(["datetime", "instrument"]).sort_index()
    joined = ens.join(lab[["label"]], how="inner")
    cov = len(joined) / max(len(lab), 1)
    if abs(cov - 1.0) > 1e-12 or len(joined) != COMMON_N:
        fail(f"prediction_coverage={cov}, joined={len(joined)}")
    else:
        pass_(f"prediction_coverage=1.0 on common sample (joined={len(joined)})")

    lines.append("")
    lines.append(f"## Verdict: {'ALL PASSED' if ok else 'FAILED — STOP'}")
    (out_root / "input_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok, "\n".join(lines)


def train_f1c_r50(out_root: Path, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = list(F1C_FEATURES)
    missing = [c for c in features if c not in dataset.columns]
    if missing:
        raise ValueError(f"missing F1C features: {missing}")

    pred_dir = out_root / "f1c_r50_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    eval_raw = dataset[dataset["split"].isin(["train", "valid", "test"])].copy()
    assert_no_holdout(eval_raw["datetime"], "F1C dataset")

    metric_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    seed_preds: list[pd.DataFrame] = []

    for seed in SEEDS:
        log.info("[F1C-R50] train seed=%s", seed)
        model = train_fixed(
            dataset=dataset, features=features, seed=seed, num_boost_round=NUM_BOOST_ROUND
        )
        with open(pred_dir / f"model_seed{seed}.pkl", "wb") as f:
            pickle.dump(model, f)
        pred = predict_fixed(model, eval_raw, features, NUM_BOOST_ROUND)
        pred.to_pickle(pred_dir / f"pred_seed{seed}.pkl")
        seed_preds.append(pred)
        capacity_rows.append(
            {
                "model_id": "F1C_R50",
                "seed": seed,
                "feature_count": len(features),
                "num_boost_round": NUM_BOOST_ROUND,
                "early_stopping": False,
                "best_iteration": int(model.best_iteration) if model.best_iteration else 0,
                "actual_num_trees": int(model.num_trees()),
                "total_split_count": int(np.sum(model.feature_importance("split"))),
                "total_gain": float(np.sum(model.feature_importance("gain"))),
            }
        )
        for period in ("valid", "test"):
            s, e = PERIODS[period]
            met, _ = daily_ic_metrics(pred, eval_raw, period, s, e, model_id="F1C_R50", seed=str(seed))
            met["num_boost_round"] = NUM_BOOST_ROUND
            met["early_stopping"] = False
            met["feature_count"] = len(features)
            metric_rows.append(met)
        del model

    ens = ensemble_predictions(seed_preds)
    ens.to_pickle(pred_dir / "pred_ensemble.pkl")
    for period, (s, e) in PERIODS.items():
        met, _ = daily_ic_metrics(ens, eval_raw, period, s, e, model_id="F1C_R50", seed="ensemble")
        met["num_boost_round"] = NUM_BOOST_ROUND
        met["early_stopping"] = False
        met["feature_count"] = len(features)
        metric_rows.append(met)

    metrics_df = pd.DataFrame(metric_rows)
    capacity_df = pd.DataFrame(capacity_rows)
    metrics_df.to_csv(out_root / "f1c_r50_model_metrics.csv", index=False)
    capacity_df.to_csv(out_root / "f1c_r50_capacity_audit.csv", index=False)
    return ens, metrics_df, capacity_df


def build_ew_composite(dataset: pd.DataFrame) -> pd.DataFrame:
    """Literature-signed equal-weight composite; not fit on data."""
    rows = dataset[["datetime", "instrument"]].copy()
    parts = []
    for c in EW_POS:
        parts.append(dataset[c])
    for c in EW_NEG:
        parts.append(-dataset[c])
    mat = pd.concat(parts, axis=1)
    # mean over available (non-NaN) signed features; all-missing -> 0
    score = mat.mean(axis=1, skipna=True)
    all_miss = mat.isna().all(axis=1)
    score = score.fillna(0.0)
    score = score.where(~all_miss, 0.0)
    out = rows.copy()
    out["score"] = score.to_numpy()
    return out.set_index(["datetime", "instrument"]).sort_index()


def daily_cs_z(score: pd.Series) -> pd.Series:
    """Cross-sectional z-score by date, ddof=0. Constant/empty day -> NaN."""
    s = score.copy()
    df = s.rename("score").reset_index()
    g = df.groupby("datetime", sort=False)["score"]
    mu = g.transform("mean")
    # ddof=0 via population std
    sd = g.transform(lambda x: float(np.nanstd(x.to_numpy(dtype=float), ddof=0)))
    z = (df["score"] - mu) / sd
    z = z.where(sd > 0)
    out = df[["datetime", "instrument"]].copy()
    out["z"] = z.to_numpy()
    return out.set_index(["datetime", "instrument"])["z"].sort_index()


def blend_scores(z_pv: pd.Series, z_fund: pd.Series, w: float) -> pd.Series:
    return w * z_pv + (1.0 - w) * z_fund


def daily_ic_from_scores(
    score: pd.Series,
    label: pd.Series,
    period: str,
) -> pd.DataFrame:
    s, e = PERIODS[period]
    start_ts, end_ts = pd.Timestamp(s), pd.Timestamp(e)
    df = pd.DataFrame({"score": score, "label": label}).dropna()
    df = df.reset_index()
    df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)]
    if df.empty:
        return pd.DataFrame(columns=["datetime", "ic", "rank_ic", "n_obs"])

    def _one(g: pd.DataFrame) -> pd.Series:
        if len(g) < 30 or g["score"].nunique() < 2 or g["label"].nunique() < 2:
            return pd.Series({"ic": np.nan, "rank_ic": np.nan, "n_obs": len(g)})
        return pd.Series(
            {
                "ic": g["score"].corr(g["label"], method="pearson"),
                "rank_ic": g["score"].corr(g["label"], method="spearman"),
                "n_obs": len(g),
            }
        )

    daily = (
        df.groupby("datetime", sort=True)
        .apply(_one, include_groups=False)
        .reset_index()
        .dropna(subset=["ic", "rank_ic"])
    )
    return daily


def summarize_daily(daily: pd.DataFrame, *, pair: str, w: float, period: str) -> dict[str, Any]:
    if daily.empty:
        return {
            "signal_pair": pair,
            "w": w,
            "period": period,
            "ic_mean": np.nan,
            "rank_ic_mean": np.nan,
            "icir": np.nan,
            "rank_icir": np.nan,
            "ic_tstat": np.nan,
            "rank_ic_tstat": np.nan,
            "n_dates": 0,
        }

    def _t(s: pd.Series) -> float:
        if len(s) <= 1:
            return float("nan")
        sd = s.std(ddof=1)
        return float(s.mean() / (sd / np.sqrt(len(s)))) if sd > 0 else float("nan")

    ic, ric = daily["ic"], daily["rank_ic"]
    return {
        "signal_pair": pair,
        "w": w,
        "period": period,
        "ic_mean": float(ic.mean()),
        "rank_ic_mean": float(ric.mean()),
        "icir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else float("nan"),
        "rank_icir": float(ric.mean() / ric.std(ddof=1)) if ric.std(ddof=1) > 0 else float("nan"),
        "ic_tstat": _t(ic),
        "rank_ic_tstat": _t(ric),
        "n_dates": int(len(daily)),
    }


def select_w_star(valid_rank_ic_by_w: dict[float, float]) -> float:
    """Pre-registered: max valid mean Rank IC; tie-break |w-0.5| minimal."""
    items = sorted(valid_rank_ic_by_w.items(), key=lambda kv: kv[0])
    best_ric = max(v for _, v in items)
    candidates = [w for w, v in items if abs(v - best_ric) < TIE_EPS]
    return min(candidates, key=lambda w: abs(w - 0.5))


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    starts = rng.integers(0, n, size=int(np.ceil(n / block)) + 5)
    pieces = []
    for st in starts:
        pieces.append(np.arange(st, min(st + block, n)))
        if sum(len(p) for p in pieces) >= n:
            break
    idx = np.concatenate(pieces)[:n]
    return idx.astype(int)


def run_bootstrap(
    daily_by_w: dict[float, pd.DataFrame],
    *,
    pair: str,
    period: str,
    w_star: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap CI for each w mean RankIC; argmax distribution on valid only."""
    seed_off = {"PV_F1C_R50": 11, "PV_EW_composite": 22}[pair]
    period_off = 0 if period == "valid" else 1
    rng = np.random.default_rng(BOOT_SEED + seed_off + period_off)
    # align dates: use intersection of all w daily dates
    date_sets = [set(df["datetime"]) for df in daily_by_w.values()]
    common_dates = sorted(set.intersection(*date_sets)) if date_sets else []
    common_dates = pd.to_datetime(common_dates)
    n = len(common_dates)
    ric_mat = {}
    for w, df in daily_by_w.items():
        m = df.set_index("datetime")["rank_ic"].reindex(common_dates)
        ric_mat[w] = m.to_numpy(dtype=float)

    ci_rows = []
    argmax_counts = {w: 0 for w in W_GRID}
    boot_means: dict[float, list[float]] = {w: [] for w in W_GRID}

    for _b in range(BOOT_B):
        idx = moving_block_indices(n, BOOT_BLOCK, rng)
        means = {w: float(np.nanmean(ric_mat[w][idx])) for w in W_GRID}
        for w in W_GRID:
            boot_means[w].append(means[w])
        if period == "valid":
            w_b = select_w_star(means)
            argmax_counts[w_b] += 1

    for w in W_GRID:
        arr = np.asarray(boot_means[w], dtype=float)
        ci_rows.append(
            {
                "signal_pair": pair,
                "period": period,
                "w": w,
                "rank_ic_mean_point": float(np.nanmean(ric_mat[w])),
                "boot_mean": float(np.nanmean(arr)),
                "ci95_low": float(np.nanpercentile(arr, 2.5)),
                "ci95_high": float(np.nanpercentile(arr, 97.5)),
                "B": BOOT_B,
                "block_len": BOOT_BLOCK,
                "is_w_star": w == w_star,
            }
        )

    argmax_rows = []
    if period == "valid":
        for w in W_GRID:
            argmax_rows.append(
                {
                    "signal_pair": pair,
                    "w": w,
                    "selected_count": argmax_counts[w],
                    "selected_freq": argmax_counts[w] / BOOT_B,
                    "is_point_w_star": w == w_star,
                    "B": BOOT_B,
                    "block_len": BOOT_BLOCK,
                }
            )
    return pd.DataFrame(ci_rows), pd.DataFrame(argmax_rows)


def signal_corr(z_a: pd.Series, z_b: pd.Series, period: str) -> dict[str, Any]:
    s, e = PERIODS[period]
    df = pd.DataFrame({"a": z_a, "b": z_b}).dropna().reset_index()
    df = df[(df["datetime"] >= pd.Timestamp(s)) & (df["datetime"] <= pd.Timestamp(e))]

    def _one(g: pd.DataFrame) -> pd.Series:
        if len(g) < 30 or g["a"].nunique() < 2 or g["b"].nunique() < 2:
            return pd.Series({"pearson": np.nan, "spearman": np.nan})
        return pd.Series(
            {
                "pearson": g["a"].corr(g["b"], method="pearson"),
                "spearman": g["a"].corr(g["b"], method="spearman"),
            }
        )

    if df.empty:
        return {
            "period": period,
            "mean_daily_pearson": np.nan,
            "mean_daily_spearman": np.nan,
            "n_dates": 0,
        }
    daily = df.groupby("datetime", sort=True).apply(_one, include_groups=False).dropna()
    return {
        "period": period,
        "mean_daily_pearson": float(daily["pearson"].mean()) if len(daily) else np.nan,
        "mean_daily_spearman": float(daily["spearman"].mean()) if len(daily) else np.nan,
        "n_dates": int(len(daily)),
    }


def make_figure(
    out_root: Path,
    grid: pd.DataFrame,
    ci: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    pairs = ["PV_F1C_R50", "PV_EW_composite"]
    colors = {"PV_F1C_R50": "#1f77b4", "PV_EW_composite": "#ff7f0e"}
    for ax, period in zip(axes, ["valid", "test"]):
        for pair in pairs:
            g = grid[(grid["signal_pair"] == pair) & (grid["period"] == period)].sort_values("w")
            c = ci[(ci["signal_pair"] == pair) & (ci["period"] == period)].sort_values("w")
            ax.plot(g["w"], g["rank_ic_mean"], color=colors[pair], label=pair, lw=1.8)
            ax.fill_between(
                c["w"], c["ci95_low"], c["ci95_high"], color=colors[pair], alpha=0.18
            )
            wstar = float(selected.loc[selected["signal_pair"] == pair, "w_star"].iloc[0])
            ystar = float(g.loc[g["w"] == wstar, "rank_ic_mean"].iloc[0])
            ax.axvline(wstar, color=colors[pair], ls="--", lw=1.0, alpha=0.7)
            ax.scatter([wstar], [ystar], color=colors[pair], zorder=5, s=36)
        ax.set_title(period)
        ax.set_xlabel("w (weight on PV / B1-R50)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean Rank IC")
    axes[0].legend(fontsize=8)
    fig.suptitle("Signal blending: Rank IC vs w (95% block-bootstrap CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_root / "fig_ic_vs_w.png", dpi=150)
    plt.close(fig)


def write_readme(
    out_root: Path,
    *,
    stamp: str,
    exp_id: str,
    selected: pd.DataFrame,
    capacity: pd.DataFrame,
    audit_ok: bool,
) -> None:
    lines = [
        f"# {exp_id}",
        "",
        "## Identity",
        f"- Experiment ID: `{exp_id}`",
        f"- Timestamp: {stamp}",
        f"- Git commit: `{git_hash()}`",
        f"- Principal (read-only): `{PRINCIPAL_ID}`",
        f"- Fixed-capacity (read-only): `{FIXED_ID}`",
        "- Holdout 2024–2025: **CLOSED**",
        "- Selection window: valid 2018–2019 only; test reported only",
        f"- Input audit: {'ALL PASSED' if audit_ok else 'FAILED'}",
        "",
        "## Reused functions",
    ]
    for p in REUSED_FUNCTIONS:
        lines.append(f"- `{p}`")
    lines.extend(
        [
            "",
            "## Signals",
            "- PV: B1-R50 ensemble (`Alpha20 + Loop9 RD8`, `num_boost_round=50`, no early stopping)",
            "- Fund main: F1C-R50 ensemble (11 `fund_*_z`)",
            "- Fund control: literature-signed equal-weight composite (not fit on data)",
            "",
            "### EW composite signs (pre-specified)",
            f"- Positive (+1): {', '.join(EW_POS)}",
            f"- Negative (−1): {', '.join(EW_NEG)}",
            f"- Excluded: {', '.join(EW_EXCLUDED)}",
            "- Per-row score = mean of available signed features; all-missing → 0",
            "",
            "## F1C-R50 capacity (facts)",
        ]
    )
    for _, r in capacity.iterrows():
        lines.append(
            f"- seed={int(r['seed'])}: trees={int(r['actual_num_trees'])}, "
            f"total_split={int(r['total_split_count'])}"
        )
    lines.extend(["", "## Selected w* (valid Rank IC rule)", ""])
    for _, r in selected.iterrows():
        lines.append(
            f"- {r['signal_pair']}: w*={r['w_star']}, "
            f"valid RankIC={r['valid_rank_ic_mean']:.6f}, "
            f"test RankIC={r['test_rank_ic_mean']:.6f}"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "- `input_audit.md`",
            "- `f1c_r50_predictions/`, `f1c_r50_model_metrics.csv`, `f1c_r50_capacity_audit.csv`",
            "- `blend_grid_metrics.csv`, `blend_selected_w.csv`",
            "- `blend_bootstrap_ci.csv`, `blend_argmax_distribution.csv`",
            "- `blend_paired_metrics.csv`, `signal_correlation.csv`",
            "- `fig_ic_vs_w.png`",
            "- `decision_log.md`, `run_manifest.json`",
            "",
            "This README states facts and numbers only.",
        ]
    )
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    t0 = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"pv_fund_blend_r50_{stamp}"
    out_root = PROJECT_ROOT / "reports" / exp_id
    if out_root.exists():
        raise FileExistsError(out_root)
    out_root.mkdir(parents=True, exist_ok=False)

    # --- 1. Pre-registration BEFORE any blending ---
    append_preregistration(out_root / "decision_log.md", stamp)

    manifest = {
        "experiment_id": exp_id,
        "timestamp": stamp,
        "git_commit": git_hash(),
        "principal_id": PRINCIPAL_ID,
        "fixed_capacity_id": FIXED_ID,
        "holdout_2024_2025": "closed",
        "num_boost_round": NUM_BOOST_ROUND,
        "seeds": SEEDS,
        "w_grid": W_GRID,
        "bootstrap_B": BOOT_B,
        "bootstrap_block": BOOT_BLOCK,
        "bootstrap_seed": BOOT_SEED,
        "common_sample_n_expected": COMMON_N,
        "reused_functions": REUSED_FUNCTIONS,
        "selection_rule": "max valid mean RankIC; tie |w-0.5|",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"EXPERIMENT_ID: {exp_id}")
    print(f"OUT_ROOT: {out_root}")
    print("Pre-registration written to decision_log.md")
    print("=" * 72)

    # Load common-sample labels/features
    log.info("Loading B3 common-sample dataset (read-only principal)")
    usecols = ["datetime", "instrument", "split", "label"] + F1C_FEATURES
    dataset = pd.read_parquet(B3_DATASET, columns=usecols, engine="pyarrow")
    dataset["datetime"] = pd.to_datetime(dataset["datetime"])
    assert_no_holdout(dataset["datetime"], "B3 dataset")
    if len(dataset) != COMMON_N:
        raise AssertionError(f"common sample size {len(dataset)} != {COMMON_N}")
    if dataset.duplicated(["datetime", "instrument"]).any():
        raise AssertionError("duplicate keys in dataset")
    label_df = dataset[["datetime", "instrument", "label", "split"]].copy()

    # --- 2. Input audit ---
    log.info("Input audit B1-R50")
    audit_ok, _ = audit_b1_r50(out_root, label_df)
    if not audit_ok:
        log.error("Input audit FAILED — stopping before training/blending")
        manifest["status"] = "STOPPED_INPUT_AUDIT_FAILED"
        manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return 2

    # --- 3. Train F1C-R50 ---
    log.info("Training F1C-R50")
    f1c_ens, f1c_metrics, f1c_cap = train_f1c_r50(out_root, dataset)

    # --- 4. EW composite ---
    log.info("Building EW composite")
    ew = build_ew_composite(dataset)
    ew.to_pickle(out_root / "ew_composite_score.pkl")

    # PV and B3 baselines
    pv = load_pred(B1_R50_DIR / "pred_ensemble.pkl")
    b3 = load_pred(B3_R50_DIR / "pred_ensemble.pkl")
    assert_no_holdout(pv.index.get_level_values("datetime"), "PV")
    assert_no_holdout(f1c_ens.index.get_level_values("datetime"), "F1C")
    assert_no_holdout(ew.index.get_level_values("datetime"), "EW")

    # Align to common index
    common_idx = pv.index.intersection(f1c_ens.index).intersection(ew.index).intersection(b3.index)
    if len(common_idx) != COMMON_N:
        raise AssertionError(f"aligned common idx {len(common_idx)} != {COMMON_N}")
    pv = pv.reindex(common_idx)
    f1c_ens = f1c_ens.reindex(common_idx)
    ew = ew.reindex(common_idx)
    b3 = b3.reindex(common_idx)
    label_s = label_df.set_index(["datetime", "instrument"])["label"].reindex(common_idx)

    pairs = {
        "PV_F1C_R50": (pv["score"], f1c_ens["score"]),
        "PV_EW_composite": (pv["score"], ew["score"]),
    }

    grid_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    ci_parts: list[pd.DataFrame] = []
    argmax_parts: list[pd.DataFrame] = []
    paired_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []

    report_periods = ["valid", "test", "covid", "post_covid"]

    for pair_name, (s_pv, s_fund) in pairs.items():
        log.info("Blending pair=%s", pair_name)
        z_pv = daily_cs_z(s_pv)
        z_fund = daily_cs_z(s_fund)

        for period in ("valid", "test"):
            c = signal_corr(z_pv, z_fund, period)
            c["signal_pair"] = pair_name
            corr_rows.append(c)

        daily_by_w: dict[float, dict[str, pd.DataFrame]] = {w: {} for w in W_GRID}
        valid_ric: dict[float, float] = {}

        for w in W_GRID:
            blended = blend_scores(z_pv, z_fund, w)
            for period in report_periods:
                daily = daily_ic_from_scores(blended, label_s, period)
                daily_by_w[w][period] = daily
                grid_rows.append(summarize_daily(daily, pair=pair_name, w=w, period=period))
            valid_ric[w] = float(
                summarize_daily(daily_by_w[w]["valid"], pair=pair_name, w=w, period="valid")[
                    "rank_ic_mean"
                ]
            )

        # Selection ONLY on valid
        w_star = select_w_star(valid_ric)
        log.info("[%s] w*=%s (valid RankIC=%.6f)", pair_name, w_star, valid_ric[w_star])

        sel = {
            "signal_pair": pair_name,
            "w_star": w_star,
            "valid_rank_ic_mean": valid_ric[w_star],
            "valid_ic_mean": summarize_daily(
                daily_by_w[w_star]["valid"], pair=pair_name, w=w_star, period="valid"
            )["ic_mean"],
            "test_rank_ic_mean": summarize_daily(
                daily_by_w[w_star]["test"], pair=pair_name, w=w_star, period="test"
            )["rank_ic_mean"],
            "test_ic_mean": summarize_daily(
                daily_by_w[w_star]["test"], pair=pair_name, w=w_star, period="test"
            )["ic_mean"],
            "selection_rule": "max_valid_mean_rank_ic; tie_break_closest_to_0.5",
        }
        selected_rows.append(sel)

        # Bootstrap CI for valid & test; argmax on valid
        for period in ("valid", "test"):
            daily_map = {w: daily_by_w[w][period] for w in W_GRID}
            ci_df, arg_df = run_bootstrap(daily_map, pair=pair_name, period=period, w_star=w_star)
            ci_parts.append(ci_df)
            if not arg_df.empty:
                argmax_parts.append(arg_df)

        # Paired: blend(w*) vs B1 (w=1), vs F1C/EW (w=0), vs B3-R50
        # Build daily IC for baselines using same z-score blend endpoints and B3 raw scores
        # For w=1 and w=0 use blend daily already computed
        # For B3: daily IC of B3 score vs label
        b3_daily = {
            period: daily_ic_from_scores(b3["score"], label_s, period)
            for period in ("valid", "test")
        }
        blend_star = {
            period: daily_by_w[w_star][period] for period in ("valid", "test")
        }
        # B1 endpoint = w=1 blend (z_pv only after CS z — equivalent path)
        b1_daily = {period: daily_by_w[1.0][period] for period in ("valid", "test")}
        fund0_daily = {period: daily_by_w[0.0][period] for period in ("valid", "test")}

        for period in ("valid", "test"):
            for name, base_daily in [
                ("blend_wstar_minus_B1_R50", b1_daily[period]),
                (
                    "blend_wstar_minus_F1C_R50"
                    if pair_name == "PV_F1C_R50"
                    else "blend_wstar_minus_EW_composite",
                    fund0_daily[period],
                ),
                ("blend_wstar_minus_B3_R50", b3_daily[period]),
            ]:
                row = paired_delta(
                    blend_star[period],
                    base_daily,
                    comparison=name,
                    period=period,
                )
                row["signal_pair"] = pair_name
                row["w_star"] = w_star
                paired_rows.append(enrich_paired(row))

    grid_df = pd.DataFrame(grid_rows)
    selected_df = pd.DataFrame(selected_rows)
    ci_df = pd.concat(ci_parts, ignore_index=True)
    argmax_df = pd.concat(argmax_parts, ignore_index=True) if argmax_parts else pd.DataFrame()
    paired_df = pd.DataFrame(paired_rows)
    corr_df = pd.DataFrame(corr_rows)

    grid_df.to_csv(out_root / "blend_grid_metrics.csv", index=False)
    selected_df.to_csv(out_root / "blend_selected_w.csv", index=False)
    ci_df.to_csv(out_root / "blend_bootstrap_ci.csv", index=False)
    argmax_df.to_csv(out_root / "blend_argmax_distribution.csv", index=False)
    paired_df.to_csv(out_root / "blend_paired_metrics.csv", index=False)
    corr_df.to_csv(out_root / "signal_correlation.csv", index=False)

    make_figure(out_root, grid_df, ci_df, selected_df)
    write_readme(
        out_root,
        stamp=stamp,
        exp_id=exp_id,
        selected=selected_df,
        capacity=f1c_cap,
        audit_ok=audit_ok,
    )

    elapsed = time.time() - t0
    manifest["status"] = "COMPLETED"
    manifest["elapsed_sec"] = elapsed
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["w_star"] = selected_df.set_index("signal_pair")["w_star"].to_dict()
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("DONE %s elapsed=%.1fs", out_root, elapsed)
    print(f"DONE: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
