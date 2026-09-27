#!/usr/bin/env python3
"""2026 APPROXIMATE_CURRENT_DATA: W1a (Amihud = |r|/(P·V)) Top5/Top20 main window
versus the same placebo as W1 (b).

Reads W1 (b) artefacts from reports/matched_2x2_2026_20260923_173800/ and D2a
models from the RD13_v2a ablation. Placebo CSVs / seeds match (b)."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "portfolio_experiments" / "market_risk"))
sys.path.insert(0, str(PROJECT_ROOT / "portfolio_experiments" / "matched_2x2_2026"))

import approx_2026_helpers as v  # noqa: E402
from run_approximate_b_only import (  # noqa: E402
    ANN_FACTOR,
    COST_BP,
    EXT_START,
    HOLD_THRESH,
    LABEL,
    MAIN_END,
    MAIN_START,
    PLACEBO_N,
    PLACEBO_SEED,
    RISK_DEGREE,
    attach_spy,
    load_membership_carry,
    percentile_of,
    portfolio_from_weights,
    sha256_file,
    summarize_daily,
    topk_dropout_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("matched_2x2_2026_w1a")

BASELINE_B = PROJECT_ROOT / "reports/matched_2x2_2026_20260923_173800"
D2A_MODELS = PROJECT_ROOT / "reports/rd13_v2a_amihud_ablation_20260923_221045/models"
AMIHUD_COL = "rd13_daily_amihud_illiquidity"
SEEDS = [42, 2026, 3407]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def recompute_amihud_pv(px: pd.DataFrame) -> pd.DataFrame:
    """Return date/ticker/amihud_v2a = |r| / (close × volume)."""
    df = px.sort_values(["ticker", "date"]).copy()
    parts = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date")
        close = g["close"].astype(float)
        vol = g["volume"].astype(float).replace(0, np.nan)
        r = close.pct_change().abs()
        amihud = r / (close * vol)
        parts.append(pd.DataFrame({"date": g["date"].to_numpy(), "ticker": ticker, "amihud_v2a": amihud.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def load_d2a_models() -> dict[int, Any]:
    models = {}
    for seed in SEEDS:
        path = D2A_MODELS / f"model_D2a_seed{seed}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as f:
            models[seed] = pickle.load(f)
    return models


def build_w1a_scores(
    px: pd.DataFrame,
    dates: list[pd.Timestamp],
    *,
    feat_base: pd.DataFrame,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feat_lists = json.loads(v.FEATURE_JSON.read_text())
    alpha20_cols = feat_lists["alpha20"]

    amihud = recompute_amihud_pv(px)
    feat = feat_base.copy()
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
    old_amihud = feat[["date", "ticker", AMIHUD_COL]].rename(columns={AMIHUD_COL: "old"})
    feat = feat.drop(columns=[AMIHUD_COL]).merge(amihud, on=["date", "ticker"], how="left")
    feat = feat.rename(columns={"amihud_v2a": AMIHUD_COL})
    both = old_amihud.merge(feat[["date", "ticker", AMIHUD_COL]], on=["date", "ticker"], how="inner").dropna()
    both = both.rename(columns={AMIHUD_COL: "new"})
    spearman = float(both["old"].corr(both["new"], method="spearman")) if len(both) > 100 else np.nan
    ratio = (both["new"] / both["old"]).replace([np.inf, -np.inf], np.nan)
    ratio_cv = float(ratio.std() / ratio.mean()) if ratio.notna().any() and float(ratio.mean()) != 0 else np.nan

    feat_path = cache_dir / "d2a_features_amihud_pv.parquet"
    feat.to_parquet(feat_path, index=False)

    z_path = BASELINE_B / "cache/alpha20_robust_z_2008_2017.json"
    mean, std = v.fit_robust_z_qlib(alpha20_cols, z_path)
    feat_z = v.apply_robust_z(feat, alpha20_cols, mean, std)

    models = load_d2a_models()
    d2a = v.predict_d2(feat_z, models)
    d2a = d2a.rename(columns={"d2": "d2a"})

    membership = load_membership_carry(dates)
    stmt = v.build_f1c_statement_panel()
    f1c_raw = v.f1c_raw_from_stmt(stmt, px, dates)
    f1c_models = v.train_f1c_seeds()
    f1c = v.predict_f1c(f1c_raw, f1c_models)

    # build_w1 expects column "d2"
    d2_for_w1 = d2a.rename(columns={"d2a": "d2"})
    w1a = v.build_w1(d2_for_w1, f1c, membership, px)
    meta = {
        "amihud_formula": "|r|/(close*volume)",
        "amihud_vs_old_spearman": spearman,
        "amihud_ratio_cv": ratio_cv,
        "d2a_models": [str(D2A_MODELS / f"model_D2a_seed{s}.pkl") for s in SEEDS],
        "d2a_models_sha256": {str(s): sha256_file(D2A_MODELS / f"model_D2a_seed{s}.pkl") for s in SEEDS},
        "feature_sha256": sha256_file(feat_path),
        "w1a_rows": int(len(w1a)),
        "f1c_rows": int(len(f1c)),
    }
    return w1a, d2a, meta


def main() -> None:
    if not BASELINE_B.exists():
        raise FileNotFoundError(BASELINE_B)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / f"reports/matched_2x2_2026_w1a_amihud_{ts}"
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    log.info("output %s", out)

    # copy placebo + W1(b) summary for side-by-side (read-only provenance)
    for name in ("placebo_main_Top5.csv", "placebo_main_Top20.csv", "placebo_summary.csv", "main_window_results.csv"):
        shutil.copy2(BASELINE_B / name, out / f"baseline_b_{name}")

    px = v.load_prices()
    spy_px = pd.read_parquet(v.MASSIVE_SPY)
    spy_px["date"] = pd.to_datetime(spy_px["date"]).dt.normalize()

    px_max = px["date"].max()
    all_decision_dates = sorted(d for d in px["date"].drop_duplicates() if MAIN_START <= d <= px_max)
    cal = sorted(px["date"].drop_duplicates())
    if all_decision_dates and all_decision_dates[-1] == cal[-1]:
        all_decision_dates = all_decision_dates[:-1]
    main_decision = [d for d in all_decision_dates if d <= MAIN_END]
    main_trade_end = MAIN_END

    feat_base = pd.read_parquet(BASELINE_B / "cache/d2_features_current_ohlcv.parquet")
    log.info("building W1a on APPROXIMATE_CURRENT_DATA through %s", px_max.date())
    w1a, d2a, meta = build_w1a_scores(px, all_decision_dates, feat_base=feat_base, cache_dir=cache)
    w1a.to_parquet(cache / "w1a_scores.parquet", index=False)
    d2a.to_parquet(cache / "d2a_scores.parquet", index=False)

    w1a_main = w1a[w1a["date"].isin(main_decision)].copy()
    w_top5 = topk_dropout_weights(w1a_main[["date", "ticker", "score"]], main_decision, topk=5, n_drop=1)
    w_top20 = topk_dropout_weights(w1a_main[["date", "ticker", "score"]], main_decision, topk=20, n_drop=2)
    daily_top5 = attach_spy(
        portfolio_from_weights(w_top5, px, eval_start=MAIN_START, eval_end=main_trade_end), spy_px
    )
    daily_top20 = attach_spy(
        portfolio_from_weights(w_top20, px, eval_start=MAIN_START, eval_end=main_trade_end), spy_px
    )
    daily_top5.to_csv(out / "daily_W1a_Top5.csv", index=False)
    daily_top20.to_csv(out / "daily_W1a_Top20.csv", index=False)
    w_top5.to_csv(out / "weights_W1a_Top5.csv", index=False)
    w_top20.to_csv(out / "weights_W1a_Top20.csv", index=False)

    rows = []
    for name, daily, weights in [
        ("W1a_Top5_drop1", daily_top5, w_top5),
        ("W1a_Top20_drop2", daily_top20, w_top20),
    ]:
        m = summarize_daily(daily, daily.set_index("trade_date")["r_spy"], weights)
        m["strategy"] = name
        m["signal"] = "W1a"
        m["data_label"] = LABEL
        m["window"] = f"main_{m['first_trade']}_to_{m['last_trade']}"
        m["amihud"] = "|r|/(close*volume)"
        rows.append(m)
    w1a_main_df = pd.DataFrame(rows)

    # W1 (b) rows from baseline
    b_main = pd.read_csv(BASELINE_B / "main_window_results.csv")
    b_keep = b_main[b_main["strategy"].isin(["B_Top5_regen_current", "B_Top20_regen_current"])].copy()
    b_keep["signal"] = "W1"
    b_keep["amihud"] = "|r|/volume"
    b_keep.loc[b_keep["strategy"] == "B_Top5_regen_current", "strategy"] = "W1_Top5_drop1_(b)"
    b_keep.loc[b_keep["strategy"] == "B_Top20_regen_current", "strategy"] = "W1_Top20_drop2_(b)"

    compare = pd.concat([b_keep, w1a_main_df], ignore_index=True, sort=False)
    compare.to_csv(out / "main_window_W1_vs_W1a.csv", index=False)

    # Same placebo distribution as (b)
    plac5 = pd.read_csv(BASELINE_B / "placebo_main_Top5.csv")
    plac20 = pd.read_csv(BASELINE_B / "placebo_main_Top20.csv")
    assert len(plac5) == PLACEBO_N and len(plac20) == PLACEBO_N

    w1_5 = float(b_keep.loc[b_keep["strategy"] == "W1_Top5_drop1_(b)", "cum_net"].iloc[0])
    w1_20 = float(b_keep.loc[b_keep["strategy"] == "W1_Top20_drop2_(b)", "cum_net"].iloc[0])
    w1a_5 = float(w1a_main_df.loc[w1a_main_df["strategy"] == "W1a_Top5_drop1", "cum_net"].iloc[0])
    w1a_20 = float(w1a_main_df.loc[w1a_main_df["strategy"] == "W1a_Top20_drop2", "cum_net"].iloc[0])

    def plac_row(dist: pd.DataFrame, rule: str, refs: dict[str, float]) -> dict:
        cn = dist["cum_net"].values.astype(float)
        row = {
            "window": "main",
            "rule": rule,
            "n": int(len(dist)),
            "cum_net_mean": float(np.mean(cn)),
            "cum_net_median": float(np.median(cn)),
            "cum_net_p05": float(np.percentile(cn, 5)),
            "cum_net_p95": float(np.percentile(cn, 95)),
            "placebo_source": str(BASELINE_B / ("placebo_main_Top5.csv" if "Top5" in rule else "placebo_main_Top20.csv")),
            "placebo_seed": PLACEBO_SEED if "Top5" in rule else PLACEBO_SEED + 1,
            "data_label": LABEL,
        }
        for k, val in refs.items():
            row[f"pctile_cum_net_{k}"] = percentile_of(val, cn)
            row[f"cum_net_{k}"] = val
        return row

    plac_sum = pd.DataFrame(
        [
            plac_row(
                plac5,
                "Top5/drop1",
                {"W1_b": w1_5, "W1a": w1a_5},
            ),
            plac_row(
                plac20,
                "Top20/drop2",
                {"W1_b": w1_20, "W1a": w1a_20},
            ),
        ]
    )
    plac_sum.to_csv(out / "placebo_percentile_W1_vs_W1a.csv", index=False)

    manifest = {
        "status": "COMPLETED_APPROXIMATE_CURRENT_DATA_W1a",
        "data_label": LABEL,
        "generated_utc": utc_now(),
        "git_commit": git_commit(),
        "baseline_b": str(BASELINE_B),
        "amihud_w1": "|r|/volume",
        "amihud_w1a": "|r|/(close*volume)",
        "engines": ["Top5/drop1", "Top20/drop2"],
        "main_window": {"start": str(MAIN_START.date()), "end": str(MAIN_END.date())},
        "placebo": {
            "n": PLACEBO_N,
            "reused_from_baseline_b": True,
            "seed_top5": PLACEBO_SEED,
            "seed_top20": PLACEBO_SEED + 1,
            "note": "Same random-signal placebo draws as W1 (b); W1a scored into that distribution.",
        },
        "cost_bp": COST_BP,
        "risk_degree": RISK_DEGREE,
        "hold_thresh": HOLD_THRESH,
        "ann_factor": ANN_FACTOR,
        "meta_scores": meta,
        "w1a_scores_sha256": sha256_file(cache / "w1a_scores.parquet"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # SUMMARY
    def fmt_pct(x: float) -> str:
        return f"{100 * x:.2f}%"

    s5 = plac_sum[plac_sum.rule == "Top5/drop1"].iloc[0]
    s20 = plac_sum[plac_sum.rule == "Top20/drop2"].iloc[0]
    m5w = w1a_main_df[w1a_main_df.strategy == "W1a_Top5_drop1"].iloc[0]
    m20w = w1a_main_df[w1a_main_df.strategy == "W1a_Top20_drop2"].iloc[0]
    b5 = b_keep[b_keep.strategy == "W1_Top5_drop1_(b)"].iloc[0]
    b20 = b_keep[b_keep.strategy == "W1_Top20_drop2_(b)"].iloc[0]

    md = f"""# W1a Amihud ablation on APPROXIMATE_CURRENT_DATA (main window)

**Directory:** `{out.name}`  
**Generated:** `{manifest['generated_utc']}`  
**Label:** `{LABEL}`  
**Baseline (b):** `{BASELINE_B.name}`  
**Amihud:** W1 = `|r|/volume`；W1a = `|r|/(close×volume)`（D2a models from `rd13_v2a_amihud_ablation_20260923_221045`）  
**Placebo:** **reused** from (b) — N={PLACEBO_N}, seeds {PLACEBO_SEED}/{PLACEBO_SEED + 1}（未重抽）

## Main window 2026-01-05 → 2026-08-24

| signal | engine | cum_net | spy_cum | cum_excess_geom | qlib_excess_ARR | Sharpe | turnover | placebo 百分位 (cum_net) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W1 (b) | Top5/drop1 | {fmt_pct(b5.cum_net)} | {fmt_pct(b5.spy_cum)} | {fmt_pct(b5.cum_excess_geom)} | {fmt_pct(b5.qlib_excess_arr)} | {b5.sharpe_rf0:.3f} | {b5.mean_daily_turnover:.4f} | **{s5.pctile_cum_net_W1_b:.1f}** |
| W1a | Top5/drop1 | {fmt_pct(m5w.cum_net)} | {fmt_pct(m5w.spy_cum)} | {fmt_pct(m5w.cum_excess_geom)} | {fmt_pct(m5w.qlib_excess_arr)} | {m5w.sharpe_rf0:.3f} | {m5w.mean_daily_turnover:.4f} | **{s5.pctile_cum_net_W1a:.1f}** |
| W1 (b) | Top20/drop2 | {fmt_pct(b20.cum_net)} | {fmt_pct(b20.spy_cum)} | {fmt_pct(b20.cum_excess_geom)} | {fmt_pct(b20.qlib_excess_arr)} | {b20.sharpe_rf0:.3f} | {b20.mean_daily_turnover:.4f} | **{s20.pctile_cum_net_W1_b:.1f}** |
| W1a | Top20/drop2 | {fmt_pct(m20w.cum_net)} | {fmt_pct(m20w.spy_cum)} | {fmt_pct(m20w.cum_excess_geom)} | {fmt_pct(m20w.qlib_excess_arr)} | {m20w.sharpe_rf0:.3f} | {m20w.mean_daily_turnover:.4f} | **{s20.pctile_cum_net_W1a:.1f}** |

### Placebo 分布（与 (b) 相同）

| rule | mean / median / p05–p95 cum_net |
|---|---|
| Top5/drop1 | {fmt_pct(s5.cum_net_mean)} / {fmt_pct(s5.cum_net_median)} / {fmt_pct(s5.cum_net_p05)}–{fmt_pct(s5.cum_net_p95)} |
| Top20/drop2 | {fmt_pct(s20.cum_net_mean)} / {fmt_pct(s20.cum_net_median)} / {fmt_pct(s20.cum_net_p05)}–{fmt_pct(s20.cum_net_p95)} |

### Formula audit（Massive feature book）

- Spearman(old Amihud, v2a) = {meta['amihud_vs_old_spearman']:.4f}
- ratio CV (v2a/old) = {meta['amihud_ratio_cv']:.4f}（非常数 → 非无害重标度）

## Files

- `main_window_W1_vs_W1a.csv`, `placebo_percentile_W1_vs_W1a.csv`
- `daily_W1a_Top5.csv`, `daily_W1a_Top20.csv`
- `baseline_b_*` copies of (b) placebo/main results
- `run_manifest.json`
"""
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    print(md)
    print("OUT", out)


if __name__ == "__main__":
    main()
