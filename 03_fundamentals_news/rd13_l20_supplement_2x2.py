#!/usr/bin/env python3
"""Supplement: $factor audit notes + IC CSV export + RD8 vs RD13_v2 TopK 2×2 (2020–2023)."""
from __future__ import annotations
import os

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_US
from qlib.contrib.evaluate import risk_analysis
from qlib.data import D

PROJECT = Path(os.environ["PROJECT_ROOT"]) if os.environ.get("PROJECT_ROOT") else Path(__file__).resolve().parents[1]
CMP = PROJECT / "reports/rd13_l20_downstream_cmp_20260923_190325"
QLIB_DATA = PROJECT / "staging/qlib_data"
ANN = 252
COST_BP = 1e-4  # 1bp per unit turnover (sum |Δw|)
RISK_DEGREE = 0.95
HOLD_THRESH = 1
BOOT_BLOCK = 21
BOOT_N = 2000
BOOT_SEED = 20260923
EVAL_START = pd.Timestamp("2020-01-01")
EVAL_END = pd.Timestamp("2023-12-31")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rd13_supplement")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def block_bootstrap_mean(x: np.ndarray, seed: int) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < BOOT_BLOCK:
        return {"mean": float(np.mean(x)) if n else np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "n": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BOOT_BLOCK))
    boots = []
    for _ in range(BOOT_N):
        starts = rng.integers(0, n - BOOT_BLOCK + 1, size=n_blocks)
        sample = np.concatenate([x[s : s + BOOT_BLOCK] for s in starts])[:n]
        boots.append(sample.mean())
    boots = np.asarray(boots)
    mean = float(np.mean(x))
    p = float((boots <= 0).mean()) if mean >= 0 else float((boots >= 0).mean())
    p = min(2 * p, 1.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": mean, "ci_lo": float(lo), "ci_hi": float(hi), "p": p, "n": n}


def topk_dropout_weights(scores: pd.DataFrame, dates: list, topk: int, n_drop: int) -> pd.DataFrame:
    by_date = {d: g.set_index("instrument")["score"] for d, g in scores.groupby("datetime")}
    held: list[str] = []
    age: dict[str, int] = {}
    rows = []
    for dt in dates:
        pred = by_date.get(dt)
        if pred is None or pred.dropna().empty:
            for tkr in held:
                rows.append({"datetime": dt, "instrument": tkr, "weight": RISK_DEGREE / max(len(held), 1)})
            continue
        pred = pred.dropna().sort_values(ascending=False)
        last = [t for t in pred.reindex(held).sort_values(ascending=False).index.tolist() if t in pred.index]
        n_buy_cap = n_drop + topk - len(last)
        today = [t for t in pred.index if t not in last][: max(n_buy_cap, 0)]
        comb = pred.reindex(list(dict.fromkeys(last + today))).dropna().sort_values(ascending=False)
        bottom = set(comb.index[-n_drop:]) if len(comb) else set()
        sell = [t for t in last if t in bottom and age.get(t, 0) >= HOLD_THRESH]
        buy = today[: len(sell) + topk - len(last)]
        new_held = [t for t in last if t not in sell] + list(buy)
        new_held = new_held[:topk]
        age = {t: (age.get(t, 0) + 1 if t in last and t not in sell else 1) for t in new_held}
        held = new_held
        w = RISK_DEGREE / max(len(held), 1)
        for tkr in held:
            rows.append({"datetime": dt, "instrument": tkr, "weight": w})
    return pd.DataFrame(rows)


def portfolio_daily(weights: pd.DataFrame, ret: pd.DataFrame, cal: list) -> pd.DataFrame:
    loc = {d: i for i, d in enumerate(cal)}
    w = weights.rename(columns={"datetime": "decision_date"})
    rows = []
    prev_w: dict[str, float] = {}
    for dec in sorted(w["decision_date"].unique()):
        i = loc.get(pd.Timestamp(dec))
        if i is None or i + 1 >= len(cal):
            continue
        ret_date = cal[i + 1]
        if ret_date < EVAL_START or ret_date > EVAL_END:
            continue
        wd = w[w["decision_date"] == dec]
        cur = dict(zip(wd["instrument"], wd["weight"]))
        day_ret = ret[ret["datetime"] == ret_date].set_index("instrument")["ret"]
        gross = 0.0
        for tkr, wt in cur.items():
            r = day_ret.get(tkr, np.nan)
            if np.isfinite(r):
                gross += wt * float(r)
        names = set(cur) | set(prev_w)
        turnover = float(sum(abs(cur.get(t, 0.0) - prev_w.get(t, 0.0)) for t in names))
        cost = COST_BP * turnover
        rows.append(
            {
                "trade_date": ret_date,
                "decision_date": dec,
                "r_gross": gross,
                "r_net": gross - cost,
                "turnover": turnover,
                "cost": cost,
                "n_hold": len(cur),
            }
        )
        prev_w = cur
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame, spy: pd.Series) -> dict:
    d = daily.sort_values("trade_date").drop_duplicates("trade_date")
    r = d.set_index("trade_date")["r_net"].astype(float)
    s = spy.reindex(r.index).astype(float)
    ex = r - s
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    # qlib-style excess ARR
    ra = risk_analysis(ex, mode="sum")
    # risk_analysis returns DataFrame
    arr = float(ra.loc["annualized_return", "risk"]) if "annualized_return" in ra.index else float(ex.mean() * ANN)
    mdd = float(ra.loc["max_drawdown", "risk"]) if "max_drawdown" in ra.index else float(dd.min())
    cagr = float(wealth.iloc[-1] ** (ANN / len(r)) - 1) if len(r) else np.nan
    return {
        "n_days": int(len(r)),
        "cum_net": float(wealth.iloc[-1] - 1),
        "cagr": cagr,
        "excess_ARR": arr,
        "excess_MDD": mdd,
        "wealth_MDD": float(dd.min()),
        "mean_turnover": float(d["turnover"].mean()),
        "sharpe_rf0": float(r.mean() / r.std(ddof=1) * np.sqrt(ANN)) if r.std(ddof=1) > 0 else np.nan,
        "cum_excess_geom": float((1 + r).prod() / (1 + s.fillna(0)).prod() - 1),
    }


def write_factor_audit(out: Path) -> None:
    md = f"""# RD13 Loop12–13 `$factor` 来源核查

**Generated:** {utc_now()}

## 结论

`$factor`（`daily_pv.h5` / Qlib）是本仓库 WRDS→Qlib 链路写入的**累计价格调整因子**，来自 CRSP **`DlyCumFacPr`**：

```text
factor = 1.0 / DlyCumFacPr
adj_OHLC = raw_OHLC / DlyCumFacPr
```

代码：`convert_wrds_stage1.py` `clean_and_adjust()`（约 L161–170）；Stage2 再按**该标的样本首日 close** 把 OHLC 与 `factor` 同除（`convert_wrds_stage2.normalize_instrument`，约 L90–98），使首日 close=1。

## Loop 12–13 如何使用

均直接读 `df['$factor']`（非自算）：

| Loop | 因子 | 用法 |
|---|---|---|
| 12 | `factor_raw_zscore` | 当日 `$factor` 横截面 z |
| 12 | `factor_ma5_zscore` | `$factor` 5 日均后再横截面 z |
| 12 | `factor_mom5_zscore` | `$factor_t/$factor_{{t-5}}-1` 再横截面 z |
| 13 | `factor_neutralized_*` | 对 `$factor` 做 size/vol 中性化后再变换 |

实现原文在 `docs/experiment_log/0710_log/phase2_experiment_data/rdagent_40loop_factor_detail.csv`（loop∈{{12,13}} 的 `code` 列）；workspace 例：Loop12 `73ed1fd75eba47909c6af609a4772f5f`。

## 以哪一日为基准？

1. **CRSP `DlyCumFacPr`**：把价格调到与**数据库末端股本口径**可比的累计因子（拆合股进入累计乘积）。
2. **本仓库 Stage2**：再除以**该 instrument CSV 第一根有效 close**，故 Qlib 里 `$factor` 的绝对水平还依赖样本起始日归一。

二者都不是“以交易日 t 为 PIT 基准、仅用 t 及以前公司行动”的实时因子。

## 是否含 t 日之后的公司行动信息？

**含（水平口径）。** 在一次完整 dump（本项目日历至约 2025-12）中，日期 t 上的 `$factor` 水平已嵌入 **t 之后直至 dump 末端** 的拆合股对累计因子的影响（CRSP 累计调整惯例）。Stage2 的全局缩放不能消除跨股票相对水平中的“未来拆股次数”差异。

- **拆股发生日**：`$factor` 序列会出现跳跃（PIT 上该日事件可知）。
- **拆股前若干日的 `$factor` 绝对水平**：在含未来拆股的 dump 中，已与“当时实时可见的累计因子”不同。

因此 Loop12–13 把 `$factor` 当原始信号（尤其 raw z / 水平相关变换）存在**调整因子水平前视**风险；`$close` 等已复权价量本身用于收益类因子通常更安全。

## 拆股样例：AAPL（PERMNO → `P14593`）

Qlib `staging/qlib_data`，4:1 拆股 **2020-08-31**：

| 日期 | `$factor` | 备注 |
|---|---:|---|
| 2020-08-28 | 0.035927 | 拆股前平台 |
| 2020-08-31 | 0.143708 | ≈ 0.035927 × **4** |

另有 2014-06-09 附近 `$factor` 约 ×7 跳跃（7:1 拆股）。验证命令：`D.features(['P14593'], ['$close','$factor'], ...)`。
"""
    (out / "FACTOR_DOLLAR_FACTOR_AUDIT.md").write_text(md)


def export_ic_csvs(out: Path) -> None:
    summary = pd.read_csv(CMP / "ic_summary_by_spec_rounds.csv")
    pairs = pd.read_csv(CMP / "paired_delta_rankic_bootstrap.csv")
    # tidy wide tables
    for period in ("test_2020_2023", "holdout_2024_2025"):
        sub = summary[summary.period == period].copy()
        sub.to_csv(out / f"ic_rankic_exact_{period}.csv", index=False)
    pairs.to_csv(out / "paired_delta_rankic_all_rounds_periods.csv", index=False)
    # convenience: only holdout + only R10/R20
    pairs.to_csv(CMP / "paired_delta_rankic_bootstrap.csv")  # already there
    summary.to_csv(out / "ic_summary_by_spec_rounds.csv", index=False)


def write_a20_rd6_note(out: Path) -> None:
    md = """# Phase 2 A20_RD6：Loop2 ATR×成交量交互项身份

## 结论：**修正前（absolute ATR × volume ratio）**

证据：

1. `reports/corrected_downstream_loop2_20260917/phase_a/rd_factor_sha256.csv`  
   `atr_20d_x_volume_ratio_20d` → workspace `3f9b4b2805d7420a80cebd090d6a1afa`。

2. 该 workspace `factor.py` 使用**绝对** True Range 的 20 日均值 × `volume/rolling_mean(volume,20)`（**无** `/prev_close`）：

```24:30:RDAGENT_ROOT/git_ignore_folder/RD-Agent_workspace/3f9b4b2805d7420a80cebd090d6a1afa/factor.py
    # Compute 20-day ATR (absolute price units)
    atr_20d = true_range.rolling(window=20, min_periods=20).mean()
    ...
    factor = atr_20d * vol_ratio_20d
```

3. **修正后**实现在 `reports/rdagent_corrected_semantic_rerun_20260918/factors/atr_20d_x_volume_ratio_20d_corrected_factor.py`（`rel_tr = true_range/prev_close`），用于 corrected semantic 20-loop，**未**写入 Phase2 `A20_RD6` 面板。

故 A20_RD6 的交互项 = matched 20-loop **原始** Loop2 实现，不是 corrected semantic 的相对 ATR 版。
"""
    (out / "A20_RD6_INTERACTION_PRE_VS_POST.md").write_text(md)


def run_2x2(out: Path) -> None:
    qlib.init(provider_uri=str(QLIB_DATA), region=REG_US, kernels=1)
    pred_rd8 = pd.read_parquet(CMP / "pred_Alpha20_RD8_R50.parquet")
    pred_rd13 = pd.read_parquet(CMP / "pred_Alpha20_RD13_v2_R50.parquet")
    for p in (pred_rd8, pred_rd13):
        p["datetime"] = pd.to_datetime(p["datetime"]).dt.normalize()
        p["instrument"] = p["instrument"].astype(str)

    # prices / returns for instruments in union
    insts = sorted(set(pred_rd8["instrument"]) | set(pred_rd13["instrument"]))
    log.info("loading closes for %d instruments", len(insts))
    # chunk to avoid huge requests
    chunks = []
    for i in range(0, len(insts), 80):
        part = insts[i : i + 80]
        feat = D.features(part, ["$close"], start_time="2019-12-01", end_time="2024-01-05")
        feat = feat.reset_index() if isinstance(feat.index, pd.MultiIndex) else feat
        feat["datetime"] = pd.to_datetime(feat["datetime"]).dt.normalize()
        chunks.append(feat)
    px = pd.concat(chunks, ignore_index=True)
    px = px.rename(columns={"$close": "close"})
    px["ret"] = px.groupby("instrument")["close"].pct_change()

    spy = D.features(["P84398"], ["$close"], start_time="2019-12-01", end_time="2024-01-05").reset_index()
    spy["datetime"] = pd.to_datetime(spy["datetime"]).dt.normalize()
    spy = spy.sort_values("datetime")
    spy["r_spy"] = spy["$close"].pct_change()
    spy_s = spy.set_index("datetime")["r_spy"]

    cal = sorted(px["datetime"].drop_duplicates())
    decision_dates = [d for d in cal if EVAL_START <= d <= EVAL_END]
    # last day needs next return
    if decision_dates and decision_dates[-1] == cal[-1]:
        decision_dates = decision_dates[:-1]

    configs = [
        ("RD8", pred_rd8, 20, 2),
        ("RD8", pred_rd8, 5, 1),
        ("RD13_v2", pred_rd13, 20, 2),
        ("RD13_v2", pred_rd13, 5, 1),
    ]
    summary_rows = []
    daily_map = {}
    for name, pred, topk, n_drop in configs:
        tag = f"{name}_Top{topk}_drop{n_drop}"
        log.info("portfolio %s", tag)
        sc = pred[(pred["datetime"] >= decision_dates[0]) & (pred["datetime"] <= decision_dates[-1])][
            ["datetime", "instrument", "score"]
        ]
        dates = [d for d in decision_dates if d in set(sc["datetime"])]
        w = topk_dropout_weights(sc, dates, topk=topk, n_drop=n_drop)
        w.to_csv(out / f"weights_{tag}.csv", index=False)
        daily = portfolio_daily(w, px[["datetime", "instrument", "ret"]], cal)
        daily = daily.merge(spy[["datetime", "r_spy"]].rename(columns={"datetime": "trade_date"}), on="trade_date", how="left")
        daily.to_csv(out / f"daily_{tag}.csv", index=False)
        m = summarize(daily, spy_s)
        m.update({"signal": name, "topk": topk, "n_drop": n_drop, "tag": tag})
        summary_rows.append(m)
        daily_map[tag] = daily.set_index("trade_date")["r_net"]

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "portfolio_2x2_summary.csv", index=False)

    # bootstrap pairwise diffs on daily excess
    boot_rows = []
    pairs = [
        ("RD13_v2_Top20_drop2", "RD8_Top20_drop2"),
        ("RD13_v2_Top5_drop1", "RD8_Top5_drop1"),
        ("RD8_Top5_drop1", "RD8_Top20_drop2"),
        ("RD13_v2_Top5_drop1", "RD13_v2_Top20_drop2"),
    ]
    for a, b in pairs:
        # align on common trade dates; compare excess
        da = daily_map[a]
        db = daily_map[b]
        idx = da.index.intersection(db.index)
        ex_a = da.loc[idx] - spy_s.reindex(idx)
        ex_b = db.loc[idx] - spy_s.reindex(idx)
        delta = (ex_a - ex_b).to_numpy()
        boot = block_bootstrap_mean(delta, seed=BOOT_SEED + hash(a + b) % 10000)
        boot_rows.append(
            {
                "contrast": f"{a}_minus_{b}",
                "delta_mean_daily_excess": boot["mean"],
                "delta_ARR_approx": boot["mean"] * ANN,
                "boot_ci_lo_daily": boot["ci_lo"],
                "boot_ci_hi_daily": boot["ci_hi"],
                "boot_ci_lo_ARR": boot["ci_lo"] * ANN,
                "boot_ci_hi_ARR": boot["ci_hi"] * ANN,
                "p_two_sided": boot["p"],
                "n_days": boot["n"],
                "block": BOOT_BLOCK,
                "n_boot": BOOT_N,
            }
        )
    boots = pd.DataFrame(boot_rows)
    boots.to_csv(out / "portfolio_2x2_bootstrap_diffs.csv", index=False)

    # markdown
    lines = [
        "# Alpha20+RD8 vs Alpha20+RD13_v2：2×2 TopK（2020–2023）",
        "",
        f"**Generated:** {utc_now()}  ",
        "**Pred:** R50，3 seeds 均值（`rd13_l20_downstream_cmp_20260923_190325`）  ",
        f"**Engine:** TopkDropout risk_degree={RISK_DEGREE}, hold_thresh={HOLD_THRESH}; cost={COST_BP}×turnover; SPY=P84398; deal lag=next-session close return",
        "",
        "| signal | rule | excess_ARR | CAGR | excess_MDD | wealth_MDD | mean_TO | cum_net |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.signal} | Top{int(r.topk)}/drop{int(r.n_drop)} | {r.excess_ARR:.4%} | {r.cagr:.4%} | {r.excess_MDD:.4%} | {r.wealth_MDD:.4%} | {r.mean_turnover:.3f} | {r.cum_net:.4%} |"
        )
    lines += ["", "## Bootstrap Δ daily excess（→×252 ≈ ARR）", "", boots.to_string(index=False), ""]
    (out / "PORTFOLIO_2X2_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT / f"reports/rd13_l20_supplement_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    write_factor_audit(out)
    export_ic_csvs(out)
    write_a20_rd6_note(out)
    run_2x2(out)
    # also copy key IC tables into CMP for convenience
    for f in ["ic_rankic_exact_test_2020_2023.csv", "ic_rankic_exact_holdout_2024_2025.csv", "paired_delta_rankic_all_rounds_periods.csv"]:
        src = out / f
        if src.exists():
            (CMP / f).write_bytes(src.read_bytes())
    manifest = {"generated_at": utc_now(), "cmp_source": str(CMP), "out": str(out)}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("done %s", out)
    print(out)


if __name__ == "__main__":
    main()
