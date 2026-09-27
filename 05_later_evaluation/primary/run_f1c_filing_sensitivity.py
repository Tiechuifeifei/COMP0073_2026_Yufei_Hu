#!/usr/bin/env python3
"""APPROXIMATE_CURRENT_DATA: F1C filing_date leakage audit + filing-safe sensitivity re-run.

1) Count (trade_date, ticker) where used Massive statement has filing_date > trade_date.
2) Rebuild F1C requiring massive_filing <= trade_date; re-run B×Top5 / B×Top20 (b) + placebo.
3) Confirm / recompute IC with label close(t+1)→close(t+2).
4) P5 name contribution concentration.
"""
from __future__ import annotations

import json
import logging
import shutil
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

import run_approximate_b_only as base  # noqa: E402
import approx_2026_helpers as v  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("f1c_filing_sens")

LABEL = "APPROXIMATE_CURRENT_DATA"
MAIN_START = base.MAIN_START
MAIN_END = base.MAIN_END
PLACEBO_N = base.PLACEBO_N
PLACEBO_SEED = base.PLACEBO_SEED + 100
BOOT_SEED = base.BOOT_SEED
PRIOR_RUN = PROJECT_ROOT / "reports/matched_2x2_2026_20260923_173800"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def f1c_asof_panel(
    stmt: pd.DataFrame,
    px: pd.DataFrame,
    dates: list[pd.Timestamp],
    *,
    require_filing_le_date: bool,
) -> pd.DataFrame:
    """merge_asof on avail_date (same as production F1C).

    If require_filing_le_date: when the asof-chosen Massive row has
    massive_filing > trade_date, drop that row and fall back to the latest
    prior statement with avail_date <= T and massive_filing <= T.
    Non-leaking rows are unchanged (not re-ranked by filing_date).
    """
    s = stmt.dropna(subset=["avail_date", "ticker"]).copy()
    s["avail_date"] = pd.to_datetime(s["avail_date"]).dt.normalize()
    s["massive_filing"] = pd.to_datetime(s["massive_filing"], errors="coerce").dt.normalize()
    date_index = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    px_keys = px[px["date"].isin(date_index)][["date", "ticker"]].drop_duplicates()
    pieces = []
    meta_cols = [c for c in s.columns if c != "ticker"]
    for ticker, g in s.groupby("ticker", sort=False):
        g = g.sort_values("avail_date").reset_index(drop=True)
        d = pd.DataFrame({"date": date_index})
        m = pd.merge_asof(d, g, left_on="date", right_on="avail_date")
        m["ticker"] = ticker
        if require_filing_le_date:
            leak = m["massive_filing"].notna() & (m["massive_filing"] > m["date"])
            if leak.any():
                g_ok = g.dropna(subset=["massive_filing"]).copy()
                for idx in m.index[leak]:
                    dt = m.at[idx, "date"]
                    elig = g_ok[(g_ok["avail_date"] <= dt) & (g_ok["massive_filing"] <= dt)]
                    if elig.empty:
                        for c in meta_cols:
                            m.at[idx, c] = pd.NA
                        m.at[idx, "ticker"] = ticker
                    else:
                        pick = elig.iloc[-1]
                        for c in meta_cols:
                            m.at[idx, c] = pick[c]
                        m.at[idx, "ticker"] = ticker
        pieces.append(m)
    if not pieces:
        return pd.DataFrame()
    asof = pd.concat(pieces, ignore_index=True)
    asof = asof.merge(px_keys, on=["date", "ticker"], how="inner")
    asof["leak_filing_after_trade"] = (
        asof["massive_filing"].notna() & (asof["massive_filing"] > asof["date"])
    )
    return asof


def f1c_raw_from_asof(asof: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Mirror v.f1c_raw_from_stmt feature construction from an as-of panel."""
    px_sub = px[["date", "ticker", "close"]].copy()
    asof = asof.merge(px_sub, on=["date", "ticker"], how="inner")
    asof = asof[asof["close"] > 0].copy()
    mcap = asof["shares"] * asof["close"]
    at_avg = asof["at_avg"].where(asof["at_avg"].notna() & (asof["at_avg"] != 0), asof["at"])
    book_avg = asof["book_avg"].where(asof["book_avg"].notna() & (asof["book_avg"] != 0), asof["book"])
    raw = pd.DataFrame(
        {
            "date": asof["date"].to_numpy(),
            "ticker": asof["ticker"].to_numpy(),
            "roe": asof["ttm_ni"] / book_avg,
            "roa": asof["ttm_ni"] / at_avg,
            "gross_profitability": asof["ttm_gp"] / at_avg,
            "sales_growth_yoy": asof["ttm_sale"] / asof["sale_lag4_ttm"] - 1.0,
            "asset_growth_yoy": asof["at"] / asof["at_lag4"] - 1.0,
            "accruals": (asof["ttm_ni"] - asof["ttm_oancf"]) / at_avg,
            "leverage": asof["debt"] / asof["at"],
            "current_ratio": asof["act"] / asof["lct"],
            "book_to_market": asof["book"] / mcap,
            "earnings_yield": asof["ttm_ni"] / mcap,
            "sales_to_price": asof["ttm_sale"] / mcap,
        }
    )
    raw = raw.replace([np.inf, -np.inf], np.nan)
    rename = {
        "roe": "roe_z",
        "roa": "roa_z",
        "gross_profitability": "gross_profitability_z",
        "sales_growth_yoy": "sales_growth_yoy_z",
        "asset_growth_yoy": "asset_growth_yoy_z",
        "accruals": "accruals_z",
        "leverage": "leverage_z",
        "current_ratio": "current_ratio_z",
        "book_to_market": "book_to_market_z",
        "earnings_yield": "earnings_yield_z",
        "sales_to_price": "sales_to_price_z",
    }
    raw = raw.rename(columns=rename)
    for col in v.F1C_FEATURES:
        if col not in raw.columns:
            raw[col] = np.nan
        raw[col] = raw.groupby("date")[col].transform(
            lambda s: v.zscore_series(v.winsorize_series(s.astype(float)))
            if s.notna().sum() >= v.CS_Z_MIN_F1C
            else np.nan
        )
    return raw


def ic_table_lag2(scores: pd.DataFrame, px: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """IC vs close(t+1)→close(t+2): label return on calendar day t+2 given decision t."""
    ret = px[["date", "ticker", "close"]].copy()
    ret["ret"] = ret.groupby("ticker")["close"].pct_change()
    cal = sorted(px["date"].drop_duplicates())
    loc = {d: i for i, d in enumerate(cal)}
    rows = []
    for dec in dates:
        i = loc.get(pd.Timestamp(dec))
        if i is None or i + 2 >= len(cal):
            continue
        label_date = cal[i + 2]  # return close(t+2)/close(t+1)-1 sits on date t+2
        s = scores[scores["date"] == dec][["ticker", "score"]]
        r = ret[ret["date"] == label_date][["ticker", "ret"]]
        m = s.merge(r, on="ticker").dropna()
        if len(m) < 20:
            continue
        rows.append(
            {
                "decision_date": dec,
                "label_date": label_date,
                "ic": float(m["score"].corr(m["ret"])),
                "rank_ic": float(m["score"].corr(m["ret"], method="spearman")),
                "n": int(len(m)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ic_summary(ic: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for metric in ("ic", "rank_ic"):
        x = ic[metric].dropna().to_numpy()
        rng = np.random.default_rng(BOOT_SEED)
        boots = []
        block = 10
        n = len(x)
        if n >= block:
            n_blocks = int(np.ceil(n / block))
            for _ in range(2000):
                starts = rng.integers(0, n - block + 1, size=n_blocks)
                sample = np.concatenate([x[s : s + block] for s in starts])[:n]
                boots.append(sample.mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
        else:
            lo = hi = np.nan
        rows.append(
            {
                "window": "main",
                "metric": metric,
                "mean": float(np.mean(x)) if len(x) else np.nan,
                "n_days": int(n),
                "boot_ci_lo": float(lo),
                "boot_ci_hi": float(hi),
                "boot_block": 10,
                "boot_n": 2000,
                "boot_seed": BOOT_SEED,
                "label": label,
                "data_label": LABEL,
            }
        )
    return pd.DataFrame(rows)


def name_contributions(
    weights: pd.DataFrame,
    px: pd.DataFrame,
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily name contribution w*r and ticker totals for main window."""
    ret = px[["date", "ticker", "close"]].copy()
    ret["ret"] = ret.groupby("ticker")["close"].pct_change()
    w = weights.rename(columns={"date": "decision_date"})
    cal = sorted(px["date"].drop_duplicates())
    loc = {d: i for i, d in enumerate(cal)}
    rows = []
    for dec in sorted(w["decision_date"].unique()):
        i = loc.get(pd.Timestamp(dec))
        if i is None or i + 1 >= len(cal):
            continue
        ret_date = cal[i + 1]
        if ret_date < eval_start or ret_date > eval_end:
            continue
        wd = w[w["decision_date"] == dec]
        day_ret = ret[ret["date"] == ret_date].set_index("ticker")["ret"]
        for tkr, wt in zip(wd["ticker"], wd["weight"]):
            r = day_ret.get(tkr, np.nan)
            if not np.isfinite(r):
                continue
            rows.append(
                {
                    "trade_date": ret_date,
                    "decision_date": dec,
                    "ticker": tkr,
                    "weight": float(wt),
                    "ret": float(r),
                    "contrib": float(wt) * float(r),
                }
            )
    daily_names = pd.DataFrame(rows)
    tot = (
        daily_names.groupby("ticker", as_index=False)["contrib"]
        .sum()
        .sort_values("contrib", ascending=False)
        .reset_index(drop=True)
    )
    tot["rank"] = np.arange(1, len(tot) + 1)
    port_sum = float(tot["contrib"].sum())
    tot["share_of_sum_contrib"] = tot["contrib"] / port_sum if port_sum != 0 else np.nan
    return daily_names, tot


def cum_from_contrib(daily_names: pd.DataFrame, exclude: set[str] | None = None, *, renormalize: bool) -> float:
    d = daily_names.copy()
    if exclude:
        d = d[~d["ticker"].isin(exclude)]
    if d.empty:
        return np.nan
    if renormalize:
        # scale remaining weights on each trade day to original day sum of weights
        day_w = daily_names.groupby("trade_date")["weight"].sum()
        rem_w = d.groupby("trade_date")["weight"].sum()
        scale = (day_w / rem_w).reindex(d["trade_date"]).to_numpy()
        d = d.copy()
        d["contrib"] = d["contrib"] * scale
    g = d.groupby("trade_date")["contrib"].sum().sort_index()
    return float((1.0 + g).prod() - 1.0)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / f"reports/matched_2x2_2026_f1c_filing_sens_{ts}"
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if PRIOR_RUN.exists():
        for f in (PRIOR_RUN / "cache").iterdir():
            if f.is_file() and f.suffix in {".parquet", ".json"}:
                shutil.copy2(f, cache / f.name)
    log.info("output %s LABEL=%s", out, LABEL)

    px = v.load_prices()
    spy_px = pd.read_parquet(v.MASSIVE_SPY)
    spy_px["date"] = pd.to_datetime(spy_px["date"]).dt.normalize()
    spy_ret = spy_px.sort_values("date").set_index("date")["close"].pct_change()

    px_max = px["date"].max()
    all_decision_dates = sorted(d for d in px["date"].drop_duplicates() if MAIN_START <= d <= px_max)
    cal = sorted(px["date"].drop_duplicates())
    if all_decision_dates and all_decision_dates[-1] == cal[-1]:
        all_decision_dates = all_decision_dates[:-1]
    main_decision = [d for d in all_decision_dates if d <= MAIN_END]

    # Membership universe for leakage denominator (stock-days with price in SP500 carry)
    membership = base.load_membership_carry(main_decision)
    universe = membership.merge(px[["date", "ticker"]].drop_duplicates(), on=["date", "ticker"], how="inner")

    log.info("building statement panel")
    stmt = v.build_f1c_statement_panel()

    # --- 1) Leakage under ACTUAL F1C as-of (avail_date clock, no filing filter) ---
    log.info("asof panel (actual avail_date clock)")
    asof_actual = f1c_asof_panel(stmt, px, main_decision, require_filing_le_date=False)
    asof_actual = asof_actual.merge(universe, on=["date", "ticker"], how="inner")
    # Only rows that actually feed F1C (have statement attached)
    used = asof_actual.dropna(subset=["period_end"]).copy()
    leak = used["leak_filing_after_trade"]
    stock_day_rate = float(leak.mean()) if len(used) else np.nan
    stock_day_n = int(len(used))
    stock_day_leak_n = int(leak.sum())

    w_p5 = pd.read_csv(PRIOR_RUN / "weights_B_Top5_regen.csv")
    w_p5["date"] = pd.to_datetime(w_p5["date"]).dt.normalize()
    w_p5 = w_p5[w_p5["date"].isin(main_decision)]
    hold_asof = w_p5.merge(
        used[["date", "ticker", "massive_filing", "avail_date", "period_end", "leak_filing_after_trade"]],
        on=["date", "ticker"],
        how="left",
    )
    # holdings without matched stmt → not counted as leak (no F1C row used)
    hold_matched = hold_asof.dropna(subset=["period_end"])
    hold_rate = float(hold_matched["leak_filing_after_trade"].mean()) if len(hold_matched) else np.nan
    hold_n = int(len(hold_matched))
    hold_leak_n = int(hold_matched["leak_filing_after_trade"].sum()) if len(hold_matched) else 0
    affected = hold_matched[hold_matched["leak_filing_after_trade"]][
        ["date", "ticker", "period_end", "avail_date", "massive_filing", "weight"]
    ].sort_values(["date", "ticker"])
    affected.to_csv(out / "p5_holdings_filing_leak_days.csv", index=False)

    leak_summary = pd.DataFrame(
        [
            {
                "scope": "universe_stock_day_with_stmt",
                "n": stock_day_n,
                "n_leak": stock_day_leak_n,
                "rate": stock_day_rate,
                "definition": "massive_filing > trade_date on F1C asof row (avail_date clock)",
                "data_label": LABEL,
            },
            {
                "scope": "p5_holdings_with_stmt",
                "n": hold_n,
                "n_leak": hold_leak_n,
                "rate": hold_rate,
                "definition": "same, restricted to B×Top5 (b) holdings",
                "data_label": LABEL,
            },
            {
                "scope": "p5_holdings_including_no_stmt",
                "n": int(len(hold_asof)),
                "n_leak": hold_leak_n,
                "rate": float(hold_leak_n / len(hold_asof)) if len(hold_asof) else np.nan,
                "definition": "leak / all P5 holding rows (unmatched stmt count as non-leak)",
                "data_label": LABEL,
            },
        ]
    )
    leak_summary.to_csv(out / "filing_leak_summary.csv", index=False)
    log.info(
        "leak stock-day %.4f (%d/%d); P5 holdings %.4f (%d/%d)",
        stock_day_rate,
        stock_day_leak_n,
        stock_day_n,
        hold_rate,
        hold_leak_n,
        hold_n,
    )

    # --- 2) Filing-safe F1C + W1 rebuild ---
    log.info("asof panel (filing-safe: massive_filing <= trade_date)")
    asof_safe = f1c_asof_panel(stmt, px, all_decision_dates, require_filing_le_date=True)
    f1c_raw_safe = f1c_raw_from_asof(asof_safe, px)
    f1c_models = v.train_f1c_seeds()
    f1c_safe = v.predict_f1c(f1c_raw_safe, f1c_models)

    # D2 from prior cache
    d2_path = cache / "d2_scores.parquet"
    if d2_path.exists():
        d2 = pd.read_parquet(d2_path)
        d2["date"] = pd.to_datetime(d2["date"]).dt.normalize()
    else:
        _, d2, _ = base.build_scores(px, all_decision_dates, cache_dir=cache, include_f1c=False)

    mem_all = base.load_membership_carry(all_decision_dates)
    w1_safe = v.build_w1(d2, f1c_safe, mem_all, px)
    w1_safe.to_parquet(cache / "w1_scores_filing_safe.parquet", index=False)

    w1_main = w1_safe[w1_safe["date"].isin(main_decision)].copy()
    w5 = base.topk_dropout_weights(w1_main[["date", "ticker", "score"]], main_decision, topk=5, n_drop=1)
    w20 = base.topk_dropout_weights(w1_main[["date", "ticker", "score"]], main_decision, topk=20, n_drop=2)
    w5.to_csv(out / "weights_B_Top5_filing_safe.csv", index=False)
    w20.to_csv(out / "weights_B_Top20_filing_safe.csv", index=False)

    daily5 = base.attach_spy(
        base.portfolio_from_weights(w5, px, eval_start=MAIN_START, eval_end=MAIN_END), spy_px
    )
    daily20 = base.attach_spy(
        base.portfolio_from_weights(w20, px, eval_start=MAIN_START, eval_end=MAIN_END), spy_px
    )
    daily5.to_csv(out / "daily_B_Top5_filing_safe.csv", index=False)
    daily20.to_csv(out / "daily_B_Top20_filing_safe.csv", index=False)

    m5 = base.summarize_daily(daily5, spy_ret, w5)
    m20 = base.summarize_daily(daily20, spy_ret, w20)
    # baseline (b) from prior
    base5 = pd.read_csv(PRIOR_RUN / "daily_B_Top5_regen.csv")
    base20 = pd.read_csv(PRIOR_RUN / "daily_B_Top20_regen.csv")
    for d in (base5, base20):
        d["trade_date"] = pd.to_datetime(d["trade_date"])
    m5_base = base.summarize_daily(base5, spy_ret, w_p5)
    w20_base = pd.read_csv(PRIOR_RUN / "weights_B_Top20_regen.csv")
    w20_base["date"] = pd.to_datetime(w20_base["date"])
    m20_base = base.summarize_daily(base20, spy_ret, w20_base)

    results = pd.DataFrame(
        [
            {"strategy": "B×Top5 (b) baseline avail_date", "data_label": LABEL, **m5_base},
            {"strategy": "B×Top5 (b) filing_safe", "data_label": LABEL, **m5},
            {"strategy": "B×Top20 (b) baseline avail_date", "data_label": LABEL, **m20_base},
            {"strategy": "B×Top20 (b) filing_safe", "data_label": LABEL, **m20},
        ]
    )
    results.to_csv(out / "main_window_filing_safe_results.csv", index=False)

    # Placebo on same universe
    log.info("placebo filing-safe window")
    univ = {
        d: g["ticker"].tolist()
        for d, g in mem_all[mem_all["date"].isin(main_decision)].groupby("date")
    }
    plac5 = base.placebo_runs(
        univ, main_decision, px, topk=5, n_drop=1, eval_start=MAIN_START, eval_end=MAIN_END, n=PLACEBO_N, seed=PLACEBO_SEED
    )
    plac20 = base.placebo_runs(
        univ,
        main_decision,
        px,
        topk=20,
        n_drop=2,
        eval_start=MAIN_START,
        eval_end=MAIN_END,
        n=PLACEBO_N,
        seed=PLACEBO_SEED + 1,
    )
    plac5.to_csv(out / "placebo_main_Top5.csv", index=False)
    plac20.to_csv(out / "placebo_main_Top20.csv", index=False)

    def plac_sum(dist: pd.DataFrame, rule: str, refs: dict[str, float]) -> dict[str, Any]:
        c = dist["cum_net"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "rule": rule,
            "window": "main",
            "n": int(len(c)),
            "cum_mean": float(np.mean(c)),
            "cum_p50": float(np.median(c)),
            "cum_p05": float(np.percentile(c, 5)),
            "cum_p95": float(np.percentile(c, 95)),
            "data_label": LABEL,
        }
        for k, v_ in refs.items():
            row[f"pctile_{k}"] = base.percentile_of(v_, c)
            row[f"ref_cum_{k}"] = v_
        return row

    plac_rows = [
        plac_sum(
            plac5,
            "Top5/drop1",
            {"baseline_P5": m5_base["cum_net"], "filing_safe_P5": m5["cum_net"], "filing_safe_Top20": m20["cum_net"]},
        ),
        plac_sum(
            plac20,
            "Top20/drop2",
            {"baseline_Top20": m20_base["cum_net"], "filing_safe_Top20": m20["cum_net"], "filing_safe_P5": m5["cum_net"]},
        ),
    ]
    pd.DataFrame(plac_rows).to_csv(out / "placebo_summary.csv", index=False)

    # --- 3) IC labels ---
    # Prior ic: close(t)→close(t+1) on label_date=t+1
    ic_lag1 = base.ic_table(w1_main, px, main_decision)
    ic_lag1.to_csv(out / "ic_daily_main_lag1_close_t_to_t1.csv", index=False)
    sum_lag1 = bootstrap_ic_summary(ic_lag1, "close(t)→close(t+1)_on_label_date_t+1__NOT_requested")
    ic_lag2 = ic_table_lag2(w1_main, px, main_decision)
    ic_lag2.to_csv(out / "ic_daily_main_lag2_close_t1_to_t2.csv", index=False)
    sum_lag2 = bootstrap_ic_summary(ic_lag2, "close(t+1)→close(t+2)_on_label_date_t+2")
    # Also IC on baseline W1 from prior cache if present
    w1_base_path = cache / "w1_scores.parquet"
    ic_compare_rows = [sum_lag1, sum_lag2]
    if w1_base_path.exists():
        w1b = pd.read_parquet(w1_base_path)
        w1b["date"] = pd.to_datetime(w1b["date"]).dt.normalize()
        w1b_main = w1b[w1b["date"].isin(main_decision)]
        ic_b1 = base.ic_table(w1b_main, px, main_decision)
        ic_b2 = ic_table_lag2(w1b_main, px, main_decision)
        s1 = bootstrap_ic_summary(ic_b1, "baseline_W1|close(t)→close(t+1)")
        s2 = bootstrap_ic_summary(ic_b2, "baseline_W1|close(t+1)→close(t+2)")
        s1["score_source"] = "baseline_avail_date"
        s2["score_source"] = "baseline_avail_date"
        sum_lag1 = sum_lag1.copy()
        sum_lag2 = sum_lag2.copy()
        sum_lag1["score_source"] = "filing_safe"
        sum_lag2["score_source"] = "filing_safe"
        ic_compare_rows = [s1, s2, sum_lag1, sum_lag2]
    else:
        sum_lag1 = sum_lag1.copy()
        sum_lag2 = sum_lag2.copy()
        sum_lag1["score_source"] = "filing_safe"
        sum_lag2["score_source"] = "filing_safe"
        ic_compare_rows = [sum_lag1, sum_lag2]
    ic_sum = pd.concat(ic_compare_rows, ignore_index=True)
    ic_sum.to_csv(out / "ic_summary.csv", index=False)

    # --- 4) Contribution concentration on baseline P5 (b) ---
    daily_names, tot = name_contributions(w_p5, px, eval_start=MAIN_START, eval_end=MAIN_END)
    daily_names.to_csv(out / "p5_daily_name_contrib.csv", index=False)
    tot.to_csv(out / "p5_ticker_contrib_ranked.csv", index=False)
    port_sum = float(tot["contrib"].sum())
    top1 = tot.head(1)
    top3 = tot.head(3)
    top5 = tot.head(5)
    excl5 = set(top5["ticker"].tolist())
    conc = {
        "data_label": LABEL,
        "portfolio": "B×Top5 (b) regen baseline",
        "sum_contrib": port_sum,
        "cum_net_full": m5_base["cum_net"],
        "top1_tickers": top1["ticker"].tolist(),
        "top1_contrib": float(top1["contrib"].sum()),
        "top1_share": float(top1["contrib"].sum() / port_sum) if port_sum else np.nan,
        "top3_tickers": top3["ticker"].tolist(),
        "top3_contrib": float(top3["contrib"].sum()),
        "top3_share": float(top3["contrib"].sum() / port_sum) if port_sum else np.nan,
        "top5_tickers": top5["ticker"].tolist(),
        "top5_contrib": float(top5["contrib"].sum()),
        "top5_share": float(top5["contrib"].sum() / port_sum) if port_sum else np.nan,
        "cum_net_drop_top5_no_renorm": cum_from_contrib(daily_names, excl5, renormalize=False),
        "cum_net_drop_top5_renorm": cum_from_contrib(daily_names, excl5, renormalize=True),
    }
    pd.DataFrame([conc]).to_csv(out / "p5_contribution_concentration.csv", index=False)
    with (out / "p5_contribution_concentration.json").open("w") as f:
        json.dump(conc, f, indent=2, default=str)

    # Manifest + SUMMARY
    manifest = {
        "status": "COMPLETED_APPROXIMATE_CURRENT_DATA",
        "data_label": LABEL,
        "generated_at": utc_now(),
        "prior_baseline_run": str(PRIOR_RUN),
        "leak_stock_day_rate": stock_day_rate,
        "leak_p5_holding_rate": hold_rate,
        "filing_safe_top5_cum": m5["cum_net"],
        "filing_safe_top20_cum": m20["cum_net"],
        "baseline_top5_cum": m5_base["cum_net"],
        "baseline_top20_cum": m20_base["cum_net"],
        "ic_prior_label_was": "close(t)→close(t+1) via next_session_return_Massive",
        "ic_requested_label": "close(t+1)→close(t+2)",
        "contribution_top5": conc["top5_tickers"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Holding overlap vs baseline
    def hold_sets(w: pd.DataFrame) -> dict:
        return w.groupby("date")["ticker"].apply(lambda s: tuple(sorted(s))).to_dict()

    a, b = hold_sets(w5), hold_sets(w_p5[w_p5["date"].isin(main_decision)])
    n_mis = sum(1 for d in set(a) | set(b) if a.get(d) != b.get(d))

    md = f"""# F1C filing_date sensitivity — APPROXIMATE_CURRENT_DATA

**Directory:** `{out.name}`  
**Generated:** {utc_now()}  
**Label:** `{LABEL}`  
**Baseline (b):** `{PRIOR_RUN.name}`

## 1. Restatement / filing_date leakage (主窗决策日)

定义：F1C 实际 `merge_asof(avail_date)` 命中的 Massive 报表行，其 `massive_filing` **晚于**该交易日。

| 口径 | n | n_leak | rate |
|---|---:|---:|---:|
| 股票-日（有 stmt） | {stock_day_n} | {stock_day_leak_n} | **{stock_day_rate:.2%}** |
| P5 持仓（有 stmt） | {hold_n} | {hold_leak_n} | **{hold_rate:.2%}** |

P5 受影响明细：`p5_holdings_filing_leak_days.csv`（{hold_leak_n} 行）。

## 2. Filing-safe 敏感性（剔除 filing_date>交易日，回退上一期已申报）

| strategy | cum_net | spy_cum | cum_excess_geom | qlib_excess_ARR | Sharpe |
|---|---:|---:|---:|---:|---:|
| B×Top5 (b) baseline | {m5_base['cum_net']:.2%} | {m5_base['spy_cum']:.2%} | {m5_base['cum_excess_geom']:.2%} | {m5_base['qlib_excess_arr']:.2%} | {m5_base['sharpe_rf0']:.2f} |
| B×Top5 (b) filing_safe | {m5['cum_net']:.2%} | {m5['spy_cum']:.2%} | {m5['cum_excess_geom']:.2%} | {m5['qlib_excess_arr']:.2%} | {m5['sharpe_rf0']:.2f} |
| B×Top20 (b) baseline | {m20_base['cum_net']:.2%} | {m20_base['spy_cum']:.2%} | {m20_base['cum_excess_geom']:.2%} | {m20_base['qlib_excess_arr']:.2%} | {m20_base['sharpe_rf0']:.2f} |
| B×Top20 (b) filing_safe | {m20['cum_net']:.2%} | {m20['spy_cum']:.2%} | {m20['cum_excess_geom']:.2%} | {m20['qlib_excess_arr']:.2%} | {m20['sharpe_rf0']:.2f} |

持仓与 baseline 决策日不一致数：{n_mis} / {len(main_decision)}。

### Placebo（500×，主窗）

见 `placebo_summary.csv`。

## 3. IC 标签

先验 `ic_summary` 标签为 **`next_session_return_Massive` = close(t)→close(t+1)**（decision=t，label_date=t+1 上的 pct_change），**不是** close(t+1)→close(t+2)。

按请求定义重算见 `ic_summary.csv`（含 baseline W1 与 filing_safe W1）。

## 4. P5 贡献集中度（baseline b，Σ w·r）

| k | tickers | share of Σcontrib |
|---|---|---:|
| 1 | {conc['top1_tickers']} | {conc['top1_share']:.2%} |
| 3 | {conc['top3_tickers']} | {conc['top3_share']:.2%} |
| 5 | {conc['top5_tickers']} | {conc['top5_share']:.2%} |

- 全样本 cum_net：{conc['cum_net_full']:.2%}
- 剔除 top5 贡献（不重标权）cum：{conc['cum_net_drop_top5_no_renorm']:.2%}
- 剔除 top5 后同日剩余权重重标到原权重和：{conc['cum_net_drop_top5_renorm']:.2%}
"""
    (out / "SUMMARY.md").write_text(md)
    log.info("done → %s", out)
    print(out)


if __name__ == "__main__":
    main()
