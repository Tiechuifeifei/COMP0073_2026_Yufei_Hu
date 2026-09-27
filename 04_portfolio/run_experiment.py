#!/usr/bin/env python3
"""TopK breadth experiment: K20 parity check, then Phase A/B/C robustness grids
around the frozen H0 engine (no score-threshold search)."""

from __future__ import annotations
import os

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qlib.contrib.evaluate import risk_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "portfolio_experiments"))
sys.path.insert(0, str(Path(os.environ["RDAGENT_ROOT"]) / "phase3_portfolio_ablation" / "scripts"))

import run_portfolio_ablation as rpa  # noqa: E402

from portfolio_experiments.adaptive_turnover_ma120.run_a0_parity import (  # noqa: E402
    compare_window,
    flatten_checks,
    json_safe as a0_json_safe,
)
from portfolio_experiments.final_holdout import holdout_runner as hr  # noqa: E402
from portfolio_experiments.final_holdout.config import HOLDOUT_END, HOLDOUT_START  # noqa: E402
from portfolio_experiments.hmm_regime.config import (  # noqa: E402
    SPY_CSV_PATH,
    TEST_END,
    TEST_START,
    VALID_END,
    VALID_START,
)
from portfolio_experiments.hmm_regime.metrics_utils import load_r1r2p_module, mdd_from_returns, run_qlib_backtest  # noqa: E402
from portfolio_experiments.hmm_regime.run_portfolio_test import (  # noqa: E402
    build_scheme_pred,
    frozen_hmm_timeline,
    infer_score_sign,
)
from portfolio_experiments.hmm_regime.run_validation import attach_states_to_panel, load_common_panel  # noqa: E402
from portfolio_experiments.hmm_regime.signal_utils import preprocess_scores  # noqa: E402
from portfolio_experiments.topk_breadth.strategies import (  # noqa: E402
    AUDIT_LOG,
    AlternatingDropTopkStrategy,  # noqa: F401
    reset_audit_log,
)

OUT_ROOT = PROJECT_ROOT / "data/portfolio_experiments/topk_breadth"
REPORT_ROOT = PROJECT_ROOT / "reports/portfolio_experiments/topk_breadth"
PRECOMMIT_PATH = OUT_ROOT / "precommit.json"
STRATEGY_MODULE = "portfolio_experiments.topk_breadth.strategies"

T0_FROZEN = PROJECT_ROOT / "data/portfolio_experiments/hmm_regime/portfolio_test_daily_returns.parquet"
H0_FROZEN = PROJECT_ROOT / "data/portfolio_experiments/final_holdout/attribution/holdout_h0_daily_returns.parquet"
H0_HOLDINGS = PROJECT_ROOT / "data/portfolio_experiments/final_holdout/attribution/h0_daily_holdings.csv"

STATUS_LABEL = "POST_HOC EXPLORATORY / ROBUSTNESS"
DAILY_ABS_TOL = 1e-10
BUCKETS = (("B1", 1, 5), ("B2", 6, 10), ("B3", 11, 20), ("B4", 21, 50))

WINDOWS = [
    {"id": "2018-2019", "split": "valid", "start": VALID_START, "end": VALID_END, "parity": None},
    {"id": "2020-2023", "split": "test", "start": TEST_START, "end": TEST_END, "parity": "T0"},
    {"id": "2024-2025", "split": "holdout", "start": HOLDOUT_START, "end": HOLDOUT_END, "parity": "H0"},
]

SLICE_PERIODS = [
    {"id": "2018-2019", "start": "2018-01-01", "end": "2019-12-31", "source_window": "2018-2019", "concat": False},
    {"id": "2020-2023", "start": "2020-01-01", "end": "2023-12-31", "source_window": "2020-2023", "concat": False},
    {"id": "2024", "start": "2024-01-01", "end": "2024-12-31", "source_window": "2024-2025", "concat": False},
    {"id": "2025", "start": "2025-01-01", "end": "2025-12-31", "source_window": "2024-2025", "concat": False},
    {"id": "2024-2025", "start": "2024-01-01", "end": "2025-12-31", "source_window": "2024-2025", "concat": False},
    {
        "id": "2020-2025",
        "start": "2020-01-01",
        "end": "2025-12-31",
        "source_window": None,
        "concat": True,
        "concat_windows": ["2020-2023", "2024-2025"],
    },
]

PHASE_B_ARMS = {
    "K20": {"topk": 20, "n_drop": 2, "strategy": "native", "phase": "B"},
    "K10": {"topk": 10, "n_drop": 1, "strategy": "native", "phase": "B"},
    "K5": {"topk": 5, "n_drop": 0, "strategy": "alternating", "phase": "B"},
}
PHASE_C_ARMS = {
    "P20": {"topk": 20, "n_drop": 2, "strategy": "reuse_k20", "phase": "C"},
    "P10": {"topk": 10, "n_drop": 2, "strategy": "native", "phase": "C"},
    "P5": {"topk": 5, "n_drop": 1, "strategy": "native", "phase": "C"},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(obj: Any) -> Any:
    return a0_json_safe(obj)


def save_frame(df: pd.DataFrame, stem: str) -> None:
    path = OUT_ROOT / stem
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path.with_suffix(".csv"), index=False)
    if len(df.columns):
        df.to_parquet(path.with_suffix(".parquet"), index=False)


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "_none_"
    view = df if cols is None else df[[c for c in cols if c in df.columns]]
    try:
        return view.to_markdown(index=False)
    except Exception:
        return "```\n" + view.to_string(index=False) + "\n```"


def fmt_pct(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{100.0 * float(x):.{nd}f}%"


def fmt_num(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{float(x):.{nd}f}"


def report_to_daily(report: pd.DataFrame, arm_id: str, window: str, phase: str) -> pd.DataFrame:
    rep = report.copy()
    rep.index = pd.to_datetime(rep.index).normalize()
    df = rep[["return", "bench", "cost", "turnover"]].reset_index()
    df = df.rename(columns={df.columns[0]: "trade_date"})
    df["arm_id"] = arm_id
    df["window"] = window
    df["phase"] = phase
    df["status"] = STATUS_LABEL
    df["excess_net"] = df["return"] - df["bench"] - df["cost"]
    df["nav"] = (1.0 + df["return"]).cumprod()
    df["bench_nav"] = (1.0 + df["bench"]).cumprod()
    return df


def stock_weights_from_positions(positions: dict, arm_id: str, window: str, phase: str) -> pd.DataFrame:
    rows = []
    for dt, pos in positions.items():
        ts = pd.Timestamp(dt).normalize()
        total = float(pos.calculate_value())
        if total <= 0:
            continue
        cash = float(pos.get_cash())
        n = len(pos.get_stock_list())
        weight_dict = pos.get_stock_weight_dict(only_stock=False)
        amounts = pos.get_stock_amount_dict()
        for code in pos.get_stock_list():
            w = float(weight_dict.get(code, 0.0))
            amt = float(amounts.get(code, 0.0))
            val = w * total
            rows.append(
                {
                    "trade_date": ts,
                    "instrument": str(code),
                    "stock_value": val,
                    "weight": w,
                    "amount": amt,
                    "price": (val / amt) if amt else np.nan,
                    "cash_weight": cash / total,
                    "n_holdings": n,
                    "arm_id": arm_id,
                    "window": window,
                    "phase": phase,
                    "status": STATUS_LABEL,
                }
            )
    return pd.DataFrame(rows)


def holdings_sets(holdings: pd.DataFrame) -> dict[pd.Timestamp, set[str]]:
    if holdings is None or holdings.empty:
        return {}
    h = holdings.copy()
    h["trade_date"] = pd.to_datetime(h["trade_date"]).dt.normalize()
    h["instrument"] = h["instrument"].astype(str)
    return {dt: set(g["instrument"]) for dt, g in h.groupby("trade_date")}


def membership_trades(hold_map: dict[pd.Timestamp, set[str]], dates: list[pd.Timestamp]) -> tuple[int, int]:
    n_buys = 0
    n_sells = 0
    prev: set[str] = set()
    for dt in dates:
        cur = hold_map.get(dt, set())
        n_buys += len(cur - prev)
        n_sells += len(prev - cur)
        prev = cur
    return n_buys, n_sells


def average_holding_period_days(hold_map: dict[pd.Timestamp, set[str]], dates: list[pd.Timestamp]) -> float:
    if not dates:
        return float("nan")
    cal = {d: i for i, d in enumerate(dates)}
    spells: list[int] = []
    by_inst: dict[str, list[pd.Timestamp]] = {}
    for dt in dates:
        for inst in hold_map.get(dt, set()):
            by_inst.setdefault(inst, []).append(dt)
    for _inst, held_dates in by_inst.items():
        held_dates = sorted(held_dates)
        start = held_dates[0]
        prev = held_dates[0]
        for dt in held_dates[1:]:
            if cal[dt] == cal[prev] + 1:
                prev = dt
            else:
                spells.append(cal[prev] - cal[start] + 1)
                start = dt
                prev = dt
        spells.append(cal[prev] - cal[start] + 1)
    return float(np.mean(spells)) if spells else float("nan")


def slice_daily(daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame()
    d = daily.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.normalize()
    mask = (d["trade_date"] >= pd.Timestamp(start)) & (d["trade_date"] <= pd.Timestamp(end))
    out = d.loc[mask].sort_values("trade_date").reset_index(drop=True)
    if out.empty:
        return out
    out = out.copy()
    out["nav"] = (1.0 + out["return"]).cumprod()
    out["bench_nav"] = (1.0 + out["bench"]).cumprod()
    return out


def slice_holdings(holdings: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if holdings is None or holdings.empty:
        return pd.DataFrame()
    h = holdings.copy()
    h["trade_date"] = pd.to_datetime(h["trade_date"]).dt.normalize()
    mask = (h["trade_date"] >= pd.Timestamp(start)) & (h["trade_date"] <= pd.Timestamp(end))
    return h.loc[mask].copy()


def concat_independent(frames: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    out = out.sort_values("trade_date").reset_index(drop=True)
    if "return" in out.columns:
        out["nav"] = (1.0 + out["return"]).cumprod()
        out["bench_nav"] = (1.0 + out["bench"]).cumprod()
        out["excess_net"] = out["return"] - out["bench"] - out["cost"]
    return out


def summary_from_daily(daily: pd.DataFrame, holdings: pd.DataFrame | None = None) -> dict[str, Any]:
    empty = {
        "n_days": 0,
        "arr": np.nan,
        "benchmark_arr": np.nan,
        "excess_arr": np.nan,
        "cumulative_return": np.nan,
        "sharpe": np.nan,
        "volatility": np.nan,
        "mdd": np.nan,
        "turnover": np.nan,
        "transaction_cost_daily_mean": np.nan,
        "transaction_cost_total": np.nan,
        "average_holding_period": np.nan,
        "n_buys": np.nan,
        "n_sells": np.nan,
        "final_nav": np.nan,
    }
    if daily is None or daily.empty:
        return empty
    r = daily["return"].astype(float)
    b = daily["bench"].astype(float)
    c = daily["cost"].astype(float)
    ex = daily["excess_net"].astype(float)
    port_ra = risk_analysis(r, freq="day")
    bench_ra = risk_analysis(b, freq="day")
    ex_ra = risk_analysis(ex, freq="day")
    dates = list(pd.to_datetime(daily["trade_date"]).dt.normalize())
    hold_map = holdings_sets(holdings) if holdings is not None and not holdings.empty else {}
    n_buys, n_sells = membership_trades(hold_map, dates) if hold_map else (np.nan, np.nan)
    avg_hp = average_holding_period_days(hold_map, dates) if hold_map else np.nan
    return {
        "n_days": int(len(daily)),
        "arr": float(port_ra.loc["annualized_return", "risk"]),
        "benchmark_arr": float(bench_ra.loc["annualized_return", "risk"]),
        "excess_arr": float(ex_ra.loc["annualized_return", "risk"]),
        "cumulative_return": float((1.0 + r).prod() - 1.0),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if float(r.std()) > 0 else np.nan,
        "volatility": float(r.std() * np.sqrt(252)) if len(r) > 1 else np.nan,
        "mdd": mdd_from_returns(r),
        "turnover": float(daily["turnover"].mean()),
        "transaction_cost_daily_mean": float(c.mean()),
        "transaction_cost_total": float(c.sum()),
        "average_holding_period": float(avg_hp) if pd.notna(avg_hp) else np.nan,
        "n_buys": n_buys if n_buys == n_buys else np.nan,
        "n_sells": n_sells if n_sells == n_sells else np.nan,
        "final_nav": float((1.0 + r).prod()),
    }


def theory_from_pred(pred: pd.DataFrame, trade_dates: list[pd.Timestamp], topk: int) -> pd.DataFrame:
    p = pred.reset_index()
    p["datetime"] = pd.to_datetime(p["datetime"]).dt.normalize()
    p["instrument"] = p["instrument"].astype(str)
    by_date = {dt: g for dt, g in p.groupby("datetime")}
    score_cal = sorted(by_date)
    rows = []
    for t in trade_dates:
        t = pd.Timestamp(t).normalize()
        earlier = [d for d in score_cal if d < t]
        if not earlier:
            continue
        sdate = earlier[-1]
        g = by_date[sdate].sort_values("score", ascending=False).head(int(topk))
        for rank, rec in enumerate(g.itertuples(index=False), 1):
            rows.append(
                {
                    "trade_date": t,
                    "score_date": sdate,
                    "instrument": str(rec.instrument),
                    "rank": int(rank),
                    "score": float(rec.score),
                    "topk": int(topk),
                }
            )
    return pd.DataFrame(rows)


def delay_stats(
    hold_map: dict[pd.Timestamp, set[str]],
    theory_map: dict[pd.Timestamp, set[str]],
    dates: list[pd.Timestamp],
) -> dict[str, Any]:
    pending_entry: dict[str, int] = {}
    pending_exit: dict[str, int] = {}
    entry_delays: list[int] = []
    exit_delays: list[int] = []
    missed_entries = 0
    cancelled_exits = 0
    prev_held: set[str] = set()

    for i, dt in enumerate(dates):
        t_set = theory_map.get(dt, set())
        h_eod = hold_map.get(dt, set())
        h_open = prev_held
        for inst in t_set:
            if inst not in h_open and inst not in pending_entry:
                pending_entry[inst] = i
        resolved_entry = []
        for inst, start in pending_entry.items():
            if inst in h_eod:
                entry_delays.append(i - start)
                resolved_entry.append(inst)
            elif inst not in t_set:
                missed_entries += 1
                resolved_entry.append(inst)
        for inst in resolved_entry:
            del pending_entry[inst]
        for inst in h_open - t_set:
            if inst not in pending_exit:
                pending_exit[inst] = i
        resolved_exit = []
        for inst, start in pending_exit.items():
            if inst not in h_eod:
                exit_delays.append(i - start)
                resolved_exit.append(inst)
            elif inst in t_set:
                cancelled_exits += 1
                resolved_exit.append(inst)
        for inst in resolved_exit:
            del pending_exit[inst]
        prev_held = h_eod

    def _mean(xs: list[int]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    overlap_ratios = []
    overlap_counts = []
    for dt in dates:
        t_set = theory_map.get(dt, set())
        h_eod = hold_map.get(dt, set())
        if not t_set:
            continue
        overlap_counts.append(len(h_eod & t_set))
        overlap_ratios.append(len(h_eod & t_set) / float(len(t_set)))
    return {
        "theoretical_topk_overlap_mean": float(np.mean(overlap_ratios)) if overlap_ratios else np.nan,
        "theoretical_topk_overlap_count_mean": float(np.mean(overlap_counts)) if overlap_counts else np.nan,
        "entry_delay_mean": _mean(entry_delays),
        "entry_delay_median": float(np.median(entry_delays)) if entry_delays else np.nan,
        "n_completed_entries": int(len(entry_delays)),
        "n_missed_entries": int(missed_entries),
        "exit_delay_mean": _mean(exit_delays),
        "exit_delay_median": float(np.median(exit_delays)) if exit_delays else np.nan,
        "n_completed_exits": int(len(exit_delays)),
        "n_cancelled_exits": int(cancelled_exits),
    }


def spy_label0_map(spy_path: Path) -> dict[pd.Timestamp, float]:
    spy = pd.read_csv(spy_path)
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
    spy = spy.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    closes = spy["close"].astype(float).to_numpy()
    dates = list(spy["date"])
    out: dict[pd.Timestamp, float] = {}
    for i, d in enumerate(dates):
        if i + 2 >= len(dates):
            continue
        c1 = closes[i + 1]
        c2 = closes[i + 2]
        if c1 > 0:
            out[pd.Timestamp(d).normalize()] = float(c2 / c1 - 1.0)
    return out


def score_w1(panel: pd.DataFrame, *, flip: bool, split: str) -> pd.DataFrame:
    sub = panel[panel["split"] == split].copy()
    sub = preprocess_scores(sub, ["quant_score", "fundamental_score"])
    sub["score"] = 0.7 * sub["quant_score_z"] + 0.3 * sub["fundamental_score_z"]
    if flip:
        sub["score"] = -sub["score"]
    sub["instrument"] = sub["instrument"].astype(str)
    sub["datetime"] = pd.to_datetime(sub["datetime"]).dt.normalize()
    return sub


def paired_tests(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    aligned = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
    diff = aligned["a"] - aligned["b"]
    out = {
        "n": int(len(diff)),
        "mean_diff": float(diff.mean()) if len(diff) else np.nan,
        "median_diff": float(diff.median()) if len(diff) else np.nan,
        "share_a_gt_b": float((diff > 0).mean()) if len(diff) else np.nan,
        "t_stat": np.nan,
        "t_pvalue": np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_pvalue": np.nan,
    }
    if len(diff) < 8:
        return out
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan
    if se and se > 0:
        t_stat = float(diff.mean() / se)
        out["t_stat"] = t_stat
        try:
            from scipy import stats

            out["t_pvalue"] = float(stats.ttest_rel(aligned["a"], aligned["b"]).pvalue)
            if float(np.abs(diff).sum()) > 0:
                w = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                out["wilcoxon_stat"] = float(w.statistic)
                out["wilcoxon_pvalue"] = float(w.pvalue)
        except Exception:
            from math import erf, sqrt

            # two-sided normal approximation if scipy missing
            out["t_pvalue"] = float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0)))))
    return out


def phase_a_daily(scored: pd.DataFrame, spy_fwd: dict[pd.Timestamp, float]) -> pd.DataFrame:
    rows = []
    for dt, g in scored.groupby("datetime"):
        sub = g.dropna(subset=["score", "label"]).copy()
        if len(sub) < 50:
            continue
        sub["rank"] = sub["score"].rank(method="first", ascending=False)
        univ = float(sub["label"].mean())
        spy = spy_fwd.get(pd.Timestamp(dt).normalize(), np.nan)
        ric = float(sub["score"].corr(sub["label"], method="spearman"))
        bucket_means = {}
        for bid, lo, hi in BUCKETS:
            b = sub[(sub["rank"] >= lo) & (sub["rank"] <= hi)]
            if b.empty:
                continue
            mean_fwd = float(b["label"].mean())
            bucket_means[bid] = mean_fwd
            q = b["score"].quantile
            rows.append(
                {
                    "datetime": pd.Timestamp(dt).normalize(),
                    "bucket": bid,
                    "rank_lo": lo,
                    "rank_hi": hi,
                    "n": int(len(b)),
                    "mean_fwd": mean_fwd,
                    "median_fwd": float(b["label"].median()),
                    "hit_rate": float((b["label"] > 0).mean()),
                    "excess_univ": mean_fwd - univ,
                    "excess_spy": (mean_fwd - spy) if pd.notna(spy) else np.nan,
                    "hit_rate_excess_univ": float(((b["label"] - univ) > 0).mean()),
                    "score_mean": float(b["score"].mean()),
                    "score_std": float(b["score"].std(ddof=0)),
                    "score_iqr": float(q(0.75) - q(0.25)),
                    "score_min": float(b["score"].min()),
                    "score_max": float(b["score"].max()),
                    "univ_mean_fwd": univ,
                    "spy_fwd": spy,
                    "rank_ic": ric,
                    "n_universe": int(len(sub)),
                    "status": STATUS_LABEL,
                }
            )
        mono = (
            bucket_means.get("B1", np.nan) >= bucket_means.get("B2", np.nan) >= bucket_means.get("B3", np.nan) >= bucket_means.get("B4", np.nan)
            if len(bucket_means) == 4
            else np.nan
        )
        b1s = sub[(sub["rank"] >= 1) & (sub["rank"] <= 5)]["score"].mean() if len(sub) else np.nan
        b4s = sub[(sub["rank"] >= 21) & (sub["rank"] <= 50)]["score"].mean() if len(sub) else np.nan
        spread = float(b1s - b4s) if pd.notna(b1s) and pd.notna(b4s) else np.nan
        for i in range(len(rows) - 1, -1, -1):
            if rows[i]["datetime"] != pd.Timestamp(dt).normalize():
                break
            rows[i]["monotonic_b1_ge_b2_ge_b3_ge_b4"] = bool(mono) if mono == mono else np.nan
            rows[i]["score_spread_b1_minus_b4"] = spread
    return pd.DataFrame(rows)


def phase_a_period_summary(daily: pd.DataFrame, period_id: str, start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    d = daily.copy()
    d["datetime"] = pd.to_datetime(d["datetime"]).dt.normalize()
    d = d[(d["datetime"] >= pd.Timestamp(start)) & (d["datetime"] <= pd.Timestamp(end))]
    rows = []
    tests: dict[str, Any] = {"period": period_id}
    if d.empty:
        return pd.DataFrame(), tests
    for bid, lo, hi in BUCKETS:
        b = d[d["bucket"] == bid]
        if b.empty:
            continue
        rows.append(
            {
                "period": period_id,
                "bucket": bid,
                "rank_lo": lo,
                "rank_hi": hi,
                "n_days": int(b["datetime"].nunique()),
                "mean_fwd": float(b["mean_fwd"].mean()),
                "median_fwd": float(b["median_fwd"].median()),
                "hit_rate": float(b["hit_rate"].mean()),
                "hit_rate_daily_positive": float((b["mean_fwd"] > 0).mean()),
                "volatility": float(b["mean_fwd"].std(ddof=1) * np.sqrt(252)) if len(b) > 1 else np.nan,
                "mean_excess_univ": float(b["excess_univ"].mean()),
                "mean_excess_spy": float(b["excess_spy"].mean()) if b["excess_spy"].notna().any() else np.nan,
                "score_mean": float(b["score_mean"].mean()),
                "score_std": float(b["score_std"].mean()),
                "score_iqr": float(b["score_iqr"].mean()),
                "score_min": float(b["score_min"].min()),
                "score_max": float(b["score_max"].max()),
                "score_spread_b1_minus_b4": float(b["score_spread_b1_minus_b4"].mean()),
                "mean_rank_ic": float(b["rank_ic"].mean()),
                "monotonic_day_share": float(b["monotonic_b1_ge_b2_ge_b3_ge_b4"].mean()),
                "status": STATUS_LABEL,
            }
        )
    wide = d.pivot_table(index="datetime", columns="bucket", values="mean_fwd", aggfunc="mean")
    for left, right, key in (("B1", "B2", "b1_vs_b2"), ("B2", "B3", "b2_vs_b3"), ("B3", "B4", "b3_vs_b4")):
        if left in wide.columns and right in wide.columns:
            tests[key] = paired_tests(wide[left], wide[right])
    if {"B1", "B2", "B3", "B4"}.issubset(wide.columns):
        tests["mean_order"] = [float(wide[c].mean()) for c in ("B1", "B2", "B3", "B4")]
        tests["monotonic_means"] = bool(wide["B1"].mean() >= wide["B2"].mean() >= wide["B3"].mean() >= wide["B4"].mean())
        tests["monotonic_day_share"] = float(
            ((wide["B1"] >= wide["B2"]) & (wide["B2"] >= wide["B3"]) & (wide["B3"] >= wide["B4"])).mean()
        )
    return pd.DataFrame(rows), tests


def contribution_stats(weights: pd.DataFrame) -> dict[str, Any]:
    if weights is None or weights.empty:
        return {
            "top1_share_signed": np.nan,
            "top3_share_signed": np.nan,
            "top5_share_signed": np.nan,
            "top1_share_positive": np.nan,
            "top3_share_positive": np.nan,
            "top5_share_positive": np.nan,
        }
    w = weights.copy()
    w["trade_date"] = pd.to_datetime(w["trade_date"]).dt.normalize()
    w = w.sort_values(["instrument", "trade_date"])
    w["prev_price"] = w.groupby("instrument")["price"].shift(1)
    w["prev_weight"] = w.groupby("instrument")["weight"].shift(1)
    w["prev_date"] = w.groupby("instrument")["trade_date"].shift(1)
    cal = sorted(w["trade_date"].unique())
    loc = {d: i for i, d in enumerate(cal)}
    w["date_gap"] = w["trade_date"].map(loc) - w["prev_date"].map(loc)
    valid = w[(w["date_gap"] == 1) & w["prev_price"].gt(0) & w["price"].notna() & w["prev_weight"].notna()].copy()
    if valid.empty:
        return {
            "top1_share_signed": np.nan,
            "top3_share_signed": np.nan,
            "top5_share_signed": np.nan,
            "top1_share_positive": np.nan,
            "top3_share_positive": np.nan,
            "top5_share_positive": np.nan,
        }
    valid["stock_ret"] = valid["price"] / valid["prev_price"] - 1.0
    valid["daily_contribution"] = valid["prev_weight"] * valid["stock_ret"]
    stock = valid.groupby("instrument")["daily_contribution"].sum()
    signed_total = float(stock.sum())
    pos = stock[stock > 0]
    out = {}
    for k in (1, 3, 5):
        if signed_total != 0:
            out[f"top{k}_share_signed"] = float(stock.nlargest(k).sum() / signed_total)
        else:
            out[f"top{k}_share_signed"] = np.nan
        if len(pos) and float(pos.sum()) > 0:
            out[f"top{k}_share_positive"] = float(pos.nlargest(min(k, len(pos))).sum() / pos.sum())
        else:
            out[f"top{k}_share_positive"] = np.nan
    return out


def concentration_from_weights(weights: pd.DataFrame) -> dict[str, Any]:
    if weights is None or weights.empty:
        return {
            "mean_n_holdings": np.nan,
            "mean_position_weight": np.nan,
            "max_position_weight": np.nan,
            "mean_daily_max_weight": np.nan,
            "hhi": np.nan,
            "effective_n": np.nan,
            "mean_cash_weight": np.nan,
        }
    w = weights.copy()
    w["trade_date"] = pd.to_datetime(w["trade_date"]).dt.normalize()
    g = w.groupby("trade_date")
    hhi = g["weight"].apply(lambda s: float((s.astype(float) ** 2).sum()))
    mean_w = g["weight"].mean()
    max_w = g["weight"].max()
    n_hold = g["n_holdings"].first() if "n_holdings" in w.columns else g["instrument"].nunique()
    cash = g["cash_weight"].first() if "cash_weight" in w.columns else pd.Series(dtype=float)
    hhi = hhi.replace(0, np.nan)
    return {
        "mean_n_holdings": float(n_hold.mean()),
        "mean_position_weight": float(mean_w.mean()),
        "max_position_weight": float(max_w.max()),
        "mean_daily_max_weight": float(max_w.mean()),
        "hhi": float(hhi.mean()),
        "effective_n": float((1.0 / hhi).mean()) if hhi.notna().any() else np.nan,
        "mean_cash_weight": float(cash.mean()) if len(cash) else np.nan,
        **contribution_stats(w),
    }


def winner_metrics(weights: pd.DataFrame, scored: pd.DataFrame, dates: list[pd.Timestamp]) -> dict[str, Any]:
    if scored is None or scored.empty:
        return {
            "winner_capture": np.nan,
            "benchmark_winner_exposure": np.nan,
            "n_period_winners": 0,
        }
    s = scored.copy()
    s["datetime"] = pd.to_datetime(s["datetime"]).dt.normalize()
    if dates:
        dmin, dmax = min(dates), max(dates)
        s = s[(s["datetime"] >= dmin) & (s["datetime"] <= dmax)]
    else:
        s = s.iloc[0:0]
    if s.empty:
        return {"winner_capture": np.nan, "benchmark_winner_exposure": np.nan, "n_period_winners": 0}
    stock = (
        s.groupby("instrument")
        .agg(mean_fwd=("label", "mean"), n_days=("datetime", "nunique"))
        .reset_index()
    )
    min_days = max(20, int(0.5 * s["datetime"].nunique()))
    stock = stock[stock["n_days"] >= min_days]
    if stock.empty:
        return {"winner_capture": np.nan, "benchmark_winner_exposure": np.nan, "n_period_winners": 0}
    stock["pct"] = stock["mean_fwd"].rank(pct=True)
    winners = stock[stock["pct"] >= 0.90].copy()
    held = set()
    wmean: dict[str, float] = {}
    if weights is not None and not weights.empty:
        ww = weights.copy()
        ww["trade_date"] = pd.to_datetime(ww["trade_date"]).dt.normalize()
        ww["instrument"] = ww["instrument"].astype(str)
        n_days = ww["trade_date"].nunique()
        held = set(ww["instrument"])
        wmean = (ww.groupby("instrument")["weight"].sum() / max(n_days, 1)).to_dict()
    winners["ever_held"] = winners["instrument"].isin(held)
    winners["avg_weight"] = winners["instrument"].map(lambda x: float(wmean.get(x, 0.0)))
    return {
        "winner_capture": float(winners["ever_held"].mean()) if len(winners) else np.nan,
        "benchmark_winner_exposure": float(winners["avg_weight"].mean()) if len(winners) else np.nan,
        "n_period_winners": int(len(winners)),
    }


def run_arm(
    r1r2p,
    pred: pd.DataFrame,
    start: str,
    end: str,
    *,
    arm_id: str,
    window: str,
    phase: str,
    strategy_class: str,
    strategy_module: str,
    extra_kwargs: dict[str, Any] | None = None,
    collect_audit: bool = False,
) -> dict[str, Any]:
    if collect_audit:
        reset_audit_log()
    bt = run_qlib_backtest(
        r1r2p,
        rpa,
        pred,
        start,
        end,
        strategy_class=strategy_class,
        strategy_module=strategy_module,
        extra_kwargs=extra_kwargs,
    )
    daily = report_to_daily(bt["report"], arm_id, window, phase)
    weights = stock_weights_from_positions(bt["result"]["positions"], arm_id, window, phase)
    holdings = weights[["trade_date", "instrument", "arm_id", "window", "phase", "status"]].copy() if not weights.empty else pd.DataFrame()
    audit = list(AUDIT_LOG) if collect_audit else []
    return {"daily": daily, "holdings": holdings, "weights": weights, "audit": audit, "bt": bt}


def build_inputs() -> dict[str, Any]:
    print("Building 2018-2019 / 2020-2023 W1 scores...", flush=True)
    rpa.init_qlib({"provider_uri": str(PROJECT_ROOT / "staging/qlib_data"), "region": "us"})
    r1r2p = load_r1r2p_module(PROJECT_ROOT)
    hmm_online, _states, label_map, _n = frozen_hmm_timeline()
    panel = load_common_panel()
    panel = attach_states_to_panel(panel, hmm_online, label_map)
    flip_w1 = infer_score_sign(panel, "W1")
    pred_valid = build_scheme_pred(panel, "W1", flip=flip_w1, split="valid")
    pred_test = build_scheme_pred(panel, "W1", flip=flip_w1, split="test")
    scored_valid = score_w1(panel, flip=flip_w1, split="valid")
    scored_test = score_w1(panel, flip=flip_w1, split="test")

    print("Building 2024-2025 W1 scores...", flush=True)
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(PROJECT_ROOT / "staging/qlib_data"), region=REG_US, kernels=1)
    panel_h = hr.load_holdout_panel()
    hmm2 = hr.build_2state_hmm()
    label_map_h = hr.load_2state_label_map()
    flip_h = hr.infer_flip(panel_h, "W1", hmm2, label_map_h)
    pred_holdout, sub_h = hr.build_holdout_pred(panel_h, "W1", hmm2, label_map_h, flip=flip_h)
    holdout_end = min(pd.Timestamp(HOLDOUT_END), pred_holdout.index.get_level_values("datetime").max())
    scored_holdout = sub_h.copy()
    scored_holdout["instrument"] = scored_holdout["instrument"].astype(str)
    scored_holdout["datetime"] = pd.to_datetime(scored_holdout["datetime"]).dt.normalize()
    return {
        "r1r2p": r1r2p,
        "flip_w1": bool(flip_w1),
        "flip_h0": bool(flip_h),
        "pred_valid": pred_valid,
        "pred_test": pred_test,
        "pred_holdout": pred_holdout,
        "scored_valid": scored_valid,
        "scored_test": scored_test,
        "scored_holdout": scored_holdout,
        "holdout_end": holdout_end.strftime("%Y-%m-%d"),
    }


def pred_for_window(inp: dict[str, Any], window_id: str) -> pd.DataFrame:
    if window_id == "2018-2019":
        return inp["pred_valid"]
    if window_id == "2020-2023":
        return inp["pred_test"]
    if window_id == "2024-2025":
        return inp["pred_holdout"]
    raise KeyError(window_id)


def scored_for_window(inp: dict[str, Any], window_id: str) -> pd.DataFrame:
    if window_id == "2018-2019":
        return inp["scored_valid"]
    if window_id == "2020-2023":
        return inp["scored_test"]
    if window_id == "2024-2025":
        return inp["scored_holdout"]
    raise KeyError(window_id)


def write_fail_report(payload: dict[str, Any], rows: pd.DataFrame) -> str:
    lines = [
        "# TopK breadth experiment report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        f"Status: `{STATUS_LABEL}`",
        "",
        "**K20 parity FAIL. Experiment stopped. Phase A / B / C were not run.**",
        "",
        "2024–2025 is not a new untouched holdout.",
        "",
        f"- T0 (2020–2023): **{'PASS' if payload['t0']['pass'] else 'FAIL'}**",
        f"- H0 (2024–2025): **{'PASS' if payload['h0']['pass'] else 'FAIL'}**",
        f"- T0 max daily-return |diff|: `{payload['t0']['max_daily_return_abs_diff']}`",
        f"- H0 max daily-return |diff|: `{payload['h0']['max_daily_return_abs_diff']}`",
        "",
        "## Daily series checks",
        "",
        md_table(rows[rows["kind"] == "daily_series"] if not rows.empty else rows),
        "",
    ]
    return "\n".join(lines) + "\n"


def get_row(df: pd.DataFrame, **kwargs) -> pd.Series | None:
    q = df
    for k, v in kwargs.items():
        q = q[q[k] == v]
    if q.empty:
        return None
    return q.iloc[0]


def write_full_report(
    *,
    payload: dict[str, Any],
    parity_rows: pd.DataFrame,
    phase_a_summary: pd.DataFrame,
    phase_a_tests: dict[str, Any],
    period_summary: pd.DataFrame,
    conc_summary: pd.DataFrame,
    resp_summary: pd.DataFrame,
    k5_audit: pd.DataFrame,
    runtimes: dict[str, float],
    verdict: dict[str, Any],
) -> str:
    def rget(arm: str, period: str, col: str, table: pd.DataFrame = period_summary):
        row = get_row(table, arm_id=arm, period=period)
        return None if row is None else row.get(col)

    lines = [
        "# TopK Portfolio Breadth Experiment Report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        f"**Status: `{STATUS_LABEL}`**",
        "",
        "Frozen H0/T0 未被改写。本实验不是新的 primary specification。",
        "**2024–2025 不得称为新的 untouched holdout。** Holdout 已打开；全部结果均为 post-hoc exploratory / robustness。",
        "",
        "研究问题：frozen `0.7 D2 + 0.3 F1C` 的有效信息是否主要集中在最前排名，以至于 Top20 稀释了预测强度？",
        "",
        "两个机制必须分开：",
        "",
        "1. **signal breadth / concentration**（Phase A；Phase B 在约 10% 日替换容量下比较 TopK）",
        "2. **portfolio turnover / responsiveness**（Phase C；不声称 isolate TopK）",
        "",
        "未运行 score-threshold experiment。未搜索参数。未加入 W1/W2/W3 组合加权、MA120、sentiment 或 HMM 动态权重。",
        "",
        "## Runtime",
        "",
        md_table(pd.DataFrame([{"stage": k, "seconds": round(v, 1)} for k, v in runtimes.items()])),
        "",
        "## 1. K20 parity gate",
        "",
        f"- T0 (2020–2023 canonical): **{'PASS' if payload['t0']['pass'] else 'FAIL'}**, max |Δ daily return| = `{payload['t0']['max_daily_return_abs_diff']}` (threshold 1e-10)",
        f"- H0 (2024–2025 canonical): **{'PASS' if payload['h0']['pass'] else 'FAIL'}**, max |Δ daily return| = `{payload['h0']['max_daily_return_abs_diff']}` (threshold 1e-10)",
        f"- Flip flags: validation/test `{payload['flip_w1']}`, holdout `{payload['flip_h0']}`",
        "",
        md_table(
            parity_rows[parity_rows["kind"] == "daily_series"][
                ["window", "metric", "pass", "max_abs_diff", "mean_abs_diff", "n_matched"]
            ]
            if not parity_rows.empty
            else parity_rows
        ),
        "",
        "## 2. Phase A — rank bucket diagnostic",
        "",
        "路径无关。每日用与 frozen H0 相同的 W1 `shift=1` 分数截面排名；前瞻收益是同一分数日的 LABEL0 = `close_{t+2}/close_{t+1}-1`。禁止把已实现的 `close_t/close_{t-1}` 配进当日排名。",
        "",
        "桶内等权，不是 Qlib 组合。",
        "",
        "### Period × bucket",
        "",
        md_table(
            phase_a_summary,
            [
                "period",
                "bucket",
                "n_days",
                "mean_fwd",
                "median_fwd",
                "hit_rate",
                "volatility",
                "mean_excess_univ",
                "mean_excess_spy",
                "score_mean",
                "monotonic_day_share",
            ],
        ),
        "",
        "### Rank-return tests (daily equal-weight bucket means, paired)",
        "",
    ]
    test_rows = []
    for period, blob in phase_a_tests.items():
        for key in ("b1_vs_b2", "b2_vs_b3", "b3_vs_b4"):
            t = blob.get(key, {})
            test_rows.append(
                {
                    "period": period,
                    "contrast": key,
                    "n": t.get("n"),
                    "mean_diff": t.get("mean_diff"),
                    "share_left_gt_right": t.get("share_a_gt_b"),
                    "t_pvalue": t.get("t_pvalue"),
                    "wilcoxon_pvalue": t.get("wilcoxon_pvalue"),
                    "monotonic_means": blob.get("monotonic_means"),
                    "monotonic_day_share": blob.get("monotonic_day_share"),
                }
            )
    lines += [md_table(pd.DataFrame(test_rows)), ""]

    lines += [
        "## 3. Phase B — turnover-controlled TopK (~10% replacement capacity)",
        "",
        "| Arm | TopK | n_drop rule | capacity |",
        "|---|---:|---|---|",
        "| K20 | 20 | 2 every day | 10% |",
        "| K10 | 10 | 1 every day | 10% |",
        "| K5 | 5 | trade_step even → 0, odd → 1 | 10% |",
        "",
        "K5 日程只依赖 Qlib `trade_step`，不随市场状态改变。`hold_thresh=1` 仍可阻断卖出，因此 intended n_drop 与成交卖单可能不一致。",
        "",
        "**实现细节：** Qlib `TopkDropoutStrategy` 的 `get_last_n(li, 0)` 等于 `list(li)[-0:]`，即整张列表，原生 `n_drop=0` 会清空持仓。K5 在已有持仓的 even `trade_step` 上改为直接返回空决策（真正 hold）。这是对冻结日程的实现修正，不是事后搜参。",
        "",
        "窗口独立初始化（2019 不带入 2020，2023 不带入 2024）。2020–2025 是独立窗口日收益拼接，**不是**连续 NAV。",
        "",
        "### Phase B portfolio metrics",
        "",
        md_table(
            period_summary[period_summary["phase"] == "B"],
            [
                "period",
                "arm_id",
                "n_days",
                "arr",
                "benchmark_arr",
                "excess_arr",
                "cumulative_return",
                "sharpe",
                "volatility",
                "mdd",
                "turnover",
                "transaction_cost_total",
                "average_holding_period",
                "n_buys",
                "n_sells",
            ],
        ),
        "",
        "### Phase B concentration",
        "",
        md_table(
            conc_summary[conc_summary["phase"] == "B"],
            [
                "period",
                "arm_id",
                "mean_n_holdings",
                "mean_position_weight",
                "max_position_weight",
                "mean_daily_max_weight",
                "hhi",
                "effective_n",
                "top1_share_signed",
                "top3_share_signed",
                "top5_share_signed",
            ],
        ),
        "",
        "### Phase B responsiveness / selection",
        "",
        md_table(
            resp_summary[resp_summary["phase"] == "B"],
            [
                "period",
                "arm_id",
                "theoretical_topk_overlap_mean",
                "entry_delay_mean",
                "exit_delay_mean",
                "winner_capture",
                "benchmark_winner_exposure",
            ],
        ),
        "",
    ]
    if k5_audit is not None and not k5_audit.empty:
        lines += [
            "### K5 intended vs executed n_drop",
            "",
            md_table(
                k5_audit.groupby(["window", "n_drop_intended"], as_index=False).agg(
                    n_days=("trade_date", "count"),
                    mean_executed_sells=("n_drop_executed_sells", "mean"),
                    mean_buys=("n_buy_orders", "mean"),
                )
            ),
            "",
        ]

    lines += [
        "## 4. Phase C — practical faster update (NOT a pure TopK test)",
        "",
        "Phase C **必须与 Phase B 分开阅读**。P10 = Top10/`n_drop=2`（约 20%/日）；P5 = Top5/`n_drop=1`（约 20%/日）；P20 复用 K20。",
        "",
        "若 C 改善而 B 不改善，解释应偏向更快换仓，而不是更强的 top-rank 信号。",
        "",
        "### Phase C portfolio metrics",
        "",
        md_table(
            period_summary[period_summary["phase"] == "C"],
            [
                "period",
                "arm_id",
                "n_days",
                "arr",
                "benchmark_arr",
                "excess_arr",
                "cumulative_return",
                "sharpe",
                "volatility",
                "mdd",
                "turnover",
                "transaction_cost_total",
                "average_holding_period",
                "n_buys",
                "n_sells",
            ],
        ),
        "",
        "### Phase C concentration",
        "",
        md_table(
            conc_summary[conc_summary["phase"] == "C"],
            [
                "period",
                "arm_id",
                "mean_n_holdings",
                "mean_position_weight",
                "max_position_weight",
                "hhi",
                "effective_n",
                "top1_share_signed",
                "top3_share_signed",
                "top5_share_signed",
            ],
        ),
        "",
        "### Phase C responsiveness / selection",
        "",
        md_table(
            resp_summary[resp_summary["phase"] == "C"],
            [
                "period",
                "arm_id",
                "theoretical_topk_overlap_mean",
                "entry_delay_mean",
                "exit_delay_mean",
                "winner_capture",
                "benchmark_winner_exposure",
            ],
        ),
        "",
        "## 5. Answers A–J",
        "",
        f"**A. Rank 1–5 的 forward return 是否显著高于 6–10？** {verdict['answers']['A']}",
        "",
        f"**B. Rank 6–10 是否优于 11–20？** {verdict['answers']['B']}",
        "",
        f"**C. Rank 11–20 是否仍有经济价值？** {verdict['answers']['C']}",
        "",
        f"**D. 是否存在明显 rank-return monotonicity？** {verdict['answers']['D']}",
        "",
        f"**E. 控制约 10% replacement capacity 后，Top5 / Top10 / Top20 哪个收益最好？** {verdict['answers']['E']}",
        "",
        f"**F. 哪个 Sharpe 最好？** {verdict['answers']['F']}",
        "",
        f"**G. 哪个 MDD 最差？** {verdict['answers']['G']}",
        "",
        f"**H. TopK 缩小带来的收益是否主要来自 signal concentration，还是更高 responsiveness？** {verdict['answers']['H']}",
        "",
        f"**I. Phase C 允许更快换仓后，Top5/Top10 是否出现额外收益？** {verdict['answers']['I']}",
        "",
        f"**J. concentration risk 是否明显恶化？** {verdict['answers']['J']}",
        "",
        "## 6. Evidence verdict",
        "",
        f"**`{verdict['label']}`**",
        "",
        verdict["explanation"],
        "",
        "Winner 不以单一 ARR（尤其不是 2024–2025 ARR）决定。本 verdict 不把任何 arm 提升为新的 frozen H0。",
        "",
        "## 7. Constraints restated",
        "",
        "- `combined_score = 0.7 D2 + 0.3 F1C` after daily winsor-z (W1)",
        "- `hold_thresh = 1`",
        "- original H0 allocation mechanics (buy-only cash split; no daily equal-weight rebalance)",
        "- original `risk_degree = 0.95`",
        "- original 1bp open/close cost, `deal_price=close`, benchmark P84398",
        "- no W1/W2/W3 portfolio weighting, no MA120, no sentiment, no HMM dynamic weighting, no parameter search",
        "",
        "Score-threshold experiment: **not started**.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _sign_consistent(values: list[float], positive: bool = True) -> bool:
    xs = [v for v in values if v is not None and np.isfinite(v)]
    if len(xs) < 2:
        return False
    return all((v > 0) if positive else (v < 0) for v in xs)


def build_verdict(
    phase_a_summary: pd.DataFrame,
    phase_a_tests: dict[str, Any],
    period_summary: pd.DataFrame,
    conc_summary: pd.DataFrame,
    resp_summary: pd.DataFrame,
) -> dict[str, Any]:
    core_periods = ["2020-2023", "2024", "2025"]

    def a_metric(period: str, bucket: str, col: str) -> float:
        row = get_row(phase_a_summary, period=period, bucket=bucket)
        return float("nan") if row is None else float(row[col])

    def p_metric(arm: str, period: str, col: str, phase: str = "B") -> float:
        row = get_row(period_summary, arm_id=arm, period=period, phase=phase)
        return float("nan") if row is None else float(row[col])

    def c_metric(arm: str, period: str, col: str, phase: str = "B") -> float:
        row = get_row(conc_summary, arm_id=arm, period=period, phase=phase)
        return float("nan") if row is None else float(row[col])

    # A–D from Phase A, emphasizing 2020-2023 plus whether holdout agrees in sign
    tests_core = {p: phase_a_tests.get(p, {}) for p in ["2018-2019", "2020-2023", "2024", "2025", "2024-2025", "2020-2025"]}
    b1b2_p = []
    b2b3_p = []
    b1b2_diff = []
    b2b3_diff = []
    b3_excess = []
    mono_means = []
    mono_days = []
    for p in ["2018-2019", "2020-2023", "2024", "2025"]:
        t = tests_core.get(p, {})
        b1b2 = t.get("b1_vs_b2", {})
        b2b3 = t.get("b2_vs_b3", {})
        if b1b2:
            b1b2_p.append(b1b2.get("t_pvalue", np.nan))
            b1b2_diff.append(b1b2.get("mean_diff", np.nan))
        if b2b3:
            b2b3_p.append(b2b3.get("t_pvalue", np.nan))
            b2b3_diff.append(b2b3.get("mean_diff", np.nan))
        b3_excess.append(a_metric(p, "B3", "mean_excess_univ"))
        mono_means.append(bool(t.get("monotonic_means")))
        mono_days.append(t.get("monotonic_day_share", np.nan))

    def majority_sig(diffs, pvals, alpha=0.05) -> str:
        pos = [d for d in diffs if pd.notna(d)]
        sig = [pv for d, pv in zip(diffs, pvals) if pd.notna(d) and pd.notna(pv) and d > 0 and pv < alpha]
        if not pos:
            return "样本不足，无法判断。"
        p_txt = ", ".join("NA" if not pd.notna(pv) else f"{pv:.3f}" for pv in pvals)
        d_txt = ", ".join("NA" if not pd.notna(d) else f"{100*d:.4f}%" for d in diffs)
        if sum(d > 0 for d in pos) >= max(2, len(pos) - 1) and len(sig) >= 1:
            return (
                f"是。多数时期 B1 日均前瞻收益高于 B2（mean diff={d_txt}），"
                f"至少一段配对 t 检验 p<0.05（p={p_txt}）。"
            )
        if sum(d > 0 for d in pos) >= max(2, len(pos) - 1):
            return (
                f"方向上 B1 高于 B2，但统计显著性不稳定（p={p_txt}）。"
                "不能称为稳健显著。"
            )
        return f"否。B1 并不稳定高于 B2（mean diff={d_txt}；p={p_txt}）。"

    ans_a = majority_sig(b1b2_diff, b1b2_p)
    # reuse text pattern for B
    pos_b = [d for d in b2b3_diff if pd.notna(d)]
    sig_b = [pv for d, pv in zip(b2b3_diff, b2b3_p) if pd.notna(d) and pd.notna(pv) and d > 0 and pv < 0.05]
    if not pos_b:
        ans_b = "样本不足，无法判断。"
    elif sum(d > 0 for d in pos_b) >= max(2, len(pos_b) - 1) and len(sig_b) >= 1:
        ans_b = "是。多数时期 B2 日均前瞻收益高于 B3，且至少一段显著。"
    elif sum(d > 0 for d in pos_b) >= max(2, len(pos_b) - 1):
        ans_b = "方向上 B2 优于 B3，但显著性不稳定。"
    else:
        ans_b = "否。B2 并不稳定优于 B3。"

    b3_pos = [x for x in b3_excess if pd.notna(x)]
    if b3_pos and sum(x > 0 for x in b3_pos) >= max(2, len(b3_pos) - 1):
        ans_c = (
            f"有。Rank 11–20 相对当日 eligible universe 等权的超额均值在多数时期为正"
            f"（{', '.join('%.3f%%' % (100*x) for x in b3_pos)}）。经济幅度小于 B1，但不能把 11–20 写成纯噪声。"
        )
        b3_valuable = True
    elif b3_pos and np.mean(b3_pos) > 0:
        ans_c = "弱。11–20 平均超额略正，但跨期不稳定，经济价值有限。"
        b3_valuable = False
    else:
        ans_c = "弱/无。11–20 相对 universe 等权没有稳定正超额。"
        b3_valuable = False

    if sum(mono_means) >= 3 and np.nanmean(mono_days) >= 0.35:
        ans_d = (
            f"存在单调性，但是日度完全单调（B1≥B2≥B3≥B4）只占约 {100*np.nanmean(mono_days):.1f}% 的交易日；"
            "均值顺序在多数时期成立，属于中等而非极强的 rank–return 关系。"
        )
        mono_ok = True
    elif sum(mono_means) >= 2:
        ans_d = "部分单调：时期均值顺序经常成立，但日度单调比例不高，不能写成严格 staircase。"
        mono_ok = False
    else:
        ans_d = "没有明显的稳健 rank–return monotonicity。"
        mono_ok = False

    # Phase B return / sharpe / mdd on core periods
    def series(arm, col, phase="B"):
        return [p_metric(arm, p, col, phase) for p in core_periods]

    k20_ex = series("K20", "excess_arr")
    k10_ex = series("K10", "excess_arr")
    k5_ex = series("K5", "excess_arr")
    k20_arr = series("K20", "arr")
    k10_arr = series("K10", "arr")
    k5_arr = series("K5", "arr")
    k20_sh = series("K20", "sharpe")
    k10_sh = series("K10", "sharpe")
    k5_sh = series("K5", "sharpe")
    k20_mdd = series("K20", "mdd")
    k10_mdd = series("K10", "mdd")
    k5_mdd = series("K5", "mdd")

    def mean_finite(xs):
        v = [x for x in xs if pd.notna(x)]
        return float(np.mean(v)) if v else float("nan")

    # E: which return best at matched 10% — use excess ARR mean across core, require not a single-year pick
    ex_means = {"K20": mean_finite(k20_ex), "K10": mean_finite(k10_ex), "K5": mean_finite(k5_ex)}
    arr_means = {"K20": mean_finite(k20_arr), "K10": mean_finite(k10_arr), "K5": mean_finite(k5_arr)}
    best_ex = max(ex_means, key=lambda k: ex_means[k] if pd.notna(ex_means[k]) else -1e9)
    k20_2023 = p_metric("K20", "2020-2023", "excess_arr")
    k10_2023 = p_metric("K10", "2020-2023", "excess_arr")
    k5_2023 = p_metric("K5", "2020-2023", "excess_arr")
    # count periods arm has highest excess
    wins = {"K20": 0, "K10": 0, "K5": 0}
    for p in core_periods:
        vals = {"K20": p_metric("K20", p, "excess_arr"), "K10": p_metric("K10", p, "excess_arr"), "K5": p_metric("K5", p, "excess_arr")}
        if all(pd.isna(v) for v in vals.values()):
            continue
        winner = max(vals, key=lambda k: vals[k] if pd.notna(vals[k]) else -1e9)
        wins[winner] += 1
    ans_e = (
        f"Phase B 核心期平均 excess ARR：K20={fmt_pct(ex_means['K20'])}, K10={fmt_pct(ex_means['K10'])}, K5={fmt_pct(ex_means['K5'])}。"
        f"2020–2023 excess：K20={fmt_pct(k20_2023)}, K10={fmt_pct(k10_2023)}, K5={fmt_pct(k5_2023)}。"
        f"各时期 excess 最高次数：{wins}。"
        "不以 2024–2025 单独定胜负。"
    )

    sh_means = {"K20": mean_finite(k20_sh), "K10": mean_finite(k10_sh), "K5": mean_finite(k5_sh)}
    best_sh = max(sh_means, key=lambda k: sh_means[k] if pd.notna(sh_means[k]) else -1e9)
    ans_f = f"Phase B 核心期平均 Sharpe：K20={fmt_num(sh_means['K20'],3)}, K10={fmt_num(sh_means['K10'],3)}, K5={fmt_num(sh_means['K5'],3)}。最高为 **{best_sh}**。"

    # G worst MDD = most negative
    mdd_means = {"K20": mean_finite(k20_mdd), "K10": mean_finite(k10_mdd), "K5": mean_finite(k5_mdd)}
    worst_mdd = min(mdd_means, key=lambda k: mdd_means[k] if pd.notna(mdd_means[k]) else 1e9)
    ans_g = f"Phase B 核心期平均 MDD：K20={fmt_pct(mdd_means['K20'])}, K10={fmt_pct(mdd_means['K10'])}, K5={fmt_pct(mdd_means['K5'])}。最差（回撤最大）为 **{worst_mdd}**。"

    # H: B vs C
    p10_ex = series("P10", "excess_arr", "C")
    p5_ex = series("P5", "excess_arr", "C")
    k10_vs_k20 = mean_finite(k10_ex) - mean_finite(k20_ex)
    k5_vs_k20 = mean_finite(k5_ex) - mean_finite(k20_ex)
    p10_vs_k10 = mean_finite(p10_ex) - mean_finite(k10_ex)
    p5_vs_k5 = mean_finite(p5_ex) - mean_finite(k5_ex)
    if (k10_vs_k20 > 0 or k5_vs_k20 > 0) and p10_vs_k10 <= 0 and p5_vs_k5 <= 0:
        ans_h = (
            "更接近 **signal concentration**：Phase B 在匹配约 10% 替换后已改善，而 Phase C 更快换仓没有额外稳定增益。"
        )
        mech = "concentration"
    elif (k10_vs_k20 <= 0 and k5_vs_k20 <= 0) and (p10_vs_k10 > 0 or p5_vs_k5 > 0):
        ans_h = (
            "更接近 **更高 responsiveness**：匹配 turnover 的 Phase B 没有稳定优于 K20，而 Phase C 加快换仓后才出现额外收益。"
        )
        mech = "responsiveness"
    elif (k10_vs_k20 > 0 or k5_vs_k20 > 0) and (p10_vs_k10 > 0 or p5_vs_k5 > 0):
        ans_h = "两者都有：Phase B 显示一定的 top-rank 集中效应，Phase C 加快换仓还有增量。不能把收益全部归给其中一个机制。"
        mech = "mixed"
    else:
        ans_h = "没有清楚的 TopK 缩小增益。K10/K5 在匹配 turnover 下未稳定优于 K20，加快换仓也没有稳定额外收益。"
        mech = "none"

    ans_i = (
        f"Phase C 相对同 TopK 的 Phase B：P10−K10 核心期平均 excess ARR = {fmt_pct(p10_vs_k10)}；"
        f"P5−K5 = {fmt_pct(p5_vs_k5)}。"
        + ("出现额外收益。" if (p10_vs_k10 > 0 or p5_vs_k5 > 0) else "没有稳定的额外收益。")
    )

    k20_maxw = [c_metric("K20", p, "max_position_weight") for p in core_periods]
    k10_maxw = [c_metric("K10", p, "max_position_weight") for p in core_periods]
    k5_maxw = [c_metric("K5", p, "max_position_weight") for p in core_periods]
    k20_hhi = [c_metric("K20", p, "hhi") for p in core_periods]
    k5_hhi = [c_metric("K5", p, "hhi") for p in core_periods]
    k20_en = [c_metric("K20", p, "effective_n") for p in core_periods]
    k5_en = [c_metric("K5", p, "effective_n") for p in core_periods]
    conc_worse = mean_finite(k5_maxw) > 1.5 * mean_finite(k20_maxw) or mean_finite(k5_hhi) > 2.0 * mean_finite(k20_hhi)
    ans_j = (
        f"K20/K10/K5 核心期平均 max weight = {fmt_pct(mean_finite(k20_maxw))} / {fmt_pct(mean_finite(k10_maxw))} / {fmt_pct(mean_finite(k5_maxw))}；"
        f"HHI = {fmt_num(mean_finite(k20_hhi),4)} / {fmt_num(mean_finite([c_metric('K10', p, 'hhi') for p in core_periods]),4)} / {fmt_num(mean_finite(k5_hhi),4)}；"
        f"effective N = {fmt_num(mean_finite(k20_en),2)} / {fmt_num(mean_finite([c_metric('K10', p, 'effective_n') for p in core_periods]),2)} / {fmt_num(mean_finite(k5_en),2)}。"
        + ("**是，K5 的集中度明显恶化。**" if conc_worse else "有上升，但按 1.5× max-weight / 2× HHI 门槛，尚未写成灾难性恶化。")
    )

    # Verdict
    k5_better_ret = wins["K5"] >= 2 and k5_vs_k20 > 0
    k10_better_ret = wins["K10"] >= 2 and k10_vs_k20 > 0
    k5_better_sh = sh_means["K5"] >= sh_means["K20"] if all(pd.notna(x) for x in (sh_means["K5"], sh_means["K20"])) else False
    k10_better_sh = sh_means["K10"] >= sh_means["K20"] if all(pd.notna(x) for x in (sh_means["K10"], sh_means["K20"])) else False
    k5_mdd_ok = (not pd.isna(mdd_means["K5"])) and (mdd_means["K5"] >= mdd_means["K20"] - 0.05)
    k10_mdd_ok = (not pd.isna(mdd_means["K10"])) and (mdd_means["K10"] >= mdd_means["K20"] - 0.03)
    test_pref_k5 = (not pd.isna(k5_2023)) and (not pd.isna(k20_2023)) and k5_2023 > k20_2023
    test_pref_k10 = (not pd.isna(k10_2023)) and (not pd.isna(k20_2023)) and k10_2023 > k20_2023

    label = "NO_CLEAR_WINNER"
    if (
        k5_better_ret
        and k5_better_sh
        and k5_mdd_ok
        and not conc_worse
        and mono_ok
        and test_pref_k5
        and mech in {"concentration", "mixed"}
    ):
        label = "TOP5_SUPPORTED"
    elif (
        k10_better_ret
        and k10_better_sh
        and k10_mdd_ok
        and test_pref_k10
        and mech in {"concentration", "mixed"}
    ):
        label = "TOP10_SUPPORTED"
    elif (not k5_better_ret and not k10_better_ret) or (b3_valuable and mech in {"none", "responsiveness"}):
        if b3_valuable or (wins["K20"] >= wins["K10"] and wins["K20"] >= wins["K5"]):
            label = "TOP20_SUPPORTED"
        else:
            label = "NO_CLEAR_WINNER"
    else:
        label = "NO_CLEAR_WINNER"

    explanation = (
        f"Return：Phase B 核心期平均 excess ARR K20/K10/K5 = {fmt_pct(ex_means['K20'])} / {fmt_pct(ex_means['K10'])} / {fmt_pct(ex_means['K5'])}；"
        f"时期胜场 {wins}。2020–2023 仍是未打开 holdout 前的主对照窗口。\n\n"
        f"Risk：平均 Sharpe K20/K10/K5 = {fmt_num(sh_means['K20'],3)} / {fmt_num(sh_means['K10'],3)} / {fmt_num(sh_means['K5'],3)}；"
        f"平均 MDD {fmt_pct(mdd_means['K20'])} / {fmt_pct(mdd_means['K10'])} / {fmt_pct(mdd_means['K5'])}。\n\n"
        f"Turnover：Phase B 设计为约 10% 日替换；Phase C 把 Top10/Top5 提到约 20%。机制判断：{mech}。\n\n"
        f"Concentration：K5 max weight / HHI / effective N 相对 K20 上升"
        f"{'，已明显恶化' if conc_worse else '，幅度需在报告表中阅读，但未单独推翻 K20'}。\n\n"
        f"Cross-period stability：同时看 2018–2019（若有）、2020–2023、2024、2025。2024–2025 只作 opened-holdout robustness，"
        f"不能当作新的 untouched holdout，也不能用它改写 frozen H0。\n\n"
        f"综合：`{label}`。"
    )
    return {
        "label": label,
        "explanation": explanation,
        "mechanism": mech,
        "answers": {
            "A": ans_a,
            "B": ans_b,
            "C": ans_c,
            "D": ans_d,
            "E": ans_e,
            "F": ans_f,
            "G": ans_g,
            "H": ans_h,
            "I": ans_i,
            "J": ans_j,
        },
        "phase_b_excess_means": ex_means,
        "phase_b_sharpe_means": sh_means,
        "phase_b_mdd_means": mdd_means,
        "period_excess_wins": wins,
    }


def period_bundle(
    daily_by_arm: dict[str, pd.DataFrame],
    hold_by_arm: dict[str, pd.DataFrame],
    weights_by_arm: dict[str, pd.DataFrame],
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pid = spec["id"]
    if spec.get("concat"):
        daily = concat_independent([slice_daily(daily_by_arm[w], spec["start"], spec["end"]) for w in spec["concat_windows"]])
        hold = concat_independent([slice_holdings(hold_by_arm[w], spec["start"], spec["end"]) for w in spec["concat_windows"]])
        weights = concat_independent([slice_holdings(weights_by_arm[w], spec["start"], spec["end"]) for w in spec["concat_windows"]])
    else:
        src = spec["source_window"]
        daily = slice_daily(daily_by_arm.get(src, pd.DataFrame()), spec["start"], spec["end"])
        hold = slice_holdings(hold_by_arm.get(src, pd.DataFrame()), spec["start"], spec["end"])
        weights = slice_holdings(weights_by_arm.get(src, pd.DataFrame()), spec["start"], spec["end"])
    return daily, hold, weights


def main() -> None:
    t_all = time.perf_counter()
    runtimes: dict[str, float] = {}
    precommit = json.loads(PRECOMMIT_PATH.read_text(encoding="utf-8"))
    if "POST_HOC" not in str(precommit.get("status", "")):
        raise RuntimeError("precommit status is not POST_HOC")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    inp = build_inputs()
    runtimes["build_scores"] = time.perf_counter() - t0
    r1r2p = inp["r1r2p"]

    # ---- K20 parity (native TopkDropout only) ----
    print("Replaying K20 / T0 2020-2023...", flush=True)
    t0 = time.perf_counter()
    k20_t0 = run_arm(
        r1r2p,
        inp["pred_test"],
        TEST_START,
        TEST_END,
        arm_id="K20",
        window="2020-2023",
        phase="B",
        strategy_class="TopkDropoutStrategy",
        strategy_module="qlib.contrib.strategy",
        extra_kwargs={"topk": 20, "n_drop": 2, "hold_thresh": 1, "risk_degree": 0.95},
    )
    frozen_t0 = pd.read_parquet(T0_FROZEN)
    frozen_t0 = frozen_t0[frozen_t0["arm_id"] == "T0"].copy()
    t0_result = compare_window(
        window="2020-2023",
        arm_id="T0",
        replay_daily=k20_t0["daily"],
        frozen_daily=frozen_t0,
        replay_holdings=k20_t0["holdings"],
        frozen_holdings=None,
        expected_days=1006,
    )
    print(
        f"T0 parity: {'PASS' if t0_result['pass'] else 'FAIL'} max_return_diff={t0_result['max_daily_return_abs_diff']}",
        flush=True,
    )

    print("Replaying K20 / H0 2024-2025...", flush=True)
    k20_h0 = run_arm(
        r1r2p,
        inp["pred_holdout"],
        HOLDOUT_START,
        inp["holdout_end"],
        arm_id="K20",
        window="2024-2025",
        phase="B",
        strategy_class="TopkDropoutStrategy",
        strategy_module="qlib.contrib.strategy",
        extra_kwargs={"topk": 20, "n_drop": 2, "hold_thresh": 1, "risk_degree": 0.95},
    )
    frozen_h0 = pd.read_parquet(H0_FROZEN)
    if "arm_id" in frozen_h0.columns:
        frozen_h0 = frozen_h0[frozen_h0["arm_id"] == "H0"].copy()
    frozen_h0_hold = pd.read_csv(H0_HOLDINGS)
    h0_result = compare_window(
        window="2024-2025",
        arm_id="H0",
        replay_daily=k20_h0["daily"],
        frozen_daily=frozen_h0,
        replay_holdings=k20_h0["holdings"],
        frozen_holdings=frozen_h0_hold,
        expected_days=500,
    )
    print(
        f"H0 parity: {'PASS' if h0_result['pass'] else 'FAIL'} max_return_diff={h0_result['max_daily_return_abs_diff']}",
        flush=True,
    )
    runtimes["k20_parity"] = time.perf_counter() - t0

    parity_rows = pd.DataFrame(flatten_checks(t0_result) + flatten_checks(h0_result))
    all_pass = bool(t0_result["pass"] and h0_result["pass"])
    parity_payload = {
        "generated_at": _utc_now(),
        "experiment_id": "POSTHOC-TOPK-BREADTH",
        "status": STATUS_LABEL,
        "strategy": {"class": "TopkDropoutStrategy", "topk": 20, "n_drop": 2, "hold_thresh": 1, "risk_degree": 0.95},
        "flip_w1": inp["flip_w1"],
        "flip_h0": inp["flip_h0"],
        "t0": t0_result,
        "h0": h0_result,
        "t0_pass": bool(t0_result["pass"]),
        "h0_pass": bool(h0_result["pass"]),
        "max_daily_return_abs_diff_t0": t0_result["max_daily_return_abs_diff"],
        "max_daily_return_abs_diff_h0": h0_result["max_daily_return_abs_diff"],
        "threshold": DAILY_ABS_TOL,
        "runtime_seconds": runtimes["k20_parity"],
    }
    parity_rows.to_csv(OUT_ROOT / "k20_parity_check.csv", index=False)
    (OUT_ROOT / "k20_parity_check.json").write_text(json.dumps(json_safe(parity_payload), indent=2), encoding="utf-8")

    if not all_pass:
        report = write_fail_report(parity_payload, parity_rows)
        (REPORT_ROOT / "topk_breadth_experiment_report.md").write_text(report, encoding="utf-8")
        print(json.dumps({"t0_pass": False if not t0_result["pass"] else True, "h0_pass": bool(h0_result["pass"]), "stopped": True}, indent=2))
        raise SystemExit(2)

    # ---- Phase A ----
    print("Running Phase A rank-bucket diagnostic...", flush=True)
    t0 = time.perf_counter()
    spy_fwd = spy_label0_map(SPY_CSV_PATH)
    scored_all = pd.concat(
        [
            inp["scored_valid"][["datetime", "instrument", "score", "label"]].assign(split="valid"),
            inp["scored_test"][["datetime", "instrument", "score", "label"]].assign(split="test"),
            inp["scored_holdout"][["datetime", "instrument", "score", "label"]].assign(split="holdout"),
        ],
        ignore_index=True,
    )
    daily_a = phase_a_daily(scored_all, spy_fwd)
    a_rows = []
    a_tests: dict[str, Any] = {}
    for spec in SLICE_PERIODS:
        part, tests = phase_a_period_summary(daily_a, spec["id"], spec["start"], spec["end"])
        if not part.empty:
            a_rows.append(part)
        a_tests[spec["id"]] = tests
    phase_a_summary = pd.concat(a_rows, ignore_index=True) if a_rows else pd.DataFrame()
    save_frame(daily_a, "rank_bucket_daily")
    phase_a_summary.to_csv(OUT_ROOT / "rank_bucket_diagnostic.csv", index=False)
    (OUT_ROOT / "rank_bucket_diagnostic.json").write_text(
        json.dumps(
            json_safe(
                {
                    "status": STATUS_LABEL,
                    "timing": "rank on score_t; forward return = panel LABEL0 at same t",
                    "buckets": {b[0]: f"{b[1]}-{b[2]}" for b in BUCKETS},
                    "summary": phase_a_summary.to_dict(orient="records"),
                    "tests": a_tests,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    runtimes["phase_a"] = time.perf_counter() - t0

    # ---- Phase B / C backtests ----
    results: dict[str, dict[str, dict[str, Any]]] = {}
    t_bc = time.perf_counter()
    t_b = time.perf_counter()
    for w in WINDOWS:
        wid = w["id"]
        pred = pred_for_window(inp, wid)
        start, end = w["start"], (inp["holdout_end"] if wid == "2024-2025" else w["end"])
        results[wid] = {}
        print(f"=== Window {wid} {start} -> {end} ===", flush=True)

        if wid == "2020-2023":
            results[wid]["K20"] = k20_t0
        elif wid == "2024-2025":
            results[wid]["K20"] = k20_h0
        else:
            print("  K20 ...", flush=True)
            results[wid]["K20"] = run_arm(
                r1r2p,
                pred,
                start,
                end,
                arm_id="K20",
                window=wid,
                phase="B",
                strategy_class="TopkDropoutStrategy",
                strategy_module="qlib.contrib.strategy",
                extra_kwargs={"topk": 20, "n_drop": 2, "hold_thresh": 1, "risk_degree": 0.95},
            )

        print("  K10 ...", flush=True)
        results[wid]["K10"] = run_arm(
            r1r2p,
            pred,
            start,
            end,
            arm_id="K10",
            window=wid,
            phase="B",
            strategy_class="TopkDropoutStrategy",
            strategy_module="qlib.contrib.strategy",
            extra_kwargs={"topk": 10, "n_drop": 1, "hold_thresh": 1, "risk_degree": 0.95},
        )

        print("  K5 alternating n_drop ...", flush=True)
        results[wid]["K5"] = run_arm(
            r1r2p,
            pred,
            start,
            end,
            arm_id="K5",
            window=wid,
            phase="B",
            strategy_class="AlternatingDropTopkStrategy",
            strategy_module=STRATEGY_MODULE,
            extra_kwargs={"topk": 5, "n_drop": 0, "hold_thresh": 1, "risk_degree": 0.95},
            collect_audit=True,
        )
    runtimes["phase_b"] = time.perf_counter() - t_b

    t_c = time.perf_counter()
    for w in WINDOWS:
        wid = w["id"]
        pred = pred_for_window(inp, wid)
        start, end = w["start"], (inp["holdout_end"] if wid == "2024-2025" else w["end"])
        print(f"  P20 reuse K20 in {wid}", flush=True)
        p20 = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in results[wid]["K20"].items()}
        for key in ("daily", "holdings", "weights"):
            if isinstance(p20.get(key), pd.DataFrame) and not p20[key].empty:
                p20[key] = p20[key].copy()
                p20[key]["arm_id"] = "P20"
                p20[key]["phase"] = "C"
        results[wid]["P20"] = p20

        print("  P10 ...", flush=True)
        results[wid]["P10"] = run_arm(
            r1r2p,
            pred,
            start,
            end,
            arm_id="P10",
            window=wid,
            phase="C",
            strategy_class="TopkDropoutStrategy",
            strategy_module="qlib.contrib.strategy",
            extra_kwargs={"topk": 10, "n_drop": 2, "hold_thresh": 1, "risk_degree": 0.95},
        )
        print("  P5 ...", flush=True)
        results[wid]["P5"] = run_arm(
            r1r2p,
            pred,
            start,
            end,
            arm_id="P5",
            window=wid,
            phase="C",
            strategy_class="TopkDropoutStrategy",
            strategy_module="qlib.contrib.strategy",
            extra_kwargs={"topk": 5, "n_drop": 1, "hold_thresh": 1, "risk_degree": 0.95},
        )
    runtimes["phase_c"] = time.perf_counter() - t_c
    runtimes["phase_b_and_c_backtests"] = time.perf_counter() - t_bc

    def stack_arm(arm: str, key: str) -> pd.DataFrame:
        dfs = []
        for wid in results:
            obj = results[wid][arm].get(key)
            if isinstance(obj, pd.DataFrame) and not obj.empty:
                dfs.append(obj)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    phase_b_daily = pd.concat([stack_arm(a, "daily") for a in ("K20", "K10", "K5")], ignore_index=True)
    phase_b_hold = pd.concat([stack_arm(a, "holdings") for a in ("K20", "K10", "K5")], ignore_index=True)
    phase_b_w = pd.concat([stack_arm(a, "weights") for a in ("K20", "K10", "K5")], ignore_index=True)
    phase_c_daily = pd.concat([stack_arm(a, "daily") for a in ("P20", "P10", "P5")], ignore_index=True)
    phase_c_hold = pd.concat([stack_arm(a, "holdings") for a in ("P20", "P10", "P5")], ignore_index=True)
    phase_c_w = pd.concat([stack_arm(a, "weights") for a in ("P20", "P10", "P5")], ignore_index=True)
    save_frame(phase_b_daily, "phase_b_daily_returns")
    save_frame(phase_b_hold, "phase_b_holdings")
    save_frame(phase_b_w, "phase_b_weights")
    save_frame(phase_c_daily, "phase_c_daily_returns")
    save_frame(phase_c_hold, "phase_c_holdings")
    save_frame(phase_c_w, "phase_c_weights")

    k5_audit_rows = []
    for wid in results:
        for rec in results[wid]["K5"].get("audit", []):
            k5_audit_rows.append(
                {
                    "window": wid,
                    "trade_date": pd.Timestamp(rec["trade_date"]).normalize(),
                    "trade_step": rec.get("trade_step"),
                    "n_drop_intended": rec.get("n_drop_intended"),
                    "n_drop_executed_sells": rec.get("n_drop_executed_sells"),
                    "n_buy_orders": rec.get("n_buy_orders"),
                    "n_h_open": rec.get("n_h_open"),
                    "status": STATUS_LABEL,
                }
            )
    k5_audit = pd.DataFrame(k5_audit_rows)
    if not k5_audit.empty:
        save_frame(k5_audit, "k5_n_drop_audit")
        hold_days = k5_audit[(k5_audit["n_drop_intended"] == 0) & (k5_audit["n_h_open"] > 0)]
        if not hold_days.empty and float(hold_days["n_drop_executed_sells"].mean()) > 0.05:
            raise RuntimeError(
                "K5 n_drop=0 days still selling; qlib n_drop=0 hold-day guard failed."
            )

    # summaries
    print("Computing period / concentration / responsiveness summaries...", flush=True)
    t0 = time.perf_counter()
    arm_meta = {
        **{k: {**v, "phase": "B"} for k, v in PHASE_B_ARMS.items()},
        **{k: {**v, "phase": "C"} for k, v in PHASE_C_ARMS.items()},
    }
    period_rows = []
    conc_rows = []
    resp_rows = []
    for arm_id, meta in arm_meta.items():
        phase = meta["phase"]
        topk = int(meta["topk"])
        daily_by_w = {wid: results[wid][arm_id]["daily"] for wid in results}
        hold_by_w = {wid: results[wid][arm_id]["holdings"] for wid in results}
        w_by_w = {wid: results[wid][arm_id]["weights"] for wid in results}
        for spec in SLICE_PERIODS:
            daily, hold, weights = period_bundle(daily_by_w, hold_by_w, w_by_w, spec)
            summ = summary_from_daily(daily, hold)
            period_rows.append(
                {
                    "period": spec["id"],
                    "arm_id": arm_id,
                    "phase": phase,
                    "topk": topk,
                    "status": STATUS_LABEL,
                    "concat_independent_windows": bool(spec.get("concat")),
                    **summ,
                }
            )
            conc = concentration_from_weights(weights)
            conc_rows.append(
                {
                    "period": spec["id"],
                    "arm_id": arm_id,
                    "phase": phase,
                    "topk": topk,
                    "status": STATUS_LABEL,
                    **conc,
                }
            )
            dates = list(pd.to_datetime(daily["trade_date"]).dt.normalize()) if not daily.empty else []
            # theory from the window that owns these dates
            if spec.get("concat"):
                theory_parts = []
                for wid in spec["concat_windows"]:
                    dsub = slice_daily(daily_by_w[wid], spec["start"], spec["end"])
                    dts = list(pd.to_datetime(dsub["trade_date"]).dt.normalize()) if not dsub.empty else []
                    theory_parts.append(theory_from_pred(pred_for_window(inp, wid), dts, topk))
                theory = pd.concat([t for t in theory_parts if t is not None and not t.empty], ignore_index=True) if theory_parts else pd.DataFrame()
                scored_parts = []
                for wid in spec["concat_windows"]:
                    scored_parts.append(scored_for_window(inp, wid))
                scored = pd.concat(scored_parts, ignore_index=True)
            else:
                wid = spec["source_window"]
                theory = theory_from_pred(pred_for_window(inp, wid), dates, topk)
                scored = scored_for_window(inp, wid)
            theory_map = holdings_sets(theory.rename(columns={"trade_date": "trade_date"}) if not theory.empty else pd.DataFrame())
            hold_map = holdings_sets(hold)
            delay = delay_stats(hold_map, theory_map, dates) if dates else {}
            win = winner_metrics(weights, scored, dates)
            resp_rows.append(
                {
                    "period": spec["id"],
                    "arm_id": arm_id,
                    "phase": phase,
                    "topk": topk,
                    "status": STATUS_LABEL,
                    **delay,
                    **win,
                }
            )

    period_summary = pd.DataFrame(period_rows)
    conc_summary = pd.DataFrame(conc_rows)
    resp_summary = pd.DataFrame(resp_rows)
    period_summary.to_csv(OUT_ROOT / "period_summary.csv", index=False)
    conc_summary.to_csv(OUT_ROOT / "concentration_summary.csv", index=False)
    resp_summary.to_csv(OUT_ROOT / "responsiveness_summary.csv", index=False)
    runtimes["summaries"] = time.perf_counter() - t0

    verdict = build_verdict(phase_a_summary, a_tests, period_summary, conc_summary, resp_summary)
    runtimes["total"] = time.perf_counter() - t_all

    report = write_full_report(
        payload=parity_payload,
        parity_rows=parity_rows,
        phase_a_summary=phase_a_summary,
        phase_a_tests=a_tests,
        period_summary=period_summary,
        conc_summary=conc_summary,
        resp_summary=resp_summary,
        k5_audit=k5_audit,
        runtimes=runtimes,
        verdict=verdict,
    )
    (REPORT_ROOT / "topk_breadth_experiment_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "generated_at": _utc_now(),
        "status": STATUS_LABEL,
        "k20_parity_pass": True,
        "score_threshold_experiment_started": False,
        "frozen_h0_modified": False,
        "parameter_search": False,
        "verdict": verdict["label"],
        "mechanism": verdict["mechanism"],
        "runtimes_seconds": runtimes,
        "flip_w1": inp["flip_w1"],
        "flip_h0": inp["flip_h0"],
        "holdout_end_used": inp["holdout_end"],
        "note_2024_2025": "Opened holdout robustness only; not a new untouched holdout.",
    }
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")

    precommit["phase"] = "RUN_COMPLETE"
    precommit["run_portfolios_this_phase"] = True
    precommit["completed_at"] = _utc_now()
    precommit["verdict"] = verdict["label"]
    PRECOMMIT_PATH.write_text(json.dumps(precommit, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "t0_pass": True,
                "h0_pass": True,
                "verdict": verdict["label"],
                "runtimes_seconds": {k: round(v, 1) for k, v in runtimes.items()},
                "report": str(REPORT_ROOT / "topk_breadth_experiment_report.md"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
