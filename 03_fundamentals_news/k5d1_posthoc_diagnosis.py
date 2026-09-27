#!/usr/bin/env python3
"""Post-hoc mechanism diagnostics for K5_D1 on the 2020–2023 development window.
Does not reselect HMM parameters or open the 2024–2025 holdout for selection."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_US
from qlib.contrib.evaluate import risk_analysis
from qlib.data import D

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str((Path(os.environ["RDAGENT_ROOT"]) / "phase3_portfolio_ablation/scripts")))

import F2P_portfolio_backtest as f2p  # noqa: E402
import run_portfolio_ablation as rpa  # noqa: E402
from loop9_fund_incremental_ablation import PERIODS  # noqa: E402
from portfolio_topk_drop_joint import (  # noqa: E402
    ACCOUNT,
    BENCHMARK,
    CLOSE_COST,
    HOLD_THRESH,
    MODELLING_END,
    OPEN_COST,
    QLIB_DATA,
    RISK_DEGREE,
    cfg_id,
    holdings_diagnostics,
    metrics_row,
    net_excess,
    run_bt,
)

S1 = PROJECT_ROOT / "reports/portfolio_topk_drop_joint_20260919_141851"
OUT = PROJECT_ROOT / "reports/k5d1_posthoc_diagnosis_20260920_110303"
DECISION_COMMIT = "32ec618e3938a34f715bef63a8e590753b1d857f"
HOLDOUT = pd.Timestamp("2024-01-01")
HARD_END = pd.Timestamp(MODELLING_END)  # 2023-12-31
ANN = 238.0
BOOT_B = 2000
BOOT_BLOCK = 20
BOOT_SEED = 20260920
RECON_TOL = 1e-6
N_PLACEBO = 300
N_WORKERS = 6

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("k5d1_posthoc")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_holdout(dates, label: str) -> pd.Timestamp:
    mx = pd.to_datetime(pd.Series(list(dates))).max()
    if pd.isna(mx) or mx >= HOLDOUT or mx > HARD_END:
        raise AssertionError(f"HOLDOUT STOP: {label} max_date={mx}")
    return pd.Timestamp(mx)


def filter_holdout_idx(obj):
    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.index, pd.MultiIndex):
            dt = obj.index.get_level_values(0)
            return obj.loc[dt <= HARD_END]
        if "datetime" in obj.columns:
            return obj.loc[pd.to_datetime(obj["datetime"]) <= HARD_END]
        if "date" in obj.columns:
            return obj.loc[pd.to_datetime(obj["date"]) <= HARD_END]
        return obj.loc[pd.to_datetime(obj.index) <= HARD_END]
    if isinstance(obj, pd.Series):
        return obj.loc[pd.to_datetime(obj.index) <= HARD_END]
    return obj


def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(n - block_size, 0)
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])
    return idx[:n]


def excess_arr(s: pd.Series) -> float:
    return float(risk_analysis(s, freq="day").loc["annualized_return", "risk"])


def load_daily(cid: str, period: str) -> pd.DataFrame:
    p = S1 / "daily_returns" / f"{cid}_{period}.csv"
    df = pd.read_csv(p, parse_dates=["datetime"]).set_index("datetime").sort_index()
    df = filter_holdout_idx(df)
    assert_holdout(df.index, f"daily:{cid}:{period}")
    return df


def load_positions(cid: str, period: str) -> dict:
    p = S1 / "positions" / f"{cid}_{period}.pkl"
    with p.open("rb") as f:
        pos = pickle.load(f)
    # filter keys
    out = {pd.Timestamp(k): v for k, v in pos.items() if pd.Timestamp(k) <= HARD_END}
    assert_holdout(out.keys(), f"pos:{cid}:{period}")
    return out


def load_score() -> pd.DataFrame:
    pred = pd.read_pickle(S1 / "F_BASE_W010_score.pkl")
    pred = filter_holdout_idx(pred)
    assert_holdout(pred.index.get_level_values(0), "F_BASE_score")
    return pred


def load_label() -> pd.Series:
    path = PROJECT_ROOT / "reports/loop9_fund_incremental_20260918_222943/datasets/B3_alpha20_loop9lib8_fund.parquet"
    lab = pd.read_parquet(path, columns=["datetime", "instrument", "label"])
    lab["datetime"] = pd.to_datetime(lab["datetime"])
    lab = lab[lab["datetime"] <= HARD_END]
    assert_holdout(lab["datetime"], "LABEL0")
    return lab.set_index(["datetime", "instrument"])["label"].sort_index()


# ---------------------------------------------------------------------------
# Input audit
# ---------------------------------------------------------------------------
def write_input_audit(score: pd.DataFrame, label: pd.Series) -> None:
    rows = []
    items = [
        ("S1_valid_metrics", S1 / "validation_portfolio_metrics.csv"),
        ("S1_test_metrics", S1 / "test_portfolio_metrics.csv"),
        ("F_BASE_score", S1 / "F_BASE_W010_score.pkl"),
        ("B3_label", PROJECT_ROOT / "reports/loop9_fund_incremental_20260918_222943/datasets/B3_alpha20_loop9lib8_fund.parquet"),
        ("SPY_csv", PROJECT_ROOT / "staging/csv_benchmark/p84398.csv"),
    ]
    for cid in ("K5_D1", "K20_D2"):
        for per in ("valid", "test"):
            items.append((f"daily_{cid}_{per}", S1 / "daily_returns" / f"{cid}_{per}.csv"))
            items.append((f"pos_{cid}_{per}", S1 / "positions" / f"{cid}_{per}.pkl"))

    lines = ["# Input audit — K5_D1 post-hoc diagnosis", "", f"- Decision freeze commit: `{DECISION_COMMIT}`", ""]
    for name, path in items:
        if not path.exists():
            raise FileNotFoundError(path)
        info: dict[str, Any] = {"name": name, "path": str(path), "sha256": sha256_file(path)}
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            info["rows"] = len(df)
            info["columns"] = list(df.columns)
            for c in ("datetime", "date"):
                if c in df.columns:
                    d = pd.to_datetime(df[c])
                    d = d[d <= HARD_END] if name.startswith("SPY") or True else d
                    if name.startswith("SPY"):
                        d = pd.to_datetime(df[c])
                        d_f = d[d <= HARD_END]
                        info["min_date_raw"] = str(d.min().date())
                        info["max_date_raw"] = str(d.max().date())
                        info["min_date"] = str(d_f.min().date())
                        info["max_date"] = str(d_f.max().date())
                        info["rows_filtered"] = int(len(d_f))
                        if d.max() >= HOLDOUT:
                            info["holdout_note"] = "RAW contains 2024+; MUST filter before use"
                    else:
                        info["min_date"] = str(d.min().date())
                        info["max_date"] = str(d.max().date())
                        assert_holdout(d, name)
        elif path.suffix == ".pkl":
            obj = pd.read_pickle(path) if "score" in name or "F_BASE" in name else None
            if obj is not None:
                dt = obj.index.get_level_values(0)
                info["rows"] = len(obj)
                info["columns"] = list(obj.columns)
                info["min_date"] = str(pd.to_datetime(dt).min().date())
                info["max_date"] = str(pd.to_datetime(dt).max().date())
                assert_holdout(dt, name)
            else:
                with path.open("rb") as f:
                    pos = pickle.load(f)
                keys = sorted(pd.Timestamp(k) for k in pos)
                info["rows"] = len(keys)
                info["columns"] = ["Position.amount/price/weight/cash"]
                info["min_date"] = str(keys[0].date())
                info["max_date"] = str(keys[-1].date())
                assert_holdout(keys, name)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path, columns=["datetime"])
            d = pd.to_datetime(df["datetime"])
            info["rows"] = len(df)
            info["columns"] = ["datetime", "instrument", "label"]
            info["min_date"] = str(d.min().date())
            info["max_date"] = str(d.max().date())
            assert_holdout(d, name)
        rows.append(info)
        lines.append(f"## {name}")
        for k, v in info.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    # score/label already filtered
    lines.append("## In-memory filtered objects")
    lines.append(f"- score max: {score.index.get_level_values(0).max()}")
    lines.append(f"- label max: {label.index.get_level_values(0).max()}")
    lines.append("")
    lines.append("## Verdict: ALL INPUTS PASS HOLDOUT ASSERTIONS (SPY requires filter)")
    (OUT / "input_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "input_inventory_runtime.csv", index=False)


# ---------------------------------------------------------------------------
# Rank buckets
# ---------------------------------------------------------------------------
def rank_bucket_analysis(score: pd.DataFrame, label: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = score["score"]
    common = s.index.intersection(label.index)
    s = s.reindex(common)
    y = label.reindex(common)
    df = pd.DataFrame({"score": s, "label": y}).dropna()
    df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["period"] = np.where(df["datetime"] < "2020-01-01", "valid", "test")
    # ranks within day
    df["rank"] = df.groupby("datetime")["score"].rank(ascending=False, method="first")

    def bucket(r):
        if r <= 5:
            return "1-5"
        if r <= 10:
            return "6-10"
        if r <= 20:
            return "11-20"
        if r <= 50:
            return "21-50"
        if r <= 100:
            return "51-100"
        return "remaining"

    df["bucket"] = df["rank"].map(bucket)
    # universe EW mean per day
    uni = df.groupby("datetime")["label"].mean().rename("uni_ew")
    df = df.merge(uni, on="datetime", how="left")
    df["label_ex_uni"] = df["label"] - df["uni_ew"]

    rows = []
    for period in ("valid", "test", "all"):
        subp = df if period == "all" else df[df["period"] == period]
        years = sorted(subp["year"].unique())
        scopes = [("period", period, subp)] + [("year", str(y), subp[subp["year"] == y]) for y in years]
        for scope_type, scope_id, sub in scopes:
            for b in ["1-5", "6-10", "11-20", "21-50", "51-100", "remaining"]:
                g = sub[sub["bucket"] == b]
                if g.empty:
                    continue
                # equal-weight daily: mean label within bucket each day, then mean across days
                daily = g.groupby("datetime")["label"].mean()
                daily_ex = g.groupby("datetime")["label_ex_uni"].mean()
                rows.append(
                    {
                        "scope_type": scope_type,
                        "scope": scope_id,
                        "bucket": b,
                        "n_obs": len(g),
                        "n_ticker_days": g[["datetime", "instrument"]].drop_duplicates().shape[0],
                        "n_days": g["datetime"].nunique(),
                        "mean_label0": float(g["label"].mean()),
                        "median_label0": float(g["label"].median()),
                        "ew_daily_mean_label0": float(daily.mean()),
                        "ew_daily_mean_ex_universe": float(daily_ex.mean()),
                        "cs_dispersion_mean": float(g.groupby("datetime")["label"].std().mean()),
                    }
                )
    bucket_df = pd.DataFrame(rows)
    bucket_df.to_csv(OUT / "rank_bucket_returns.csv", index=False)

    # D_top bootstrap
    boot_rows = []
    for period, (start, end) in {"valid": PERIODS["valid"], "test": PERIODS["test"]}.items():
        sub = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
        d1 = sub[sub["rank"] <= 5].groupby("datetime")["label"].mean()
        d620 = sub[(sub["rank"] >= 6) & (sub["rank"] <= 20)].groupby("datetime")["label"].mean()
        common_d = d1.index.intersection(d620.index)
        diff = (d1.reindex(common_d) - d620.reindex(common_d)).dropna()
        assert_holdout(diff.index, f"D_top:{period}")
        arr = diff.to_numpy(dtype=float)
        n = len(arr)
        point = float(arr.mean())
        rng = np.random.default_rng(BOOT_SEED + (0 if period == "valid" else 1))
        boots = np.empty(BOOT_B)
        for i in range(BOOT_B):
            idx = block_bootstrap_indices(n, BOOT_BLOCK, rng)
            boots[i] = arr[idx].mean()
        boot_rows.append(
            {
                "period": period,
                "n_days": n,
                "mean_D_top": point,
                "annualised_conditional_mean_diff_238x": point * ANN,
                "note_annualised": "238 × mean daily difference; NOT portfolio ARR",
                "ci95_low": float(np.percentile(boots, 2.5)),
                "ci95_high": float(np.percentile(boots, 97.5)),
                "boot_p_one_sided_le0": float(np.mean(boots <= 0)),
                "boot_p_two_sided": float(2 * min(np.mean(boots <= 0), np.mean(boots >= 0))),
                "bootstrap_B": BOOT_B,
                "bootstrap_block": BOOT_BLOCK,
            }
        )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT / "rank_bucket_topdiff_bootstrap.csv", index=False)
    return bucket_df, boot_df


# ---------------------------------------------------------------------------
# Contribution
# ---------------------------------------------------------------------------
def stock_close_panel(instruments: list[str], start: str, end: str) -> pd.DataFrame:
    # pad one day before for returns
    start_pad = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    feat = D.features(instruments, ["$close"], start_time=start_pad, end_time=end)
    feat = feat.reset_index().rename(columns={"instrument": "instrument", "datetime": "datetime", "$close": "close"})
    feat["datetime"] = pd.to_datetime(feat["datetime"])
    feat = feat[feat["datetime"] <= HARD_END]
    return feat


def contribution_analysis(cid: str, period: str, daily: pd.DataFrame, positions: dict) -> dict[str, Any]:
    dates = sorted(positions.keys())
    # instruments ever held
    insts = sorted({s for d in dates for s in positions[d].get_stock_list()})
    start, end = str(dates[0].date()), str(dates[-1].date())
    closes = stock_close_panel(insts, start, end)
    close_wide = closes.pivot(index="datetime", columns="instrument", values="close").sort_index()
    ret_wide = close_wide.pct_change(fill_method=None)

    recon_rows = []
    contrib_records = []  # per stock-day
    for i in range(1, len(dates)):
        t_prev, t = dates[i - 1], dates[i]
        pos_prev = positions[t_prev]
        w = pos_prev.get_stock_weight_dict()  # EOD t-1, fraction of account
        cash_w = float(pos_prev.get_cash() / pos_prev.calculate_value()) if pos_prev.calculate_value() else 0.0
        stock_sum = 0.0
        if t not in ret_wide.index:
            continue
        day_ret = ret_wide.loc[t]
        for code, wi in w.items():
            ri = day_ret.get(code, np.nan)
            if not np.isfinite(ri):
                # try position price path
                if code in positions[t].get_stock_list() and code in pos_prev.get_stock_list():
                    p0 = pos_prev.get_stock_price(code)
                    p1 = positions[t].get_stock_price(code)
                    ri = p1 / p0 - 1.0 if p0 else np.nan
                else:
                    ri = np.nan
            if not np.isfinite(ri):
                continue
            c = float(wi) * float(ri)
            stock_sum += c
            contrib_records.append(
                {
                    "datetime": t,
                    "instrument": code,
                    "weight_tm1": float(wi),
                    "stock_return": float(ri),
                    "contribution": c,
                }
            )
        cash_c = cash_w * 0.0
        approx = stock_sum + cash_c
        # gross portfolio return from daily file
        if t not in daily.index:
            continue
        gross = float(daily.loc[t, "portfolio_return"])
        recon_rows.append(
            {
                "datetime": t,
                "gross_portfolio_return": gross,
                "sum_stock_contrib": stock_sum,
                "cash_contrib": cash_c,
                "approx_return": approx,
                "abs_error": abs(approx - gross),
            }
        )

    recon = pd.DataFrame(recon_rows)
    max_err = float(recon["abs_error"].max()) if len(recon) else float("inf")
    mean_err = float(recon["abs_error"].mean()) if len(recon) else float("nan")
    p99 = float(recon["abs_error"].quantile(0.99)) if len(recon) else float("nan")
    recon_summary = {
        "config_id": cid,
        "period": period,
        "n_days": len(recon),
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "p99_abs_error": p99,
        "pass": max_err <= RECON_TOL,
        "cash_return_assumption": 0.0,
        "weight_timing": "EOD_t-1 × close_to_close_t",
        "reconcile_target": "GROSS portfolio_return",
    }
    return {
        "recon": recon,
        "recon_summary": recon_summary,
        "contrib": pd.DataFrame(contrib_records),
    }


def summarize_contributions(cid: str, period: str, contrib: pd.DataFrame, daily: pd.DataFrame, recon_ok: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not recon_ok or contrib.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    g = contrib.groupby("instrument").agg(
        total_contribution=("contribution", "sum"),
        positive_contribution=("contribution", lambda x: float(x[x > 0].sum())),
        negative_contribution=("contribution", lambda x: float(x[x < 0].sum())),
        holding_days=("datetime", "nunique"),
        average_weight=("weight_tm1", "mean"),
        max_weight=("weight_tm1", "max"),
    ).reset_index()
    # entries/exits approximate from day-to-day presence
    # skip detailed entries for speed; set placeholder from holding pattern
    g["number_of_entries"] = np.nan
    g["number_of_exits"] = np.nan
    g = g.sort_values("total_contribution", ascending=False)
    total_pnl = float(g["total_contribution"].sum())
    pos_pnl = float(g["positive_contribution"].sum())
    abs_pnl = float(g["total_contribution"].abs().sum())
    rows = []
    for k in (1, 3, 5):
        top = g.head(k)
        rows.append(
            {
                "config_id": cid,
                "period": period,
                "top_k": k,
                "share_of_total_pnl": float(top["total_contribution"].sum() / total_pnl) if total_pnl != 0 else np.nan,
                "share_of_positive_contribution": float(top["positive_contribution"].sum() / pos_pnl) if pos_pnl != 0 else np.nan,
                "share_of_absolute_contribution": float(top["total_contribution"].abs().sum() / abs_pnl) if abs_pnl != 0 else np.nan,
                "tickers": ",".join(top["instrument"].tolist()),
            }
        )
    conc = pd.DataFrame(rows)

    # counterfactual removal approx for K5 only
    cf = {}
    if cid == "K5_D1":
        # rebuild daily contrib sum, subtract top names
        daily_c = contrib.groupby("datetime")["contribution"].sum()
        cost = daily["cost"]
        bench = daily["benchmark_return"]
        for k in (1, 2):
            top_names = set(g.head(k)["instrument"])
            rem = contrib[~contrib["instrument"].isin(top_names)].groupby("datetime")["contribution"].sum()
            # approx net excess = rem - bench - cost (cash already 0 in rem if excluded only stocks)
            # Note: removing stock contrib doesn't redistribute weights — APPROXIMATION
            aligned = rem.reindex(daily.index).fillna(0.0)
            net_ex = aligned - bench - cost
            cf[f"ex_top{k}"] = {
                "approx_net_excess_arr": excess_arr(net_ex.dropna()),
                "note": "COUNTERFACTUAL CONTRIBUTION-REMOVAL APPROXIMATION; not rebalanced backtest",
                "removed": ",".join(g.head(k)["instrument"].tolist()),
            }
    g.insert(0, "config_id", cid)
    g.insert(1, "period", period)
    return g, conc, cf


# ---------------------------------------------------------------------------
# Time concentration
# ---------------------------------------------------------------------------
def time_concentration(cid: str, period: str, daily: pd.DataFrame) -> dict[str, Any]:
    d = daily.copy()
    d["year"] = d.index.year
    d["month"] = d.index.to_period("M").astype(str)
    d["net_excess"] = d["net_excess"]
    d["gross_excess"] = d["gross_excess"]
    # annual
    ann_rows = []
    for y, g in d.groupby("year"):
        ann_rows.append(
            {
                "config_id": cid,
                "period": period,
                "year": int(y),
                "cumulative_return": float((1 + g["portfolio_return"]).prod() - 1),
                "cumulative_net_excess_sum": float(g["net_excess"].sum()),
                "net_excess_arr": excess_arr(g["net_excess"]),
                "n_days": len(g),
            }
        )
    monthly = []
    for m, g in d.groupby("month"):
        monthly.append(
            {
                "config_id": cid,
                "period": period,
                "month": m,
                "monthly_return": float((1 + g["portfolio_return"]).prod() - 1),
                "monthly_net_excess_sum": float(g["net_excess"].sum()),
                "n_days": len(g),
            }
        )
    mdf = pd.DataFrame(monthly).sort_values("monthly_net_excess_sum", ascending=False)
    total_ex_sum = float(d["net_excess"].sum())
    best10 = d["net_excess"].nlargest(10)
    worst10 = d["net_excess"].nsmallest(10)
    ex_best10 = d.loc[~d.index.isin(best10.index), "net_excess"]
    out = {
        "config_id": cid,
        "period": period,
        "best10_share_of_net_excess_sum": float(best10.sum() / total_ex_sum) if total_ex_sum != 0 else np.nan,
        "worst10_share_of_net_excess_sum": float(worst10.sum() / total_ex_sum) if total_ex_sum != 0 else np.nan,
        "best_month": mdf.iloc[0]["month"] if len(mdf) else None,
        "best_month_net_excess_sum": float(mdf.iloc[0]["monthly_net_excess_sum"]) if len(mdf) else np.nan,
        "best_month_share": float(mdf.iloc[0]["monthly_net_excess_sum"] / total_ex_sum) if len(mdf) and total_ex_sum != 0 else np.nan,
        "worst_month": mdf.iloc[-1]["month"] if len(mdf) else None,
        "top3_months": ",".join(mdf.head(3)["month"].tolist()),
        "approx_net_excess_arr_ex_best10": excess_arr(ex_best10) if len(ex_best10) > 50 else np.nan,
        "full_net_excess_arr": excess_arr(d["net_excess"]),
        "note": "ex-best10 is path-concentration diagnostic, not tradable",
    }
    # temporal label for validation
    if period == "valid":
        y2018 = next((r for r in ann_rows if r["year"] == 2018), None)
        y2019 = next((r for r in ann_rows if r["year"] == 2019), None)
        broad = (
            y2018
            and y2019
            and y2018["cumulative_net_excess_sum"] > 0
            and y2019["cumulative_net_excess_sum"] > 0
            and out["best_month_share"] < 0.5
            and out["approx_net_excess_arr_ex_best10"] > 0.5 * out["full_net_excess_arr"]
        )
        out["temporal_label"] = "TEMPORALLY_BROAD" if broad else "TEMPORALLY_CONCENTRATED"
    return {"summary": out, "annual": ann_rows, "monthly": mdf}


# ---------------------------------------------------------------------------
# Backtest helpers for neighbourhood / placebo
# ---------------------------------------------------------------------------
def run_one_bt(pred: pd.DataFrame, topk: int, n_drop: int, start: str, end: str) -> dict[str, Any]:
    end = min(pd.Timestamp(end), HARD_END).strftime("%Y-%m-%d")
    res = run_bt(pred, topk, n_drop, start, end)
    assert_holdout(res["report"].index, f"bt:K{topk}_D{n_drop}:{start}:{end}")
    return res


def _placebo_worker(args: tuple) -> dict[str, Any]:
    """Module-level for ProcessPoolExecutor pickling."""
    run_id, period, score_path, out_dir = args
    try:
        import os
        import sys
        from pathlib import Path

        project = Path(os.environ["PROJECT_ROOT"]) if os.environ.get("PROJECT_ROOT") else Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(project / "03_fundamentals_news"))
        sys.path.insert(0, str(project))
        sys.path.insert(0, str(Path(os.environ["RDAGENT_ROOT"]) / "phase3_portfolio_ablation" / "scripts"))
        os.environ.setdefault("QLIB_LOGGING_LEVEL", "WARNING")

        import pandas as pd
        import numpy as np
        import qlib
        from qlib.constant import REG_US
        from portfolio_topk_drop_joint import QLIB_DATA, metrics_row, run_bt
        from loop9_fund_incremental_ablation import PERIODS

        HARD_END = pd.Timestamp("2023-12-31")
        qlib.init(provider_uri=str(QLIB_DATA), region=REG_US, kernels=1)
        score = pd.read_pickle(score_path)
        score = score.loc[score.index.get_level_values(0) <= HARD_END]
        start, end = PERIODS[period]
        end = min(pd.Timestamp(end), HARD_END).strftime("%Y-%m-%d")
        dt = score.index.get_level_values(0)
        mask = (dt >= pd.Timestamp(start) - pd.Timedelta(days=10)) & (dt <= pd.Timestamp(end))
        sub = score.loc[mask].copy()
        rng = np.random.default_rng(int(run_id))
        pieces = []
        for d, g in sub.groupby(level=0):
            s = g["score"].to_numpy().copy()
            order = np.argsort(-s)
            top20 = order[: min(20, len(order))]
            perm = top20.copy()
            rng.shuffle(perm)
            new_s = s.copy()
            new_s[top20] = s[perm]
            gg = g.copy()
            gg["score"] = new_s
            pieces.append(gg)
        shuffled = pd.concat(pieces).sort_index()
        res = run_bt(shuffled, 5, 1, start, end)
        m = metrics_row(res, topk=5, n_drop=1, period=period, role="placebo")
        return {
            "run_id": int(run_id),
            "seed": int(run_id),
            "period": period,
            "net_excess_arr": m["excess_arr_net"],
            "net_ir": m["ir_net"],
            "absolute_mdd": m["mdd"],
            "mean_turnover": m["mean_daily_turnover"],
            "realised_replacement": m["realised_replacement_fraction"],
            "mean_daily_cost": m["mean_daily_cost"],
            "status": "ok",
        }
    except Exception as e:
        return {
            "run_id": int(run_id),
            "seed": int(run_id),
            "period": period,
            "status": "error",
            "error": repr(e),
        }


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def style_exposure(daily_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spy = pd.read_csv(PROJECT_ROOT / "staging/csv_benchmark/p84398.csv")
    spy.columns = [c.lower() for c in spy.columns]
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy[spy["date"] <= HARD_END].copy()
    assert_holdout(spy["date"], "SPY_filtered")
    spy = spy.set_index("date").sort_index()
    spy["spy_ret"] = spy["close"].pct_change()
    rows = []
    for key, daily in daily_map.items():
        cid, period = key.split("|")
        y = daily["portfolio_return"]
        x = spy["spy_ret"].reindex(y.index)
        ok = y.notna() & x.notna()
        yv, xv = y[ok].to_numpy(), x[ok].to_numpy()
        if len(yv) < 30:
            continue
        X = np.column_stack([np.ones(len(xv)), xv])
        beta_hat, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta_hat
        s2 = float(resid.var(ddof=2))
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(xtx_inv) * s2)
        tstat = beta_hat / se
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        r2 = 1 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan
        rows.append(
            {
                "config_id": cid,
                "period": period,
                "model": "full",
                "alpha": float(beta_hat[0]),
                "beta": float(beta_hat[1]),
                "alpha_t": float(tstat[0]),
                "beta_t": float(tstat[1]),
                "r2": r2,
                "n": int(len(yv)),
            }
        )
        up = xv > 0
        for name, mask in (("up", up), ("down", ~up)):
            if mask.sum() < 20:
                continue
            X2 = np.column_stack([np.ones(mask.sum()), xv[mask]])
            b2, *_ = np.linalg.lstsq(X2, yv[mask], rcond=None)
            rows.append(
                {
                    "config_id": cid,
                    "period": period,
                    "model": name,
                    "alpha": float(b2[0]),
                    "beta": float(b2[1]),
                    "n": int(mask.sum()),
                }
            )
    # characteristics: vol20 / mom 12-1 from closes of holdings — PIT from past closes
    # size: NOT_AVAILABLE ($market_value empty)
    rows.append(
        {
            "config_id": "ALL",
            "period": "ALL",
            "model": "FF_MOM",
            "note": "NOT_AVAILABLE — no Ken French / FF factors in repo; no download",
        }
    )
    rows.append(
        {
            "config_id": "ALL",
            "period": "ALL",
            "model": "size_log_mcap",
            "note": "NOT_AVAILABLE — $market_value/$cap empty in qlib provider",
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("daily_outputs", "positions_if_new", "placebo_logs", "figures"):
        (OUT / sub).mkdir(exist_ok=True)

    log.info("Init qlib")
    qlib.init(provider_uri=str(QLIB_DATA), region=REG_US, kernels=1)

    log.info("Load score/label + input audit")
    score = load_score()
    label = load_label()
    write_input_audit(score, label)

    # Step 3 reproduce
    log.info("Reproduce original metrics")
    repro_rows = []
    daily_map = {}
    pos_map = {}
    expected = {
        ("K5_D1", "valid"): 0.184532,
        ("K20_D2", "valid"): 0.071628,
        ("K5_D1", "test"): 0.110896,
        ("K20_D2", "test"): 0.100720,
    }
    vm = pd.read_csv(S1 / "validation_portfolio_metrics.csv")
    tm = pd.read_csv(S1 / "test_portfolio_metrics.csv")
    for cid in ("K5_D1", "K20_D2"):
        for period, metrics_df in (("valid", vm), ("test", tm)):
            row = metrics_df[metrics_df.config_id == cid].iloc[0]
            val = float(row["excess_arr_net"])
            exp = expected[(cid, period)]
            ok = abs(val - exp) < 5e-4
            repro_rows.append({"config_id": cid, "period": period, "excess_arr_net": val, "expected": exp, "ok": ok})
            if not ok:
                raise AssertionError(f"Reproduce STOP: {cid} {period} {val} != {exp}")
            daily_map[f"{cid}|{period}"] = load_daily(cid, period)
            pos_map[f"{cid}|{period}"] = load_positions(cid, period)
    pd.DataFrame(repro_rows).to_csv(OUT / "reproduction_check.csv", index=False)
    log.info("Reproduction PASS")

    # Step 4 rank buckets
    log.info("Rank bucket + topdiff bootstrap")
    bucket_df, boot_df = rank_bucket_analysis(score, label)

    # Step 5 contribution
    log.info("Contribution reconciliation")
    all_recon_sum = []
    all_conc = []
    all_top = []
    cf_store = {}
    for cid in ("K5_D1", "K20_D2"):
        for period in ("valid", "test"):
            key = f"{cid}|{period}"
            result = contribution_analysis(cid, period, daily_map[key], pos_map[key])
            result["recon"].to_csv(OUT / "daily_outputs" / f"contrib_recon_{cid}_{period}.csv", index=False)
            all_recon_sum.append(result["recon_summary"])
            if not result["recon_summary"]["pass"]:
                pd.DataFrame(all_recon_sum).to_csv(OUT / "contribution_reconciliation.csv", index=False)
                (OUT / "STOP_REASON.md").write_text(
                    f"STOP: contribution reconciliation max_abs_error="
                    f"{result['recon_summary']['max_abs_error']} > {RECON_TOL} for {cid} {period}\n",
                    encoding="utf-8",
                )
                raise SystemExit(f"STOP recon {cid} {period} err={result['recon_summary']['max_abs_error']}")
            g, conc, cf = summarize_contributions(cid, period, result["contrib"], daily_map[key], True)
            g.to_csv(OUT / "daily_outputs" / f"contrib_by_stock_{cid}_{period}.csv", index=False)
            all_top.append(g.head(20))
            all_conc.append(conc)
            if cf:
                cf_store[f"{cid}|{period}"] = cf
    pd.DataFrame(all_recon_sum).to_csv(OUT / "contribution_reconciliation.csv", index=False)
    pd.concat(all_conc, ignore_index=True).to_csv(OUT / "contribution_concentration.csv", index=False)
    pd.concat(all_top, ignore_index=True).to_csv(OUT / "top_contributors.csv", index=False)
    (OUT / "contribution_counterfactual.json").write_text(json.dumps(cf_store, indent=2), encoding="utf-8")

    # Step 6 time
    log.info("Time concentration")
    time_sum = []
    monthly_all = []
    annual_all = []
    for cid in ("K5_D1", "K20_D2"):
        for period in ("valid", "test"):
            tc = time_concentration(cid, period, daily_map[f"{cid}|{period}"])
            time_sum.append(tc["summary"])
            monthly_all.append(tc["monthly"])
            annual_all.extend(tc["annual"])
    pd.DataFrame(time_sum).to_csv(OUT / "time_concentration.csv", index=False)
    pd.concat(monthly_all, ignore_index=True).to_csv(OUT / "monthly_returns.csv", index=False)
    pd.DataFrame(annual_all).to_csv(OUT / "annual_returns.csv", index=False)

    # Step 7–8 neighbourhoods
    log.info("Neighbourhood backtests")
    nb_rows = []
    # load existing K5_D1 from metrics
    for period, mdf in (("valid", vm), ("test", tm)):
        r = mdf[mdf.config_id == "K5_D1"].iloc[0]
        nb_rows.append(
            {
                "config_id": "K5_D1",
                "topk": 5,
                "n_drop": 1,
                "nominal_replacement": 0.2,
                "axis": "both",
                "period": period,
                "source": "S1_artifact",
                "net_excess_arr": float(r["excess_arr_net"]),
                "net_ir": float(r["ir_net"]),
                "absolute_mdd": float(r["mdd"]),
                "excess_mdd_net": float(r["excess_mdd_net"]),
                "mean_turnover": float(r["mean_daily_turnover"]),
                "realised_replacement": float(r["realised_replacement_fraction"]),
                "mean_holding_duration": float(r["mean_holding_duration_days"]),
                "median_holding_duration": float(r["median_holding_duration_days"]),
                "mean_retained_frac": float(r["mean_retained_frac"]),
                "mean_daily_cost": float(r["mean_daily_cost"]),
                "avg_holdings": float(r["avg_holdings"]),
            }
        )

    repl_cfgs = [(5, d) for d in (2, 3, 4, 5)]
    breadth_cfgs = [(k, 1) for k in (3, 4, 6, 7)]
    for topk, n_drop in repl_cfgs + breadth_cfgs:
        for period in ("valid", "test"):
            start, end = PERIODS[period]
            log.info("BT %s %s", period, cfg_id(topk, n_drop))
            res = run_one_bt(score, topk, n_drop, start, end)
            m = metrics_row(res, topk=topk, n_drop=n_drop, period=period, role="neighbourhood")
            axis = "replacement" if topk == 5 else "breadth"
            nb_rows.append(
                {
                    "config_id": cfg_id(topk, n_drop),
                    "topk": topk,
                    "n_drop": n_drop,
                    "nominal_replacement": n_drop / topk,
                    "axis": axis,
                    "period": period,
                    "source": "new_backtest",
                    "net_excess_arr": m["excess_arr_net"],
                    "net_ir": m["ir_net"],
                    "absolute_mdd": m["mdd"],
                    "excess_mdd_net": m["excess_mdd_net"],
                    "mean_turnover": m["mean_daily_turnover"],
                    "realised_replacement": m["realised_replacement_fraction"],
                    "mean_holding_duration": m["mean_holding_duration_days"],
                    "median_holding_duration": m["median_holding_duration_days"],
                    "mean_retained_frac": m["mean_retained_frac"],
                    "mean_daily_cost": m["mean_daily_cost"],
                    "avg_holdings": m["avg_holdings"],
                }
            )
            # save daily
            rep = res["report"]
            pd.DataFrame(
                {
                    "datetime": rep.index,
                    "portfolio_return": rep["return"].to_numpy(),
                    "benchmark_return": rep["bench"].to_numpy(),
                    "cost": rep["cost"].to_numpy(),
                    "turnover": rep["turnover"].to_numpy(),
                    "net_excess": net_excess(rep).to_numpy(),
                }
            ).to_csv(OUT / "daily_outputs" / f"{cfg_id(topk, n_drop)}_{period}.csv", index=False)
            with (OUT / "positions_if_new" / f"{cfg_id(topk, n_drop)}_{period}.pkl").open("wb") as f:
                pickle.dump(res["positions"], f)

    nb = pd.DataFrame(nb_rows)
    nb.to_csv(OUT / "k5_neighbourhood_metrics.csv", index=False)
    nb[nb["topk"] == 5].to_csv(OUT / "k5_replacement_neighbourhood_metrics.csv", index=False)
    nb[nb["n_drop"] == 1].to_csv(OUT / "k5_breadth_neighbourhood_metrics.csv", index=False)

    # deltas K5 vs K4/K6
    delta_rows = []
    for period in ("valid", "test"):
        k5 = nb[(nb.config_id == "K5_D1") & (nb.period == period)].iloc[0]
        for other in ("K4_D1", "K6_D1"):
            o = nb[(nb.config_id == other) & (nb.period == period)]
            if o.empty:
                continue
            o = o.iloc[0]
            delta_rows.append(
                {
                    "period": period,
                    "comparison": f"K5_D1_minus_{other}",
                    "delta_arr": float(k5["net_excess_arr"] - o["net_excess_arr"]),
                    "delta_ir": float(k5["net_ir"] - o["net_ir"]),
                }
            )
    pd.DataFrame(delta_rows).to_csv(OUT / "k5_breadth_deltas.csv", index=False)

    # Step 9 placebo
    log.info("Placebo 600 runs with %d workers", N_WORKERS)
    score_path = OUT / "F_BASE_W010_score_holdout_filtered.pkl"
    score.to_pickle(score_path)
    tasks = []
    for period in ("valid", "test"):
        for run_id in range(N_PLACEBO):
            tasks.append((run_id, period, str(score_path), str(OUT)))
    placebo_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_placebo_worker, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            placebo_rows.append(fut.result())
            done += 1
            if done % 50 == 0:
                log.info("Placebo progress %d/%d", done, len(tasks))
    placebo_df = pd.DataFrame(placebo_rows).sort_values(["period", "run_id"]).reset_index(drop=True)
    placebo_df.to_csv(OUT / "placebo_results.csv", index=False)
    if (placebo_df["status"] != "ok").any():
        nerr = int((placebo_df["status"] != "ok").sum())
        log.warning("Placebo errors: %d", nerr)
        placebo_df[placebo_df.status != "ok"].to_csv(OUT / "placebo_logs" / "errors.csv", index=False)

    # true K5 into summary
    summ = []
    for period in ("valid", "test"):
        sub = placebo_df[(placebo_df.period == period) & (placebo_df.status == "ok")]
        true_arr = float(expected[("K5_D1", period)]) if period else np.nan
        # use exact from repro
        true_arr = float([r for r in repro_rows if r["config_id"] == "K5_D1" and r["period"] == period][0]["excess_arr_net"])
        true_to = float(vm[vm.config_id == "K5_D1"].iloc[0]["mean_daily_turnover"]) if period == "valid" else float(
            tm[tm.config_id == "K5_D1"].iloc[0]["mean_daily_turnover"]
        )
        arrs = sub["net_excess_arr"].to_numpy()
        pct = float(np.mean(arrs <= true_arr) * 100) if len(arrs) else np.nan
        # empirical one-sided p: share of placebo >= true
        p_one = float(np.mean(arrs >= true_arr)) if len(arrs) else np.nan
        summ.append(
            {
                "period": period,
                "n_ok": len(sub),
                "placebo_mean_arr": float(sub["net_excess_arr"].mean()),
                "placebo_median_arr": float(sub["net_excess_arr"].median()),
                "placebo_std_arr": float(sub["net_excess_arr"].std(ddof=1)),
                "p5": float(sub["net_excess_arr"].quantile(0.05)),
                "p25": float(sub["net_excess_arr"].quantile(0.25)),
                "p75": float(sub["net_excess_arr"].quantile(0.75)),
                "p95": float(sub["net_excess_arr"].quantile(0.95)),
                "true_k5_arr": true_arr,
                "true_percentile": pct,
                "empirical_one_sided_p_placebo_ge_true": p_one,
                "placebo_mean_turnover": float(sub["mean_turnover"].mean()),
                "true_k5_turnover": true_to,
                "turnover_diff_true_minus_placebo_mean": true_to - float(sub["mean_turnover"].mean()),
                "interpretation_gate_95": "withinTop20_ranking_material" if pct >= 95 else "cannot_distinguish_from_random_Top20",
            }
        )
    pd.DataFrame(summ).to_csv(OUT / "placebo_summary.csv", index=False)

    # Step 10 style
    log.info("Style exposure")
    style = style_exposure(daily_map)
    # optional vol/mom exposure for holdings — reconstruct PIT
    char_rows = []
    for cid in ("K5_D1", "K20_D2"):
        for period in ("valid", "test"):
            pos = pos_map[f"{cid}|{period}"]
            dates = sorted(pos.keys())
            insts = sorted({s for d in dates for s in pos[d].get_stock_list()})
            start, end = str(dates[0].date()), str(dates[-1].date())
            start_pad = (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
            feat = D.features(insts, ["$close"], start_time=start_pad, end_time=end)
            close = feat.reset_index().pivot(index="datetime", columns="instrument", values="$close").sort_index()
            close = close.loc[close.index <= HARD_END]
            ret = close.pct_change()
            vol20 = ret.rolling(20, min_periods=20).std()
            mom = close / close.shift(252) - 1.0  # rough 12-m; 12-1 would exclude last month
            mom_12_1 = close.shift(21) / close.shift(252) - 1.0
            port_vol = []
            port_mom = []
            uni_vol = []
            uni_mom = []
            for d in dates:
                if d not in vol20.index:
                    continue
                held = pos[d].get_stock_list()
                w = pos[d].get_stock_weight_dict()
                # PIT: use vol/mom as of d (rolling ends at d)
                vv = vol20.loc[d].reindex(held)
                mm = mom_12_1.loc[d].reindex(held)
                ww = pd.Series(w)
                if vv.notna().any():
                    port_vol.append(float((vv * ww.reindex(held)).sum() / ww.reindex(held).sum()))
                    uni_vol.append(float(vol20.loc[d].mean()))
                if mm.notna().any():
                    port_mom.append(float((mm.fillna(0) * ww.reindex(held)).sum() / ww.reindex(held).sum()))
                    uni_mom.append(float(mom_12_1.loc[d].mean()))
            char_rows.append(
                {
                    "config_id": cid,
                    "period": period,
                    "mean_port_vol20": float(np.mean(port_vol)) if port_vol else np.nan,
                    "mean_uni_vol20": float(np.mean(uni_vol)) if uni_vol else np.nan,
                    "vol20_port_minus_uni": float(np.mean(port_vol) - np.mean(uni_vol)) if port_vol else np.nan,
                    "mean_port_mom12_1": float(np.mean(port_mom)) if port_mom else np.nan,
                    "mean_uni_mom12_1": float(np.mean(uni_mom)) if uni_mom else np.nan,
                    "mom_port_minus_uni": float(np.mean(port_mom) - np.mean(uni_mom)) if port_mom else np.nan,
                    "size_log_mcap": "NOT_AVAILABLE",
                    "pit_note": "vol20 and mom12-1 from past closes only through date d",
                }
            )
    style = pd.concat([style, pd.DataFrame(char_rows)], ignore_index=True)
    style.to_csv(OUT / "style_exposure.csv", index=False)

    # Step 11 figures + analysis
    log.info("Figures + analysis")
    _make_figures(bucket_df, boot_df, placebo_df, summ, nb, daily_map, time_sum, all_conc)
    _write_analysis(
        repro_rows, boot_df, bucket_df, all_recon_sum, all_conc, cf_store, time_sum, summ, nb, style, delta_rows
    )

    manifest = {
        "experiment_id": OUT.name,
        "identity": "POST-HOC_EXPLORATORY_MECHANISM_DIAGNOSIS",
        "decision_log_commit": DECISION_COMMIT,
        "holdout_2024_2025": "closed",
        "elapsed_sec": time.time() - t0,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED",
        "script": "03_fundamentals_news/k5d1_posthoc_diagnosis.py",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # terminal summary print
    _print_summary(boot_df, summ, all_conc, cf_store, time_sum, nb, style)
    return 0


def _make_figures(bucket_df, boot_df, placebo_df, summ, nb, daily_map, time_sum, all_conc):
    # 1-2 rank buckets
    for period in ("valid", "test"):
        sub = bucket_df[(bucket_df.scope_type == "period") & (bucket_df.scope == period)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        order = ["1-5", "6-10", "11-20", "21-50", "51-100", "remaining"]
        sub = sub.set_index("bucket").reindex(order)
        ax.bar(range(len(order)), sub["ew_daily_mean_label0"], color="C0")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30)
        ax.set_ylabel("EW daily mean LABEL0")
        ax.set_title(f"Rank-bucket LABEL0 ({period})")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"rank_bucket_returns_{period}.png", dpi=140)
        plt.close(fig)

    # 3 topdiff
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        boot_df["period"],
        boot_df["mean_D_top"],
        yerr=[boot_df["mean_D_top"] - boot_df["ci95_low"], boot_df["ci95_high"] - boot_df["mean_D_top"]],
        fmt="o",
        capsize=5,
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("D_top = mean(LABEL0 1–5) − mean(LABEL0 6–20)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "topdiff_validation_test.png", dpi=140)
    plt.close(fig)

    # 4-5 placebo
    for period in ("valid", "test"):
        sub = placebo_df[(placebo_df.period == period) & (placebo_df.status == "ok")]
        true_arr = [s for s in summ if s["period"] == period][0]["true_k5_arr"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sub["net_excess_arr"], bins=30, color="0.7", edgecolor="k")
        ax.axvline(true_arr, color="C3", lw=2, label=f"true K5={true_arr:.3f}")
        ax.legend()
        ax.set_title(f"Placebo net excess ARR ({period})")
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"placebo_{period}_distribution.png", dpi=140)
        plt.close(fig)

    # 6 cumulative excess
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cid, color in (("K5_D1", "C3"), ("K20_D2", "C0")):
        d = pd.concat([daily_map[f"{cid}|valid"], daily_map[f"{cid}|test"]]).sort_index()
        ax.plot(d.index, (1 + d["net_excess"]).cumprod() - 1, label=cid, color=color)
    ax.legend()
    ax.set_title("Cumulative net excess (valid+test)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "cumulative_excess_K5_vs_K20.png", dpi=140)
    plt.close(fig)

    # 7 replacement curve
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    repl = nb[nb["topk"] == 5].copy()
    repl["nom_pct"] = repl["nominal_replacement"] * 100
    for period, axrow in (("valid", axes[0]), ("test", axes[1])):
        sub = repl[repl.period == period].sort_values("nom_pct")
        for ax, y, title in zip(
            axrow,
            ["net_excess_arr", "net_ir", "mean_turnover"],
            ["net excess ARR", "net IR", "mean turnover"],
        ):
            ax.plot(sub["nom_pct"], sub[y], marker="o")
            ax.set_title(f"{period}: {title}")
            ax.set_xlabel("nominal replacement %")
            ax.grid(True, alpha=0.3)
    fig.suptitle("K5 replacement neighbourhood")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "k5_replacement_curve.png", dpi=140)
    plt.close(fig)

    # also MDD / realised repl panel saved in same spirit — extra small
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, y in zip(axes, ["absolute_mdd", "realised_replacement"]):
        for period in ("valid", "test"):
            sub = repl[repl.period == period].sort_values("nom_pct")
            ax.plot(sub["nom_pct"], sub[y], marker="o", label=period)
        ax.set_title(y)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "k5_replacement_mdd_repl.png", dpi=140)
    plt.close(fig)

    # 8 breadth
    br = nb[nb["n_drop"] == 1].copy()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for period, axrow in (("valid", axes[0]), ("test", axes[1])):
        sub = br[br.period == period].sort_values("topk")
        axrow[0].plot(sub["topk"], sub["net_excess_arr"], marker="o")
        axrow[0].set_title(f"{period} ARR")
        axrow[1].plot(sub["topk"], sub["net_ir"], marker="o")
        axrow[1].set_title(f"{period} IR")
        for ax in axrow:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("K")
    fig.suptitle("Breadth neighbourhood (n_drop=1)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "k5_breadth_curve.png", dpi=140)
    plt.close(fig)

    # 9 contribution concentration
    if all_conc:
        conc = pd.concat(all_conc, ignore_index=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        sub = conc[(conc.period == "valid") & (conc.config_id == "K5_D1")]
        ax.bar(sub["top_k"].astype(str), sub["share_of_total_pnl"], color="C3")
        ax.set_ylabel("Share of total contribution P&L")
        ax.set_title("K5_D1 validation contribution concentration")
        fig.tight_layout()
        fig.savefig(OUT / "figures" / "contribution_concentration.png", dpi=140)
        plt.close(fig)

    # 10 monthly
    monthly = pd.read_csv(OUT / "monthly_returns.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    for cid, color in (("K5_D1", "C3"), ("K20_D2", "C0")):
        sub = monthly[(monthly.config_id == cid) & (monthly.period == "valid")].sort_values("month")
        ax.bar(sub["month"], sub["monthly_net_excess_sum"], alpha=0.5, label=cid, color=color)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    ax.set_title("Monthly net excess sum (validation)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "monthly_excess_heatmap_or_bar.png", dpi=140)
    plt.close(fig)


def _write_analysis(repro, boot_df, bucket_df, recon, conc_list, cf_store, time_sum, summ, nb, style, deltas):
    boot = {r["period"]: r for _, r in boot_df.iterrows()}
    ps = {s["period"]: s for s in summ}
    ts = {(t["config_id"], t["period"]): t for t in time_sum}
    conc = pd.concat(conc_list, ignore_index=True) if conc_list else pd.DataFrame()

    def interp_topdiff():
        v, t = boot["valid"], boot["test"]
        if v["ci95_low"] > 0 and t["ci95_low"] > 0:
            return "consistent evidence of incremental predictive information in the extreme top ranks"
        if v["ci95_low"] > 0 and not (t["ci95_low"] > 0):
            return "top-rank advantage is not stable across periods"
        return "rank 1–5 cannot be statistically distinguished from rank 6–20"

    def interp_placebo(period):
        return (
            "the within-Top20 ranking contributes materially to the K5 outcome"
            if ps[period]["true_percentile"] >= 95
            else "K5 performance cannot be clearly distinguished from random selection within the Top20 set"
        )

    repl = nb[nb.topk == 5]
    br = nb[nb.n_drop == 1]

    # replacement interpretation
    vrepl = repl[repl.period == "valid"].sort_values("n_drop")
    arrs = vrepl["net_excess_arr"].to_numpy()
    if len(arrs) >= 3 and arrs[0] > np.median(arrs[1:]) + 0.03:
        repl_interp = "performance is more consistent with a concentration × retention interaction"
    elif len(arrs) >= 3 and np.min(arrs) > 0.10:
        repl_interp = "Top5 concentration appears more important than the exact replacement intensity"
    elif len(arrs) >= 2 and arrs[-1] > arrs[0] + 0.02:
        repl_interp = "faster replacement may better match the short-lived top-rank signal"
    else:
        repl_interp = "no clear replacement-intensity mechanism is identified"

    # breadth interpretation
    vbr = br[br.period == "valid"].sort_values("topk")
    tbr = br[br.period == "test"].sort_values("topk")
    k5v = float(vbr[vbr.topk == 5]["net_excess_arr"].iloc[0])
    neighbors_v = vbr[vbr.topk.isin([4, 6])]["net_excess_arr"]
    if len(neighbors_v) and (neighbors_v.min() > k5v - 0.03):
        if len(tbr) and float(tbr[tbr.topk == 5]["net_excess_arr"].iloc[0]) > 0.05:
            breadth_interp = "cross-period local concentration stability"
        else:
            breadth_interp = "local breadth stability"
    elif len(neighbors_v) and (k5v - neighbors_v.max() > 0.04):
        if len(tbr) and abs(float(tbr[tbr.topk == 5]["net_excess_arr"].iloc[0]) - float(tbr[tbr.topk == 4]["net_excess_arr"].iloc[0] if 4 in set(tbr.topk) else 0)) > 0.03:
            breadth_interp = "isolated local peak"
        else:
            breadth_interp = "period-specific breadth advantage"
    else:
        breadth_interp = "period-specific breadth advantage"

    lines = [
        f"# K5_D1 post-hoc exploratory diagnosis — {OUT.name}",
        "",
        "**Identity:** POST-HOC EXPLORATORY / MECHANISM DIAGNOSIS. Not confirmatory selection.",
        "",
        "## 1. Original result reproduction",
        "",
    ]
    for r in repro:
        lines.append(f"- {r['config_id']} {r['period']}: excess_arr_net={r['excess_arr_net']:.6f} (expected≈{r['expected']}) OK={r['ok']}")

    lines += [
        "",
        "## 2. Rank-bucket evidence",
        "",
        "See `rank_bucket_returns.csv`. Higher buckets use LABEL0 on the same date as score (no extra shift).",
        "",
        "## 3. Rank1–5 vs Rank6–20 bootstrap",
        "",
        f"- Valid: mean D_top={boot['valid']['mean_D_top']:.6f}, CI=[{boot['valid']['ci95_low']:.6f},{boot['valid']['ci95_high']:.6f}], "
        f"one-sided p={boot['valid']['boot_p_one_sided_le0']:.4f}",
        f"- Test: mean D_top={boot['test']['mean_D_top']:.6f}, CI=[{boot['test']['ci95_low']:.6f},{boot['test']['ci95_high']:.6f}], "
        f"one-sided p={boot['test']['boot_p_one_sided_le0']:.4f}",
        f"- Interpretation: {interp_topdiff()}",
        "",
        "## 4. Contribution concentration",
        "",
        f"- Reconciliation: " + ", ".join(f"{r['config_id']}/{r['period']} max_err={r['max_abs_error']:.2e} pass={r['pass']}" for r in recon),
    ]
    if not conc.empty:
        for _, r in conc[(conc.config_id == "K5_D1") & (conc.period == "valid")].iterrows():
            lines.append(
                f"- K5 valid top{int(r.top_k)}: share_total_pnl={r.share_of_total_pnl:.3f}, "
                f"share_pos={r.share_of_positive_contribution:.3f}, share_abs={r.share_of_absolute_contribution:.3f}"
            )
    if "K5_D1|valid" in cf_store:
        cf = cf_store["K5_D1|valid"]
        lines.append(f"- Approx ex-top1 ARR (valid): {cf.get('ex_top1',{}).get('approx_net_excess_arr')}")
        lines.append(f"- Approx ex-top2 ARR (valid): {cf.get('ex_top2',{}).get('approx_net_excess_arr')}")
        lines.append("- Label: COUNTERFACTUAL CONTRIBUTION-REMOVAL APPROXIMATION")

    lines += ["", "## 5. Time concentration", ""]
    for key in (("K5_D1", "valid"), ("K5_D1", "test"), ("K20_D2", "valid")):
        t = ts[key]
        lines.append(
            f"- {key[0]} {key[1]}: best_month={t['best_month']} share={t['best_month_share']:.3f}, "
            f"best10_share={t['best10_share_of_net_excess_sum']:.3f}, "
            f"ex_best10_arr≈{t['approx_net_excess_arr_ex_best10']:.4f}, label={t.get('temporal_label','')}"
        )

    lines += [
        "",
        "## 6. Top20-internal placebo",
        "",
        f"- Valid: true percentile={ps['valid']['true_percentile']:.1f}%, p={ps['valid']['empirical_one_sided_p_placebo_ge_true']:.4f}; {interp_placebo('valid')}",
        f"- Test: true percentile={ps['test']['true_percentile']:.1f}%, p={ps['test']['empirical_one_sided_p_placebo_ge_true']:.4f}; {interp_placebo('test')}",
        f"- Valid turnover diff (true−placebo mean)={ps['valid']['turnover_diff_true_minus_placebo_mean']:.4f}",
        "",
        "## 7. K5 replacement neighbourhood",
        "",
    ]
    for _, r in repl[repl.period == "valid"].sort_values("n_drop").iterrows():
        lines.append(
            f"- {r.config_id}: ARR={r.net_excess_arr:.4f}, IR={r.net_ir:.3f}, TO={r.mean_turnover:.3f}, "
            f"realised_repl={r.realised_replacement:.3f}"
        )
    lines.append(f"- Interpretation: {repl_interp}")

    lines += ["", "## 8. K3–K7 breadth neighbourhood", ""]
    for _, r in br[br.period == "valid"].sort_values("topk").iterrows():
        lines.append(f"- {r.config_id}: ARR={r.net_excess_arr:.4f}, IR={r.net_ir:.3f}, TO={r.mean_turnover:.3f}")
    lines.append(f"- Interpretation: {breadth_interp}")
    for d in deltas:
        lines.append(f"- {d['comparison']} {d['period']}: ΔARR={d['delta_arr']:.4f}, ΔIR={d['delta_ir']:.4f}")

    lines += ["", "## 9. Style exposure", ""]
    beta = style[(style.model == "full") & (style.config_id.isin(["K5_D1", "K20_D2"]))]
    for _, r in beta.iterrows():
        lines.append(f"- {r.config_id} {r.period}: beta={r.beta:.3f}, alpha={r.alpha:.6f}, R2={r.r2:.3f}")
    lines.append("- FF/MOM: NOT_AVAILABLE; size log mcap: NOT_AVAILABLE")
    char = style.dropna(subset=["mean_port_vol20"]) if "mean_port_vol20" in style.columns else pd.DataFrame()
    for _, r in char.iterrows():
        lines.append(
            f"- {r.config_id} {r.period}: vol20 port−uni={r.vol20_port_minus_uni:.5f}, "
            f"mom12-1 port−uni={r.mom_port_minus_uni:.5f}"
        )

    # integrated
    bits = []
    bits.append("A" if "consistent" in interp_topdiff() else ("F" if "not stable" in interp_topdiff() else "mixed"))
    if ps["valid"]["true_percentile"] >= 95:
        bits.append("A/ranking")
    else:
        bits.append("concentration-without-precise-rank")
    if "retention" in repl_interp:
        bits.append("B")
    if not conc.empty:
        s1 = conc[(conc.config_id == "K5_D1") & (conc.period == "valid") & (conc.top_k == 1)]
        if len(s1) and float(s1.iloc[0]["share_of_total_pnl"]) > 0.25:
            bits.append("C")
    if ts[("K5_D1", "valid")].get("temporal_label") == "TEMPORALLY_CONCENTRATED":
        bits.append("D")
    bits.append("G_MIXED")

    lines += [
        "",
        "## 10. Integrated interpretation",
        "",
        "K5_D1's validation advantage is best read as **MIXED EVIDENCE**:",
        f"- Top-rank signal: {interp_topdiff()}.",
        f"- Placebo (valid): {interp_placebo('valid')}.",
        f"- Replacement axis: {repl_interp}.",
        f"- Breadth axis: {breadth_interp}.",
        f"- Time: {ts[('K5_D1','valid')].get('temporal_label')}.",
        "- This study does **not** declare K5_D1 an automatic final static winner.",
        "",
        f"- Tags: {', '.join(bits)}",
    ]
    (OUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # append mapping results to decision log
    with (OUT / "decision_log.md").open("a", encoding="utf-8") as f:
        f.write("\n### [POST-RUN] Results appended (rules were pre-frozen)\n")
        f.write(f"- topdiff interp: {interp_topdiff()}\n")
        f.write(f"- placebo valid: {interp_placebo('valid')}\n")
        f.write(f"- replacement: {repl_interp}\n")
        f.write(f"- breadth: {breadth_interp}\n")


def _print_summary(boot_df, summ, conc_list, cf_store, time_sum, nb, style):
    boot = {r["period"]: r for _, r in boot_df.iterrows()}
    ps = {s["period"]: s for s in summ}
    conc = pd.concat(conc_list, ignore_index=True)
    ts = {(t["config_id"], t["period"]): t for t in time_sum}
    repl = nb[(nb.topk == 5) & (nb.period == "valid")].sort_values("n_drop")
    br = nb[(nb.n_drop == 1) & (nb.period == "valid")].sort_values("topk")
    print("=" * 72)
    print("K5_D1 POST-HOC DIAGNOSIS COMPLETE")
    print()
    print("A. rank1–5 vs 6–20")
    print(f"- valid: mean={boot['valid']['mean_D_top']:.6f} CI=[{boot['valid']['ci95_low']:.6f},{boot['valid']['ci95_high']:.6f}] p={boot['valid']['boot_p_one_sided_le0']:.4f}")
    print(f"- test: mean={boot['test']['mean_D_top']:.6f} CI=[{boot['test']['ci95_low']:.6f},{boot['test']['ci95_high']:.6f}] p={boot['test']['boot_p_one_sided_le0']:.4f}")
    print()
    print("B. placebo")
    print(f"- valid true percentile: {ps['valid']['true_percentile']:.1f}%")
    print(f"- test true percentile: {ps['test']['true_percentile']:.1f}%")
    print()
    print("C. contribution concentration")
    for _, r in conc[(conc.config_id == "K5_D1") & (conc.period == "valid")].iterrows():
        if int(r.top_k) in (1, 3):
            print(f"- top{int(r.top_k)}: share_total_pnl={r.share_of_total_pnl:.3f}")
    if "K5_D1|valid" in cf_store:
        print(f"- ex-top2 result: ARR≈{cf_store['K5_D1|valid'].get('ex_top2',{}).get('approx_net_excess_arr')}")
    print()
    print("D. time concentration")
    t = ts[("K5_D1", "valid")]
    print(f"- best month contribution: {t['best_month']} share={t['best_month_share']:.3f}")
    print(f"- ex-best10-days: ARR≈{t['approx_net_excess_arr_ex_best10']:.4f} ({t.get('temporal_label')})")
    print()
    print("E. replacement neighbourhood")
    for _, r in repl.iterrows():
        print(f"{r.config_id}: ARR={r.net_excess_arr:.4f}")
    print()
    print("F. breadth neighbourhood")
    for _, r in br.iterrows():
        print(f"{r.config_id}: ARR={r.net_excess_arr:.4f}")
    print()
    print("G. style exposure")
    print("- see style_exposure.csv (FF/MOM NOT_AVAILABLE; size NOT_AVAILABLE)")
    print()
    print("H. integrated conclusion")
    print("- See analysis.md §10 (MIXED EVIDENCE; not automatic winner).")
    print()
    print("I. output directory")
    print(OUT)
    print("=" * 72)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
