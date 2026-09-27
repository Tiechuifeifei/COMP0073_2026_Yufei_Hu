#!/usr/bin/env python3
"""2024–2025 matched 2×2: signals A/B × Top20/drop2 and Top5/drop1.

Evaluation against frozen prediction artefacts (no retrain). Step 1 checks
native-universe parity; on pass, builds the unified-intersection main table."""

from __future__ import annotations
import os

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_US
from qlib.contrib.evaluate import risk_analysis
from qlib.data import D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "03_fundamentals_news"))
sys.path.insert(0, str(PROJECT_ROOT / "04_portfolio"))
sys.path.insert(0, str(PROJECT_ROOT / "05_later_evaluation" / "precommit"))
sys.path.insert(0, str(Path(os.environ["RDAGENT_ROOT"]) / "phase3_portfolio_ablation" / "scripts"))

import F2P_portfolio_backtest as f2p  # noqa: E402
import run_portfolio_ablation as rpa  # noqa: E402
import holdout_runner as hr  # noqa: E402
try:
    from portfolio_topk_drop_joint import holdings_diagnostics  # noqa: E402
except ImportError:  # pragma: no cover
    holdings_diagnostics = None
from metrics_utils import (  # noqa: E402
    load_r1r2p_module,
    run_qlib_backtest,
)
from signal_utils import pred_df_to_qlib  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("matched_2x2")

QLIB_DATA = PROJECT_ROOT / "staging/qlib_data"
CAL_PATH = QLIB_DATA / "calendars/day.txt"
SENTINEL = "2026-01-02"

A_HOLDOUT = PROJECT_ROOT / "reports/holdout_2024_2025_K5D1_WD_no_retrain_20260921_000350"
A_PRED = A_HOLDOUT / "predictions.parquet"
H0_DAILY = PROJECT_ROOT / "data/portfolio_experiments/final_holdout/attribution/holdout_h0_daily_returns.parquet"
P5_DAILY = PROJECT_ROOT / "data/portfolio_experiments/topk_breadth/phase_c_daily_returns.parquet"
TOPK_PERIOD = PROJECT_ROOT / "data/portfolio_experiments/topk_breadth/period_summary.csv"
TOPK_MANIFEST = PROJECT_ROOT / "data/portfolio_experiments/topk_breadth/run_manifest.json"

HOLD_THRESH = 1
RISK_DEGREE = 0.95
OPEN_COST = 0.0001
CLOSE_COST = 0.0001
ACCOUNT = 1e8
BENCHMARK = "P84398"
N_CAGR = 252
BOOT_SEED = 20260923
BOOT_BLOCK = 21
BOOT_N = 2000
MET_TOL_PP = 0.05  # ±0.05 percentage points
DAILY_TOL = 1e-10

EXPECTED_A = {
    "K5_D1": {"cagr": 0.0850, "wealth_mdd": -0.2073, "sharpe": 0.51},
    "K20_D2": {"cagr": 0.1220, "wealth_mdd": -0.1631, "sharpe": 0.79},
    "SPY": {"cagr": 0.1986},
}

EVIDENCE_TAG = {
    ("A", "K20_D2"): "locked-specification later-period",
    ("A", "K5_D1"): "locked-specification later-period",
    ("B", "K20_D2"): "H0 frozen engine",
    ("B", "K5_D1"): "post-hoc（规则形成于 H0 打开 2024–2025 之后）",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_series(s: pd.Series) -> str:
    arr = pd.to_numeric(s, errors="coerce").astype("float64").fillna(0).values.tobytes()
    idx = "\n".join(map(str, s.index.tolist())).encode()
    return hashlib.sha256(arr + idx).hexdigest()


def wealth_metrics(r: pd.Series, bench: pd.Series | None = None) -> dict[str, Any]:
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {"n_days": n, "status": "TOO_SHORT"}
    W = (1.0 + r).cumprod()
    w0 = pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)])
    W_ext = pd.concat([w0, W])
    peak = W_ext.cummax()
    dd = W_ext / peak - 1.0
    mdd = float(dd.min())
    cum = float(W.iloc[-1] - 1.0)
    cagr = float(W.iloc[-1] ** (N_CAGR / n) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(N_CAGR))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(N_CAGR)) if r.std(ddof=1) > 0 else np.nan
    out: dict[str, Any] = {
        "n_days": n,
        "cum_return": cum,
        "cagr": cagr,
        "ann_vol": vol,
        "wealth_mdd": mdd,
        "sharpe_rf0": sharpe,
        "first_date": str(pd.Timestamp(r.index[0]).date()),
        "last_date": str(pd.Timestamp(r.index[-1]).date()),
    }
    if bench is not None:
        bb = bench.reindex(r.index)
        if bb.isna().any():
            out["spy_note"] = "bench has NA on strategy dates"
        bb2 = bb.dropna()
        rr = r.reindex(bb2.index)
        Wb = (1.0 + bb2).cumprod()
        out["spy_cagr"] = float(Wb.iloc[-1] ** (N_CAGR / len(bb2)) - 1.0)
        out["spy_cum"] = float(Wb.iloc[-1] - 1.0)
        out["cagr_minus_spy"] = float(out["cagr"] - out["spy_cagr"])
        # legacy ARR excess = qlib risk_analysis annualized_return on (return - bench - cost)
        # caller may pass net already; here excess on wealth net vs bench
        ex = rr - bb2
        try:
            out["qlib_excess_arr_mean_x_238"] = float(
                risk_analysis(ex, freq="day").loc["annualized_return", "risk"]
            )
        except Exception:
            out["qlib_excess_arr_mean_x_238"] = float(ex.mean() * 238)
    return out


def load_port_cfg(topk: int, n_drop: int) -> dict[str, Any]:
    cfg = f2p.load_port_config()
    cfg["strategy"]["kwargs"]["topk"] = int(topk)
    cfg["strategy"]["kwargs"]["n_drop"] = int(n_drop)
    cfg["strategy"]["kwargs"]["hold_thresh"] = HOLD_THRESH
    cfg["strategy"]["kwargs"]["risk_degree"] = RISK_DEGREE
    for k, v in rpa.STRATEGY_DEFAULTS.items():
        cfg["strategy"]["kwargs"].setdefault(k, v)
    cfg["backtest"]["account"] = ACCOUNT
    cfg["backtest"]["benchmark"] = BENCHMARK
    cfg["backtest"]["exchange_kwargs"]["deal_price"] = "close"
    cfg["backtest"]["exchange_kwargs"]["open_cost"] = OPEN_COST
    cfg["backtest"]["exchange_kwargs"]["close_cost"] = CLOSE_COST
    cfg["backtest"]["exchange_kwargs"]["min_cost"] = 0
    cfg["backtest"]["exchange_kwargs"]["limit_threshold"] = None
    return cfg


def run_bt(pred: pd.DataFrame, topk: int, n_drop: int, start: str, end: str) -> dict[str, Any]:
    cfg = load_port_cfg(topk, n_drop)
    cfg["backtest"]["start_time"] = start
    cfg["backtest"]["end_time"] = end
    dates = pred.index.get_level_values("datetime")
    sub = pred.loc[(dates >= pd.Timestamp(start) - pd.Timedelta(days=10)) & (dates <= pd.Timestamp(end))]
    return rpa.run_backtest(cfg, sub)


def with_calendar_sentinel(fn):
    text = CAL_PATH.read_text()
    bak = CAL_PATH.with_suffix(".txt.bak_matched_2x2")
    shutil.copy2(CAL_PATH, bak)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if SENTINEL not in lines:
        lines.append(SENTINEL)
        CAL_PATH.write_text("\n".join(lines) + "\n")
        log.info("Appended calendar sentinel %s", SENTINEL)
    try:
        try:
            qlib.reset()
        except Exception:
            pass
        qlib.init(
            provider_uri=str(QLIB_DATA),
            region=REG_US,
            kernels=1,
            expression_cache=None,
            dataset_cache=None,
        )
        return fn()
    finally:
        shutil.copy2(bak, CAL_PATH)
        bak.unlink(missing_ok=True)
        try:
            qlib.reset()
        except Exception:
            pass
        qlib.init(
            provider_uri=str(QLIB_DATA),
            region=REG_US,
            kernels=1,
            expression_cache=None,
            dataset_cache=None,
        )
        log.info("Restored calendar")


def init_qlib_dedicated(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Disable redis/expression/dataset caches; local scratch only for any pickle we write.
    try:
        qlib.reset()
    except Exception:
        pass
    qlib.init(
        provider_uri=str(QLIB_DATA),
        region=REG_US,
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
    )
    (cache_dir / "README.txt").write_text(
        "Dedicated cache for matched_2x2. expression_cache=None, dataset_cache=None. "
        "Do not reuse old qlib redis/disk caches. Cache keys intentionally omit window.\n"
    )


def median_holding_days(positions: dict, n_drop: int = 1) -> dict[str, float]:
    hd = holdings_diagnostics(positions, n_drop)
    return {
        "median_holding_duration_days": float(hd.get("median_holding_duration_days", float("nan"))),
        "mean_holding_duration_days": float(hd.get("mean_holding_duration_days", float("nan"))),
        "avg_holdings": float(hd.get("avg_holdings", float("nan"))),
    }


def ols_alpha_beta(y: pd.Series, x: pd.Series) -> dict[str, float]:
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 10:
        return {"alpha": np.nan, "beta": np.nan, "t_alpha": np.nan, "t_beta": np.nan, "n": len(df)}
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    yv = df["y"].values
    beta_hat, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta_hat
    dof = max(len(df) - 2, 1)
    s2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    t = beta_hat / se
    return {
        "alpha": float(beta_hat[0]),
        "beta": float(beta_hat[1]),
        "t_alpha": float(t[0]),
        "t_beta": float(t[1]),
        "n": int(len(df)),
    }


def block_bootstrap_ci(
    series: pd.Series, *, block: int, n_boot: int, seed: int, stat: str = "mean"
) -> dict[str, float]:
    x = series.dropna().values.astype(float)
    n = len(x)
    if n < block + 2:
        return {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))

    def _stat(a: np.ndarray) -> float:
        if stat == "mean":
            return float(a.mean())
        if stat == "cagr_from_daily":
            # a is daily returns
            return float((1.0 + a).prod() ** (N_CAGR / len(a)) - 1.0)
        raise ValueError(stat)

    point = _stat(x)
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        chunks = [x[s : s + block] for s in starts]
        sample = np.concatenate(chunks)[:n]
        boots.append(_stat(sample))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi), "n": n}


def daily_ic(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """pred MultiIndex score; labels datetime/instrument/label."""
    p = pred.reset_index()
    lab = labels.copy()
    lab["datetime"] = pd.to_datetime(lab["datetime"]).dt.normalize()
    lab["instrument"] = lab["instrument"].astype(str)
    m = p.merge(lab, on=["datetime", "instrument"], how="inner")
    rows = []
    for dt, g in m.groupby("datetime"):
        g = g.dropna(subset=["score", "label"])
        if len(g) < 20:
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


def build_signal_b_holdout() -> tuple[pd.DataFrame, pd.DataFrame, bool, dict]:
    """Return (qlib_pred, scored_long, flip, meta). Flip from valid IC; freeze applied to holdout."""
    panel = hr.load_holdout_panel()
    hmm2 = hr.build_2state_hmm()
    lmap = hr.load_2state_label_map()
    flip = bool(hr.infer_flip(panel, "W1", hmm2, lmap))
    pred, sub = hr.build_holdout_pred(panel, "W1", hmm2, lmap, flip=flip)
    meta = {
        "flip_w1_holdout_infer": flip,
        "flip_actually_applied": flip,
        "formula": "W1 = 0.7*z(D2)+0.3*z(F1C) after daily winsorize+z",
        "n_rows": int(len(sub)),
        "n_dates": int(sub["datetime"].nunique()),
        "date_min": str(pd.to_datetime(sub["datetime"]).min().date()),
        "date_max": str(pd.to_datetime(sub["datetime"]).max().date()),
    }
    return pred, sub, flip, meta


def report_to_frame(report: pd.DataFrame) -> pd.DataFrame:
    d = report.copy()
    d.index = pd.to_datetime(d.index).normalize()
    d["net"] = d["return"] - d["cost"]
    d["excess_net"] = d["return"] - d["bench"] - d["cost"]
    return d


def metrics_bundle(d: pd.DataFrame, positions: dict | None = None) -> dict[str, Any]:
    m = wealth_metrics(d["net"], d["bench"])
    m["mean_daily_turnover"] = float(d["turnover"].mean()) if "turnover" in d else np.nan
    m["mean_cost"] = float(d["cost"].mean())
    try:
        m["qlib_excess_arr_legacy"] = float(
            risk_analysis(d["excess_net"], freq="day").loc["annualized_return", "risk"]
        )
    except Exception:
        m["qlib_excess_arr_legacy"] = float(d["excess_net"].mean() * 238)
    # CAGR-gap excess (same spirit as ~−5.4% H0 note)
    m["cagr_gap_vs_spy"] = m.get("cagr_minus_spy", np.nan)
    ab = ols_alpha_beta(d["net"], d["bench"])
    m.update({f"ols_{k}": v for k, v in ab.items()})
    if positions is not None:
        # n_drop unknown here; pass 1 as unused for duration stats
        m.update(median_holding_days(positions, n_drop=1))
    return m


def period_slice(d: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return d.loc[(d.index >= start) & (d.index <= end)]


def compare_daily(a: pd.Series, b: pd.Series, tol: float) -> dict[str, Any]:
    aa = a.copy()
    bb = b.copy()
    aa.index = pd.to_datetime(aa.index).normalize()
    bb.index = pd.to_datetime(bb.index).normalize()
    common = aa.index.intersection(bb.index)
    if len(common) == 0:
        return {"pass": False, "reason": "no_common_dates", "max_abs": np.nan, "n": 0}
    diff = (aa.reindex(common) - bb.reindex(common)).abs()
    mx = float(diff.max())
    return {
        "pass": bool(mx <= tol) and len(common) == len(aa) == len(bb),
        "max_abs": mx,
        "n_common": int(len(common)),
        "n_a": int(len(aa)),
        "n_b": int(len(bb)),
        "n_mismatch_dates": int(abs(len(aa) - len(bb)) + abs(len(aa) - len(common))),
        "tol": tol,
    }


def extract_topk_breadth_controls() -> pd.DataFrame:
    ps = pd.read_csv(TOPK_PERIOD)
    arms = ["K10", "K5", "P10"]
    periods = ["2024", "2025", "2024-2025"]
    sub = ps[ps["arm_id"].isin(arms) & ps["period"].isin(periods)].copy()
    sub["source"] = "topk_breadth/period_summary.csv (as-is, not rerun)"
    return sub.sort_values(["arm_id", "period"])


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------

def step1_native(out: Path, cache_dir: Path) -> dict[str, Any]:
    init_qlib_dedicated(cache_dir)
    results: dict[str, Any] = {"gates": [], "all_pass": True, "flip_b": None}
    mismatches: list[str] = []

    # --- Signal A predictions ---
    pred_a = pd.read_parquet(A_PRED)
    pred_a = pred_a.copy()
    pred_a.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(pred_a.index.get_level_values(0)).normalize(),
            pred_a.index.get_level_values(1).astype(str),
        ],
        names=["datetime", "instrument"],
    )
    results["signal_a_pred_sha256"] = sha256_file(A_PRED)
    results["signal_a_score_sha256"] = sha256_series(pred_a["score"])

    # --- Signal B ---
    pred_b, scored_b, flip_b, meta_b = build_signal_b_holdout()
    results["flip_b"] = meta_b
    results["signal_b_score_sha256"] = sha256_series(pred_b["score"])
    scored_b.to_parquet(out / "signal_b_w1_scored_native.parquet", index=False)
    pred_b.to_pickle(out / "signal_b_w1_pred_qlib.pkl")
    log.info("W1 flip on 2024–2025: applied=%s (infer_flip=%s)", flip_b, flip_b)

    # Persist scored A for universe intersection later
    scored_a = pred_a.reset_index()
    scored_a.to_parquet(out / "signal_a_fbase_scored_native.parquet", index=False)

    # === A × K5 / K20 with YE end (known metrics) ===
    def _run_a():
        outs = {}
        for cid, topk, nd in [("K5_D1", 5, 1), ("K20_D2", 20, 2)]:
            log.info("Step1 A native %s …", cid)
            bt = run_bt(pred_a, topk, nd, "2024-01-01", "2025-12-31")
            d = report_to_frame(bt["report"])
            m = metrics_bundle(d, bt.get("positions"))
            outs[cid] = {"daily": d, "metrics": m, "positions": bt.get("positions")}
        return outs

    a_runs = with_calendar_sentinel(_run_a)

    # SPY from A K5 bench path
    spy_d = a_runs["K5_D1"]["daily"]["bench"]
    spy_m = wealth_metrics(spy_d)
    results["spy_metrics"] = spy_m

    for cid, exp in [("K5_D1", EXPECTED_A["K5_D1"]), ("K20_D2", EXPECTED_A["K20_D2"])]:
        m = a_runs[cid]["metrics"]
        row = {
            "cell": f"A×{cid}",
            "metric": "cagr/mdd/sharpe",
            "expected": exp,
            "got": {
                "cagr": m["cagr"],
                "wealth_mdd": m["wealth_mdd"],
                "sharpe": m["sharpe_rf0"],
            },
        }
            # ±0.05pp on CAGR/MDD (0.0005 in decimal); ±0.05 absolute on Sharpe
        ok = (
            abs(m["cagr"] - exp["cagr"]) <= 0.0005
            and abs(m["wealth_mdd"] - exp["wealth_mdd"]) <= 0.0005
            and abs(m["sharpe_rf0"] - exp["sharpe"]) <= 0.05
        )
        row["pass"] = bool(ok)
        row["delta"] = {
            "cagr_pp": (m["cagr"] - exp["cagr"]) * 100,
            "mdd_pp": (m["wealth_mdd"] - exp["wealth_mdd"]) * 100,
            "sharpe": m["sharpe_rf0"] - exp["sharpe"],
        }
        results["gates"].append(row)
        if not ok:
            results["all_pass"] = False
            mismatches.append(f"A×{cid} metrics fail: {row}")
        a_runs[cid]["daily"].to_csv(out / f"step1_native_A_{cid}_daily.csv")
        pd.Series(m).to_json(out / f"step1_native_A_{cid}_metrics.json")

    spy_ok = abs(spy_m["cagr"] - EXPECTED_A["SPY"]["cagr"]) <= 0.0005
    spy_row = {
        "cell": "SPY",
        "expected_cagr": EXPECTED_A["SPY"]["cagr"],
        "got_cagr": spy_m["cagr"],
        "delta_pp": (spy_m["cagr"] - EXPECTED_A["SPY"]["cagr"]) * 100,
        "pass": bool(spy_ok),
    }
    results["gates"].append(spy_row)
    if not spy_ok:
        results["all_pass"] = False
        mismatches.append(f"SPY CAGR fail: {spy_row}")

    # === B × Top20 vs H0 daily ===
    log.info("Step1 B native Top20/drop2 …")
    init_qlib_dedicated(cache_dir)
    r1r2p = load_r1r2p_module(PROJECT_ROOT)
    # Use same mechanical path as holdout (run_qlib_backtest)
    bt_h0 = run_qlib_backtest(
        r1r2p,
        rpa,
        pred_b,
        "2024-01-01",
        "2025-12-29",
        extra_kwargs={"topk": 20, "n_drop": 2, "hold_thresh": 1, "risk_degree": 0.95},
    )
    d_h0 = report_to_frame(bt_h0["report"])
    ref_h0 = pd.read_parquet(H0_DAILY).set_index("trade_date")
    ref_h0.index = pd.to_datetime(ref_h0.index).normalize()
    cmp_h0 = compare_daily(d_h0["return"], ref_h0["return"], DAILY_TOL)
    cmp_h0["cell"] = "B×Top20/drop2 vs holdout_h0_daily_returns"
    cmp_h0["flip"] = flip_b
    results["gates"].append(cmp_h0)
    if not cmp_h0["pass"]:
        results["all_pass"] = False
        mismatches.append(f"H0 daily parity fail: {cmp_h0}")
    d_h0.to_csv(out / "step1_native_B_Top20_daily.csv")

    # === B × Top5 vs Phase C P5 ===
    log.info("Step1 B native Top5/drop1 …")
    bt_p5 = run_qlib_backtest(
        r1r2p,
        rpa,
        pred_b,
        "2024-01-01",
        "2025-12-29",
        extra_kwargs={"topk": 5, "n_drop": 1, "hold_thresh": 1, "risk_degree": 0.95},
    )
    d_p5 = report_to_frame(bt_p5["report"])
    ref_p5 = pd.read_parquet(P5_DAILY)
    ref_p5 = ref_p5[(ref_p5["arm_id"] == "P5") & (ref_p5["window"] == "2024-2025")].set_index("trade_date")
    ref_p5.index = pd.to_datetime(ref_p5.index).normalize()
    cmp_p5 = compare_daily(d_p5["return"], ref_p5["return"], DAILY_TOL)
    cmp_p5["cell"] = "B×Top5/drop1 vs phase_c P5"
    cmp_p5["flip"] = flip_b
    results["gates"].append(cmp_p5)
    if not cmp_p5["pass"]:
        results["all_pass"] = False
        mismatches.append(f"P5 daily parity fail: {cmp_p5}")
    d_p5.to_csv(out / "step1_native_B_Top5_daily.csv")

    # Save native runs for later
    results["a_runs_paths"] = {
        "K5_D1": str(out / "step1_native_A_K5_D1_daily.csv"),
        "K20_D2": str(out / "step1_native_A_K20_D2_daily.csv"),
    }
    results["b_native"] = {
        "Top20": metrics_bundle(d_h0, bt_h0["result"].get("positions")),
        "Top5": metrics_bundle(d_p5, bt_p5["result"].get("positions")),
    }
    # also store objects for step2 if pass — pickle dailies already saved
    with (out / "step1_a_runs_metrics.json").open("w") as f:
        json.dump({k: v["metrics"] for k, v in a_runs.items()}, f, indent=2, default=str)

    results["mismatches"] = mismatches
    results["a_runs"] = {k: {"metrics": v["metrics"]} for k, v in a_runs.items()}
    # Keep full daily frames for step2 via CSV reload
    return results


def build_unified_preds(
    scored_a: pd.DataFrame, scored_b: pd.DataFrame, out: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Intersect universes by day; keep native z-scores (no re-z)."""
    a = scored_a.copy()
    b = scored_b.copy()
    a["datetime"] = pd.to_datetime(a["datetime"]).dt.normalize()
    b["datetime"] = pd.to_datetime(b["datetime"]).dt.normalize()
    a["instrument"] = a["instrument"].astype(str)
    b["instrument"] = b["instrument"].astype(str)
    # B scored has 'score'; A has 'score'
    a_keys = a[["datetime", "instrument", "score"]].rename(columns={"score": "score_a"})
    b_keys = b[["datetime", "instrument", "score"]].rename(columns={"score": "score_b"})
    inter = a_keys.merge(b_keys, on=["datetime", "instrument"], how="inner")

    # diagnostics
    rows = []
    for dt, g in inter.groupby("datetime"):
        na = int((a["datetime"] == dt).sum())
        nb = int((b["datetime"] == dt).sum())
        ni = int(len(g))
        rows.append(
            {
                "datetime": dt,
                "n_intersect": ni,
                "n_a_native": na,
                "n_b_native": nb,
                "n_a_dropped": na - ni,
                "n_b_dropped": nb - ni,
            }
        )
    diag = pd.DataFrame(rows).sort_values("datetime")
    diag.to_csv(out / "universe_intersection_daily.csv", index=False)
    summary = {
        "n_days": int(len(diag)),
        "intersect_min": int(diag["n_intersect"].min()) if len(diag) else 0,
        "intersect_p50": float(diag["n_intersect"].median()) if len(diag) else 0,
        "intersect_mean": float(diag["n_intersect"].mean()) if len(diag) else 0,
        "intersect_max": int(diag["n_intersect"].max()) if len(diag) else 0,
        "a_dropped_mean": float(diag["n_a_dropped"].mean()) if len(diag) else 0,
        "b_dropped_mean": float(diag["n_b_dropped"].mean()) if len(diag) else 0,
        "a_dropped_max": int(diag["n_a_dropped"].max()) if len(diag) else 0,
        "b_dropped_max": int(diag["n_b_dropped"].max()) if len(diag) else 0,
        "note": "Native z-scores preserved; ranking restricted to intersection; NO re-zscore",
    }
    (out / "universe_intersection_summary.json").write_text(json.dumps(summary, indent=2))

    pred_a_u = pred_df_to_qlib(inter.rename(columns={"score_a": "score"})[["datetime", "instrument", "score"]])
    pred_b_u = pred_df_to_qlib(inter.rename(columns={"score_b": "score"})[["datetime", "instrument", "score"]])
    return pred_a_u, pred_b_u, diag


def run_cell(
    pred: pd.DataFrame, topk: int, n_drop: int, start: str, end: str, use_sentinel: bool
) -> tuple[pd.DataFrame, dict, dict]:
    def _go():
        bt = run_bt(pred, topk, n_drop, start, end)
        d = report_to_frame(bt["report"])
        m = metrics_bundle(d, bt.get("positions"))
        return d, m, bt

    if use_sentinel:
        return with_calendar_sentinel(_go)
    return _go()


def step2_and_3(out: Path, cache_dir: Path, step1: dict) -> None:
    init_qlib_dedicated(cache_dir)
    scored_a = pd.read_parquet(out / "signal_a_fbase_scored_native.parquet")
    scored_b = pd.read_parquet(out / "signal_b_w1_scored_native.parquet")
    pred_a_u, pred_b_u, diag = build_unified_preds(scored_a, scored_b, out)

    # Also keep native preds
    pred_a_n = pd.read_parquet(A_PRED)
    pred_b_n = pd.read_pickle(out / "signal_b_w1_pred_qlib.pkl")

    cells = [
        ("A", "K20_D2", 20, 2, pred_a_n, pred_a_u),
        ("A", "K5_D1", 5, 1, pred_a_n, pred_a_u),
        ("B", "K20_D2", 20, 2, pred_b_n, pred_b_u),
        ("B", "K5_D1", 5, 1, pred_b_n, pred_b_u),
    ]

    rows = []
    daily_store = {}
    for sig, rule, topk, nd, pred_n, pred_u in cells:
        for univ, pred in [("native", pred_n), ("unified", pred_u)]:
            log.info("Step2 %s × %s (%s) …", sig, rule, univ)
            d, m, bt = run_cell(pred, topk, nd, "2024-01-01", "2025-12-31", use_sentinel=True)
            key = f"{sig}_{rule}_{univ}"
            d.to_csv(out / f"daily_{key}.csv")
            daily_store[key] = d
            for period, (s, e) in {
                "2024": ("2024-01-01", "2024-12-31"),
                "2025": ("2025-01-01", "2025-12-31"),
                "2024_2025": ("2024-01-01", "2025-12-31"),
            }.items():
                sub = period_slice(d, s, e)
                mm = metrics_bundle(sub, None if period != "2024_2025" else bt.get("positions"))
                rows.append(
                    {
                        "signal": sig,
                        "rule": rule,
                        "universe": univ,
                        "period": period,
                        "evidence_tag": EVIDENCE_TAG[(sig, rule)],
                        **mm,
                    }
                )

    res_df = pd.DataFrame(rows)
    res_df.to_csv(out / "matched_2x2_results.csv", index=False)

    # IC tables
    # Labels: from holdout panel for B; for A use same LABEL0 from panel merge if possible
    panel = hr.load_holdout_panel()
    labels = panel[["datetime", "instrument", "label"]].copy()
    ic_rows = []
    for sig, pred, scored in [
        ("A", pred_a_u, scored_a),
        ("B", pred_b_u, scored_b),
    ]:
        # use native scores on their native dates but only where label realized
        p = pred_df_to_qlib(scored[["datetime", "instrument", "score"]]) if "score" in scored.columns else pred
        # For A scored from parquet has score; for B too
        p = pred_df_to_qlib(
            scored.assign(
                datetime=pd.to_datetime(scored["datetime"]).dt.normalize(),
                instrument=scored["instrument"].astype(str),
            )[["datetime", "instrument", "score"]]
        )
        dic = daily_ic(p, labels)
        dic.to_csv(out / f"daily_ic_signal_{sig}.csv", index=False)
        for period, (s, e) in {
            "2024": ("2024-01-01", "2024-12-31"),
            "2025": ("2025-01-01", "2025-12-31"),
            "2024_2025": ("2024-01-01", "2025-12-31"),
        }.items():
            sub = dic[(dic["datetime"] >= s) & (dic["datetime"] <= e)].dropna(subset=["ic"])
            # realization filter: label present already; exclude days without label via dropna
            for col in ("ic", "rank_ic"):
                boot = block_bootstrap_ci(sub[col], block=BOOT_BLOCK, n_boot=BOOT_N, seed=BOOT_SEED + hash(sig + period + col) % 10000)
                ic_rows.append(
                    {
                        "signal": sig,
                        "period": period,
                        "metric": col,
                        "mean": float(sub[col].mean()) if len(sub) else np.nan,
                        "n_days": int(len(sub)),
                        "boot_ci_lo": boot["ci_lo"],
                        "boot_ci_hi": boot["ci_hi"],
                        "boot_block": BOOT_BLOCK,
                        "boot_n": BOOT_N,
                        "boot_seed": BOOT_SEED,
                    }
                )
    pd.DataFrame(ic_rows).to_csv(out / "ic_summary.csv", index=False)

    # Step3 diffs on unified universe full period
    def net_series(sig, rule):
        return daily_store[f"{sig}_{rule}_unified"]["net"]

    def excess_series(sig, rule):
        return daily_store[f"{sig}_{rule}_unified"]["excess_net"]

    diffs = []
    pairs = [
        ("A-B @Top20", ("A", "K20_D2"), ("B", "K20_D2")),
        ("A-B @Top5", ("A", "K5_D1"), ("B", "K5_D1")),
        ("Top5-Top20 @A", ("A", "K5_D1"), ("A", "K20_D2")),
        ("Top5-Top20 @B", ("B", "K5_D1"), ("B", "K20_D2")),
    ]
    for name, left, right in pairs:
        la, lr = left
        ra, rr = right
        n1, n2 = net_series(la, lr), net_series(ra, rr)
        e1, e2 = excess_series(la, lr), excess_series(ra, rr)
        common = n1.index.intersection(n2.index)
        dn = n1.reindex(common) - n2.reindex(common)
        de = e1.reindex(common) - e2.reindex(common)
        # CAGR diff point
        c1 = wealth_metrics(n1.reindex(common))["cagr"]
        c2 = wealth_metrics(n2.reindex(common))["cagr"]
        cagr_diff = c1 - c2
        boot_cagr = block_bootstrap_ci(
            # bootstrap on paired daily net of left, reconstruct CAGR diff vs right path is hard;
            # use bootstrap on daily net-diff CAGR of (1+dn) as approximation of relative path
            dn,
            block=BOOT_BLOCK,
            n_boot=BOOT_N,
            seed=BOOT_SEED + abs(hash(name)) % 100000,
            stat="cagr_from_daily",
        )
        boot_ex = block_bootstrap_ci(
            de, block=BOOT_BLOCK, n_boot=BOOT_N, seed=BOOT_SEED + abs(hash(name + "ex")) % 100000, stat="mean"
        )
        # t on daily excess-diff
        se = float(de.std(ddof=1) / np.sqrt(len(de))) if len(de) > 1 else np.nan
        t_ex = float(de.mean() / se) if se and se > 0 else np.nan
        diffs.append(
            {
                "contrast": name,
                "cagr_diff_point": cagr_diff,
                "cagr_diff_boot_from_daily_net_diff": boot_cagr["point"],
                "cagr_diff_ci_lo": boot_cagr["ci_lo"],
                "cagr_diff_ci_hi": boot_cagr["ci_hi"],
                "daily_excess_diff_mean": float(de.mean()),
                "daily_excess_diff_ci_lo": boot_ex["ci_lo"],
                "daily_excess_diff_ci_hi": boot_ex["ci_hi"],
                "daily_excess_diff_t": t_ex,
                "n_days": int(len(common)),
                "boot_block": BOOT_BLOCK,
                "boot_n": BOOT_N,
                "boot_seed": BOOT_SEED,
            }
        )
    pd.DataFrame(diffs).to_csv(out / "difference_inference.csv", index=False)

    # Controls from topk_breadth
    ctrl = extract_topk_breadth_controls()
    ctrl.to_csv(out / "topk_breadth_K10_K5alt_P10_2024_2025_as_is.csv", index=False)

    write_summary_md(out, step1, res_df, ic_rows, diffs, ctrl)


def write_summary_md(out, step1, res_df, ic_rows, diffs, ctrl):
    lines = [
        "# Matched 2×2 2024–2025 — summary",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Output: `{out}`",
        f"- Engine commit: `{git_commit()}`",
        f"- W1 flip applied on 2024–2025: **{step1['flip_b']['flip_actually_applied']}**",
        "",
        "## Step1 reproduction gates",
        "",
    ]
    for g in step1["gates"]:
        lines.append(f"- `{json.dumps(g, default=str)}`")
    lines += ["", "## Main table (unified universe, full period)", ""]
    main = res_df[(res_df["universe"] == "unified") & (res_df["period"] == "2024_2025")][
        [
            "signal",
            "rule",
            "evidence_tag",
            "cum_return",
            "cagr",
            "wealth_mdd",
            "sharpe_rf0",
            "cagr_minus_spy",
            "qlib_excess_arr_legacy",
            "mean_daily_turnover",
            "median_holding_days",
            "ols_beta",
            "ols_t_beta",
            "ols_alpha",
            "ols_t_alpha",
        ]
    ]
    lines.append(main.to_markdown(index=False))
    lines += ["", "## Native universe control (full period)", ""]
    nat = res_df[(res_df["universe"] == "native") & (res_df["period"] == "2024_2025")][
        ["signal", "rule", "cagr", "wealth_mdd", "sharpe_rf0", "cagr_minus_spy", "qlib_excess_arr_legacy"]
    ]
    lines.append(nat.to_markdown(index=False))
    lines += ["", "## IC summary", ""]
    lines.append(pd.DataFrame(ic_rows).to_markdown(index=False))
    lines += ["", "## Difference inference (unified)", ""]
    lines.append(pd.DataFrame(diffs).to_markdown(index=False))
    lines += [
        "",
        "## Excess ARR definition (legacy)",
        "",
        "- `qlib_excess_arr_legacy` = Qlib `risk_analysis(return − bench − cost, freq=day)['annualized_return']`",
        "  which equals `mean(daily excess_net) × 238` (not 252; not geometric CAGR gap).",
        "- `cagr_minus_spy` / `cagr_gap_vs_spy` = wealth CAGR(net) − wealth CAGR(SPY) (geometric, N=252 convention).",
        "- H0 archived ≈ −5.4% refers to **geometric CAGR gap** (~−5.57%); Qlib excess ARR ≈ −4.52%.",
        "",
        "## topk_breadth controls (as-is, not rerun)",
        "",
    ]
    lines.append(ctrl.to_markdown(index=False))
    if step1.get("mismatches"):
        lines += ["", "## Mismatches (unfixed)", ""]
        for m in step1["mismatches"]:
            lines.append(f"- {m}")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / "reports" / f"matched_2x2_2024_2025_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    cache_dir = out / "cache_dedicated"
    cache_dir.mkdir()
    log.info("OUTPUT %s", out)

    manifest: dict[str, Any] = {
        "started_at_utc": utc_now(),
        "output_dir": str(out),
        "git_commit": git_commit(),
        "costs": {"open_cost": OPEN_COST, "close_cost": CLOSE_COST, "deal_price": "close"},
        "seed": BOOT_SEED,
        "boot_block": BOOT_BLOCK,
        "boot_n": BOOT_N,
        "lgbm_fit_calls": 0,
        "mechanical_kernel": "qlib TopkDropoutStrategy shared",
        "evidence_tags": {f"{a}×{b}": EVIDENCE_TAG[(a, b)] for a, b in EVIDENCE_TAG},
        "signal_a_pred_path": str(A_PRED),
        "signal_a_pred_sha256": sha256_file(A_PRED),
    }

    try:
        step1 = step1_native(out, cache_dir)
        manifest["step1"] = {
            "all_pass": step1["all_pass"],
            "gates": step1["gates"],
            "flip_b": step1["flip_b"],
            "signal_a_score_sha256": step1["signal_a_score_sha256"],
            "signal_b_score_sha256": step1["signal_b_score_sha256"],
            "mismatches": step1["mismatches"],
        }
        (out / "step1_reproduction.json").write_text(json.dumps(manifest["step1"], indent=2, default=str))

        # write step1 table csv
        pd.json_normalize(step1["gates"]).to_csv(out / "step1_reproduction_table.csv", index=False)

        if not step1["all_pass"]:
            manifest["status"] = "STOPPED_STEP1_FAIL"
            manifest["ended_at_utc"] = utc_now()
            (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
            (out / "SUMMARY.md").write_text(
                "# STOPPED at Step1\n\n"
                + "\n".join(f"- {m}" for m in step1["mismatches"])
                + "\n\nGates:\n"
                + json.dumps(step1["gates"], indent=2, default=str)
                + "\n"
            )
            log.error("Step1 FAILED — stopping. See %s", out)
            return 2

        log.info("Step1 PASS — continuing unified + steps 2–3")
        step2_and_3(out, cache_dir, step1)
        manifest["status"] = "COMPLETE"
        manifest["ended_at_utc"] = utc_now()
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        log.info("DONE %s", out)
        return 0
    except Exception:
        manifest["status"] = "ERROR"
        manifest["error"] = traceback.format_exc()
        manifest["ended_at_utc"] = utc_now()
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
