#!/usr/bin/env python3
"""2026 B-only matched engines under APPROXIMATE_CURRENT_DATA (Option I degraded).
Writes under a new reports/matched_2x2_2026_* directory."""

from __future__ import annotations

import hashlib
import json
import logging
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

import approx_2026_helpers as v  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("matched_2x2_2026_approx")

LABEL = "APPROXIMATE_CURRENT_DATA"
MAIN_START = pd.Timestamp("2026-01-02")
MAIN_END = pd.Timestamp("2026-08-24")
EXT_START = pd.Timestamp("2026-08-25")
COST_BP = float(v.MR4A_COST_BP)
RISK_DEGREE = 0.95
HOLD_THRESH = 1
ANN_FACTOR = 238
PLACEBO_N = 500
PLACEBO_SEED = 20260923
BOOT_SEED = 20260923

ARCH_DAILY = PROJECT_ROOT / "data/portfolio_experiments/market_risk/vt0_2026/vt0_2026_daily.csv"
ARCH_WEIGHTS = PROJECT_ROOT / "data/portfolio_experiments/market_risk/vt0_2026/vt0_2026_p5_weights.csv"


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


def load_membership_carry(dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Carry-forward membership from last archived in_index day (2026-08-18)."""
    m = pd.read_csv(v.MEMBER)
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    m["ticker"] = m["ticker"].astype(str).str.upper().map(v.map_ticker)
    m["in_index"] = m["in_index"].astype(bool)
    last_day = m["date"].max()
    last = m[(m["date"] == last_day) & m["in_index"]]
    rows = [m[m["in_index"]][["date", "ticker"]]]
    for d in dates:
        if d <= last_day:
            continue
        add = last[["ticker"]].copy()
        add["date"] = d
        rows.append(add)
    out = pd.concat(rows, ignore_index=True)
    out = out[out["date"].isin(dates)].drop_duplicates(["date", "ticker"])
    return out


def topk_dropout_weights(
    scores: pd.DataFrame,
    dates: list[pd.Timestamp],
    *,
    topk: int,
    n_drop: int,
    hold_thresh: int = HOLD_THRESH,
    risk_degree: float = RISK_DEGREE,
) -> pd.DataFrame:
    by_date = {d: g.set_index("ticker")["score"] for d, g in scores.groupby("date")}
    held: list[str] = []
    age: dict[str, int] = {}
    rows = []
    for dt in dates:
        pred = by_date.get(dt)
        if pred is None or pred.dropna().empty:
            for tkr in held:
                rows.append({"date": dt, "ticker": tkr, "weight": risk_degree / max(len(held), 1)})
            continue
        pred = pred.dropna().sort_values(ascending=False)
        last = [t for t in pred.reindex(held).sort_values(ascending=False).index.tolist() if t in pred.index]
        n_buy_cap = n_drop + topk - len(last)
        today = [t for t in pred.index if t not in last][: max(n_buy_cap, 0)]
        comb = pred.reindex(list(dict.fromkeys(last + today))).dropna().sort_values(ascending=False)
        bottom = set(comb.index[-n_drop:]) if len(comb) else set()
        sell = [t for t in last if t in bottom and age.get(t, 0) >= hold_thresh]
        buy = today[: len(sell) + topk - len(last)]
        new_held = [t for t in last if t not in sell] + list(buy)
        new_held = new_held[:topk]
        age = {t: (age.get(t, 0) + 1 if t in last and t not in sell else 1) for t in new_held}
        held = new_held
        w = risk_degree / max(len(held), 1)
        for tkr in held:
            rows.append({"date": dt, "ticker": tkr, "weight": w})
    return pd.DataFrame(rows)


def portfolio_from_weights(
    weights: pd.DataFrame,
    px: pd.DataFrame,
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> pd.DataFrame:
    ret = px[["date", "ticker", "close"]].copy()
    ret["ret"] = ret.groupby("ticker")["close"].pct_change()
    w = weights.rename(columns={"date": "decision_date"})
    cal = sorted(px["date"].drop_duplicates())
    loc = {d: i for i, d in enumerate(cal)}
    rows = []
    prev_w: dict[str, float] = {}
    for dec in sorted(w["decision_date"].unique()):
        i = loc.get(pd.Timestamp(dec))
        if i is None or i + 1 >= len(cal):
            continue
        ret_date = cal[i + 1]
        if ret_date < eval_start or ret_date > eval_end:
            continue
        wd = w[w["decision_date"] == dec]
        cur = dict(zip(wd["ticker"], wd["weight"]))
        day_ret = ret[ret["date"] == ret_date].set_index("ticker")["ret"]
        gross = 0.0
        eq = 0.0
        name_rets = {}
        for tkr, wt in cur.items():
            r = day_ret.get(tkr, np.nan)
            if not np.isfinite(r):
                continue
            gross += wt * float(r)
            eq += wt
            name_rets[tkr] = float(r) * wt
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
                "equity_weight": eq,
                "n_hold": int(len(cur)),
            }
        )
        prev_w = cur
    return pd.DataFrame(rows)


def holding_durations(weights: pd.DataFrame) -> tuple[float, float]:
    if weights.empty:
        return np.nan, np.nan
    w = weights.sort_values(["ticker", "date"])
    spells = []
    for tkr, g in w.groupby("ticker"):
        dates = sorted(pd.to_datetime(g["date"]).unique())
        if not dates:
            continue
        run = 1
        for i in range(1, len(dates)):
            # trading-day continuity approximated by consecutive decision dates in panel
            if (dates[i] - dates[i - 1]).days <= 5:
                run += 1
            else:
                spells.append(run)
                run = 1
        spells.append(run)
    if not spells:
        return np.nan, np.nan
    return float(np.mean(spells)), float(np.median(spells))


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


def summarize_daily(daily: pd.DataFrame, spy: pd.Series, weights: pd.DataFrame | None = None) -> dict[str, Any]:
    d = daily.drop_duplicates(subset=["trade_date"]).sort_values("trade_date").copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.normalize()
    r = d.set_index("trade_date")["r_net"]
    if isinstance(r, pd.DataFrame):
        r = r.iloc[:, 0]
    r = r.astype(float)
    if isinstance(spy, pd.DataFrame):
        spy = spy.squeeze(axis=1)
    s = pd.Series(spy).astype(float).copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].reindex(r.index)
    ex = pd.Series(r.to_numpy() - s.to_numpy(), index=r.index, dtype=float)
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    mean_h, med_h = (np.nan, np.nan)
    if weights is not None and not weights.empty:
        ww = weights.copy()
        if "date" not in ww.columns and "decision_date" in ww.columns:
            ww = ww.rename(columns={"decision_date": "date"})
        mean_h, med_h = holding_durations(ww)
    ols = ols_alpha_beta(r, s)
    return {
        "n_days": int(len(r)),
        "first_trade": str(r.index.min().date()) if len(r) else None,
        "last_trade": str(r.index.max().date()) if len(r) else None,
        "cum_net": float(wealth.iloc[-1] - 1.0) if len(r) else np.nan,
        "spy_cum": float((1.0 + s.fillna(0)).prod() - 1.0) if s.notna().any() else np.nan,
        "cum_excess_geom": float((1.0 + r).prod() / (1.0 + s.fillna(0)).prod() - 1.0) if len(r) else np.nan,
        "qlib_excess_arr": float(ex.mean() * ANN_FACTOR) if len(ex.dropna()) else np.nan,
        "wealth_mdd": float(dd.min()) if len(dd) else np.nan,
        "sharpe_rf0": float(r.mean() / r.std(ddof=1) * np.sqrt(ANN_FACTOR)) if r.std(ddof=1) > 0 else np.nan,
        "mean_daily_turnover": float(d["turnover"].mean()) if "turnover" in d.columns else np.nan,
        "mean_holding_days": mean_h,
        "median_holding_days": med_h,
        "ols_beta": ols["beta"],
        "ols_t_beta": ols["t_beta"],
        "ols_alpha": ols["alpha"],
        "ols_t_alpha": ols["t_alpha"],
    }


def attach_spy(daily: pd.DataFrame, spy_px: pd.DataFrame) -> pd.DataFrame:
    spy = spy_px.sort_values("date").copy()
    spy["r_spy"] = spy["close"].pct_change()
    out = daily.merge(spy[["date", "r_spy"]].rename(columns={"date": "trade_date"}), on="trade_date", how="left")
    return out


def build_scores(
    px: pd.DataFrame,
    dates: list[pd.Timestamp],
    *,
    cache_dir: Path,
    include_f1c: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (w1_or_d2_scores, d2_raw_scores, meta)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat_path = cache_dir / "d2_features_current_ohlcv.parquet"
    feat_lists = json.loads(v.FEATURE_JSON.read_text())
    alpha20_cols = feat_lists["alpha20"]
    all_cols = feat_lists["all"]

    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
        log.info("loaded feature cache %s rows=%d", feat_path, len(feat))
    else:
        log.info("computing Massive features on current OHLCV (full book)")
        feat = v.compute_price_features(px)
        feat.to_parquet(feat_path, index=False)

    mean, std = v.fit_robust_z_qlib(alpha20_cols, cache_dir / "alpha20_robust_z_2008_2017.json")
    # Prefer replay robust-z params if present for continuity
    replay_z = v.OUT_DIR / "alpha20_robust_z_2008_2017.json"
    if replay_z.exists():
        mean, std = v.fit_robust_z_qlib(alpha20_cols, replay_z)
    feat_z = v.apply_robust_z(feat, alpha20_cols, mean, std)
    models = v.complete_d2_seeds(all_cols)
    d2 = v.predict_d2(feat_z, models)

    membership = load_membership_carry(dates)
    meta = {
        "feature_rows": int(len(feat)),
        "feature_sha256": sha256_file(feat_path),
        "include_f1c": include_f1c,
        "lgbm_fit_calls": 0,
        "d2_models_loaded": sorted(models.keys()),
    }

    if not include_f1c:
        panel = membership.merge(px[["date", "ticker"]].drop_duplicates(), on=["date", "ticker"], how="inner")
        panel = panel.merge(d2, on=["date", "ticker"], how="inner")
        panel["score"] = panel.groupby("date")["d2"].transform(v.cs_winsor_z)
        return panel.dropna(subset=["score"]), d2, meta

    stmt = v.build_f1c_statement_panel()
    f1c_raw = v.f1c_raw_from_stmt(stmt, px, dates)
    f1c_models = v.train_f1c_seeds()
    f1c = v.predict_f1c(f1c_raw, f1c_models)
    w1 = v.build_w1(d2, f1c, membership, px)
    meta["f1c_rows"] = int(len(f1c))
    meta["w1_rows"] = int(len(w1))
    return w1, d2, meta


def monthly_table(dailies: dict[str, pd.DataFrame], spy: pd.Series) -> pd.DataFrame:
    rows = []
    for name, d in dailies.items():
        x = d.drop_duplicates(subset=["trade_date"]).copy()
        x["trade_date"] = pd.to_datetime(x["trade_date"]).dt.normalize()
        x["ym"] = x["trade_date"].dt.to_period("M").astype(str)
        for ym, g in x.groupby("ym", sort=True):
            vals = g["r_net"].to_numpy(dtype=float)
            rows.append({"series": name, "month": ym, "cum_net": float(np.prod(1.0 + vals) - 1.0)})
    sp = spy.dropna().copy()
    if not isinstance(sp, pd.Series):
        sp = pd.Series(sp)
    sp.index = pd.to_datetime(sp.index).normalize()
    sp = sp[~sp.index.duplicated(keep="last")]
    sp_df = sp.to_frame("r_spy")
    sp_df["ym"] = sp_df.index.to_period("M").astype(str)
    for ym, g in sp_df.groupby("ym", sort=True):
        vals = g["r_spy"].to_numpy(dtype=float)
        rows.append({"series": "SPY", "month": ym, "cum_net": float(np.prod(1.0 + vals) - 1.0)})
    wide = pd.DataFrame(rows).pivot_table(index="month", columns="series", values="cum_net", aggfunc="first")
    return wide.reset_index()


def ic_table(scores: pd.DataFrame, px: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Cross-sectional IC vs next-session return (Massive), decision-date aligned."""
    ret = px[["date", "ticker", "close"]].copy()
    ret["ret"] = ret.groupby("ticker")["close"].pct_change()
    cal = sorted(px["date"].drop_duplicates())
    loc = {d: i for i, d in enumerate(cal)}
    rows = []
    for dec in dates:
        i = loc.get(pd.Timestamp(dec))
        if i is None or i + 1 >= len(cal):
            continue
        nxt = cal[i + 1]
        s = scores[scores["date"] == dec][["ticker", "score"]]
        r = ret[ret["date"] == nxt][["ticker", "ret"]]
        m = s.merge(r, on="ticker").dropna()
        if len(m) < 20:
            continue
        rows.append(
            {
                "decision_date": dec,
                "label_date": nxt,
                "ic": float(m["score"].corr(m["ret"])),
                "rank_ic": float(m["score"].corr(m["ret"], method="spearman")),
                "n": int(len(m)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # summary + bootstrap on full window
    for metric in ("ic", "rank_ic"):
        x = out[metric].dropna().values
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
        out.attrs[f"{metric}_mean"] = float(np.mean(x)) if len(x) else np.nan
        out.attrs[f"{metric}_ci"] = (float(lo), float(hi))
    return out


def placebo_runs(
    universe_by_date: dict[pd.Timestamp, list[str]],
    dates: list[pd.Timestamp],
    px: pd.DataFrame,
    *,
    topk: int,
    n_drop: int,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    n: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spy_px = pd.read_parquet(v.MASSIVE_SPY)
    spy_px["date"] = pd.to_datetime(spy_px["date"]).dt.normalize()
    results = []
    for i in range(n):
        rows = []
        for dt in dates:
            tickers = universe_by_date.get(dt, [])
            if not tickers:
                continue
            scores = rng.random(len(tickers))
            for tkr, sc in zip(tickers, scores):
                rows.append({"date": dt, "ticker": tkr, "score": float(sc)})
        sc = pd.DataFrame(rows)
        if sc.empty:
            continue
        w = topk_dropout_weights(sc, dates, topk=topk, n_drop=n_drop)
        daily = portfolio_from_weights(w, px, eval_start=eval_start, eval_end=eval_end)
        if daily.empty:
            continue
        daily = attach_spy(daily, spy_px)
        r = daily["r_net"].astype(float)
        s = daily["r_spy"].astype(float)
        results.append(
            {
                "run": i,
                "cum_net": float((1 + r).prod() - 1),
                "cum_excess_geom": float((1 + r).prod() / (1 + s).prod() - 1),
            }
        )
        if (i + 1) % 100 == 0:
            log.info("placebo topk=%d n_drop=%d %d/%d", topk, n_drop, i + 1, n)
    return pd.DataFrame(results)


def percentile_of(x: float, dist: np.ndarray) -> float:
    if not np.isfinite(x) or len(dist) == 0:
        return np.nan
    return float((dist <= x).mean() * 100.0)


def mismatch_days_contrib(
    w_new: pd.DataFrame,
    w_arch: pd.DataFrame,
    daily_new: pd.DataFrame,
    daily_arch: pd.DataFrame,
) -> pd.DataFrame:
    def sets(w: pd.DataFrame) -> dict:
        return w.groupby("date")["ticker"].apply(lambda s: tuple(sorted(s))).to_dict()

    a, b = sets(w_new), sets(w_arch)
    days = sorted(d for d in set(a) | set(b) if a.get(d) != b.get(d))
    rows = []
    arch_d = daily_arch.set_index("decision_date") if "decision_date" in daily_arch.columns else None
    new_d = daily_new.set_index("decision_date")
    for d in days:
        row = {
            "decision_date": str(pd.Timestamp(d).date()),
            "holdings_regen": ",".join(a.get(d, ())),
            "holdings_archived": ",".join(b.get(d, ())),
        }
        if d in new_d.index:
            row["r_net_regen"] = float(new_d.loc[d, "r_net"]) if not isinstance(new_d.loc[d, "r_net"], pd.Series) else float(new_d.loc[d, "r_net"].iloc[0])
            row["trade_date"] = str(pd.Timestamp(new_d.loc[d, "trade_date"]).date()) if "trade_date" in new_d.columns else None
        if arch_d is not None and d in arch_d.index:
            val = arch_d.loc[d, "r_p0"] if "r_p0" in arch_d.columns else arch_d.loc[d, "r_net"]
            row["r_net_archived"] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
        if "r_net_regen" in row and "r_net_archived" in row:
            row["delta_r_net"] = row["r_net_regen"] - row["r_net_archived"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / f"reports/matched_2x2_2026_{ts}"
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    # Reuse feature/score cache from any prior partial/complete run if available
    priors = sorted(
        PROJECT_ROOT.glob("reports/matched_2x2_2026_*/cache/d2_features_current_ohlcv.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if priors:
        import shutil

        prior = priors[0].parent
        for f in prior.iterdir():
            dest = cache / f.name
            if f.is_file() and not dest.exists():
                shutil.copy2(f, dest)
                log.info("reused cache file %s from %s", f.name, prior.parent.name)
    log.info("output %s", out)

    px = v.load_prices()
    spy_px = pd.read_parquet(v.MASSIVE_SPY)
    spy_px["date"] = pd.to_datetime(spy_px["date"]).dt.normalize()
    spy_ret = spy_px.sort_values("date").set_index("date")["close"].pct_change()

    px_max = px["date"].max()
    all_decision_dates = sorted(d for d in px["date"].drop_duplicates() if MAIN_START <= d <= px_max)
    # last decision needs a next session for return
    cal = sorted(px["date"].drop_duplicates())
    if all_decision_dates and all_decision_dates[-1] == cal[-1]:
        all_decision_dates = all_decision_dates[:-1]

    main_decision = [d for d in all_decision_dates if d <= MAIN_END]
    ext_decision = [d for d in all_decision_dates if d >= EXT_START]
    main_trade_end = MAIN_END
    ext_trade_start = EXT_START
    ext_trade_end = px_max

    log.info("building W1 scores on current data through %s", px_max.date())
    w1, d2, meta = build_scores(px, all_decision_dates, cache_dir=cache, include_f1c=True)
    w1_path = cache / "w1_scores.parquet"
    w1.to_parquet(w1_path, index=False)
    d2.to_parquet(cache / "d2_scores.parquet", index=False)

    # --- Main window strategies ---
    w1_main = w1[w1["date"].isin(main_decision)].copy()
    w_top5 = topk_dropout_weights(w1_main[["date", "ticker", "score"]], main_decision, topk=5, n_drop=1)
    w_top20 = topk_dropout_weights(w1_main[["date", "ticker", "score"]], main_decision, topk=20, n_drop=2)
    daily_top5 = attach_spy(
        portfolio_from_weights(w_top5, px, eval_start=MAIN_START, eval_end=main_trade_end), spy_px
    )
    daily_top20 = attach_spy(
        portfolio_from_weights(w_top20, px, eval_start=MAIN_START, eval_end=main_trade_end), spy_px
    )

    # Archived P5
    arch = pd.read_csv(ARCH_DAILY)
    arch["trade_date"] = pd.to_datetime(arch["trade_date"])
    arch["decision_date"] = pd.to_datetime(arch["decision_date"])
    arch_w = pd.read_csv(ARCH_WEIGHTS)
    arch_w["date"] = pd.to_datetime(arch_w["date"])
    # Archived P0 path: use r_p0 (equals r_net for unscaled P5); avoid duplicate r_net cols
    arch_daily_mapped = arch[["trade_date", "decision_date", "r_p0", "r_spy", "turnover", "cost"]].copy()
    arch_daily_mapped = arch_daily_mapped.rename(columns={"r_p0": "r_net"})

    # D2-only POST_HOC (same current chain, no F1C leg)
    log.info("D2-only diagnostic")
    membership = load_membership_carry(main_decision)
    panel = membership.merge(px[["date", "ticker"]].drop_duplicates(), on=["date", "ticker"], how="inner")
    panel = panel.merge(d2, on=["date", "ticker"], how="inner")
    panel["score"] = panel.groupby("date")["d2"].transform(v.cs_winsor_z)
    d2_only = panel.dropna(subset=["score"])
    w_d2only = topk_dropout_weights(d2_only[["date", "ticker", "score"]], main_decision, topk=5, n_drop=1)
    daily_d2only = attach_spy(
        portfolio_from_weights(w_d2only, px, eval_start=MAIN_START, eval_end=main_trade_end), spy_px
    )

    # Mismatch contribution
    mm = mismatch_days_contrib(w_top5, arch_w, daily_top5, arch)
    mm.to_csv(out / "holding_mismatch_days.csv", index=False)

    # Drop non-scalar cols before metrics
    for col in ("name_contrib",):
        if col in daily_top5.columns:
            daily_top5 = daily_top5.drop(columns=[col])
        if col in daily_top20.columns:
            daily_top20 = daily_top20.drop(columns=[col])
        if col in daily_d2only.columns:
            daily_d2only = daily_d2only.drop(columns=[col])

    # Summaries
    spy_main = daily_top5.drop_duplicates("trade_date").set_index("trade_date")["r_spy"]
    rows = []
    for name, daily, weights, tag in [
        ("B_Top5_archived_r_p0", arch_daily_mapped, arch_w, "ARCHIVED_ORIGINAL"),
        ("B_Top5_regen_current", daily_top5, w_top5, LABEL),
        ("B_Top20_regen_current", daily_top20, w_top20, LABEL),
        ("D2only_Top5_POST_HOC", daily_d2only, w_d2only, "POST_HOC|" + LABEL),
    ]:
        dly = daily.drop_duplicates(subset=["trade_date"]).copy()
        if "r_spy" in dly.columns:
            spy_s = dly.set_index(pd.to_datetime(dly["trade_date"]).dt.normalize())["r_spy"]
        else:
            spy_s = spy_main
        m = summarize_daily(dly, spy_s, weights)
        m["strategy"] = name
        m["data_label"] = tag
        m["window"] = "main_2026-01-05_to_2026-08-24"
        rows.append(m)
    main_summary = pd.DataFrame(rows)
    main_summary.to_csv(out / "main_window_results.csv", index=False)

    # Daily dumps
    daily_top5.to_csv(out / "daily_B_Top5_regen.csv", index=False)
    daily_top20.to_csv(out / "daily_B_Top20_regen.csv", index=False)
    daily_d2only.to_csv(out / "daily_D2only_Top5.csv", index=False)
    w_top5.to_csv(out / "weights_B_Top5_regen.csv", index=False)
    w_top20.to_csv(out / "weights_B_Top20_regen.csv", index=False)

    # Monthly
    monthly = monthly_table(
        {
            "B_Top5_regen": daily_top5,
            "B_Top20_regen": daily_top20,
            "B_Top5_archived": arch_daily_mapped,
            "D2only_Top5": daily_d2only,
        },
        spy_main,
    )
    monthly.to_csv(out / "monthly_returns_main.csv", index=False)

    # IC
    ic = ic_table(w1_main, px, main_decision)
    ic.to_csv(out / "ic_daily_main.csv", index=False)
    ic_sum = pd.DataFrame(
        [
            {
                "window": "main",
                "metric": "ic",
                "mean": float(ic["ic"].mean()) if len(ic) else np.nan,
                "n_days": int(len(ic)),
                "boot_ci_lo": ic.attrs.get("ic_ci", (np.nan, np.nan))[0] if hasattr(ic, "attrs") else np.nan,
                "boot_ci_hi": ic.attrs.get("ic_ci", (np.nan, np.nan))[1] if hasattr(ic, "attrs") else np.nan,
                "boot_block": 10,
                "boot_n": 2000,
                "boot_seed": BOOT_SEED,
                "label": "next_session_return_Massive",
            },
            {
                "window": "main",
                "metric": "rank_ic",
                "mean": float(ic["rank_ic"].mean()) if len(ic) else np.nan,
                "n_days": int(len(ic)),
                "boot_ci_lo": ic.attrs.get("rank_ic_ci", (np.nan, np.nan))[0] if hasattr(ic, "attrs") else np.nan,
                "boot_ci_hi": ic.attrs.get("rank_ic_ci", (np.nan, np.nan))[1] if hasattr(ic, "attrs") else np.nan,
                "boot_block": 10,
                "boot_n": 2000,
                "boot_seed": BOOT_SEED,
                "label": "next_session_return_Massive",
            },
        ]
    )
    # recompute CIs cleanly
    if len(ic):
        for metric in ("ic", "rank_ic"):
            x = ic[metric].dropna().values
            rng = np.random.default_rng(BOOT_SEED)
            block = 10
            n = len(x)
            n_blocks = int(np.ceil(n / block))
            boots = []
            for _ in range(2000):
                starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
                sample = np.concatenate([x[s : s + block] for s in starts])[:n]
                boots.append(sample.mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ic_sum.loc[ic_sum.metric == metric, "boot_ci_lo"] = lo
            ic_sum.loc[ic_sum.metric == metric, "boot_ci_hi"] = hi
    ic_sum.to_csv(out / "ic_summary.csv", index=False)

    # Differences (b-only): Top5 - Top20
    diff_rows = []
    for name, a, b in [
        ("Top5-Top20 @B_regen", daily_top5, daily_top20),
        ("Top5_regen - Top5_archived", daily_top5, arch_daily_mapped),
        ("D2onlyTop5 - W1Top5_regen", daily_d2only, daily_top5),
    ]:
        aa = a.drop_duplicates("trade_date")[["trade_date", "r_net"]].copy()
        bb = b.drop_duplicates("trade_date")[["trade_date", "r_net"]].copy()
        aa["trade_date"] = pd.to_datetime(aa["trade_date"]).dt.normalize()
        bb["trade_date"] = pd.to_datetime(bb["trade_date"]).dt.normalize()
        aa = aa.rename(columns={"r_net": "r_net_a"})
        bb = bb.rename(columns={"r_net": "r_net_b"})
        m = aa.merge(bb, on="trade_date", how="inner")
        m["diff"] = m["r_net_a"].to_numpy(dtype=float) - m["r_net_b"].to_numpy(dtype=float)
        diff_std = float(np.std(m["diff"], ddof=1))
        t = float(m["diff"].mean() / (diff_std / np.sqrt(len(m)))) if diff_std > 0 else np.nan
        x = m["diff"].to_numpy(dtype=float)
        rng = np.random.default_rng(BOOT_SEED)
        block = 10
        n = len(x)
        n_blocks = int(np.ceil(n / block))
        boots = []
        for _ in range(2000):
            starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
            sample = np.concatenate([x[s : s + block] for s in starts])[:n]
            boots.append(sample.mean() * ANN_FACTOR)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        diff_rows.append(
            {
                "contrast": name,
                "cum_a": float(np.prod(1.0 + m["r_net_a"].to_numpy(dtype=float)) - 1.0),
                "cum_b": float(np.prod(1.0 + m["r_net_b"].to_numpy(dtype=float)) - 1.0),
                "cum_diff": float(
                    np.prod(1.0 + m["r_net_a"].to_numpy(dtype=float))
                    - np.prod(1.0 + m["r_net_b"].to_numpy(dtype=float))
                ),
                "qlib_excess_arr_diff": float(m["diff"].mean() * ANN_FACTOR),
                "daily_diff_mean": float(m["diff"].mean()),
                "daily_diff_t": t,
                "boot_ci_lo_arr": float(lo),
                "boot_ci_hi_arr": float(hi),
                "boot_block": 10,
                "boot_n": 2000,
                "boot_seed": BOOT_SEED,
                "n_days": int(len(m)),
                "data_label": LABEL,
            }
        )
    pd.DataFrame(diff_rows).to_csv(out / "difference_inference_main.csv", index=False)

    # Placebo main
    log.info("placebo main window")
    univ = {
        d: g["ticker"].tolist()
        for d, g in load_membership_carry(main_decision).merge(
            px[["date", "ticker"]].drop_duplicates(), on=["date", "ticker"], how="inner"
        ).groupby("date")
    }
    plac5 = placebo_runs(
        univ, main_decision, px, topk=5, n_drop=1, eval_start=MAIN_START, eval_end=main_trade_end, n=PLACEBO_N, seed=PLACEBO_SEED
    )
    plac20 = placebo_runs(
        univ,
        main_decision,
        px,
        topk=20,
        n_drop=2,
        eval_start=MAIN_START,
        eval_end=main_trade_end,
        n=PLACEBO_N,
        seed=PLACEBO_SEED + 1,
    )
    plac5.to_csv(out / "placebo_main_Top5.csv", index=False)
    plac20.to_csv(out / "placebo_main_Top20.csv", index=False)

    def placebo_summary(dist: pd.DataFrame, rule: str, window: str, refs: dict[str, float]) -> dict:
        cn = dist["cum_net"].values
        ce = dist["cum_excess_geom"].values
        row = {
            "window": window,
            "rule": rule,
            "n": int(len(dist)),
            "cum_net_mean": float(np.mean(cn)),
            "cum_net_median": float(np.median(cn)),
            "cum_net_p05": float(np.percentile(cn, 5)),
            "cum_net_p95": float(np.percentile(cn, 95)),
            "cum_excess_mean": float(np.mean(ce)),
            "cum_excess_median": float(np.median(ce)),
            "cum_excess_p05": float(np.percentile(ce, 5)),
            "cum_excess_p95": float(np.percentile(ce, 95)),
            "seed": PLACEBO_SEED if rule.startswith("Top5") else PLACEBO_SEED + 1,
            "data_label": LABEL,
        }
        for k, val in refs.items():
            row[f"pctile_cum_net_{k}"] = percentile_of(val, cn)
        return row

    arch_cum = float((1 + arch_daily_mapped["r_net"]).prod() - 1)
    regen5_cum = float((1 + daily_top5["r_net"]).prod() - 1)
    regen20_cum = float((1 + daily_top20["r_net"]).prod() - 1)
    plac_sum = [
        placebo_summary(
            plac5,
            "Top5/drop1",
            "main",
            {"archived_P5": arch_cum, "regen_P5": regen5_cum, "regen_Top20": regen20_cum},
        ),
        placebo_summary(
            plac20,
            "Top20/drop2",
            "main",
            {"archived_P5": arch_cum, "regen_P5": regen5_cum, "regen_Top20": regen20_cum},
        ),
    ]

    # Extension window
    log.info("extension window %s .. %s", EXT_START.date(), ext_trade_end.date())
    ext_rows = []
    plac_ext_sum = []
    if ext_decision:
        w1_ext = w1[w1["date"].isin(ext_decision)].copy()
        w5e = topk_dropout_weights(w1_ext[["date", "ticker", "score"]], ext_decision, topk=5, n_drop=1)
        w20e = topk_dropout_weights(w1_ext[["date", "ticker", "score"]], ext_decision, topk=20, n_drop=2)
        d5e = attach_spy(portfolio_from_weights(w5e, px, eval_start=ext_trade_start, eval_end=ext_trade_end), spy_px)
        d20e = attach_spy(portfolio_from_weights(w20e, px, eval_start=ext_trade_start, eval_end=ext_trade_end), spy_px)
        d5e.to_csv(out / "daily_B_Top5_regen_extension.csv", index=False)
        d20e.to_csv(out / "daily_B_Top20_regen_extension.csv", index=False)
        for name, daily, weights in [
            ("B_Top5_regen_current", d5e, w5e),
            ("B_Top20_regen_current", d20e, w20e),
        ]:
            m = summarize_daily(daily, daily.set_index("trade_date")["r_spy"], weights)
            m["strategy"] = name
            m["data_label"] = LABEL
            m["window"] = f"extension_{ext_trade_start.date()}_to_{ext_trade_end.date()}"
            ext_rows.append(m)
        pd.DataFrame(ext_rows).to_csv(out / "extension_window_results.csv", index=False)

        univ_e = {
            d: g["ticker"].tolist()
            for d, g in load_membership_carry(ext_decision).merge(
                px[["date", "ticker"]].drop_duplicates(), on=["date", "ticker"], how="inner"
            ).groupby("date")
        }
        p5e = placebo_runs(
            univ_e,
            ext_decision,
            px,
            topk=5,
            n_drop=1,
            eval_start=ext_trade_start,
            eval_end=ext_trade_end,
            n=PLACEBO_N,
            seed=PLACEBO_SEED + 10,
        )
        p20e = placebo_runs(
            univ_e,
            ext_decision,
            px,
            topk=20,
            n_drop=2,
            eval_start=ext_trade_start,
            eval_end=ext_trade_end,
            n=PLACEBO_N,
            seed=PLACEBO_SEED + 11,
        )
        p5e.to_csv(out / "placebo_extension_Top5.csv", index=False)
        p20e.to_csv(out / "placebo_extension_Top20.csv", index=False)
        c5e = float((1 + d5e["r_net"]).prod() - 1)
        c20e = float((1 + d20e["r_net"]).prod() - 1)
        plac_ext_sum = [
            placebo_summary(p5e, "Top5/drop1", "extension", {"regen_P5": c5e, "regen_Top20": c20e}),
            placebo_summary(p20e, "Top20/drop2", "extension", {"regen_P5": c5e, "regen_Top20": c20e}),
        ]
    else:
        pd.DataFrame(columns=["strategy"]).to_csv(out / "extension_window_results.csv", index=False)

    pd.DataFrame(plac_sum + plac_ext_sum).to_csv(out / "placebo_summary.csv", index=False)

    # Manifest
    manifest = {
        "experiment_id": out.name,
        "status": "COMPLETED_APPROXIMATE_CURRENT_DATA",
        "generated": utc_now(),
        "data_label": LABEL,
        "option": "I_degraded_reopen",
        "signal_A": {"built": False, "reason": "frozen_chain_unavailable"},
        "signal_B": "legacy W1 approximate on Massive current OHLCV + Massive/EDGAR F1C",
        "windows": {
            "main": {"decision": f"{MAIN_START.date()}..{MAIN_END.date()}", "trade_end": str(MAIN_END.date())},
            "extension": {
                "decision_from": str(EXT_START.date()),
                "trade_end": str(ext_trade_end.date()),
                "membership": "carry_forward from 2026-08-18 snapshot events",
            },
        },
        "engine_commit": git_commit(),
        "lgbm_fit_calls_this_run": 0,
        "historical_refit_at_replay": {
            "d2_seeds_2026_3407": True,
            "f1c_approx_three_seeds": True,
            "trained_on_2026": False,
        },
        "ohlcv_sha256_current": sha256_file(v.MASSIVE_PX),
        "spy_sha256_current": sha256_file(v.MASSIVE_SPY),
        "w1_scores_sha256": sha256_file(w1_path),
        "archives_untouched": True,
        "placebo": {"n": PLACEBO_N, "seed_main_top5": PLACEBO_SEED, "seed_main_top20": PLACEBO_SEED + 1},
        "meta_scores": meta,
        "comparison_rule": "Top5 vs Top20 uses regen (b) only; archived r_p0 reported separately",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Summary markdown
    s5 = main_summary[main_summary.strategy == "B_Top5_regen_current"].iloc[0]
    s5a = main_summary[main_summary.strategy == "B_Top5_archived_r_p0"].iloc[0]
    s20 = main_summary[main_summary.strategy == "B_Top20_regen_current"].iloc[0]
    sd2 = main_summary[main_summary.strategy == "D2only_Top5_POST_HOC"].iloc[0]
    ps5 = [p for p in plac_sum if p["rule"] == "Top5/drop1"][0]
    ps20 = [p for p in plac_sum if p["rule"] == "Top20/drop2"][0]

    md = f"""# Matched 2×2 2026 — B-only APPROXIMATE_CURRENT_DATA

**Directory:** `{out}`  
**Generated:** {manifest['generated']}  
**Label:** `{LABEL}` — 近似数据链 + 2026-09-14 修订后价量。不修改任何归档文件。  
**Signal A:** 未构建（冻结 Compustat/RD8 链不可得）。

## Main window (trade ~2026-01-05 .. 2026-08-24)

| strategy | cum_net | spy_cum | cum_excess_geom | qlib_excess_ARR | wealth_MDD | Sharpe | mean_TO |
|---|---:|---:|---:|---:|---:|---:|---:|
| B×Top5 **archived** r_p0 | {s5a.cum_net:.4%} | {s5a.spy_cum:.4%} | {s5a.cum_excess_geom:.4%} | {s5a.qlib_excess_arr:.4%} | {s5a.wealth_mdd:.4%} | {s5a.sharpe_rf0:.3f} | {s5a.mean_daily_turnover:.4f} |
| B×Top5 **regen (b)** | {s5.cum_net:.4%} | {s5.spy_cum:.4%} | {s5.cum_excess_geom:.4%} | {s5.qlib_excess_arr:.4%} | {s5.wealth_mdd:.4%} | {s5.sharpe_rf0:.3f} | {s5.mean_daily_turnover:.4f} |
| B×Top20 regen (b) | {s20.cum_net:.4%} | {s20.spy_cum:.4%} | {s20.cum_excess_geom:.4%} | {s20.qlib_excess_arr:.4%} | {s20.wealth_mdd:.4%} | {s20.sharpe_rf0:.3f} | {s20.mean_daily_turnover:.4f} |
| D2-only×Top5 POST_HOC | {sd2.cum_net:.4%} | {sd2.spy_cum:.4%} | {sd2.cum_excess_geom:.4%} | {sd2.qlib_excess_arr:.4%} | {sd2.wealth_mdd:.4%} | {sd2.sharpe_rf0:.3f} | {sd2.mean_daily_turnover:.4f} |

- Top5 vs Top20 对比 **仅用 (b)**。  
- 持仓不同决策日：见 `holding_mismatch_days.csv`（n={len(mm)}）。

## Placebo main (500 runs)

| rule | cum_net mean / p50 / p05 / p95 | pctile archived P5 | pctile regen P5 | pctile regen Top20 |
|---|---|---:|---:|---:|
| Top5/drop1 | {ps5['cum_net_mean']:.2%} / {ps5['cum_net_median']:.2%} / {ps5['cum_net_p05']:.2%} / {ps5['cum_net_p95']:.2%} | {ps5.get('pctile_cum_net_archived_P5', float('nan')):.1f} | {ps5.get('pctile_cum_net_regen_P5', float('nan')):.1f} | {ps5.get('pctile_cum_net_regen_Top20', float('nan')):.1f} |
| Top20/drop2 | {ps20['cum_net_mean']:.2%} / {ps20['cum_net_median']:.2%} / {ps20['cum_net_p05']:.2%} / {ps20['cum_net_p95']:.2%} | {ps20.get('pctile_cum_net_archived_P5', float('nan')):.1f} | {ps20.get('pctile_cum_net_regen_P5', float('nan')):.1f} | {ps20.get('pctile_cum_net_regen_Top20', float('nan')):.1f} |

## Extension ({EXT_START.date()} .. {ext_trade_end.date()})

见 `extension_window_results.csv` 与 `placebo_extension_*.csv`（单独窗口，不与主窗合并）。

## Files

- `main_window_results.csv`, `extension_window_results.csv`
- `daily_B_Top5_regen.csv`, `daily_B_Top20_regen.csv`, `daily_D2only_Top5.csv`
- `holding_mismatch_days.csv`, `monthly_returns_main.csv`
- `ic_summary.csv`, `difference_inference_main.csv`
- `placebo_summary.csv`, `run_manifest.json`
"""
    (out / "SUMMARY.md").write_text(md)
    log.info("done -> %s", out)
    print(out)


if __name__ == "__main__":
    main()
