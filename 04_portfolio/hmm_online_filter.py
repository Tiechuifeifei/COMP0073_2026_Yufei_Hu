"""Strict ex-ante online HMM filtering — no full-sample smoothing for trading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass(frozen=True)
class OnlineHMMConfig:
    n_states: int
    random_state: int = 42
    n_iter: int = 200
    min_covar: float = 1e-3
    covariance_type: str = "diag"


def _standardize_train_apply(
    train: pd.DataFrame, apply: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mu = train[["spy_ret", "spy_vol20"]].mean()
    sd = train[["spy_ret", "spy_vol20"]].std().replace(0, 1.0)
    tr = train.copy()
    ap = apply.copy()
    tr[["spy_ret", "spy_vol20"]] = (tr[["spy_ret", "spy_vol20"]] - mu) / sd
    ap[["spy_ret", "spy_vol20"]] = (ap[["spy_ret", "spy_vol20"]] - mu) / sd
    return tr, ap, {"mean": mu.to_dict(), "std": sd.to_dict()}


def fit_hmm_train_only(
    features: pd.DataFrame,
    cfg: OnlineHMMConfig,
    *,
    apply_features: pd.DataFrame | None = None,
) -> tuple[GaussianHMM, pd.DataFrame, dict]:
    """Fit HMM on train; return model, scaled features for inference, scaler."""
    apply = apply_features if apply_features is not None else features
    train_s, apply_s, scaler = _standardize_train_apply(features, apply)
    x = train_s[["spy_ret", "spy_vol20"]].to_numpy(dtype=float)
    model = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type=cfg.covariance_type,
        random_state=cfg.random_state,
        n_iter=cfg.n_iter,
        min_covar=cfg.min_covar,
    )
    model.fit(x)
    return model, apply_s, scaler


def online_filter_posteriors(model: GaussianHMM, features: pd.DataFrame) -> pd.DataFrame:
    """Return causal filtered state probabilities for each date.

    For each index t, probabilities use observations x[0:t+1] only.
    Implementation: truncated ``predict_proba`` on prefix ending at t.
    """
    x = features[["spy_ret", "spy_vol20"]].to_numpy(dtype=float)
    n = len(features)
    n_states = model.n_components
    probs = np.zeros((n, n_states), dtype=float)

    for t in range(n):
        prefix = x[: t + 1]
        # predict_proba on prefix; take last row = filtered state at t
        filtered = model.predict_proba(prefix)
        probs[t] = filtered[-1]

    out = features[["date"]].copy()
    out["information_cutoff_date"] = out["date"]
    for i in range(n_states):
        out[f"state_probability_{i}"] = probs[:, i]
    out["raw_state"] = probs.argmax(axis=1)
    return out


def attach_trade_dates(states: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Map information_cutoff_date to next trading session (trade_date)."""
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    dates = pd.to_datetime(states["information_cutoff_date"])
    trade_dates = []
    for d in dates:
        future = cal[cal > d]
        trade_dates.append(future[0] if len(future) else pd.NaT)
    out = states.copy()
    out["trade_date"] = trade_dates
    out["trade_state_lag1"] = out["raw_state"].shift(1)
    # probabilities available at t apply to trade at t+1
    for i in range(model_n_states(out)):
        out[f"trade_prob_{i}"] = out[f"state_probability_{i}"].shift(1)
    return out


def model_n_states(states: pd.DataFrame) -> int:
    return sum(1 for c in states.columns if c.startswith("state_probability_"))


def label_states_by_spy_stats(
    states: pd.DataFrame,
    spy_features: pd.DataFrame,
    *,
    n_states: int,
) -> dict[int, str]:
    """Post-hoc economic labels from SPY stats within each raw state."""
    merged = states.merge(spy_features, on="date", how="left")
    labels: dict[int, str] = {}
    stats = []
    for s in range(n_states):
        sub = merged[merged["raw_state"] == s]
        if sub.empty:
            continue
        stats.append(
            {
                "state": s,
                "mean_ret": float(sub["spy_ret"].mean()),
                "vol": float(sub["spy_ret"].std() * np.sqrt(252)),
                "count": len(sub),
            }
        )
    stats_df = pd.DataFrame(stats).sort_values("mean_ret")
    if n_states == 2:
        mapping = {int(stats_df.iloc[0]["state"]): "bear", int(stats_df.iloc[1]["state"]): "bull"}
    else:
        mapping = {
            int(stats_df.iloc[0]["state"]): "bear",
            int(stats_df.iloc[1]["state"]): "neutral",
            int(stats_df.iloc[2]["state"]): "bull",
        }
    labels.update(mapping)
    return labels


def compute_state_run_stats(state_series: pd.Series) -> dict:
    """Run-length stats for a discrete state sequence (validation period)."""
    s = state_series.dropna().astype(int).values
    if len(s) == 0:
        return {
            "n_switches": 0,
            "mean_duration": np.nan,
            "min_duration": np.nan,
            "max_duration": np.nan,
        }
    durations = []
    switches = 0
    cur = s[0]
    run = 1
    for v in s[1:]:
        if v == cur:
            run += 1
        else:
            durations.append(run)
            switches += 1
            cur = v
            run = 1
    durations.append(run)
    return {
        "n_switches": int(switches),
        "mean_duration": float(np.mean(durations)),
        "min_duration": int(np.min(durations)),
        "max_duration": int(np.max(durations)),
    }


def spy_state_summary(
    states: pd.DataFrame,
    spy: pd.DataFrame,
    *,
    state_col: str = "raw_state",
    date_col: str = "date",
) -> pd.DataFrame:
    merged = states.merge(spy, on=date_col, how="left")
    rows = []
    n = len(merged)
    for st, g in merged.groupby(state_col):
        rets = g["spy_ret"].dropna()
        wealth = (1 + rets).cumprod()
        dd = float((wealth / wealth.cummax() - 1).min()) if len(wealth) else np.nan
        rows.append(
            {
                "state": int(st),
                "n_days": len(g),
                "pct_days": len(g) / n if n else np.nan,
                "mean_daily_ret": float(rets.mean()) if len(rets) else np.nan,
                "ann_ret": float(rets.mean() * 252) if len(rets) else np.nan,
                "ann_vol": float(rets.std() * np.sqrt(252)) if len(rets) > 1 else np.nan,
                "mdd": dd,
            }
        )
    return pd.DataFrame(rows)


def validation_log_likelihood(model: GaussianHMM, features: pd.DataFrame) -> float:
    x = features[["spy_ret", "spy_vol20"]].to_numpy(dtype=float)
    return float(model.score(x))


def state_label_series(states: pd.DataFrame, label_map: dict[int, str], prob_prefix: str = "trade_prob_") -> pd.DataFrame:
    out = states.copy()
    out["state_label"] = out["trade_state_lag1"].map(label_map)
    for econ in ("bull", "neutral", "bear"):
        cols = [c for c in out.columns if c.startswith(prob_prefix)]
        mapped = []
        for _, row in out.iterrows():
            p = 0.0
            for raw, lab in label_map.items():
                if lab == econ:
                    p += float(row.get(f"{prob_prefix}{raw}", 0.0) or 0.0)
            mapped.append(p)
        out[f"p_{econ}"] = mapped
    return out


def leakage_audit(states: pd.DataFrame) -> dict:
    ok = bool((pd.to_datetime(states["information_cutoff_date"]) < pd.to_datetime(states["trade_date"])).all())
    return {
        "information_cutoff_before_trade_date": ok,
        "uses_online_filter": True,
        "uses_full_sample_smoothed_states_for_trading": False,
        "max_information_cutoff_date": str(pd.to_datetime(states["information_cutoff_date"]).max().date()),
        "max_trade_date": str(pd.to_datetime(states["trade_date"]).dropna().max().date()),
    }
