"""Cross-section signal processing for regime-weighted scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if s.notna().sum() < 5:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def zscore_series(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


def preprocess_scores(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    """Per-day winsorize + z-score for each score column."""
    out = df.copy()
    for col in score_cols:
        zcol = f"{col}_z"
        out[zcol] = np.nan
        for dt, g in out.groupby("datetime", sort=False):
            s = winsorize_series(g[col].astype(float))
            out.loc[g.index, zcol] = zscore_series(s)
    return out


def score_direction_audit(df: pd.DataFrame, score_col: str, label_col: str = "label") -> dict:
    sub = df[[score_col, label_col]].dropna()
    if sub.empty:
        return {"score_col": score_col, "pearson_ic": np.nan, "direction_ok": False}
    ic = float(sub[score_col].corr(sub[label_col]))
    return {"score_col": score_col, "pearson_ic": ic, "direction_ok": ic > 0}


WEIGHT_SCHEMES: dict[str, dict[str, tuple[float, float]]] = {
    "W0": {"bull": (1.0, 0.0), "neutral": (1.0, 0.0), "bear": (1.0, 0.0)},
    "W1": {"bull": (0.7, 0.3), "neutral": (0.7, 0.3), "bear": (0.7, 0.3)},
    "W2": {"bull": (0.8, 0.2), "neutral": (0.6, 0.4), "bear": (0.4, 0.6)},
    "W3": {"bull": (0.9, 0.1), "neutral": (0.6, 0.4), "bear": (0.2, 0.8)},
}


def build_dynamic_score(
    df: pd.DataFrame,
    scheme: dict[str, tuple[float, float]],
    state_col: str = "state_label",
) -> pd.Series:
    wq = df[state_col].map({k: v[0] for k, v in scheme.items()}).astype(float)
    wf = df[state_col].map({k: v[1] for k, v in scheme.items()}).astype(float)
    return wq * df["quant_score_z"] + wf * df["fundamental_score_z"]


def pred_df_to_qlib(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str)
    return out.set_index(["datetime", "instrument"])[["score"]].sort_index()
