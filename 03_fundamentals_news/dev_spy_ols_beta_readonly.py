#!/usr/bin/env python3
"""Read-only SPY OLS betas for development portfolios (Appendix G.6 style).

r_port,t = α + β · r_SPY,t + ε_t
- Daily returns, no risk-free subtraction
- SPY = PERMNO 84398 close-to-close (staging/csv_benchmark/p84398.csv)
- β SE / α t-stat: Newey–West HAC, lag = 5
- Annualized α = α_daily × 252
"""
from __future__ import annotations
import os

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(os.environ["PROJECT_ROOT"]) if os.environ.get("PROJECT_ROOT") else Path(__file__).resolve().parents[1]
OUT = PROJECT / "reports/dev_spy_ols_beta_20260924"
OUT.mkdir(parents=True, exist_ok=True)

SPY_PATH = PROJECT / "staging/csv_benchmark/p84398.csv"
PHASE_B = PROJECT / "data/portfolio_experiments/topk_breadth/phase_b_daily_returns.csv"
PHASE_C = PROJECT / "data/portfolio_experiments/topk_breadth/phase_c_daily_returns.csv"
MECH_DAILY = (
    PROJECT
    / "reports/portfolio_engine_pair/p20d2_vs_p20d4_main_experiments/04_w1_mechanism_daily.csv"
)
FBASE_DIR = PROJECT / "reports/portfolio_topk_drop_joint_20260919_141851/daily_returns"
G6_STYLE = PROJECT / "reports/k5d1_posthoc_diagnosis_20260920_110303/style_exposure.csv"

NW_LAGS = 5
ALPHA_SCALE = 252.0


def load_spy() -> pd.Series:
    spy = pd.read_csv(SPY_PATH)
    spy.columns = [c.lower() for c in spy.columns]
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
    spy = spy.set_index("date").sort_index()
    return spy["close"].pct_change().rename("spy_ret")


def nw_ols(y: pd.Series, x: pd.Series, lags: int = NW_LAGS) -> dict:
    """OLS with Newey–West HAC SE (Bartlett kernel)."""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1, sort=True).dropna()
    n = len(df)
    if n < 30:
        return {"n": n, "error": "too_few_obs"}
    yv = df["y"].to_numpy(dtype=float)
    xv = df["x"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), xv])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta

    # HAC middle matrix S
    u = resid.reshape(-1, 1) * X  # n x 2 score
    S = u.T @ u
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = u[lag:].T @ u[:-lag]
        S = S + w * (gamma + gamma.T)

    xtx_inv = np.linalg.inv(X.T @ X)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(cov))
    tstat = beta / se

    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan

    # classical SE for G.6 parity check (OLS iid)
    s2 = float(resid.var(ddof=2))
    se_ols = np.sqrt(np.diag(xtx_inv) * s2)

    return {
        "n": n,
        "alpha_daily": float(beta[0]),
        "beta": float(beta[1]),
        "alpha_se_nw": float(se[0]),
        "beta_se_nw": float(se[1]),
        "alpha_t_nw": float(tstat[0]),
        "beta_t_nw": float(tstat[1]),
        "alpha_ann_252": float(beta[0] * ALPHA_SCALE),
        "alpha_ann_t_nw": float(tstat[0]),  # same t on daily α; ann scale cancels in t
        "r2": float(r2),
        "beta_se_ols": float(se_ols[1]),
        "alpha_t_ols": float(beta[0] / se_ols[0]),
        "beta_t_ols": float(beta[1] / se_ols[1]),
    }


def load_topk_arm(path: Path, arm_id: str, window: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["trade_date"])
    sub = df[(df["arm_id"] == arm_id) & (df["window"] == window)].copy()
    sub = sub.set_index("trade_date").sort_index()
    return sub["return"].astype(float).rename("portfolio_return")


def load_fbase(cid: str, period: str) -> pd.Series:
    p = FBASE_DIR / f"{cid}_{period}.csv"
    df = pd.read_csv(p, parse_dates=["datetime"]).set_index("datetime").sort_index()
    return df["portfolio_return"].astype(float)


def load_p20d4_dev() -> pd.Series:
    df = pd.read_csv(MECH_DAILY, parse_dates=["date"])
    sub = df[df["engine"] == "P20D4"].set_index("date").sort_index()
    return sub["daily_portfolio_return"].astype(float).rename("portfolio_return")


def main() -> None:
    spy = load_spy()

    series = {
        "H0": {
            "dev": load_topk_arm(PHASE_B, "K20", "2020-2023"),
            "valid": load_topk_arm(PHASE_B, "K20", "2018-2019"),
            "source_dev": str(PHASE_B) + " | arm_id=K20 | window=2020-2023",
            "source_valid": str(PHASE_B) + " | arm_id=K20 | window=2018-2019",
            "note": "W1×Top20/drop2; phase_b K20 ≡ H0 ≡ T0",
        },
        "P5": {
            "dev": load_topk_arm(PHASE_C, "P5", "2020-2023"),
            "valid": load_topk_arm(PHASE_C, "P5", "2018-2019"),
            "source_dev": str(PHASE_C) + " | arm_id=P5 | window=2020-2023",
            "source_valid": str(PHASE_C) + " | arm_id=P5 | window=2018-2019",
            "note": "W1×Top5/drop1; phase_c P5",
        },
        "P20D4": {
            "dev": load_p20d4_dev(),
            "valid": None,
            "source_dev": str(MECH_DAILY) + " | engine=P20D4",
            "source_valid": "",
            "note": "W1×Top20/drop4; mechanism daily (2020–2023 only; no valid archive here)",
        },
        "K20_D2": {
            "dev": load_fbase("K20_D2", "test"),
            "valid": load_fbase("K20_D2", "valid"),
            "source_dev": str(FBASE_DIR / "K20_D2_test.csv"),
            "source_valid": str(FBASE_DIR / "K20_D2_valid.csv"),
            "note": "F_BASE_W010×Top20/drop2; G.6 check target β≈1.19",
        },
        "K5D1_WD": {
            "dev": load_fbase("K5_D1", "test"),
            "valid": load_fbase("K5_D1", "valid"),
            "source_dev": str(FBASE_DIR / "K5_D1_test.csv"),
            "source_valid": str(FBASE_DIR / "K5_D1_valid.csv"),
            "note": "F_BASE_W010×Top5/drop1 (K5_D1); G.6 check target β≈1.35",
        },
    }

    rows = []
    for name, meta in series.items():
        for period_label, y in (("2020-2023", meta["dev"]), ("2018-2019", meta["valid"])):
            if y is None:
                rows.append(
                    {
                        "portfolio": name,
                        "period": period_label,
                        "scope": "full_period",
                        "year": "",
                        "n_days": 0,
                        "beta": np.nan,
                        "beta_se_nw_lag5": np.nan,
                        "alpha_ann_252": np.nan,
                        "alpha_t_nw": np.nan,
                        "r2": np.nan,
                        "status": "NO_DAILY_RETURNS",
                        "source": meta.get("source_valid") or meta["source_dev"],
                        "note": meta["note"],
                    }
                )
                continue
            fit = nw_ols(y, spy)
            rows.append(
                {
                    "portfolio": name,
                    "period": period_label,
                    "scope": "full_period",
                    "year": "",
                    "n_days": fit["n"],
                    "beta": fit["beta"],
                    "beta_se_nw_lag5": fit["beta_se_nw"],
                    "alpha_ann_252": fit["alpha_ann_252"],
                    "alpha_t_nw": fit["alpha_t_nw"],
                    "r2": fit["r2"],
                    "alpha_daily": fit["alpha_daily"],
                    "beta_t_nw": fit["beta_t_nw"],
                    "beta_ols_classical": fit["beta"],
                    "beta_t_ols_classical": fit["beta_t_ols"],
                    "status": "OK",
                    "source": meta["source_dev"] if period_label.startswith("2020") else meta["source_valid"],
                    "note": meta["note"],
                }
            )

        # yearly betas on development sample only
        ydev = meta["dev"]
        if ydev is None:
            continue
        ydev = ydev.copy()
        ydev.index = pd.to_datetime(ydev.index)
        for year in (2020, 2021, 2022, 2023):
            yy = ydev[ydev.index.year == year]
            fit = nw_ols(yy, spy)
            if "error" in fit:
                rows.append(
                    {
                        "portfolio": name,
                        "period": "2020-2023",
                        "scope": "by_year",
                        "year": year,
                        "n_days": fit.get("n", 0),
                        "beta": np.nan,
                        "beta_se_nw_lag5": np.nan,
                        "alpha_ann_252": np.nan,
                        "alpha_t_nw": np.nan,
                        "r2": np.nan,
                        "status": fit.get("error", "FAIL"),
                        "source": meta["source_dev"],
                        "note": meta["note"],
                    }
                )
                continue
            rows.append(
                {
                    "portfolio": name,
                    "period": "2020-2023",
                    "scope": "by_year",
                    "year": year,
                    "n_days": fit["n"],
                    "beta": fit["beta"],
                    "beta_se_nw_lag5": fit["beta_se_nw"],
                    "alpha_ann_252": fit["alpha_ann_252"],
                    "alpha_t_nw": fit["alpha_t_nw"],
                    "r2": fit["r2"],
                    "alpha_daily": fit["alpha_daily"],
                    "beta_t_nw": fit["beta_t_nw"],
                    "status": "OK",
                    "source": meta["source_dev"],
                    "note": meta["note"],
                }
            )

    out = pd.DataFrame(rows)
    csv_path = OUT / "dev_spy_ols_beta_summary.csv"
    out.to_csv(csv_path, index=False)
    # also root reports pointer
    out.to_csv(PROJECT / "reports/dev_spy_ols_beta_summary.csv", index=False)

    # G.6 parity check (classical OLS β on F_BASE)
    g6 = pd.read_csv(G6_STYLE)
    g6_full = g6[(g6["model"] == "full") & (g6["config_id"].isin(["K20_D2", "K5_D1"]))]
    parity = []
    for cid, port in [("K20_D2", "K20_D2"), ("K5_D1", "K5D1_WD")]:
        for period, plabel in [("test", "2020-2023"), ("valid", "2018-2019")]:
            archived = g6_full[(g6_full.config_id == cid) & (g6_full.period == period)]
            ours = out[(out.portfolio == port) & (out.period == plabel) & (out.scope == "full_period")]
            if archived.empty or ours.empty:
                continue
            parity.append(
                {
                    "portfolio": port,
                    "period": plabel,
                    "g6_beta": float(archived.iloc[0]["beta"]),
                    "recomputed_beta": float(ours.iloc[0]["beta"]),
                    "abs_diff": abs(float(archived.iloc[0]["beta"]) - float(ours.iloc[0]["beta"])),
                    "g6_r2": float(archived.iloc[0]["r2"]),
                    "recomputed_r2": float(ours.iloc[0]["r2"]),
                }
            )
    parity_df = pd.DataFrame(parity)
    parity_df.to_csv(OUT / "g6_parity_check.csv", index=False)

    meta = {
        "method": "OLS daily r_port = a + b * r_SPY; no RF; NW HAC Bartlett lag=5 for SE/t; alpha_ann = a*252",
        "spy": str(SPY_PATH) + " close.pct_change(); PERMNO 84398",
        "g6_reference": str(G6_STYLE),
        "note_p20d4": "phase_c P20 is reuse of K20 (drop2), NOT P20D4; P20D4 taken from mechanism daily.",
        "g6_parity_max_abs_beta_diff": float(parity_df["abs_diff"].max()) if len(parity_df) else None,
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # markdown brief
    full = out[out.scope == "full_period"].copy()
    lines = [
        "# 开发期组合对 SPY 的 OLS β（只读）",
        "",
        "口径与附录 G.6 一致：日收益、无风险利率、SPY=PERMNO 84398 `close` 日收益。",
        "β 标准误与 α 的 t 值使用 **Newey–West HAC（Bartlett，滞后 5 日）**；年化 α = 日 α × 252。",
        "",
        "## 2020–2023 全样本",
        "",
        "| 组合 | β | β SE (NW5) | 年化 α | α t (NW) | R² | 天数 |",
        "|------|--:|-----------:|-------:|---------:|---:|-----:|",
    ]
    for _, r in full[full.period == "2020-2023"].iterrows():
        if r.status != "OK":
            lines.append(f"| {r.portfolio} | — | — | — | — | — | {int(r.n_days)} ({r.status}) |")
            continue
        lines.append(
            f"| {r.portfolio} | {r.beta:.3f} | {r.beta_se_nw_lag5:.4f} | "
            f"{100*r.alpha_ann_252:.2f}% | {r.alpha_t_nw:.2f} | {r.r2:.3f} | {int(r.n_days)} |"
        )
    lines += ["", "## 验证期 2018–2019（有日收益者）", ""]
    lines.append("| 组合 | β | β SE (NW5) | 年化 α | α t (NW) | R² | 天数 |")
    lines.append("|------|--:|-----------:|-------:|---------:|---:|-----:|")
    for _, r in full[full.period == "2018-2019"].iterrows():
        if r.status != "OK":
            lines.append(f"| {r.portfolio} | — | — | — | — | — | ({r.status}) |")
            continue
        lines.append(
            f"| {r.portfolio} | {r.beta:.3f} | {r.beta_se_nw_lag5:.4f} | "
            f"{100*r.alpha_ann_252:.2f}% | {r.alpha_t_nw:.2f} | {r.r2:.3f} | {int(r.n_days)} |"
        )

    lines += ["", "## 分年 β（2020–2023）", ""]
    by = out[out.scope == "by_year"].pivot(index="portfolio", columns="year", values="beta")
    lines.append("| 组合 | 2020 | 2021 | 2022 | 2023 |")
    lines.append("|------|-----:|-----:|-----:|-----:|")
    for port in by.index:
        cells = [f"{by.loc[port, y]:.3f}" if pd.notna(by.loc[port, y]) else "—" for y in (2020, 2021, 2022, 2023)]
        lines.append(f"| {port} | " + " | ".join(cells) + " |")

    lines += ["", "## G.6 核对（K20_D2 / K5_D1）", ""]
    if len(parity_df):
        lines.append("| portfolio | period | g6_beta | recomputed_beta | abs_diff |")
        lines.append("|-----------|--------|--------:|----------------:|---------:|")
        for _, r in parity_df.iterrows():
            lines.append(
                f"| {r.portfolio} | {r.period} | {r.g6_beta:.6f} | {r.recomputed_beta:.6f} | {r.abs_diff:.2e} |"
            )
        lines.append("")
        lines.append(f"最大 |Δβ| = {parity_df['abs_diff'].max():.2e}")
    else:
        lines.append("(empty)")

    lines += [
        "",
        "## 来源",
        "",
        f"- SPY：`{SPY_PATH}`",
        f"- H0：`{PHASE_B}` (K20)",
        f"- P5：`{PHASE_C}` (P5)",
        f"- P20D4：`{MECH_DAILY}` (engine=P20D4；非 phase_c 的 P20)",
        f"- K20_D2 / K5D1_WD：`{FBASE_DIR}`",
        f"- G.6 存档：`{G6_STYLE}`",
        f"- CSV：`{csv_path}`",
    ]
    md = OUT / "dev_spy_ols_beta_report.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    (PROJECT / "reports/dev_spy_ols_beta_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"out": str(OUT), "g6_parity": parity, "n_rows": len(out)}, indent=2, default=str))
    print(full[full.period == "2020-2023"][
        ["portfolio", "beta", "beta_se_nw_lag5", "alpha_ann_252", "alpha_t_nw", "r2", "n_days"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
