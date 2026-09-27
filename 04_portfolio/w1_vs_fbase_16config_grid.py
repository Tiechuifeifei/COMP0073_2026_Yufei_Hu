#!/usr/bin/env python3
"""Matched 16-config TopK×n_drop grid: frozen W1 vs archived F_BASE_W010 (S1/S2).

F_BASE metrics come from the S1/S2 artefact; W1 is backtested with the same
engine / costs / benchmark / windows as portfolio_topk_drop_joint.py
(TopkDropoutStrategy, 1bp, P84398, hold_thresh=1, risk_degree=0.95).
Holdout 2024–2025 is closed for this grid."""

from __future__ import annotations
import os

import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_US

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str((Path(os.environ["RDAGENT_ROOT"]) / "phase3_portfolio_ablation/scripts")))

from loop9_fund_incremental_ablation import PERIODS  # noqa: E402
from portfolio_topk_drop_joint import (  # noqa: E402
    BENCHMARK,
    CLOSE_COST,
    HOLD_THRESH,
    MODELLING_END,
    OPEN_COST,
    QLIB_DATA,
    RISK_DEGREE,
    assert_no_holdout,
    cfg_id,
    metrics_row,
    run_bt,
    sha256_file,
)
from portfolio_us_replacement_extension import FULL_16  # noqa: E402

# Only development windows (exclude train / covid / post_covid slices)
RUN_PERIODS = {k: PERIODS[k] for k in ("valid", "test")}

S2_ROOT = PROJECT_ROOT / "reports/portfolio_us_replacement_extension_20260919_174020"
W1_PRED = PROJECT_ROOT / "reports/legacy_RD13_system_audit_20260921_074429/predictions/pred_legacy_W1.parquet"
F_BASE_SCORE = PROJECT_ROOT / "reports/portfolio_topk_drop_joint_20260919_141851/F_BASE_W010_score.pkl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("w1_vs_fbase_16config")


def load_fbase_archived() -> pd.DataFrame:
    reg = pd.read_csv(S2_ROOT / "full_16_config_registry.csv")
    valid = pd.read_csv(S2_ROOT / "validation_metrics.csv")
    test = pd.read_csv(S2_ROOT / "test_metrics.csv")
    keep = [
        "config_id",
        "topk",
        "n_drop",
        "nominal_replacement_ratio",
        "period",
        "excess_arr_net",
        "ir_net",
        "mean_daily_turnover",
        "realised_replacement_fraction",
        "source_stage",
    ]
    long = pd.concat([valid[keep], test[keep]], ignore_index=True)
    long["signal"] = "F_BASE_W010"
    long = long.merge(
        reg[["config_id", "nominal_replacement_pct_exact", "is_reference_static_primary", "source_stage"]].rename(
            columns={"source_stage": "registry_stage"}
        ),
        on="config_id",
        how="left",
    )
    return long


def load_w1_pred() -> pd.DataFrame:
    w1 = pd.read_parquet(W1_PRED)
    w1["datetime"] = pd.to_datetime(w1["datetime"])
    pred = w1.set_index(["datetime", "instrument"])[["score"]].sort_index()
    # Frozen W1 parquet may extend into holdout; clip to modelling end for this grid.
    dates = pred.index.get_level_values("datetime")
    pred = pred.loc[dates <= pd.Timestamp(MODELLING_END)]
    assert_no_holdout(pred.index.get_level_values("datetime"), "W1_clipped")
    return pred


def run_w1_grid(pred: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    daily_dir = out_root / "daily_returns"
    daily_dir.mkdir(parents=True, exist_ok=True)
    for topk, n_drop in FULL_16:
        cid = cfg_id(topk, n_drop)
        for period, (start, end) in RUN_PERIODS.items():
            t0 = time.time()
            log.info("W1 %s %s %s..%s", cid, period, start, end)
            result = run_bt(pred, topk, n_drop, start, end)
            assert_no_holdout(result["report"].index, f"W1 {cid} {period}")
            row = metrics_row(result, topk=topk, n_drop=n_drop, period=period, role="w1_matched_16")
            row["signal"] = "W1"
            row["elapsed_sec"] = time.time() - t0
            rows.append(row)
            rep = result["report"].copy()
            rep = rep.rename(
                columns={
                    "return": "portfolio_return",
                    "bench": "benchmark_return",
                }
            )
            rep.index.name = "datetime"
            rep.to_csv(daily_dir / f"{cid}_{period}.csv")
            log.info(
                "  excess_arr_net=%.4f IR=%.3f TO=%.4f repl=%.4f (%.1fs)",
                row["excess_arr_net"],
                row["ir_net"],
                row["mean_daily_turnover"],
                row["realised_replacement_fraction"],
                row["elapsed_sec"],
            )
    return pd.DataFrame(rows)


def pivot_period_table(long: pd.DataFrame, signal: str) -> pd.DataFrame:
    sub = long[long["signal"] == signal].copy()
    pieces = []
    for period, tag in (("valid", "valid"), ("test", "test_2020_2023")):
        p = sub[sub["period"] == period][
            [
                "config_id",
                "topk",
                "n_drop",
                "nominal_replacement_ratio",
                "excess_arr_net",
                "ir_net",
                "mean_daily_turnover",
                "realised_replacement_fraction",
            ]
        ].rename(
            columns={
                "excess_arr_net": f"{tag}_excess_arr_net",
                "ir_net": f"{tag}_ir_net",
                "mean_daily_turnover": f"{tag}_mean_daily_turnover",
                "realised_replacement_fraction": f"{tag}_realised_replacement",
            }
        )
        pieces.append(p)
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on=["config_id", "topk", "n_drop", "nominal_replacement_ratio"], how="outer")
    out["signal"] = signal
    out["nominal_replacement_pct"] = (out["nominal_replacement_ratio"] * 100).round(4)
    return out.sort_values(["topk", "n_drop"]).reset_index(drop=True)


def within_k_20_vs_10_by_k(piv: pd.DataFrame, period_col: str) -> dict[int, float]:
    """Same-K Δ excess ARR: ~20% nominal minus ~10% nominal, keyed by topk."""
    deltas: dict[int, float] = {}
    for k, g in piv.groupby("topk"):
        r10 = g[np.isclose(g["nominal_replacement_ratio"], 0.1)]
        r20 = g[np.isclose(g["nominal_replacement_ratio"], 0.2)]
        if len(r10) != 1 or len(r20) != 1:
            deltas[int(k)] = float("nan")
        else:
            deltas[int(k)] = float(r20.iloc[0][period_col] - r10.iloc[0][period_col])
    return deltas


def build_comparison(fbase_piv: pd.DataFrame, w1_piv: pd.DataFrame) -> pd.DataFrame:
    # Side-by-side on config_id
    cols_core = [
        "config_id",
        "topk",
        "n_drop",
        "nominal_replacement_ratio",
        "nominal_replacement_pct",
        "valid_excess_arr_net",
        "valid_ir_net",
        "valid_mean_daily_turnover",
        "valid_realised_replacement",
        "test_2020_2023_excess_arr_net",
        "test_2020_2023_ir_net",
        "test_2020_2023_mean_daily_turnover",
        "test_2020_2023_realised_replacement",
    ]
    a = fbase_piv[cols_core].add_prefix("F_BASE_").rename(
        columns={
            "F_BASE_config_id": "config_id",
            "F_BASE_topk": "topk",
            "F_BASE_n_drop": "n_drop",
            "F_BASE_nominal_replacement_ratio": "nominal_replacement_ratio",
            "F_BASE_nominal_replacement_pct": "nominal_replacement_pct",
        }
    )
    b = w1_piv[cols_core].add_prefix("W1_").rename(
        columns={
            "W1_config_id": "config_id",
            "W1_topk": "topk",
            "W1_n_drop": "n_drop",
            "W1_nominal_replacement_ratio": "nominal_replacement_ratio",
            "W1_nominal_replacement_pct": "nominal_replacement_pct",
        }
    )
    b = b.drop(columns=["topk", "n_drop", "nominal_replacement_ratio", "nominal_replacement_pct"])
    m = a.merge(b, on="config_id", how="outer").sort_values(["topk", "n_drop"]).reset_index(drop=True)

    d_fb_t = within_k_20_vs_10_by_k(fbase_piv, "test_2020_2023_excess_arr_net")
    d_w1_t = within_k_20_vs_10_by_k(w1_piv, "test_2020_2023_excess_arr_net")
    d_fb_v = within_k_20_vs_10_by_k(fbase_piv, "valid_excess_arr_net")
    d_w1_v = within_k_20_vs_10_by_k(w1_piv, "valid_excess_arr_net")
    m["F_BASE_delta_excess_arr_20pct_minus_10pct_valid"] = m["topk"].map(d_fb_v)
    m["F_BASE_delta_excess_arr_20pct_minus_10pct_test"] = m["topk"].map(d_fb_t)
    m["W1_delta_excess_arr_20pct_minus_10pct_valid"] = m["topk"].map(d_w1_v)
    m["W1_delta_excess_arr_20pct_minus_10pct_test"] = m["topk"].map(d_w1_t)
    m["F_BASE_withinK_excessARR_20pct_minus_10pct"] = m["F_BASE_delta_excess_arr_20pct_minus_10pct_test"]
    m["W1_withinK_excessARR_20pct_minus_10pct"] = m["W1_delta_excess_arr_20pct_minus_10pct_test"]
    return m


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = PROJECT_ROOT / f"reports/w1_vs_fbase_16config_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    fbase_long = load_fbase_archived()
    fbase_long.to_csv(out_root / "fbase_16_long.csv", index=False)
    fbase_piv = pivot_period_table(fbase_long, "F_BASE_W010")
    fbase_piv.to_csv(out_root / "fbase_16_wide.csv", index=False)

    log.info("Init qlib + load W1")
    qlib.init(provider_uri=str(QLIB_DATA), region=REG_US, kernels=1)
    pred = load_w1_pred()
    log.info("W1 rows=%d sha256=%s", len(pred), sha256_file(W1_PRED)[:16])

    w1_long = run_w1_grid(pred, out_root)
    w1_long.to_csv(out_root / "w1_16_long.csv", index=False)
    w1_piv = pivot_period_table(w1_long, "W1")
    w1_piv.to_csv(out_root / "w1_16_wide.csv", index=False)

    comp = build_comparison(fbase_piv, w1_piv)
    comp.to_csv(out_root / "matched_16_comparison.csv", index=False)

    # Compact display tables
    fbase_piv.to_csv(out_root / "table1_F_BASE_W010_16configs.csv", index=False)
    w1_piv.to_csv(out_root / "w1_16_wide.csv", index=False)
    # Dual-signal table with the two within-K delta columns (test 20%−10%)
    dual = comp[
        [
            "config_id",
            "topk",
            "n_drop",
            "nominal_replacement_pct",
            "F_BASE_valid_excess_arr_net",
            "F_BASE_valid_ir_net",
            "F_BASE_valid_mean_daily_turnover",
            "F_BASE_valid_realised_replacement",
            "F_BASE_test_2020_2023_excess_arr_net",
            "F_BASE_test_2020_2023_ir_net",
            "F_BASE_test_2020_2023_mean_daily_turnover",
            "F_BASE_test_2020_2023_realised_replacement",
            "W1_valid_excess_arr_net",
            "W1_valid_ir_net",
            "W1_valid_mean_daily_turnover",
            "W1_valid_realised_replacement",
            "W1_test_2020_2023_excess_arr_net",
            "W1_test_2020_2023_ir_net",
            "W1_test_2020_2023_mean_daily_turnover",
            "W1_test_2020_2023_realised_replacement",
            "F_BASE_withinK_excessARR_20pct_minus_10pct",
            "W1_withinK_excessARR_20pct_minus_10pct",
        ]
    ]
    dual.to_csv(out_root / "table2_matched_W1_and_F_BASE_with_20vs10_delta.csv", index=False)

    manifest = {
        "generated_utc": stamp,
        "signal_W1": str(W1_PRED),
        "signal_W1_sha256": sha256_file(W1_PRED),
        "fbase_source": str(S2_ROOT),
        "fbase_score_pkl": str(F_BASE_SCORE),
        "configs": [cfg_id(k, d) for k, d in FULL_16],
        "engine": {
            "hold_thresh": HOLD_THRESH,
            "risk_degree": RISK_DEGREE,
            "open_cost": OPEN_COST,
            "close_cost": CLOSE_COST,
            "benchmark": BENCHMARK,
            "deal_price": "close",
            "periods": {k: list(v) for k, v in RUN_PERIODS.items()},
            "w1_clip": f"datetime <= {MODELLING_END}",
        },
        "note_delta_cols": (
            "F_BASE_withinK_excessARR_20pct_minus_10pct and W1_withinK_excessARR_20pct_minus_10pct "
            "are same-K (test 2020–2023) net excess ARR at ~20% nominal minus ~10% nominal; "
            "NaN if that K lacks both intensities in the 16-pool (e.g. K=5 only has 20%)."
        ),
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Markdown summary
    lines = [
        "# W1 vs F_BASE_W010 — matched 16-config grid",
        "",
        f"Generated (UTC): `{stamp}`",
        "",
        "## 1. F_BASE_W010 (archived S1/S2)",
        "",
        fbase_piv.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 2. W1 (frozen) — same engine",
        "",
        w1_piv.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 3. Matched table + within-K 20%−10% excess ARR (test)",
        "",
        dual.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (out_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("DONE %s", out_root)
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
