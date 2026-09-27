#!/usr/bin/env python3
"""F2P_portfolio_backtest.py

Fixed-configuration portfolio backtest for Alpha20 / F1C / F2 fusion using
official Alpha20 PortAna settings (test 2020–2023). Specification record:
data/fundamental_experiments/F2P_portfolio_backtest/F2P_frozen_specification_record.md"""

from __future__ import annotations
import os

import copy
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.constant import REG_US
from qlib.contrib.evaluate import risk_analysis

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_ROOT = PROJECT_ROOT / "data" / "fundamental_experiments" / "F2P_portfolio_backtest"
PHASE3_SCRIPTS = (Path(os.environ["RDAGENT_ROOT"]) / "phase3_portfolio_ablation/scripts")
WORKFLOW_YAML = PROJECT_ROOT / "experiments" / "conf_alpha20_sp500_transfer.yaml"
BASELINE_REPORT = PROJECT_ROOT / "staging" / "alpha20_baseline_p84398_report.json"
CRSP_MCAP = PROJECT_ROOT / "data" / "crsp_daily" / "crsp_daily_market.parquet"

ALPHA20_PRED = (Path(os.environ["RDAGENT_ROOT"]) / "phase3_portfolio_ablation/inputs/baseline_pred.pkl")
F1C_MAIN = (
    PROJECT_ROOT
    / "data/fundamental_experiments/F1_fundamental_lightgbm/main/F1C_no_operating_profitability_ensemble_pred.pkl"
)
F1C_NEXTDAY = (
    PROJECT_ROOT
    / "data/fundamental_experiments/F1_fundamental_lightgbm/nextday/F1C_no_operating_profitability_ensemble_pred.pkl"
)
F2_SELECTED_MAIN = PROJECT_ROOT / "data/fundamental_experiments/F2_score_fusion/main/F2_selected_fusion_pred.pkl"
F2_EQUAL_MAIN = PROJECT_ROOT / "data/fundamental_experiments/F2_score_fusion/main/F2_equal_weight_fusion_pred.pkl"
F2_SELECTED_NEXTDAY = PROJECT_ROOT / "data/fundamental_experiments/F2_score_fusion/nextday/F2_selected_fusion_pred.pkl"
F2_EQUAL_NEXTDAY = PROJECT_ROOT / "data/fundamental_experiments/F2_score_fusion/nextday/F2_equal_weight_fusion_pred.pkl"

FROZEN_ALPHA = 0.20
TOPK = 20
TEST_START = "2020-01-01"
TEST_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"

PERIODS: dict[str, tuple[str, str]] = {
    "test": (TEST_START, TEST_END),
    "covid": ("2020-01-01", "2021-12-31"),
    "post_covid": ("2022-01-01", "2023-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2021": ("2021-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
}

BOOT_BLOCK_SIZE = 21
BOOT_N = 2000
BOOT_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(PHASE3_SCRIPTS))
import run_portfolio_ablation as rpa  # noqa: E402


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(type(obj))


def load_port_config() -> dict[str, Any]:
    raw = yaml.safe_load(WORKFLOW_YAML.read_text(encoding="utf-8"))
    port = copy.deepcopy(raw["port_analysis_config"])
    port["backtest"]["benchmark"] = "P84398"
    return port


def ensure_score_df(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        df = obj.to_frame("score")
    else:
        df = obj.copy()
        if "score" not in df.columns:
            df = df.rename(columns={df.columns[0]: "score"})
    df.index = df.index.set_names(["datetime", "instrument"])
    return df.sort_index()


def load_pred(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        pred = ensure_score_df(pickle.load(handle))
    if pred.index.duplicated().any():
        raise ValueError(f"duplicate prediction keys in {path}")
    dates = pred.index.get_level_values("datetime")
    if dates.max() >= pd.Timestamp(HOLDOUT_START):
        pred = pred.loc[dates < pd.Timestamp(HOLDOUT_START)]
    pred = pred.loc[(dates >= pd.Timestamp(TEST_START)) & (dates <= pd.Timestamp(TEST_END))]
    if pred.empty:
        raise ValueError(f"no test-period predictions in {path}")
    return pred


def common_keys(*preds: pd.DataFrame) -> pd.MultiIndex:
    idx = preds[0].index
    for p in preds[1:]:
        idx = idx.intersection(p.index)
    if idx.empty:
        raise ValueError("empty common sample intersection")
    return idx.sort_values()


def filter_pred_period(pred: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pred.index.get_level_values("datetime")
    return pred.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def restrict_pred(pred: pd.DataFrame, keys: pd.MultiIndex | None = None) -> pd.DataFrame:
    out = pred.copy()
    if keys is not None:
        out = out.loc[out.index.isin(keys)]
    return out.sort_index()


def prediction_diagnostics(pred: pd.DataFrame) -> dict[str, Any]:
    daily = pred.groupby(level="datetime").size()
    return {
        "prediction_rows": int(len(pred)),
        "unique_dates": int(daily.shape[0]),
        "unique_instruments": int(pred.index.get_level_values("instrument").nunique()),
        "avg_cross_section_n": float(daily.mean()),
        "days_with_insufficient_predictions": int((daily < TOPK).sum()),
        "min_daily_predictions": int(daily.min()),
        "max_daily_predictions": int(daily.max()),
    }


def sharpe_ratio(daily: pd.Series) -> float:
    s = daily.dropna()
    if len(s) < 2 or s.std() == 0:
        return float("nan")
    return float(s.mean() / s.std() * np.sqrt(252))


def annualized_volatility(daily: pd.Series) -> float:
    s = daily.dropna()
    if len(s) < 2:
        return float("nan")
    return float(s.std() * np.sqrt(252))


def run_period_backtest(port_cfg: dict[str, Any], pred: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    cfg = copy.deepcopy(port_cfg)
    cfg["backtest"]["start_time"] = start
    cfg["backtest"]["end_time"] = end
    return rpa.run_backtest(cfg, pred)


def extract_metrics(
    result: dict[str, Any],
    pred: pd.DataFrame,
    *,
    model_id: str,
    model_label: str,
    variant: str,
    sample_type: str,
    period: str,
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    report = result["report"]
    indicators = result["indicators_df"]
    excess_wo = result["excess_without_cost"]
    excess_w = result["excess_with_cost"]
    port_abs = risk_analysis(report["return"], freq="day")
    bench_abs = risk_analysis(report["bench"], freq="day")

    pred_diag = prediction_diagnostics(pred)
    avg_pos = rpa.average_position_count(result["positions"])

    row = {
        "variant": variant,
        "model_id": model_id,
        "model_label": model_label,
        "sample_type": sample_type,
        "period": period,
        "period_start": period_start,
        "period_end": period_end,
        "annualized_return": float(port_abs.loc["annualized_return", "risk"]),
        "benchmark_annualized_return": float(bench_abs.loc["annualized_return", "risk"]),
        "excess_annualized_return_gross": float(excess_wo.loc["annualized_return", "risk"]),
        "excess_annualized_return_net": float(excess_w.loc["annualized_return", "risk"]),
        "information_ratio_gross": float(excess_wo.loc["information_ratio", "risk"]),
        "information_ratio_net": float(excess_w.loc["information_ratio", "risk"]),
        "sharpe_ratio": sharpe_ratio(report["return"]),
        "maximum_drawdown": float(port_abs.loc["max_drawdown", "risk"]),
        "excess_maximum_drawdown_gross": float(excess_wo.loc["max_drawdown", "risk"]),
        "excess_maximum_drawdown_net": float(excess_w.loc["max_drawdown", "risk"]),
        "annualized_volatility": annualized_volatility(report["return"]),
        "mean_daily_turnover": float(report["turnover"].mean()),
        "total_turnover": float(report["total_turnover"].iloc[-1]) if len(report) else float("nan"),
        "total_transaction_cost": float(report["total_cost"].iloc[-1]) if len(report) else float("nan"),
        "mean_daily_cost": float(report["cost"].mean()),
        "annualized_cost_drag": float(
            excess_wo.loc["annualized_return", "risk"] - excess_w.loc["annualized_return", "risk"]
        ),
        "number_of_trades": int(indicators["count"].fillna(0).sum()),
        "average_number_of_holdings": avg_pos,
        "portfolio_coverage": float(avg_pos / TOPK) if avg_pos is not None else float("nan"),
        **pred_diag,
    }
    return row


def holdings_frame(positions: dict[Any, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dt, pos in positions.items():
        ts = pd.Timestamp(dt)
        for inst in pos.get_stock_list():
            rows.append({"datetime": ts, "instrument": inst})
    return pd.DataFrame(rows)


def daily_topk_from_pred(pred: pd.DataFrame, k: int = TOPK) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dt, grp in pred.groupby(level="datetime"):
        top = grp["score"].nlargest(min(k, len(grp))).index.get_level_values("instrument").tolist()
        for inst in top:
            rows.append({"datetime": pd.Timestamp(dt), "instrument": inst})
    return pd.DataFrame(rows)


def rank_map(pred: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for dt, grp in pred.groupby(level="datetime"):
        ranks = grp["score"].rank(ascending=False, method="average")
        tmp = ranks.reset_index()
        tmp["datetime"] = pd.Timestamp(dt)
        tmp = tmp.rename(columns={"score": "rank"})
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True)


def holdings_overlap_stats(
    holdings_a: pd.DataFrame,
    holdings_b: pd.DataFrame,
    *,
    compare_label: str,
    reference_label: str,
    variant: str,
    period: str,
) -> dict[str, Any]:
    if holdings_a.empty or holdings_b.empty:
        return {}
    merged = holdings_a.merge(
        holdings_b,
        on=["datetime", "instrument"],
        how="outer",
        indicator=True,
    )
    daily_rows: list[dict[str, Any]] = []
    for dt, ga in holdings_a.groupby("datetime"):
        gb = holdings_b[holdings_b["datetime"] == dt]
        sa = set(ga["instrument"])
        sb = set(gb["instrument"])
        overlap = len(sa & sb)
        union = len(sa | sb) if (sa or sb) else 0
        daily_rows.append(
            {
                "datetime": dt,
                "overlap_count": overlap,
                "overlap_ratio": overlap / TOPK if TOPK else np.nan,
                "jaccard": overlap / union if union else np.nan,
                "holdings_changed_pct": 1.0 - (overlap / TOPK if TOPK else np.nan),
            }
        )
    daily = pd.DataFrame(daily_rows)
    return {
        "variant": variant,
        "period": period,
        "compare_model": compare_label,
        "reference_model": reference_label,
        "mean_overlap_count": float(daily["overlap_count"].mean()),
        "mean_overlap_ratio": float(daily["overlap_ratio"].mean()),
        "mean_jaccard": float(daily["jaccard"].mean()),
        "mean_holdings_changed_pct": float(daily["holdings_changed_pct"].mean()),
        "median_overlap_count": float(daily["overlap_count"].median()),
        "daily_detail": daily,
    }


def rank_change_stats(
    pred_ref: pd.DataFrame,
    pred_cmp: pd.DataFrame,
    *,
    variant: str,
    period: str,
    compare_label: str,
    reference_label: str,
) -> dict[str, Any]:
    ref = rank_map(pred_ref).rename(columns={"rank": "rank_ref"})
    cmp = rank_map(pred_cmp).rename(columns={"rank": "rank_cmp"})
    merged = ref.merge(cmp, on=["datetime", "instrument"], how="inner")
    merged["rank_change"] = (merged["rank_cmp"] - merged["rank_ref"]).abs()
    ref_top = daily_topk_from_pred(pred_ref)
    cmp_top = daily_topk_from_pred(pred_cmp)
    top_union = pd.concat([ref_top, cmp_top]).drop_duplicates(["datetime", "instrument"])
    top_merged = top_union.merge(ref, on=["datetime", "instrument"], how="left").merge(
        cmp,
        on=["datetime", "instrument"],
        how="left",
    )
    top_merged["rank_change"] = (top_merged["rank_cmp"] - top_merged["rank_ref"]).abs()
    return {
        "variant": variant,
        "period": period,
        "compare_model": compare_label,
        "reference_model": reference_label,
        "mean_abs_rank_change_common_universe": float(merged["rank_change"].mean()),
        "mean_abs_rank_change_topk_union": float(top_merged["rank_change"].dropna().mean()),
        "median_abs_rank_change_topk_union": float(top_merged["rank_change"].dropna().median()),
    }


def load_mcap_lookup() -> pd.DataFrame:
    mcap = pd.read_parquet(CRSP_MCAP, columns=["permno", "date", "market_cap"])
    mcap["instrument"] = "P" + mcap["permno"].astype(str)
    mcap = mcap.rename(columns={"date": "datetime"})
    return mcap[["datetime", "instrument", "market_cap"]]


def holdings_mcap_summary(holdings: pd.DataFrame, mcap_lookup: pd.DataFrame, *, model_label: str, variant: str, period: str) -> dict[str, Any]:
    if holdings.empty:
        return {}
    merged = holdings.merge(mcap_lookup, on=["datetime", "instrument"], how="left")
    valid = merged["market_cap"].notna()
    return {
        "variant": variant,
        "period": period,
        "model_label": model_label,
        "mean_holding_market_cap": float(merged.loc[valid, "market_cap"].mean()),
        "median_holding_market_cap": float(merged.loc[valid, "market_cap"].median()),
        "mcap_coverage": float(valid.mean()),
        "large_cap_share_gt_10bn": float((merged.loc[valid, "market_cap"] >= 10e9).mean()) if valid.any() else float("nan"),
        "mid_cap_share_2_10bn": float(((merged.loc[valid, "market_cap"] >= 2e9) & (merged.loc[valid, "market_cap"] < 10e9)).mean())
        if valid.any()
        else float("nan"),
        "small_cap_share_lt_2bn": float((merged.loc[valid, "market_cap"] < 2e9).mean()) if valid.any() else float("nan"),
    }


def block_bootstrap_mean(series: pd.Series, block_size: int = BOOT_BLOCK_SIZE, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict[str, float]:
    arr = series.dropna().to_numpy()
    n = len(arr)
    if n == 0:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    max_start = max(n - block_size + 1, 1)
    n_blocks = int(np.ceil(n / block_size))
    for i in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([arr[s : s + block_size] for s in starts])[:n]
        means[i] = sample.mean()
    return {
        "mean": float(arr.mean()),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
    }


def daily_excess_frame(result: dict[str, Any], *, variant: str, model_id: str, sample_type: str) -> pd.DataFrame:
    report = result["report"]
    out = pd.DataFrame(
        {
            "datetime": report.index,
            "portfolio_return": report["return"].to_numpy(),
            "benchmark_return": report["bench"].to_numpy(),
            "cost": report["cost"].to_numpy(),
            "turnover": report["turnover"].to_numpy(),
            "excess_return_gross": (report["return"] - report["bench"]).to_numpy(),
            "excess_return_net": (report["return"] - report["bench"] - report["cost"]).to_numpy(),
        }
    )
    out["variant"] = variant
    out["model_id"] = model_id
    out["sample_type"] = sample_type
    return out


def write_portfolio_configuration(path: Path, port_cfg: dict[str, Any]) -> None:
    strategy = rpa.effective_strategy_params(port_cfg["strategy"])
    backtest = port_cfg["backtest"]
    exchange = backtest["exchange_kwargs"]
    text = f"""# F2P Portfolio Configuration

Frozen official Alpha20 baseline portfolio specification.

## Source

- Workflow: `{WORKFLOW_YAML}`
- Benchmark report: `{BASELINE_REPORT}`
- Phase 3 runner defaults: `{PHASE3_SCRIPTS / 'run_portfolio_ablation.py'}`

## Strategy

| Parameter | Value |
|-----------|-------|
| class | `{strategy['class']}` |
| module_path | `{strategy['module_path']}` |
| topk | `{strategy['kwargs']['topk']}` |
| n_drop | `{strategy['kwargs']['n_drop']}` |
| hold_thresh | `{strategy['kwargs']['hold_thresh']}` |
| risk_degree | `{strategy['kwargs']['risk_degree']}` |
| method_buy | `{strategy['kwargs']['method_buy']}` |
| method_sell | `{strategy['kwargs']['method_sell']}` |
| only_tradable | `{strategy['kwargs']['only_tradable']}` |
| forbid_all_trade_at_limit | `{strategy['kwargs']['forbid_all_trade_at_limit']}` |

## Backtest

| Parameter | Value |
|-----------|-------|
| start_time | `{backtest['start_time']}` |
| end_time | `{backtest['end_time']}` |
| account | `{backtest['account']}` |
| benchmark | `{backtest['benchmark']}` |
| deal_price | `{exchange['deal_price']}` |
| limit_threshold | `{exchange['limit_threshold']}` |
| open_cost | `{exchange['open_cost']}` |
| close_cost | `{exchange['close_cost']}` |
| min_cost | `{exchange['min_cost']}` |

## Executor

- class: `{rpa.EXECUTOR_CONFIG['class']}`
- time_per_step: `{rpa.EXECUTOR_CONFIG['kwargs']['time_per_step']}`

## F2 fusion (frozen, not tuned here)

- selected alpha: `{FROZEN_ALPHA}`
- fusion formula: `{FROZEN_ALPHA} * Alpha20_cs_z + {1 - FROZEN_ALPHA:.2f} * F1C_cs_z`

## Evaluation scope

- Test only: `{TEST_START}` to `{TEST_END}`
- Holdout `{HOLDOUT_START}+` excluded
- No portfolio-parameter optimisation in F2P
"""
    path.write_text(text, encoding="utf-8")


def save_model_artifacts(out_dir: Path, summary: dict[str, Any], result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_safe), encoding="utf-8")
    with (out_dir / "report_normal_1day.pkl").open("wb") as handle:
        pickle.dump(result["report"], handle)
    with (out_dir / "positions_normal_1day.pkl").open("wb") as handle:
        pickle.dump(result["positions"], handle)
    with (out_dir / "indicators_normal_1day.pkl").open("wb") as handle:
        pickle.dump(result["indicators_df"], handle)


def build_model_specs(variant: str) -> list[dict[str, Any]]:
    f1c = F1C_MAIN if variant == "main" else F1C_NEXTDAY
    f2_sel = F2_SELECTED_MAIN if variant == "main" else F2_SELECTED_NEXTDAY
    f2_eq = F2_EQUAL_MAIN if variant == "main" else F2_EQUAL_NEXTDAY
    return [
        {
            "model_id": "alpha20_full",
            "model_label": "Alpha20 official baseline (full universe)",
            "path": ALPHA20_PRED,
            "sample_type": "full_universe",
            "common_restrict": False,
        },
        {
            "model_id": "alpha20_common",
            "model_label": "Alpha20 common-sample baseline",
            "path": ALPHA20_PRED,
            "sample_type": "common_sample",
            "common_restrict": True,
        },
        {
            "model_id": "f1c_only",
            "model_label": "F1C fundamental-only ensemble",
            "path": f1c,
            "sample_type": "common_sample",
            "common_restrict": True,
        },
        {
            "model_id": "f2_selected",
            "model_label": f"F2 selected fusion (alpha={FROZEN_ALPHA:.2f})",
            "path": f2_sel,
            "sample_type": "common_sample",
            "common_restrict": True,
        },
        {
            "model_id": "f2_equal",
            "model_label": "F2 equal-weight fusion (alpha=0.50)",
            "path": f2_eq,
            "sample_type": "common_sample",
            "common_restrict": True,
        },
    ]


def run_variant(
    variant: str,
    port_cfg: dict[str, Any],
    mcap_lookup: pd.DataFrame,
) -> dict[str, Any]:
    raw_preds = {spec["model_id"]: load_pred(spec["path"]) for spec in build_model_specs(variant)}
    common_idx = common_keys(raw_preds["alpha20_full"], raw_preds["f1c_only"])

    preds: dict[str, pd.DataFrame] = {}
    for spec in build_model_specs(variant):
        pred = raw_preds[spec["model_id"]]
        if spec["common_restrict"]:
            pred = restrict_pred(pred, common_idx)
        preds[spec["model_id"]] = pred

    full_results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []

    for spec in build_model_specs(variant):
        model_id = spec["model_id"]
        pred = preds[model_id]
        out_dir = OUT_ROOT / variant / model_id
        result = run_period_backtest(port_cfg, pred, TEST_START, TEST_END)
        full_results[model_id] = result
        summary = {
            "model_id": model_id,
            "model_label": spec["model_label"],
            "variant": variant,
            "sample_type": spec["sample_type"],
            "pred_path": str(spec["path"]),
            "pred_diagnostics": prediction_diagnostics(pred),
        }
        save_model_artifacts(out_dir, summary, result)
        daily_frames.append(daily_excess_frame(result, variant=variant, model_id=model_id, sample_type=spec["sample_type"]))

        for period, (start, end) in PERIODS.items():
            period_pred = pred
            period_result = run_period_backtest(port_cfg, period_pred, start, end)
            summary_rows.append(
                extract_metrics(
                    period_result,
                    period_pred,
                    model_id=model_id,
                    model_label=spec["model_label"],
                    variant=variant,
                    sample_type=spec["sample_type"],
                    period=period,
                    period_start=start,
                    period_end=end,
                )
            )

    overlap_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    mcap_rows: list[dict[str, Any]] = []
    overlap_daily_parts: list[pd.DataFrame] = []

    ref_id = "alpha20_common"
    cmp_id = "f2_selected"
    ref_pred = preds[ref_id]
    cmp_pred = preds[cmp_id]
    for period in ["test", "covid", "post_covid", "2020", "2021", "2022", "2023"]:
        start, end = PERIODS[period]
        ref_period_pred = filter_pred_period(ref_pred, start, end)
        cmp_period_pred = filter_pred_period(cmp_pred, start, end)
        ref_result = run_period_backtest(port_cfg, ref_period_pred, start, end)
        cmp_result = run_period_backtest(port_cfg, cmp_period_pred, start, end)

        ref_hold = holdings_frame(ref_result["positions"])
        cmp_hold = holdings_frame(cmp_result["positions"])
        overlap = holdings_overlap_stats(
            ref_hold,
            cmp_hold,
            compare_label=cmp_id,
            reference_label=ref_id,
            variant=variant,
            period=period,
        )
        if overlap:
            daily = overlap.pop("daily_detail")
            daily["variant"] = variant
            daily["period"] = period
            overlap_daily_parts.append(daily)
            overlap_rows.append({k: v for k, v in overlap.items() if k != "daily_detail"})
        rank_rows.append(
            rank_change_stats(
                ref_period_pred,
                cmp_period_pred,
                variant=variant,
                period=period,
                compare_label=cmp_id,
                reference_label=ref_id,
            )
        )
        turnover_rows.append(
            {
                "variant": variant,
                "period": period,
                "reference_model": ref_id,
                "compare_model": cmp_id,
                "alpha20_common_mean_daily_turnover": float(ref_result["report"]["turnover"].mean()),
                "f2_selected_mean_daily_turnover": float(cmp_result["report"]["turnover"].mean()),
                "turnover_difference_f2_minus_alpha20": float(
                    cmp_result["report"]["turnover"].mean() - ref_result["report"]["turnover"].mean()
                ),
                "alpha20_common_total_cost": float(ref_result["report"]["total_cost"].iloc[-1]),
                "f2_selected_total_cost": float(cmp_result["report"]["total_cost"].iloc[-1]),
                "total_cost_difference_f2_minus_alpha20": float(
                    cmp_result["report"]["total_cost"].iloc[-1] - ref_result["report"]["total_cost"].iloc[-1]
                ),
            }
        )
        if period == "test":
            for model_id, label in [(ref_id, "Alpha20 common"), (cmp_id, "F2 selected fusion")]:
                mcap_rows.append(
                    holdings_mcap_summary(
                        holdings_frame(full_results[model_id]["positions"]),
                        mcap_lookup,
                        model_label=label,
                        variant=variant,
                        period=period,
                    )
                )

    full_diag = prediction_diagnostics(preds["alpha20_full"])
    common_diag = prediction_diagnostics(preds["f1c_only"])
    recon = pd.DataFrame(
        [
            {"variant": variant, "sample": "A_alpha20_full_universe", **full_diag},
            {"variant": variant, "sample": "B_common_fundamental_sample", **common_diag},
            {
                "variant": variant,
                "sample": "delta_rows_full_minus_common",
                "prediction_rows": int(full_diag["prediction_rows"] - common_diag["prediction_rows"]),
                "unique_dates": np.nan,
                "unique_instruments": np.nan,
                "avg_cross_section_n": np.nan,
                "days_with_insufficient_predictions": np.nan,
                "min_daily_predictions": np.nan,
                "max_daily_predictions": np.nan,
            },
        ]
    )

    return {
        "preds": preds,
        "full_results": full_results,
        "summary_rows": summary_rows,
        "daily_frames": daily_frames,
        "overlap_rows": overlap_rows,
        "overlap_daily_parts": overlap_daily_parts,
        "turnover_rows": turnover_rows,
        "rank_rows": rank_rows,
        "mcap_rows": mcap_rows,
        "recon": recon,
    }


def bootstrap_uncertainty(main: dict[str, Any]) -> pd.DataFrame:
    daily = pd.concat(main["daily_frames"], ignore_index=True)
    pivot_gross = daily.pivot_table(index="datetime", columns="model_id", values="excess_return_gross", aggfunc="first")
    rows: list[dict[str, Any]] = []
    for model_id in pivot_gross.columns:
        stats = block_bootstrap_mean(pivot_gross[model_id])
        rows.append(
            {
                "variant": "main",
                "comparison": model_id,
                "metric": "daily_excess_return_gross_mean",
                "point_estimate": stats["mean"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
                "block_size": BOOT_BLOCK_SIZE,
                "n_bootstrap": BOOT_N,
            }
        )
    if {"f2_selected", "alpha20_common"}.issubset(pivot_gross.columns):
        diff = pivot_gross["f2_selected"] - pivot_gross["alpha20_common"]
        stats = block_bootstrap_mean(diff)
        rows.append(
            {
                "variant": "main",
                "comparison": "f2_selected_minus_alpha20_common",
                "metric": "daily_excess_return_gross_diff_mean",
                "point_estimate": stats["mean"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
                "block_size": BOOT_BLOCK_SIZE,
                "n_bootstrap": BOOT_N,
            }
        )
        diff_net = daily.pivot_table(index="datetime", columns="model_id", values="excess_return_net", aggfunc="first")
        stats_net = block_bootstrap_mean(diff_net["f2_selected"] - diff_net["alpha20_common"])
        rows.append(
            {
                "variant": "main",
                "comparison": "f2_selected_minus_alpha20_common",
                "metric": "daily_excess_return_net_diff_mean",
                "point_estimate": stats_net["mean"],
                "ci_lower": stats_net["ci_lower"],
                "ci_upper": stats_net["ci_upper"],
                "block_size": BOOT_BLOCK_SIZE,
                "n_bootstrap": BOOT_N,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    recon: pd.DataFrame,
    turnover: pd.DataFrame,
    qa: dict[str, Any],
) -> None:
    test = summary[summary["period"] == "test"]
    lines = [
        "# F2P Portfolio Backtest Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Frozen configuration",
        "",
        "- Official Alpha20 TopkDropout: topk=20, n_drop=2, hold_thresh=1, 1bp costs, benchmark P84398",
        f"- F2 selected alpha fixed at **{FROZEN_ALPHA:.2f}** (validation-selected; not retuned here)",
        "- Test window only: 2020-2023; 2024 holdout excluded",
        "",
        "## QA",
        "",
        "```",
        json.dumps(qa, indent=2),
        "```",
        "",
        "## Test-period portfolio summary",
        "",
        "```",
        test.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "```",
        "",
        "## Common-sample reconciliation",
        "",
        "```",
        recon.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "```",
        "",
        "## Fusion vs Alpha20 turnover / cost",
        "",
        "```",
        turnover.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "```",
        "",
        "## Block-bootstrap uncertainty (main)",
        "",
        "```",
        bootstrap.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "```",
        "",
        "## Sector attribution",
        "",
        "Sector weights are **not reported**: no frozen sector mapping is wired into the F2P "
        "portfolio pipeline. Market-cap distribution diagnostics are in model artifact summaries.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str(PROJECT_ROOT / "staging/qlib_data"), region=REG_US, kernels=1)

    port_cfg = load_port_config()
    write_portfolio_configuration(OUT_ROOT / "F2P_portfolio_configuration.md", port_cfg)
    mcap_lookup = load_mcap_lookup()

    main_res = run_variant("main", port_cfg, mcap_lookup)
    next_res = run_variant("nextday", port_cfg, mcap_lookup)

    summary = pd.concat([pd.DataFrame(main_res["summary_rows"]), pd.DataFrame(next_res["summary_rows"])])
    summary.to_csv(OUT_ROOT / "F2P_portfolio_summary.csv", index=False)

    annual = summary[summary["period"].isin(["2020", "2021", "2022", "2023"])]
    annual.to_csv(OUT_ROOT / "F2P_annual_performance.csv", index=False)

    daily = pd.concat(main_res["daily_frames"] + next_res["daily_frames"], ignore_index=True)
    daily.to_parquet(OUT_ROOT / "F2P_daily_portfolio_returns.parquet", index=False)

    overlap = pd.DataFrame(main_res["overlap_rows"] + next_res["overlap_rows"])
    overlap.to_csv(OUT_ROOT / "F2P_holdings_overlap.csv", index=False)
    overlap_daily = pd.concat(main_res["overlap_daily_parts"] + next_res["overlap_daily_parts"], ignore_index=True)
    overlap_daily.to_parquet(OUT_ROOT / "F2P_holdings_overlap_daily.parquet", index=False)

    turnover = pd.DataFrame(main_res["turnover_rows"] + next_res["turnover_rows"])
    turnover.to_csv(OUT_ROOT / "F2P_turnover_and_costs.csv", index=False)

    recon = pd.concat([main_res["recon"], next_res["recon"]], ignore_index=True)
    recon.to_csv(OUT_ROOT / "F2P_common_sample_reconciliation.csv", index=False)

    rank_df = pd.DataFrame(main_res["rank_rows"] + next_res["rank_rows"])
    rank_df.to_csv(OUT_ROOT / "F2P_rank_change_diagnostics.csv", index=False)
    mcap_df = pd.DataFrame([r for r in main_res["mcap_rows"] + next_res["mcap_rows"] if r])
    mcap_df.to_csv(OUT_ROOT / "F2P_holdings_market_cap.csv", index=False)

    bootstrap = bootstrap_uncertainty(main_res)
    bootstrap.to_csv(OUT_ROOT / "F2P_bootstrap_uncertainty.csv", index=False)

    main_test = summary[(summary["variant"] == "main") & (summary["period"] == "test")]
    next_test = summary[(summary["variant"] == "nextday") & (summary["period"] == "test")]
    cmp = main_test.merge(next_test, on=["model_id", "sample_type", "period"], suffixes=("_main", "_nextday"))
    cmp_rows = []
    for _, r in cmp.iterrows():
        cmp_rows.append(
            {
                "model_id": r["model_id"],
                "model_label": r["model_label_main"],
                "period": r["period"],
                "excess_arr_net_main": r["excess_annualized_return_net_main"],
                "excess_arr_net_nextday": r["excess_annualized_return_net_nextday"],
                "excess_arr_net_diff_main_minus_nextday": r["excess_annualized_return_net_main"]
                - r["excess_annualized_return_net_nextday"],
                "ir_net_main": r["information_ratio_net_main"],
                "ir_net_nextday": r["information_ratio_net_nextday"],
                "mdd_net_main": r["excess_maximum_drawdown_net_main"],
                "mdd_net_nextday": r["excess_maximum_drawdown_net_nextday"],
                "turnover_main": r["mean_daily_turnover_main"],
                "turnover_nextday": r["mean_daily_turnover_nextday"],
                "annualized_return_main": r["annualized_return_main"],
                "annualized_return_nextday": r["annualized_return_nextday"],
            }
        )
    overlap_cmp = overlap[(overlap["period"] == "test")]
    overlap_main = overlap_cmp[overlap_cmp["variant"] == "main"]
    overlap_next = overlap_cmp[overlap_cmp["variant"] == "nextday"]
    cmp_df = pd.DataFrame(cmp_rows)
    if not overlap_main.empty and not overlap_next.empty:
        cmp_df["holdings_overlap_ratio_main"] = float(overlap_main["mean_overlap_ratio"].iloc[0])
        cmp_df["holdings_overlap_ratio_nextday"] = float(overlap_next["mean_overlap_ratio"].iloc[0])
    cmp_df.to_csv(OUT_ROOT / "F2P_main_vs_nextday_comparison.csv", index=False)

    baseline = main_test[main_test["model_id"] == "alpha20_full"].iloc[0]
    expected = json.loads(BASELINE_REPORT.read_text())["excess_metrics_vs_p84398"]["annualized_return_with_cost"]
    qa = {
        "holdout_loaded": False,
        "portfolio_params_tuned_on_test": False,
        "frozen_alpha": FROZEN_ALPHA,
        "all_models_same_portfolio_settings": True,
        "labels_used_in_portfolio": False,
        "alpha20_full_excess_arr_net": float(baseline["excess_annualized_return_net"]),
        "alpha20_full_excess_arr_expected": float(expected),
        "alpha20_full_arr_match": abs(float(baseline["excess_annualized_return_net"]) - float(expected)) < 1e-3,
        "duplicate_prediction_keys_checked": True,
        "common_and_full_alpha20_reported_separately": True,
    }
    meta = {
        "experiment": "F2P_portfolio_backtest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_alpha": FROZEN_ALPHA,
        "portfolio_config_source": str(WORKFLOW_YAML),
        "alpha20_pred": str(ALPHA20_PRED),
        "qa": qa,
        "holdout_evaluated": False,
    }
    (OUT_ROOT / "F2P_meta.json").write_text(json.dumps(meta, indent=2, default=json_safe), encoding="utf-8")

    write_report(
        OUT_ROOT / "F2P_portfolio_backtest_report.md",
        summary,
        bootstrap,
        recon,
        turnover[turnover["period"] == "test"],
        qa,
    )

    log.info("F2P complete. alpha20_full excess ARR net=%.6f (expected %.6f)", baseline["excess_annualized_return_net"], expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
