#!/usr/bin/env python3
"""F1_fundamental_lightgbm.py

Experiment F1: fundamental-only LightGBM on Phase 5A features with Alpha20
label / split / LightGBM conventions. Train/valid/test within 2008–2023."""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.data.dataset.processor import zscore
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

WORKFLOW_YAML = PROJECT_ROOT / "experiments" / "conf_alpha20_sp500_transfer.yaml"
MAIN_DATASET = (
    PROJECT_ROOT / "data" / "fundamental_model_dataset" / "main" / "fundamental_model_dataset.parquet"
)
NEXTDAY_DATASET = (
    PROJECT_ROOT
    / "data"
    / "fundamental_model_dataset"
    / "nextday"
    / "fundamental_model_dataset.parquet"
)
ALPHA20_PRED = (
    PROJECT_ROOT
    / "mlruns"
    / "561707949424138210"
    / "11ee753279e548a99601c34ef6ca195f"
    / "artifacts"
    / "pred.pkl"
)
OUT_ROOT = PROJECT_ROOT / "data" / "fundamental_experiments" / "F1_fundamental_lightgbm"

ALL12 = [
    "roe_z",
    "roa_z",
    "gross_profitability_z",
    "operating_profitability_z",
    "sales_growth_yoy_z",
    "asset_growth_yoy_z",
    "accruals_z",
    "leverage_z",
    "current_ratio_z",
    "book_to_market_z",
    "earnings_yield_z",
    "sales_to_price_z",
]

MISSING_FLAGS = [f.replace("_z", "_missing_flag") for f in ALL12]

SPECS: dict[str, dict[str, Any]] = {
    "F1A": {
        "name": "F1A_all12",
        "features": ALL12,
        "pred_file": "F1A_all12_pred.pkl",
        "description": "All 12 fundamental z-scored features (primary)",
    },
    "F1B": {
        "name": "F1B_roa_gp",
        "features": ["roa_z", "gross_profitability_z"],
        "pred_file": "F1B_roa_gp_pred.pkl",
        "description": "F0 train/valid-selected parsimonious model",
    },
    "F1C": {
        "name": "F1C_no_operating_profitability",
        "features": [f for f in ALL12 if f != "operating_profitability_z"],
        "pred_file": "F1C_no_operating_profitability_pred.pkl",
        "description": "All 12 except operating_profitability_z (collinearity ablation)",
    },
    "F1D": {
        "name": "F1D_all12_missing_flags",
        "features": ALL12 + MISSING_FLAGS,
        "pred_file": "F1D_all12_missing_flags_pred.pkl",
        "description": "All 12 z features plus missing indicators (ablation)",
    },
}

PERIODS: dict[str, tuple[str, str]] = {
    "train": ("2008-01-01", "2017-12-31"),
    "valid": ("2018-01-01", "2019-12-31"),
    "test": ("2020-01-01", "2023-12-31"),
    "covid": ("2020-01-01", "2021-12-31"),
    "post_covid": ("2022-01-01", "2023-12-31"),
}

SEEDS = [42, 2026, 3407]
PRIMARY_SEED = 42
MIN_CROSS_SECTION = 30

# Official Alpha20 LGBModel defaults (qlib.contrib.model.gbdt.LGBModel)
OFFICIAL_LGB = {
    "objective": "regression",
    "metric": "mse",
    "verbosity": -1,
    "colsample_bytree": 0.8879,
    "learning_rate": 0.2,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 20,
}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

FORBIDDEN_FEATURE_COLS = {
    "label",
    "datetime",
    "instrument",
    "permno",
    "date",
    "qlib_instrument",
    "split",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


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
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(type(obj))


def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].copy()


def drop_holdout(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["split"] != "holdout"].copy()


def process_training_label(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate Alpha20 learn_processors: DropnaLabel + CSZScoreNorm on label."""
    out = df.dropna(subset=["label"]).copy()
    out["label_processed"] = out.groupby("datetime", group_keys=False)["label"].apply(zscore)
    return out


def lgb_params(seed: int) -> dict[str, Any]:
    p = OFFICIAL_LGB.copy()
    p.update(
        {
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        }
    )
    return p


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: list[str],
    seed: int,
) -> lgb.Booster:
    X_train = train_df[features]
    y_train = train_df["label_processed"].to_numpy()
    X_valid = valid_df[features]
    y_valid = valid_df["label_processed"].to_numpy()

    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set, free_raw_data=False)

    model = lgb.train(
        lgb_params(seed),
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def predict_frame(
    model: lgb.Booster,
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    sub = df[["datetime", "instrument", "label"] + features].copy()
    sub["score"] = model.predict(sub[features], num_iteration=model.best_iteration)
    pred = sub[["datetime", "instrument", "score"]].set_index(["datetime", "instrument"]).sort_index()
    return pred


def daily_ic_metrics(
    pred: pd.DataFrame,
    label_df: pd.DataFrame,
    period: str,
    start: str,
    end: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    lab = label_df[["datetime", "instrument", "label"]].dropna(subset=["label"])
    lab = lab[(lab["datetime"] >= start_ts) & (lab["datetime"] <= end_ts)]
    merged = pred.reset_index().merge(lab, on=["datetime", "instrument"], how="inner")
    merged = merged.dropna(subset=["score", "label"])

    daily_rows = []
    for dt, g in merged.groupby("datetime"):
        if len(g) < MIN_CROSS_SECTION:
            continue
        if g["score"].nunique() < 2 or g["label"].nunique() < 2:
            continue
        ic = g["score"].corr(g["label"], method="pearson")
        ric = g["score"].corr(g["label"], method="spearman")
        daily_rows.append({"datetime": dt, "period": period, "ic": ic, "rank_ic": ric, "n_obs": len(g)})

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return {
            "period": period,
            "ic_mean": np.nan,
            "ic_median": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "ic_tstat": np.nan,
            "ic_positive_pct": np.nan,
            "rank_ic_mean": np.nan,
            "rank_ic_median": np.nan,
            "rank_ic_std": np.nan,
            "rank_ic_ir": np.nan,
            "rank_ic_tstat": np.nan,
            "rank_ic_positive_pct": np.nan,
            "valid_ic_dates": 0,
            "avg_cross_section_n": np.nan,
            "prediction_coverage": np.nan,
            "prediction_mean": np.nan,
            "prediction_std": np.nan,
        }, daily

    ic_s = daily["ic"]
    ric_s = daily["rank_ic"]

    def _tstat(s: pd.Series) -> float:
        if len(s) <= 1:
            return float("nan")
        std = s.std(ddof=1)
        return float(s.mean() / (std / np.sqrt(len(s)))) if std > 0 else float("nan")

    pred_period = pred.reset_index()
    pred_period = pred_period[
        (pred_period["datetime"] >= start_ts) & (pred_period["datetime"] <= end_ts)
    ]
    return {
        "period": period,
        "ic_mean": float(ic_s.mean()),
        "ic_median": float(ic_s.median()),
        "ic_std": float(ic_s.std(ddof=1)),
        "ic_ir": float(ic_s.mean() / ic_s.std(ddof=1)) if ic_s.std(ddof=1) > 0 else float("nan"),
        "ic_tstat": _tstat(ic_s),
        "ic_positive_pct": float((ic_s > 0).mean()),
        "rank_ic_mean": float(ric_s.mean()),
        "rank_ic_median": float(ric_s.median()),
        "rank_ic_std": float(ric_s.std(ddof=1)),
        "rank_ic_ir": float(ric_s.mean() / ric_s.std(ddof=1)) if ric_s.std(ddof=1) > 0 else float("nan"),
        "rank_ic_tstat": _tstat(ric_s),
        "rank_ic_positive_pct": float((ric_s > 0).mean()),
        "valid_ic_dates": int(len(daily)),
        "avg_cross_section_n": float(daily["n_obs"].mean()),
        "prediction_coverage": float(len(merged) / max(len(lab), 1)),
        "prediction_mean": float(pred_period["score"].mean()),
        "prediction_std": float(pred_period["score"].std(ddof=0)),
    }, daily


def permutation_importance_valid(
    model: lgb.Booster,
    valid_df: pd.DataFrame,
    features: list[str],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_pred = predict_frame(model, valid_df, features)
    base_metrics, _ = daily_ic_metrics(
        base_pred,
        valid_df,
        "valid",
        *PERIODS["valid"],
    )
    base_rank_ic = base_metrics["rank_ic_mean"]

    rows = []
    for feat in features:
        perm_df = valid_df.copy()
        # Shuffle within each date to preserve cross-section size
        perm_values = []
        for _, g in perm_df.groupby("datetime"):
            vals = g[feat].to_numpy(copy=True)
            rng.shuffle(vals)
            perm_values.append(pd.Series(vals, index=g.index))
        perm_df[feat] = pd.concat(perm_values).sort_index()
        perm_pred = predict_frame(model, perm_df, features)
        perm_metrics, _ = daily_ic_metrics(perm_pred, valid_df, "valid", *PERIODS["valid"])
        rows.append(
            {
                "feature": feat,
                "baseline_valid_rank_ic": base_rank_ic,
                "perm_valid_rank_ic": perm_metrics["rank_ic_mean"],
                "rank_ic_drop": base_rank_ic - perm_metrics["rank_ic_mean"],
            }
        )
    return pd.DataFrame(rows).sort_values("rank_ic_drop", ascending=False)


def compare_alpha20(
    f1_pred: pd.DataFrame,
    alpha20_pred: pd.DataFrame,
    label_df: pd.DataFrame,
    spec: str,
    variant: str,
) -> list[dict[str, Any]]:
    rows = []
    for period in ["test", "covid", "post_covid"]:
        start, end = PERIODS[period]
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        a = alpha20_pred.reset_index()
        a = a[(a["datetime"] >= start_ts) & (a["datetime"] <= end_ts)]
        f = f1_pred.reset_index()
        f = f[(f["datetime"] >= start_ts) & (f["datetime"] <= end_ts)]
        common = f.merge(a, on=["datetime", "instrument"], suffixes=("_f1", "_a20"))
        if common.empty:
            continue
        pred_corr = float(common["score_f1"].corr(common["score_a20"]))
        daily_corrs = common.groupby("datetime").apply(
            lambda g: g["score_f1"].corr(g["score_a20"]) if len(g) >= MIN_CROSS_SECTION else np.nan,
            include_groups=False,
        ).dropna()

        f1_m, _ = daily_ic_metrics(f1_pred, label_df, period, start, end)
        # Alpha20 IC vs raw label
        a_pred = a.rename(columns={"score": "score"}).set_index(["datetime", "instrument"])
        a_m, _ = daily_ic_metrics(a_pred, label_df, period, start, end)

        rows.append(
            {
                "spec": spec,
                "variant": variant,
                "period": period,
                "common_observations": int(len(common)),
                "prediction_pearson_corr": pred_corr,
                "daily_prediction_corr_mean": float(daily_corrs.mean()) if len(daily_corrs) else np.nan,
                "f1_ic": f1_m["ic_mean"],
                "alpha20_ic": a_m["ic_mean"],
                "ic_difference_f1_minus_a20": f1_m["ic_mean"] - a_m["ic_mean"],
                "f1_rank_ic": f1_m["rank_ic_mean"],
                "alpha20_rank_ic": a_m["rank_ic_mean"],
                "rank_ic_difference_f1_minus_a20": f1_m["rank_ic_mean"] - a_m["rank_ic_mean"],
            }
        )
    return rows


def qa_checks(df: pd.DataFrame, features: list[str], pred: pd.DataFrame) -> dict[str, Any]:
    splits = df["split"].unique().tolist()
    holdout_used = "holdout" in splits and int((df["split"] == "holdout").sum()) > 0
    dup_pred = int(pred.reset_index().duplicated(["datetime", "instrument"]).sum())
    overlap_tv = len(
        set(df.loc[df["split"] == "train", "datetime"].dt.normalize())
        & set(df.loc[df["split"] == "valid", "datetime"].dt.normalize())
    )
    overlap_tt = len(
        set(df.loc[df["split"] == "train", "datetime"].dt.normalize())
        & set(df.loc[df["split"] == "test", "datetime"].dt.normalize())
    )
    forbidden_in_x = [c for c in features if c in FORBIDDEN_FEATURE_COLS or c == "label"]
    return {
        "holdout_rows_in_training_data": int((df["split"] == "holdout").sum()),
        "holdout_used_in_development": holdout_used,
        "duplicate_datetime_instrument_predictions": dup_pred,
        "train_valid_date_overlap": overlap_tv,
        "train_test_date_overlap": overlap_tt,
        "forbidden_columns_in_feature_list": forbidden_in_x,
        "label_in_feature_matrix": "label" in features,
        "prediction_index_names": list(pred.index.names),
        "passed": (
            not holdout_used
            and dup_pred == 0
            and overlap_tv == 0
            and overlap_tt == 0
            and not forbidden_in_x
            and "label" not in features
            and pred.index.names == ["datetime", "instrument"]
        ),
    }


def write_model_configuration_md(path: Path) -> None:
    lines = [
        "# F1 LightGBM Model Configuration",
        "",
        "Configuration aligned with the official Alpha20 baseline "
        f"(`{WORKFLOW_YAML}`) and `qlib.contrib.model.gbdt.LGBModel` defaults.",
        "",
        "## Official Alpha20 LightGBM parameters",
        "",
        "| Parameter | Value | Source |",
        "|-----------|-------|--------|",
        f"| objective | regression (qlib: mse) | task.model.kwargs.loss |",
        f"| learning_rate | {OFFICIAL_LGB['learning_rate']} | conf_alpha20_sp500_transfer.yaml |",
        f"| num_leaves | {OFFICIAL_LGB['num_leaves']} | conf_alpha20_sp500_transfer.yaml |",
        f"| max_depth | {OFFICIAL_LGB['max_depth']} | conf_alpha20_sp500_transfer.yaml |",
        f"| colsample_bytree (feature_fraction) | {OFFICIAL_LGB['colsample_bytree']} | conf_alpha20_sp500_transfer.yaml |",
        f"| subsample (bagging_fraction) | {OFFICIAL_LGB['subsample']} | conf_alpha20_sp500_transfer.yaml |",
        f"| bagging_freq | not specified (LightGBM default: 0) | — |",
        f"| lambda_l1 | {OFFICIAL_LGB['lambda_l1']} | conf_alpha20_sp500_transfer.yaml |",
        f"| lambda_l2 | {OFFICIAL_LGB['lambda_l2']} | conf_alpha20_sp500_transfer.yaml |",
        f"| min_data_in_leaf | not specified (LightGBM default: 20) | — |",
        f"| num_boost_round | {NUM_BOOST_ROUND} | LGBModel default |",
        f"| early_stopping_rounds | {EARLY_STOPPING_ROUNDS} | LGBModel default |",
        f"| num_threads | {OFFICIAL_LGB['num_threads']} | conf_alpha20_sp500_transfer.yaml |",
        f"| seeds (fixed set) | {SEEDS} | F1 experiment protocol |",
        "",
        "## Label processing (training only)",
        "",
        "Matches official `learn_processors`:",
        "",
        "1. **DropnaLabel** — drop rows with missing raw LABEL0 before training.",
        "2. **CSZScoreNorm** — cross-sectional z-score of label by `datetime` "
        "(qlib `zscore`, applied independently on train and valid dates).",
        "",
        "**Evaluation label:** raw LABEL0 (`Ref($close, -2)/Ref($close, -1) - 1`) "
        "without CSZScoreNorm, consistent with Alpha20 IC evaluation.",
        "",
        "## Feature processing",
        "",
        "No additional feature standardisation. Inputs are frozen Phase 4B daily "
        "cross-sectional z-scores. Missing values are not imputed; LightGBM handles NaN.",
        "",
        "## Splits",
        "",
        "| Split | Dates | Usage |",
        "|-------|-------|-------|",
        "| train | 2008-01-01 .. 2017-12-31 | model fitting |",
        "| valid | 2018-01-01 .. 2019-12-31 | early stopping |",
        "| test | 2020-01-01 .. 2023-12-31 | evaluation only |",
        "| holdout 2024 | 2024-01-01 .. 2024-12-31 | **excluded from F1 development** |",
        "",
        "## Model specifications",
        "",
    ]
    for spec_id, spec in SPECS.items():
        lines.append(f"- **{spec_id}** ({spec['name']}): {spec['description']}")
        lines.append(f"  - Features: `{', '.join(spec['features'])}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_variant(
    dataset_path: Path,
    variant: str,
    alpha20_pred: pd.DataFrame,
) -> dict[str, Any]:
    log.info("Loading %s dataset from %s", variant, dataset_path)
    raw = pd.read_parquet(dataset_path)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    df = drop_holdout(raw)

    train_raw = df[df["split"] == "train"]
    valid_raw = df[df["split"] == "valid"]
    test_raw = df[df["split"] == "test"]
    eval_raw = pd.concat([train_raw, valid_raw, test_raw], ignore_index=True)

    train = process_training_label(train_raw)
    valid = process_training_label(valid_raw)

    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    period_metrics_rows: list[dict[str, Any]] = []
    daily_ic_frames: list[pd.DataFrame] = []
    gain_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    alpha20_rows: list[dict[str, Any]] = []
    perm_frames: list[pd.DataFrame] = []
    spec_period_maps: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    primary_preds: dict[str, pd.DataFrame] = {}

    for spec_id, spec in SPECS.items():
        features = spec["features"]
        log.info("Variant=%s spec=%s features=%d", variant, spec_id, len(features))

        spec_period_by_seed: dict[int, dict[str, dict[str, Any]]] = {}

        for seed in SEEDS:
            model = train_model(train, valid, features, seed)
            pred_train = predict_frame(model, train_raw, features)
            pred_valid = predict_frame(model, valid_raw, features)
            pred_test = predict_frame(model, test_raw, features)
            pred_all = pd.concat([pred_train, pred_valid, pred_test]).sort_index()
            pred_all = pred_all[~pred_all.index.duplicated(keep="first")]

            best_iter = int(model.best_iteration)
            period_map: dict[str, dict[str, Any]] = {}
            for period in PERIODS:
                if period == "holdout":
                    continue
                metrics, daily = daily_ic_metrics(
                    pred_all, eval_raw, period, *PERIODS[period]
                )
                period_map[period] = metrics
                period_metrics_rows.append(
                    {
                        "variant": variant,
                        "spec": spec_id,
                        "spec_name": spec["name"],
                        "seed": seed,
                        **metrics,
                    }
                )
                if seed == PRIMARY_SEED:
                    dd = daily.copy()
                    dd["variant"] = variant
                    dd["spec"] = spec_id
                    daily_ic_frames.append(dd)

            spec_period_by_seed[seed] = period_map
            seed_rows.append(
                {
                    "variant": variant,
                    "spec": spec_id,
                    "seed": seed,
                    "best_iteration": best_iter,
                    "valid_rank_ic": period_map["valid"]["rank_ic_mean"],
                    "valid_ic": period_map["valid"]["ic_mean"],
                    "test_rank_ic": period_map["test"]["rank_ic_mean"],
                    "test_ic": period_map["test"]["ic_mean"],
                }
            )

            if seed == PRIMARY_SEED:
                primary_preds[spec_id] = pred_all
                qa = qa_checks(df, features, pred_all)
                with (out_dir / spec["pred_file"]).open("wb") as f:
                    pickle.dump(pred_all, f)
                with (out_dir / spec["pred_file"].replace("_pred.pkl", "_pred_valid.pkl")).open("wb") as f:
                    pickle.dump(pred_valid, f)
                with (out_dir / spec["pred_file"].replace("_pred.pkl", "_pred_test.pkl")).open("wb") as f:
                    pickle.dump(pred_test, f)

                imp_gain = model.feature_importance(importance_type="gain")
                imp_split = model.feature_importance(importance_type="split")
                for feat, g, s in zip(features, imp_gain, imp_split):
                    gain_rows.append(
                        {
                            "variant": variant,
                            "spec": spec_id,
                            "feature": feat,
                            "importance": float(g),
                        }
                    )
                    split_rows.append(
                        {
                            "variant": variant,
                            "spec": spec_id,
                            "feature": feat,
                            "importance": int(s),
                        }
                    )

                if spec_id == "F1A":
                    perm = permutation_importance_valid(model, valid_raw, features, seed)
                    perm["variant"] = variant
                    perm["spec"] = spec_id
                    perm_frames.append(perm)

                alpha20_rows.extend(
                    compare_alpha20(pred_all, alpha20_pred, eval_raw, spec_id, variant)
                )

        spec_period_maps[spec_id] = spec_period_by_seed

    summary_rows = []
    for spec_id in SPECS:
        seed_info = [r for r in seed_rows if r["spec"] == spec_id and r["variant"] == variant]
        pm = spec_period_maps[spec_id][PRIMARY_SEED]
        valid_ric = [r["valid_rank_ic"] for r in seed_info]
        row: dict[str, Any] = {
            "variant": variant,
            "spec": spec_id,
            "spec_name": SPECS[spec_id]["name"],
            "primary_seed": PRIMARY_SEED,
            "seeds": SEEDS,
            "valid_rank_ic_mean_across_seeds": float(np.mean(valid_ric)),
            "valid_rank_ic_std_across_seeds": float(np.std(valid_ric, ddof=0)),
            "valid_ic_mean_across_seeds": float(np.mean([r["valid_ic"] for r in seed_info])),
            "test_rank_ic_mean_across_seeds": float(np.mean([r["test_rank_ic"] for r in seed_info])),
            "test_ic_mean_across_seeds": float(np.mean([r["test_ic"] for r in seed_info])),
        }
        for period, metrics in pm.items():
            row[f"{period}_ic"] = metrics["ic_mean"]
            row[f"{period}_rank_ic"] = metrics["rank_ic_mean"]
            row[f"{period}_ic_ir"] = metrics["ic_ir"]
        summary_rows.append(row)

    return {
        "summary_rows": summary_rows,
        "period_metrics_rows": period_metrics_rows,
        "daily_ic_frames": daily_ic_frames,
        "gain_rows": gain_rows,
        "split_rows": split_rows,
        "seed_rows": seed_rows,
        "alpha20_rows": alpha20_rows,
        "perm_frames": perm_frames,
        "primary_preds": primary_preds,
    }


def select_provisional_model(summary: pd.DataFrame) -> dict[str, Any]:
    main = summary[summary["variant"] == "main"].copy()
    main = main.sort_values(
        ["valid_rank_ic_mean_across_seeds", "valid_ic_mean_across_seeds"],
        ascending=False,
    )
    top = main.iloc[0]
    return {
        "provisional_spec": top["spec"],
        "provisional_spec_name": top["spec_name"],
        "selection_basis": "validation Rank IC (mean across seeds), then validation IC",
        "valid_rank_ic_mean_across_seeds": float(top["valid_rank_ic_mean_across_seeds"]),
        "valid_ic_mean_across_seeds": float(top["valid_ic_mean_across_seeds"]),
        "valid_rank_ic_std_across_seeds": float(top["valid_rank_ic_std_across_seeds"]),
        "note": "Test and holdout not used for selection",
    }


def write_report(
    path: Path,
    summary: pd.DataFrame,
    provisional: dict[str, Any],
    alpha20_cmp: pd.DataFrame,
    main_next_cmp: pd.DataFrame,
    seed_stability: pd.DataFrame,
) -> None:
    main_sum = summary[summary["variant"] == "main"]
    lines = [
        "# F1 Fundamental LightGBM Report",
        "",
        "Standalone LightGBM on frozen fundamental features vs raw LABEL0.",
        "**2024 holdout excluded** from this development run.",
        "",
        "## Provisional model (validation-only selection)",
        "",
        f"- **Selected spec:** {provisional['provisional_spec']} ({provisional['provisional_spec_name']})",
        f"- **Basis:** {provisional['selection_basis']}",
        f"- **Valid Rank IC (mean over seeds):** {provisional['valid_rank_ic_mean_across_seeds']:.4f}",
        f"- **Valid IC (mean over seeds):** {provisional['valid_ic_mean_across_seeds']:.4f}",
        "",
        "## Main variant — validation and test IC / Rank IC (seed=42)",
        "",
        "```",
        main_sum[
            [
                "spec",
                "valid_ic",
                "valid_rank_ic",
                "test_ic",
                "test_rank_ic",
                "covid_ic",
                "covid_rank_ic",
                "post_covid_ic",
                "post_covid_rank_ic",
                "valid_rank_ic_mean_across_seeds",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "```",
        "",
        "## COVID vs post-COVID (test subperiods, evaluation only)",
        "",
        "See `F1_period_metrics.csv` for full detail.",
        "",
        "## Main vs next-day sensitivity",
        "",
        "```",
        main_next_cmp.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "```",
        "",
        "## F1 vs Alpha20 (test period, seed=42 main)",
        "",
        "```",
        alpha20_cmp[alpha20_cmp["period"] == "test"].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ),
        "```",
        "",
        "## Seed stability",
        "",
        "```",
        seed_stability.groupby(["variant", "spec"])[["valid_rank_ic", "test_rank_ic"]]
        .agg(["mean", "std"])
        .to_string(float_format=lambda x: f"{x:.4f}"),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in [MAIN_DATASET, NEXTDAY_DATASET, WORKFLOW_YAML, ALPHA20_PRED]:
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    write_model_configuration_md(OUT_ROOT / "F1_model_configuration.md")

    with ALPHA20_PRED.open("rb") as f:
        alpha20_pred = pickle.load(f)
    if "score" not in alpha20_pred.columns:
        alpha20_pred = alpha20_pred.rename(columns={alpha20_pred.columns[0]: "score"})

    main_res = run_variant(MAIN_DATASET, "main", alpha20_pred)
    next_res = run_variant(NEXTDAY_DATASET, "nextday", alpha20_pred)

    summary = pd.DataFrame(main_res["summary_rows"] + next_res["summary_rows"])
    summary.to_csv(OUT_ROOT / "F1_model_summary.csv", index=False)

    period_metrics = pd.DataFrame(main_res["period_metrics_rows"] + next_res["period_metrics_rows"])
    period_metrics.to_csv(OUT_ROOT / "F1_period_metrics.csv", index=False)

    daily_ic = pd.concat(main_res["daily_ic_frames"] + next_res["daily_ic_frames"], ignore_index=True)
    daily_ic.to_parquet(OUT_ROOT / "F1_daily_ic_timeseries.parquet", index=False)

    gain = pd.DataFrame(main_res["gain_rows"] + next_res["gain_rows"])
    gain.to_csv(OUT_ROOT / "F1_feature_importance_gain.csv", index=False)
    split_imp = pd.DataFrame(main_res["split_rows"] + next_res["split_rows"])
    split_imp.to_csv(OUT_ROOT / "F1_feature_importance_split.csv", index=False)

    perm = pd.concat(main_res["perm_frames"] + next_res["perm_frames"], ignore_index=True)
    perm.to_csv(OUT_ROOT / "F1_validation_permutation_importance.csv", index=False)

    seed_stability = pd.DataFrame(main_res["seed_rows"] + next_res["seed_rows"])
    seed_stability.to_csv(OUT_ROOT / "F1_seed_stability.csv", index=False)

    # Main vs nextday comparison (seed 42 valid/test metrics)
    cmp_rows = []
    pm = period_metrics[period_metrics["seed"] == PRIMARY_SEED]
    for spec in SPECS:
        for period in ["valid", "test", "covid", "post_covid"]:
            m = pm[(pm["variant"] == "main") & (pm["spec"] == spec) & (pm["period"] == period)]
            n = pm[(pm["variant"] == "nextday") & (pm["spec"] == spec) & (pm["period"] == period)]
            if m.empty or n.empty:
                continue
            m, n = m.iloc[0], n.iloc[0]
            p1 = main_res["primary_preds"].get(spec)
            p2 = next_res["primary_preds"].get(spec)
            daily_corr = np.nan
            if p1 is not None and p2 is not None:
                common = p1.reset_index().merge(p2.reset_index(), on=["datetime", "instrument"])
                if len(common) > 0:
                    daily_corr = float(common["score_x"].corr(common["score_y"]))
            cmp_rows.append(
                {
                    "spec": spec,
                    "period": period,
                    "main_ic": m["ic_mean"],
                    "nextday_ic": n["ic_mean"],
                    "ic_diff_main_minus_nextday": m["ic_mean"] - n["ic_mean"],
                    "main_rank_ic": m["rank_ic_mean"],
                    "nextday_rank_ic": n["rank_ic_mean"],
                    "rank_ic_diff": m["rank_ic_mean"] - n["rank_ic_mean"],
                    "main_icir": m["ic_ir"],
                    "nextday_icir": n["ic_ir"],
                    "main_valid_dates": m["valid_ic_dates"],
                    "nextday_valid_dates": n["valid_ic_dates"],
                    "full_sample_prediction_corr": daily_corr,
                }
            )
    main_next_cmp = pd.DataFrame(cmp_rows)
    main_next_cmp.to_csv(OUT_ROOT / "F1_main_vs_nextday_comparison.csv", index=False)

    alpha20_cmp = pd.DataFrame(main_res["alpha20_rows"] + next_res["alpha20_rows"])
    alpha20_cmp.to_csv(OUT_ROOT / "F1_vs_alpha20_comparison.csv", index=False)

    provisional = select_provisional_model(summary)

    write_report(
        OUT_ROOT / "F1_fundamental_lightgbm_report.md",
        summary,
        provisional,
        alpha20_cmp,
        main_next_cmp,
        seed_stability,
    )

    meta = {
        "experiment": "F1_fundamental_lightgbm",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_lgb_params": OFFICIAL_LGB,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "seeds": SEEDS,
        "primary_seed_for_predictions": PRIMARY_SEED,
        "label_training": "DropnaLabel + CSZScoreNorm (per datetime, qlib zscore)",
        "label_evaluation": "raw LABEL0",
        "holdout_evaluated": False,
        "provisional_model_selection": provisional,
        "specs": {k: v["features"] for k, v in SPECS.items()},
        "inputs": {
            "main": str(MAIN_DATASET),
            "nextday": str(NEXTDAY_DATASET),
            "alpha20_pred": str(ALPHA20_PRED),
        },
    }
    (OUT_ROOT / "F1_meta.json").write_text(
        json.dumps(meta, indent=2, default=json_safe), encoding="utf-8"
    )

    log.info("F1 complete. Provisional spec: %s", provisional["provisional_spec"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
