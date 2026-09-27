#!/usr/bin/env python3
"""14_build_fundamental_model_dataset.py

Build fundamental modelling datasets with Alpha20 LABEL0 on the same
(datetime, instrument) keys as the frozen baseline.

Run with the qlib env, e.g.:
  /opt/anaconda3/envs/qlib/bin/python fundamental_pipeline/14_build_fundamental_model_dataset.py"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

WORKFLOW_YAML = PROJECT_ROOT / "experiments" / "conf_alpha20_sp500_transfer.yaml"
QLIB_DATA_DIR = PROJECT_ROOT / "staging" / "qlib_data"
OFFICIAL_LABEL_PKL = (
    PROJECT_ROOT
    / "mlruns"
    / "561707949424138210"
    / "11ee753279e548a99601c34ef6ca195f"
    / "artifacts"
    / "label.pkl"
)
TRANSFER_REPORT = PROJECT_ROOT / "staging" / "alpha20_transfer_report.json"

MAIN_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "daily_fundamental_features_processed"
    / "main"
    / "daily_fundamental_features_processed.parquet"
)
NEXTDAY_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "daily_fundamental_features_processed"
    / "nextday"
    / "daily_fundamental_features_processed.parquet"
)
OUT_ROOT = PROJECT_ROOT / "data" / "fundamental_model_dataset"

LABEL_EXPR = "Ref($close, -2)/Ref($close, -1) - 1"
LABEL_COL = "LABEL0"
LABEL_TOLERANCE = 1e-5

FEATURES = [
    "roe",
    "roa",
    "gross_profitability",
    "operating_profitability",
    "sales_growth_yoy",
    "asset_growth_yoy",
    "accruals",
    "leverage",
    "current_ratio",
    "book_to_market",
    "earnings_yield",
    "sales_to_price",
]

# Extended modelling splits (Phase 5A). Official Alpha20 baseline only defines train/valid/test.
SPLIT_RANGES: dict[str, tuple[str, str]] = {
    "history": ("2006-01-01", "2007-12-31"),
    "train": ("2008-01-01", "2017-12-31"),
    "valid": ("2018-01-01", "2019-12-31"),
    "test": ("2020-01-01", "2023-12-31"),
    "holdout": ("2024-01-01", "2024-12-31"),
}

LABEL_START = "2006-01-01"
LABEL_END = "2024-12-31"
HANDLER_START = "2008-01-01"
HANDLER_END = "2023-12-31"

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
    if isinstance(obj, datetime):
        return obj.isoformat()
    import datetime as dt

    if isinstance(obj, dt.date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def load_handler_config() -> dict[str, Any]:
    with WORKFLOW_YAML.open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    return conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]


def init_qlib() -> None:
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_US, kernels=1)


def normalize_mi_series(s: pd.Series) -> pd.Series:
    out = s.copy()
    if out.index.names[0] == "instrument":
        out = out.reorder_levels(["datetime", "instrument"])
    out.index = out.index.set_names(["datetime", "instrument"])
    return out.sort_index()


def build_label_table() -> pd.DataFrame:
    from qlib.data import D

    log.info("Building LABEL0 via Qlib D.features (%s .. %s)", LABEL_START, LABEL_END)
    instruments = D.list_instruments(
        D.instruments("sp500"), start_time=LABEL_START, end_time=LABEL_END, as_list=True
    )
    raw = D.features(instruments, [LABEL_EXPR], LABEL_START, LABEL_END, freq="day")
    raw.columns = [LABEL_COL]
    s = normalize_mi_series(raw.iloc[:, 0])
    df = s.reset_index()
    df = df.rename(columns={LABEL_COL: "label"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def fetch_handler_raw_label() -> pd.Series:
    from qlib.contrib.data.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config

    handler_cfg = {
        "class": "DataHandlerLP",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": load_handler_config(),
    }
    handler = init_instance_by_config(handler_cfg)
    raw = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)
    return normalize_mi_series(raw.iloc[:, 0])


def load_official_label_pkl() -> pd.Series | None:
    if not OFFICIAL_LABEL_PKL.exists():
        log.warning("Official label.pkl not found: %s", OFFICIAL_LABEL_PKL)
        return None
    with OFFICIAL_LABEL_PKL.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        s = obj.iloc[:, 0]
    else:
        s = obj
    return normalize_mi_series(s)


def compare_label_series(
    reconstructed: pd.DataFrame,
    reference: pd.Series,
    reference_name: str,
) -> dict[str, Any]:
    recon = reconstructed.set_index(["datetime", "instrument"])["label"].sort_index()
    ref = reference.rename("ref")
    common = pd.concat([recon.rename("recon"), ref], axis=1, join="inner").dropna()
    if common.empty:
        return {
            "reference": reference_name,
            "common_observations": 0,
            "pearson_correlation": None,
            "max_abs_difference": None,
            "mean_abs_difference": None,
            "pct_equal_within_tolerance": None,
            "passed": False,
            "error": "no common observations",
        }
    diff = (common["recon"] - common["ref"]).abs()
    tol = LABEL_TOLERANCE
    return {
        "reference": reference_name,
        "common_observations": int(len(common)),
        "pearson_correlation": float(common["recon"].corr(common["ref"])),
        "max_abs_difference": float(diff.max()),
        "mean_abs_difference": float(diff.mean()),
        "pct_equal_within_tolerance": float((diff <= tol).mean()),
        "passed": bool(diff.max() <= tol),
    }


def verify_label_reproduction(label_df: pd.DataFrame) -> tuple[dict[str, Any], bool]:
    handler_label = fetch_handler_raw_label()
    handler_cmp = compare_label_series(label_df, handler_label, "DataHandlerLP_DK_R")

    official = load_official_label_pkl()
    official_cmp: dict[str, Any] | None = None
    if official is not None:
        official_cmp = compare_label_series(label_df, official, "mlruns_label_pkl")

    passed = handler_cmp["passed"] and (
        official_cmp is None or official_cmp.get("passed", False)
    )
    audit = {
        "label_expression": LABEL_EXPR,
        "label_tolerance": LABEL_TOLERANCE,
        "handler_comparison": handler_cmp,
        "official_pkl_comparison": official_cmp,
        "official_label_pkl_path": str(OFFICIAL_LABEL_PKL),
        "passed": passed,
    }
    return audit, passed


def assign_split(dt: pd.Timestamp) -> str:
    ts = pd.Timestamp(dt).normalize()
    for name, (start, end) in SPLIT_RANGES.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return name
    return "outside_sample"


def feature_columns() -> dict[str, list[str]]:
    raw = [f"{f}_raw" for f in FEATURES]
    win = [f"{f}_win" for f in FEATURES]
    z = [f"{f}_z" for f in FEATURES]
    miss = [f"{f}_missing_flag" for f in FEATURES]
    return {"raw": raw, "win": win, "z": z, "missing_flag": miss}


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns={"date": "datetime", "qlib_instrument": "instrument"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def join_features_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = features.merge(labels, on=["datetime", "instrument"], how="left", validate="one_to_one")
    merged["split"] = merged["datetime"].map(assign_split)
    return merged


def label_profile_by_year(label_df: pd.DataFrame) -> pd.DataFrame:
    sub = label_df.dropna(subset=["label"]).copy()
    sub["year"] = sub["datetime"].dt.year
    rows = []
    for year, g in sub.groupby("year"):
        s = g["label"]
        rows.append(
            {
                "year": int(year),
                "non_missing_labels": int(s.notna().sum()),
                "unique_instruments": int(g["instrument"].nunique()),
                "unique_dates": int(g["datetime"].nunique()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=0)),
                "p1": float(s.quantile(0.01)),
                "p99": float(s.quantile(0.99)),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def join_audit(features: pd.DataFrame, labels: pd.DataFrame, merged: pd.DataFrame) -> dict[str, Any]:
    feat_keys = features[["datetime", "instrument"]].drop_duplicates()
    lab_keys = labels[["datetime", "instrument"]].drop_duplicates()
    matched = merged[merged["label"].notna()][["datetime", "instrument"]].drop_duplicates()
    feat_only = feat_keys.merge(lab_keys, on=["datetime", "instrument"], how="left", indicator=True)
    lab_only = lab_keys.merge(feat_keys, on=["datetime", "instrument"], how="left", indicator=True)
    dup_dt_inst_feat = int(features.duplicated(["datetime", "instrument"]).sum())
    dup_dt_inst_lab = int(labels.duplicated(["datetime", "instrument"]).sum())
    dup_dt_permno = int(features.duplicated(["datetime", "permno"]).sum()) if "permno" in features else None
    return {
        "processed_feature_rows_before_join": int(len(features)),
        "label_rows_before_join": int(len(labels)),
        "matched_rows_with_label": int(len(matched)),
        "feature_rows_without_label": int((feat_only["_merge"] == "left_only").sum()),
        "label_rows_without_features": int((lab_only["_merge"] == "left_only").sum()),
        "duplicate_datetime_instrument_features": dup_dt_inst_feat,
        "duplicate_datetime_instrument_labels": dup_dt_inst_lab,
        "duplicate_datetime_permno_features": dup_dt_permno,
        "merged_rows": int(len(merged)),
    }


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    z_cols = feature_columns()["z"]
    rows = []
    for split_name in list(SPLIT_RANGES) + ["outside_sample"]:
        sub = df[df["split"] == split_name]
        if sub.empty:
            rows.append(
                {
                    "split": split_name,
                    "rows": 0,
                    "unique_dates": 0,
                    "unique_instruments": 0,
                    "label_coverage": 0.0,
                    **{f"{c}_coverage": 0.0 for c in z_cols},
                }
            )
            continue
        row: dict[str, Any] = {
            "split": split_name,
            "rows": int(len(sub)),
            "unique_dates": int(sub["datetime"].nunique()),
            "unique_instruments": int(sub["instrument"].nunique()),
            "label_coverage": float(sub["label"].notna().mean()),
        }
        for c in z_cols:
            row[f"{c}_coverage"] = float(sub[c].notna().mean()) if c in sub.columns else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def temporal_checks(df: pd.DataFrame) -> dict[str, Any]:
    before_signal = int((df["datetime"] < df["signal_start_date"]).sum()) if "signal_start_date" in df else None
    split_sets = {}
    for name in SPLIT_RANGES:
        split_sets[name] = set(df.loc[df["split"] == name, "datetime"].dt.normalize().unique())
    overlaps = {}
    names = list(SPLIT_RANGES)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlaps[f"{a}_x_{b}"] = int(len(split_sets[a] & split_sets[b]))
    return {
        "label_horizon_note": (
            "At feature date t, label = close(t+2)/close(t+1)-1 using Qlib $close; "
            "features on t use accounting signals with signal_start_date <= t."
        ),
        "date_before_signal_start_date_count": before_signal,
        "split_date_overlaps": overlaps,
        "temporal_integrity_passed": before_signal == 0 and all(v == 0 for v in overlaps.values()),
    }


def compare_variants(main_df: pd.DataFrame, next_df: pd.DataFrame) -> dict[str, Any]:
    key_cols = ["datetime", "instrument"]
    main_keys = main_df[key_cols].drop_duplicates()
    next_keys = next_df[key_cols].drop_duplicates()
    common = main_keys.merge(next_keys, on=key_cols, how="inner")
    main_only = len(main_keys) - len(common)
    next_only = len(next_keys) - len(common)

    m = main_df.set_index(key_cols)
    n = next_df.set_index(key_cols)
    idx = common.set_index(key_cols).index
    lm = m.loc[idx, "label"]
    ln = n.loc[idx, "label"]
    label_diff = (lm - ln).abs()
    label_equal = float((label_diff <= LABEL_TOLERANCE).mean()) if len(label_diff) else float("nan")

    z_cols = feature_columns()["z"]
    feat_diff_counts = {}
    for c in z_cols:
        if c not in m.columns or c not in n.columns:
            continue
        a = pd.to_numeric(m.loc[idx, c], errors="coerce")
        b = pd.to_numeric(n.loc[idx, c], errors="coerce")
        both = a.notna() & b.notna()
        if both.any():
            feat_diff_counts[c] = int((both & ~np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=False)).sum())
        else:
            feat_diff_counts[c] = 0

    split_counts_main = main_df["split"].value_counts().to_dict()
    split_counts_next = next_df["split"].value_counts().to_dict()

    return {
        "main_rows": int(len(main_df)),
        "nextday_rows": int(len(next_df)),
        "common_keys": int(len(common)),
        "keys_unique_to_main": int(main_only),
        "keys_unique_to_nextday": int(next_only),
        "label_equal_on_common_pct": label_equal,
        "label_max_abs_diff_on_common": float(label_diff.max()) if len(label_diff) else None,
        "z_feature_diff_counts_on_common": feat_diff_counts,
        "split_row_counts_main": {str(k): int(v) for k, v in split_counts_main.items()},
        "split_row_counts_nextday": {str(k): int(v) for k, v in split_counts_next.items()},
    }


def write_official_label_definition_md(handler_cfg: dict[str, Any]) -> None:
    segments = yaml.safe_load(WORKFLOW_YAML.read_text(encoding="utf-8"))["task"]["dataset"]["kwargs"]["segments"]
    lines = [
        "# Official Alpha20 Label Definition (Phase 5A evidence)",
        "",
        "This document records the **frozen official Alpha20 baseline** label and split "
        "configuration used for fundamental model dataset alignment.",
        "",
        "## Primary configuration source",
        "",
        f"- **Workflow YAML:** `{WORKFLOW_YAML}`",
        f"- **Transfer report:** `{TRANSFER_REPORT}`",
        f"- **Official cached label (test segment):** `{OFFICIAL_LABEL_PKL}`",
        f"- **Qlib provider URI:** `{QLIB_DATA_DIR}` (see `qlib_init.provider_uri` in workflow YAML)",
        "",
        "## LABEL0 expression",
        "",
        "From `experiments/conf_alpha20_sp500_transfer.yaml` (Alpha158DL label block):",
        "",
        "```yaml",
        "label:",
        '  - ["Ref($close, -2)/Ref($close, -1) - 1"]',
        '  - ["LABEL0"]',
        "```",
        "",
        "**Exact expression:** `Ref($close, -2)/Ref($close, -1) - 1`",
        "",
        "## Economic interpretation",
        "",
        "At trading date **t**, the raw label equals **close(t+2) / close(t+1) − 1**, i.e. a "
        "one-day forward return from the next session close to the session-after-next close "
        "(Qlib Alpha158 convention).",
        "",
        "## Label frequency and horizon",
        "",
        "- **Frequency:** daily (`freq=\"day\"`)",
        "- **Prediction horizon:** 2 trading days ahead (return between t+1 and t+2 closes, indexed at t)",
        "",
        "## Price field and return definition",
        "",
        "- **Price field:** Qlib `$close` from `staging/qlib_data` (not CRSP `daily_ret`)",
        "- **Return type:** close-to-close ratio across future sessions (not open-to-close)",
        "",
        "## Label processing in the official training pipeline",
        "",
        "From `learn_processors` in the same YAML:",
        "",
        "```yaml",
        "learn_processors:",
        "  - class: DropnaLabel",
        "  - class: CSZScoreNorm",
        "    kwargs:",
        "      fields_group: label",
        "```",
        "",
        "Phase 5A stores the **raw LABEL0** (before `CSZScoreNorm`). Reproduction checks use "
        "`DataHandlerLP.DK_R` and `mlruns/.../label.pkl`, which contain the raw label.",
        "",
        "## Instrument naming and universe",
        "",
        "- **Universe:** `sp500` (`market: sp500` in workflow YAML)",
        "- **Instrument format:** `P{PERMNO}` (e.g. `P10104`) from `staging/qlib_data/instruments/sp500.txt`",
        "",
        "## Calendar",
        "",
        "- **Trading calendar:** US equity sessions from Qlib `staging/qlib_data/calendars/day.txt`",
        "- **Modelling key:** `(datetime, instrument)` MultiIndex convention",
        "",
        "## Official baseline train / validation / test periods",
        "",
        "From `task.dataset.kwargs.segments` in `conf_alpha20_sp500_transfer.yaml`:",
        "",
        f"- **train:** {segments['train'][0]} .. {segments['train'][1]}",
        f"- **valid:** {segments['valid'][0]} .. {segments['valid'][1]}",
        f"- **test:** {segments['test'][0]} .. {segments['test'][1]}",
        "",
        "Handler `start_time` / `end_time`: "
        f"{handler_cfg['start_time']} .. {handler_cfg['end_time']} (no official holdout segment).",
        "",
        "## Phase 5A extended splits (not in official Alpha20 baseline)",
        "",
        "The fundamental pipeline additionally defines:",
        "",
        f"- **history (lookback):** {SPLIT_RANGES['history'][0]} .. {SPLIT_RANGES['history'][1]}",
        f"- **holdout (2024):** {SPLIT_RANGES['holdout'][0]} .. {SPLIT_RANGES['holdout'][1]}",
        "",
        "**Discrepancy note:** The official Alpha20 baseline does **not** include a 2024 holdout or "
        "2006–2007 history segment. These splits are Phase 5A extensions for fundamental modelling "
        "only. The 2024 holdout must not be used for feature selection, hyperparameter tuning, "
        "winsorisation, early stopping, or model selection.",
        "",
        "## Reproduction method",
        "",
        "Labels are computed independently via:",
        "",
        "```python",
        'D.features(instruments, ["Ref($close, -2)/Ref($close, -1) - 1"], start, end, freq="day")',
        "```",
        "",
        "Verified against `DataHandlerLP.fetch(col_set=\"label\", data_key=DK_R)` and the frozen "
        "`label.pkl` artifact from recorder `11ee753279e548a99601c34ef6ca195f`.",
        "",
    ]
    (OUT_ROOT / "official_label_definition.md").write_text("\n".join(lines), encoding="utf-8")


def save_variant_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "fundamental_model_dataset.parquet", index=False)

    pkl_df = df.set_index(["datetime", "instrument"]).sort_index()
    with (out_dir / "fundamental_model_dataset.pkl").open("wb") as f:
        pickle.dump(pkl_df, f)

    for split_name in ["train", "valid", "test"]:
        sub = df[df["split"] == split_name]
        sub.to_parquet(out_dir / f"{split_name}.parquet", index=False)

    holdout = df[df["split"] == "holdout"]
    holdout.to_parquet(out_dir / "holdout_2024.parquet", index=False)


def write_quality_report(
    path: Path,
    label_df: pd.DataFrame,
    label_audit: dict[str, Any],
    join_info: dict[str, Any],
    temporal: dict[str, Any],
    split_df: pd.DataFrame,
    variant: str,
) -> None:
    sub = label_df["label"].dropna()
    lines = [
        f"# Fundamental Model Dataset Quality Report ({variant})",
        "",
        "## 1. Label verification",
        "",
        f"- **Expression:** `{LABEL_EXPR}`",
        f"- **Source configuration:** `{WORKFLOW_YAML}`",
        f"- **Label date range:** {label_df['datetime'].min().date()} .. {label_df['datetime'].max().date()}",
        f"- **Non-missing labels:** {int(sub.notna().sum()):,}",
        f"- **Mean / median / std:** {sub.mean():.6f} / {sub.median():.6f} / {sub.std(ddof=0):.6f}",
        f"- **p1 / p99:** {sub.quantile(0.01):.6f} / {sub.quantile(0.99):.6f}",
        "",
        "### Reproduction audit",
        "",
        f"- **Handler comparison passed:** {label_audit['handler_comparison']['passed']}",
        f"- **Common observations (handler):** {label_audit['handler_comparison']['common_observations']:,}",
        f"- **Max abs diff (handler):** {label_audit['handler_comparison']['max_abs_difference']}",
        f"- **Pearson (handler):** {label_audit['handler_comparison']['pearson_correlation']}",
        "",
        "## 2. Join integrity",
        "",
    ]
    for k, v in join_info.items():
        lines.append(f"- **{k}:** {v:,}" if isinstance(v, int) else f"- **{k}:** {v}")
    lines.extend(
        [
            "",
            "## 3. Temporal integrity",
            "",
            f"- **Label horizon:** {temporal['label_horizon_note']}",
            f"- **date < signal_start_date count:** {temporal['date_before_signal_start_date_count']} "
            f"(must be 0)",
            f"- **Split date overlaps:** {temporal['split_date_overlaps']}",
            f"- **Temporal integrity passed:** {temporal['temporal_integrity_passed']}",
            "",
            "## 4. Split sizes",
            "",
            split_df.to_string(index=False),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_main_vs_nextday_md(stats: dict[str, Any]) -> None:
    lines = [
        "# Main vs Next-Day Fundamental Model Dataset Comparison",
        "",
        "## Split row counts",
        "",
        f"- **Main:** {stats['split_row_counts_main']}",
        f"- **Next-day:** {stats['split_row_counts_nextday']}",
        "",
        "## Key overlap",
        "",
        f"- **Common (datetime, instrument) keys:** {stats['common_keys']:,}",
        f"- **Unique to main:** {stats['keys_unique_to_main']:,}",
        f"- **Unique to next-day:** {stats['keys_unique_to_nextday']:,}",
        "",
        "## Label equality on common keys",
        "",
        f"- **Equal within tolerance pct:** {stats['label_equal_on_common_pct']:.6f}",
        f"- **Max abs diff:** {stats['label_max_abs_diff_on_common']}",
        "",
        "## Z-feature differences on common keys (timing-induced)",
        "",
    ]
    for k, v in stats["z_feature_diff_counts_on_common"].items():
        lines.append(f"- **{k}:** {v:,} differing rows")
    (OUT_ROOT / "main_vs_nextday_model_dataset_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def process_variant(
    features_path: Path,
    variant: str,
    label_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
    log.info("Processing variant %s", variant)
    features = load_features(features_path)
    merged = join_features_labels(features, label_df)
    join_info = join_audit(features, label_df, merged)
    temporal = temporal_checks(merged)
    splits = split_summary(merged)
    return merged, join_info, temporal, splits


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in [MAIN_FEATURES, NEXTDAY_FEATURES, WORKFLOW_YAML]:
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    handler_cfg = load_handler_config()
    write_official_label_definition_md(handler_cfg)

    init_qlib()
    label_df = build_label_table()

    label_stats = label_df["label"].dropna()
    global_label_profile = {
        "total_non_missing": int(label_stats.notna().sum()),
        "mean": float(label_stats.mean()),
        "median": float(label_stats.median()),
        "std": float(label_stats.std(ddof=0)),
        "p1": float(label_stats.quantile(0.01)),
        "p99": float(label_stats.quantile(0.99)),
    }
    profile_year = label_profile_by_year(label_df)
    profile_year.to_csv(OUT_ROOT / "label_profile_by_year.csv", index=False)

    label_audit, passed = verify_label_reproduction(label_df)
    pd.DataFrame(
        [
            label_audit["handler_comparison"],
            label_audit.get("official_pkl_comparison") or {},
        ]
    ).to_csv(OUT_ROOT / "official_label_reproduction_audit.csv", index=False)

    if not passed:
        fail_report = OUT_ROOT / "model_dataset_quality_report.md"
        fail_report.write_text(
            "# Phase 5A ABORTED: Label reproduction failed\n\n"
            + json.dumps(label_audit, indent=2, default=json_safe),
            encoding="utf-8",
        )
        log.error("Label reproduction failed — see %s", fail_report)
        return 1

    main_df, main_join, main_temporal, main_splits = process_variant(MAIN_FEATURES, "main", label_df)
    next_df, next_join, next_temporal, next_splits = process_variant(
        NEXTDAY_FEATURES, "nextday", label_df
    )

    save_variant_outputs(main_df, OUT_ROOT / "main")
    save_variant_outputs(next_df, OUT_ROOT / "nextday")

    join_audit_df = pd.DataFrame(
        [
            {"variant": "main", **main_join},
            {"variant": "nextday", **next_join},
        ]
    )
    join_audit_df.to_csv(OUT_ROOT / "feature_label_join_audit.csv", index=False)

    split_summary_df = pd.concat(
        [main_splits.assign(variant="main"), next_splits.assign(variant="nextday")],
        ignore_index=True,
    )
    split_summary_df.to_csv(OUT_ROOT / "model_split_summary.csv", index=False)

    compare_stats = compare_variants(main_df, next_df)
    write_main_vs_nextday_md(compare_stats)

    write_quality_report(
        OUT_ROOT / "model_dataset_quality_report.md",
        label_df,
        label_audit,
        main_join,
        main_temporal,
        main_splits,
        "main (primary); nextday join counts in feature_label_join_audit.csv",
    )

    meta = {
        "phase": "5A",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_expression": LABEL_EXPR,
        "label_source": str(WORKFLOW_YAML),
        "official_label_pkl": str(OFFICIAL_LABEL_PKL),
        "label_reproduction_audit": label_audit,
        "global_label_profile": global_label_profile,
        "split_ranges": SPLIT_RANGES,
        "official_baseline_segments": yaml.safe_load(WORKFLOW_YAML.read_text(encoding="utf-8"))[
            "task"
        ]["dataset"]["kwargs"]["segments"],
        "extended_splits_note": (
            "history and holdout_2024 are Phase 5A extensions; not present in official Alpha20 baseline"
        ),
        "inputs": {
            "main_features": str(MAIN_FEATURES),
            "nextday_features": str(NEXTDAY_FEATURES),
        },
        "main_rows": int(len(main_df)),
        "nextday_rows": int(len(next_df)),
        "main_vs_nextday": compare_stats,
        "join_audit": {"main": main_join, "nextday": next_join},
        "temporal_integrity": {"main": main_temporal, "nextday": next_temporal},
    }
    (OUT_ROOT / "model_dataset_meta.json").write_text(
        json.dumps(meta, indent=2, default=json_safe), encoding="utf-8"
    )

    log.info(
        "Phase 5A complete: main %d rows; label reproduction passed (handler max diff %s)",
        len(main_df),
        label_audit["handler_comparison"]["max_abs_difference"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
