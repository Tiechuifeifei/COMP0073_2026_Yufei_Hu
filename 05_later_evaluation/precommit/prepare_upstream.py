#!/usr/bin/env python3
"""Orchestrate Final Holdout upstream data + frozen model prediction extension."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from qlib.contrib.data.handler import DataHandlerLP
from qlib.data.dataset.processor import zscore
from qlib.utils import init_instance_by_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)
_port = str(PROJECT_ROOT / "04_portfolio")
if _port not in sys.path:
    sys.path.insert(0, _port)

from config import (
    D2_PRED_PATH,
    F1C_PRED_PATH,
    HOLDOUT_END,
    HOLDOUT_START,
    OUT_ROOT,
    PANEL_PATH,
)
from extend_crsp import extend_crsp
from extend_rd13_panel import extend_rd13_panel

AUDIT_PATH = OUT_ROOT / "upstream_preparation_audit.json"
HOLDOUT_SPLIT = ("2024-01-01", HOLDOUT_END)
TRAIN = ("2008-01-01", "2017-12-31")
VALID = ("2018-01-01", "2019-12-31")
TEST = ("2020-01-01", "2023-12-31")
DELAYED_SEEDS = [42, 2026, 3407]
F1C_FEATURES = [
    "roe_z", "roa_z", "gross_profitability_z", "sales_growth_yoy_z", "asset_growth_yoy_z",
    "accruals_z", "leverage_z", "current_ratio_z", "book_to_market_z", "earnings_yield_z", "sales_to_price_z",
]
OFFICIAL_LGB = {
    "objective": "regression", "metric": "mse", "verbosity": -1,
    "colsample_bytree": 0.8879, "learning_rate": 0.2, "subsample": 0.8789,
    "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8, "num_leaves": 210, "num_threads": 20,
}
NUM_BOOST = 1000
EARLY_STOP = 50
ALPHA20 = [
    "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", "ROC60",
    "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW",
]
WORKFLOW_YAML = PROJECT_ROOT / "experiments/conf_alpha20_sp500_transfer.yaml"
PHASE5A = PROJECT_ROOT / "data/fundamental_model_dataset/main/fundamental_model_dataset.parquet"
FUND_MAIN = PROJECT_ROOT / "data/daily_fundamental_features_processed/main/daily_fundamental_features_processed.parquet"
REPRO_TOL = 1e-5


def _init_qlib() -> None:
    import qlib
    from qlib.constant import REG_US
    qlib.init(provider_uri=str(PROJECT_ROOT / "staging/qlib_data"), region=REG_US, kernels=1)


def split_between(dates: pd.Series, start: str, end: str) -> pd.Series:
    ts = pd.to_datetime(dates)
    return (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))


def assign_split(dt: pd.Series) -> pd.Series:
    out = pd.Series("other", index=dt.index, dtype=object)
    for name, (s, e) in {
        "train": TRAIN, "valid": VALID, "test": TEST, "holdout": HOLDOUT_SPLIT,
    }.items():
        out[split_between(dt, s, e)] = name
    return out


def load_r05():
    path = PROJECT_ROOT / "03_fundamentals_news/R05_build_rd_fundamental_model_datasets.py"
    spec = importlib.util.spec_from_file_location("r05", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_alpha20_through(end: str) -> pd.DataFrame:
    _init_qlib()
    with WORKFLOW_YAML.open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    handler_kwargs = conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"].copy()
    handler_kwargs["end_time"] = end
    handler_cfg = {
        "class": "DataHandlerLP",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": handler_kwargs,
    }
    handler = init_instance_by_config(handler_cfg)
    feat = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    out = feat.reset_index()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str)
    out = out[out["datetime"] <= pd.Timestamp(end)]
    missing = [c for c in ALPHA20 if c not in out.columns]
    if missing:
        raise ValueError(f"Missing Alpha20 columns: {missing}")
    return out[["datetime", "instrument"] + ALPHA20]


def extend_fundamental_dataset() -> dict:
    """Extend Phase5A dataset through HOLDOUT_END with holdout split."""
    existing = pd.read_parquet(PHASE5A)
    existing["datetime"] = pd.to_datetime(existing["datetime"]).dt.normalize()
    max_dt = existing["datetime"].max()

    # Run pipeline 12+13+14 extension for dates beyond max if needed
    if max_dt >= pd.Timestamp(HOLDOUT_END):
        return {"status": "already_extended", "max_date": str(max_dt.date())}

    # Extend daily fundamentals via frozen pipelines
    import subprocess
    py = "/opt/anaconda3/envs/qlib/bin/python"
    for script in [
        "fundamental_pipeline/12_expand_quarterly_features_to_daily.py",
        "fundamental_pipeline/13_preprocess_daily_fundamental_features.py",
    ]:
        subprocess.run([py, str(PROJECT_ROOT / script)], check=True, cwd=str(PROJECT_ROOT))

    # Patch pipeline 14 constants temporarily via env override script
    subprocess.run([py, str(Path(__file__).parent / "extend_fundamental_model_dataset.py")], check=True)

    updated = pd.read_parquet(PHASE5A)
    return {
        "max_date": str(updated["datetime"].max().date()),
        "holdout_rows": int((updated["split"] == "holdout").sum()),
        "rows_2025": int((updated["datetime"] >= "2025-01-01").sum()),
    }


def build_holdout_panel() -> tuple[pd.DataFrame, dict]:
    r05 = load_r05()
    phase5a = pd.read_parquet(PHASE5A)
    phase5a["datetime"] = pd.to_datetime(phase5a["datetime"]).dt.normalize()
    phase5a["instrument"] = phase5a["instrument"].astype(str)
    phase5a = phase5a[phase5a["label"].notna()].copy()
    alpha20 = load_alpha20_through(HOLDOUT_END)
    rd_v2 = pd.read_parquet(
        PROJECT_ROOT / "data/fundamental_experiments/RD13_provenance/RD13_v2_CORRECTED/rd13_v2_panel.parquet"
    )
    rd_v2["datetime"] = pd.to_datetime(rd_v2["datetime"]).dt.normalize()
    rd_v2["instrument"] = rd_v2["instrument"].astype(str)
    rd_v2 = rd_v2.set_index(["datetime", "instrument"]).sort_index()

    keys = r05.build_common_keys(phase5a, alpha20, rd_v2)
    panel = r05.assemble_dataset(keys, phase5a, alpha20, rd_v2, include_fundamentals=True, rd_prefix="rd13")
    # Re-assign split including holdout
    panel["split"] = assign_split(panel["datetime"]).values
    panel.to_parquet(PANEL_PATH, index=False)
    audit = {
        "rows": int(len(panel)),
        "max_date": str(panel["datetime"].max().date()),
        "holdout_rows": int((panel["split"] == "holdout").sum()),
        "split_counts": panel.groupby("split").size().to_dict(),
    }
    return panel, audit


def process_train_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["label"]).copy()
    out["label_processed"] = out.groupby("datetime", group_keys=False)["label"].apply(zscore)
    return out


def train_lgb(train_df, valid_df, features, seed: int) -> lgb.Booster:
    p = OFFICIAL_LGB.copy()
    p.update({"seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed, "data_random_seed": seed})
    tr = lgb.Dataset(train_df[features], label=train_df["label_processed"], free_raw_data=False)
    va = lgb.Dataset(valid_df[features], label=valid_df["label_processed"], reference=tr, free_raw_data=False)
    return lgb.train(
        p, tr, num_boost_round=NUM_BOOST, valid_sets=[va], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(period=0)],
    )


def compare_predictions(new: pd.DataFrame, ref_path: Path, period: tuple[str, str]) -> dict:
    ref = pd.read_parquet(ref_path) if ref_path.suffix == ".parquet" else pd.read_pickle(ref_path).reset_index()
    for df in (new, ref):
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df["instrument"] = df["instrument"].astype(str)
    s, e = period
    a = new[(new["datetime"] >= s) & (new["datetime"] <= e)]
    b = ref[(ref["datetime"] >= s) & (ref["datetime"] <= e)]
    m = a.merge(b, on=["datetime", "instrument"], suffixes=("_new", "_ref"))
    diff = (m["score_new"] - m["score_ref"]).abs()
    return {
        "n": int(len(m)),
        "max_abs_diff": float(diff.max()) if len(diff) else np.nan,
        "mean_abs_diff": float(diff.mean()) if len(diff) else np.nan,
        "passed": bool(len(diff) and diff.max() <= REPRO_TOL),
    }


def reconstruct_d2(panel: pd.DataFrame) -> dict:
    rd_cols = sorted([c for c in panel.columns if c.startswith("rd13_")])
    a20_cols = [f"alpha20_{c.lower()}" for c in ALPHA20]
    feats = a20_cols + rd_cols
    train = process_train_label(panel[panel["split"] == "train"])
    valid = process_train_label(panel[panel["split"] == "valid"])
    seed_preds = []
    model_audit = []
    for seed in DELAYED_SEEDS:
        model = train_lgb(train, valid, feats, seed)
        sub = panel.dropna(subset=feats + ["label"]).copy()
        sub["score"] = model.predict(sub[feats], num_iteration=model.best_iteration)
        pred = sub[["datetime", "instrument", "score"]]
        seed_preds.append(pred)
        model_audit.append({"seed": seed, "best_iteration": int(model.best_iteration)})
    ens = pd.concat(seed_preds).groupby(["datetime", "instrument"], as_index=False)["score"].mean()
    repro = compare_predictions(ens, D2_PRED_PATH, TEST)
    if not repro["passed"]:
        raise RuntimeError(f"D2 2020-2023 reproduction failed: {repro}")
    ens.to_parquet(D2_PRED_PATH, index=False)
    return {"method": "deterministic reconstruction of the frozen training procedure", "reproduction": repro, "models": model_audit}


def reconstruct_f1c(panel: pd.DataFrame | None = None) -> dict:
    f1_path = PROJECT_ROOT / "03_fundamentals_news/F1_fundamental_lightgbm.py"
    spec = importlib.util.spec_from_file_location("f1", f1_path)
    f1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f1)
    df = pd.read_parquet(PHASE5A)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df["instrument"] = df["instrument"].astype(str)
    feats = F1C_FEATURES
    train = f1.process_training_label(df[df["split"] == "train"])
    valid = f1.process_training_label(df[df["split"] == "valid"])
    seed_preds = []
    for seed in DELAYED_SEEDS:
        model = f1.train_model(train, valid, feats, seed)
        pred = f1.predict_frame(model, df, feats)
        seed_preds.append(pred.reset_index())
    ens = pd.concat(seed_preds).groupby(["datetime", "instrument"], as_index=False)["score"].mean()
    ens_pkl = ens.set_index(["datetime", "instrument"]).sort_index()[["score"]]
    repro = compare_predictions(ens.reset_index(), F1C_PRED_PATH, TEST)
    if not repro["passed"]:
        raise RuntimeError(f"F1C 2020-2023 reproduction failed: {repro}")
    with F1C_PRED_PATH.open("wb") as f:
        pickle.dump(ens_pkl, f)
    return {"method": "deterministic reconstruction of the frozen training procedure", "reproduction": repro}


def extend_spy_features() -> dict:
    import importlib.util
    try:
        from build_spy_features import build_spy_features  # optional helper
    except ImportError:
        build_spy_features = None
    spec = importlib.util.spec_from_file_location(
        "hmm_config", PROJECT_ROOT / "04_portfolio/config.py"
    )
    hmm_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hmm_cfg)
    SPY_CSV_PATH, MAIN_HMM_ROOT = hmm_cfg.SPY_CSV_PATH, hmm_cfg.OUT_ROOT

    spy = pd.read_csv(SPY_CSV_PATH)
    spy.columns = [c.lower() for c in spy.columns]
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")
    spy = spy[spy["date"] <= pd.Timestamp(HOLDOUT_END)].copy()
    spy["spy_ret"] = spy["close"].pct_change()
    spy["spy_vol20"] = spy["spy_ret"].rolling(20, min_periods=20).std()
    spy["spy_vol60"] = spy["spy_ret"].rolling(60, min_periods=60).std()
    spy["spy_vol_ratio_20_60"] = spy["spy_vol20"] / spy["spy_vol60"].replace(0, np.nan)
    spy = spy.dropna(subset=["spy_ret", "spy_vol20"]).reset_index(drop=True)
    out = MAIN_HMM_ROOT / "spy_features_through_holdout.parquet"
    spy.to_parquet(out, index=False)
    r2_out = OUT_ROOT / "spy_features_r2_through_holdout.parquet"
    spy[["date", "spy_ret", "spy_vol20", "spy_vol60", "spy_vol_ratio_20_60"]].to_parquet(r2_out, index=False)
    return {"max_date": str(spy["date"].max().date()), "rows": int(len(spy))}


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit: dict = {"started_at": datetime.now(timezone.utc).isoformat(), "steps": {}}
    try:
        _, crsp_a = extend_crsp(HOLDOUT_END)
        audit["steps"]["crsp"] = crsp_a

        _, rd_a = extend_rd13_panel(HOLDOUT_END)
        audit["steps"]["rd13"] = rd_a

        audit["steps"]["fundamental_dataset"] = extend_fundamental_dataset()

        panel, panel_a = build_holdout_panel()
        audit["steps"]["panel"] = panel_a

        audit["steps"]["d2"] = reconstruct_d2(panel)
        audit["steps"]["f1c"] = reconstruct_f1c()
        audit["steps"]["spy"] = extend_spy_features()

        audit["status"] = "SUCCESS"
        audit["completed_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        audit["status"] = "FAILED"
        audit["error"] = str(exc)
        audit["traceback"] = traceback.format_exc()
        AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        raise

    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
