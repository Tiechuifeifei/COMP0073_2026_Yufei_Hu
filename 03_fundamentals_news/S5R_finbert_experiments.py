#!/usr/bin/env python3
"""S5R FinBERT experiments: D2/D3/D4 (delayed) and E2 (ex-ante immediate).

Uses validated full-history FinBERT scores and the corrected ex-ante
event-time-v2 pipeline. Sample ends 2023."""

from __future__ import annotations

import importlib.util
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/finbert_experiments"
PANEL_ROOT = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/finbert_daily_panel"
FINBERT_SCORES = (
    PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/finbert_full_history/S5R_finbert_scores_full_history.parquet"
)
ARTICLES_PATH = PROJECT_ROOT / "data/sentiment_raw/alpha_vantage_full/normalized/articles.parquet"
WORKFLOW_YAML = PROJECT_ROOT / "experiments/conf_alpha20_sp500_transfer.yaml"
QLIB_DATA = PROJECT_ROOT / "staging/qlib_data"
LABEL_EXPR = "Ref($close, -2)/Ref($close, -1) - 1"

POSITIVE_FB = {"positive"}
NEGATIVE_FB = {"negative"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("S5R-FINBERT-EXP")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def weighted_finbert(g: pd.DataFrame) -> float:
    rel = g["relevance_score_num"]
    score = g["signed_score"]
    denom = rel.sum()
    if denom <= 0:
        return np.nan
    return float((score * rel).sum() / denom)


def aggregate_daily_finbert(articles: pd.DataFrame) -> pd.DataFrame:
    valid = articles.dropna(subset=["eligible_signal_date", "signed_score"]).copy()
    valid = valid[valid["relevance_score_num"] > 0]
    rows = []
    for (permno, sig_date), g in valid.groupby(["permno", "eligible_signal_date"]):
        n = len(g)
        labels = g["predicted_label"].astype(str)
        pos = labels.isin(POSITIVE_FB).sum()
        neg = labels.isin(NEGATIVE_FB).sum()
        scores = g["signed_score"]
        rows.append(
            {
                "permno": int(permno),
                "date": pd.Timestamp(sig_date),
                "news_count": n,
                "log_news_count": float(np.log1p(n)),
                "ticker_mean_sentiment": float(scores.mean()),
                "relevance_weighted_sentiment": weighted_finbert(g),
                "positive_news_share": float(pos / n),
                "negative_news_share": float(neg / n),
                "net_positive_share": float((pos - neg) / n),
                "sentiment_dispersion": float(scores.std(ddof=0)) if n >= 2 else np.nan,
                "relevance_mean": float(g["relevance_score_num"].mean()),
                "no_news_flag": 0,
            }
        )
    return pd.DataFrame(rows)


def build_finbert_daily_panel(force: bool = False) -> Path:
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = PANEL_ROOT / "S5R_finbert_daily_sentiment_raw.parquet"
    if out_path.exists() and not force:
        return out_path

    s3 = load_module("s3", PROJECT_ROOT / "01_data/sentiment_experiments/S3_daily_sentiment_panel.py")
    articles = pd.read_parquet(ARTICLES_PATH)
    articles = articles[
        (articles["publication_date"] >= str(s3.PERIOD_START))
        & (articles["publication_date"] <= str(s3.PERIOD_END))
    ]
    deduped, stats = s3.dedupe_articles(articles)
    scores = pd.read_parquet(FINBERT_SCORES)
    scores = scores[scores["inference_status"] == "ok"][
        ["permno", "canonical_url_hash", "signed_score", "predicted_label"]
    ]
    deduped = deduped.merge(scores, on=["permno", "canonical_url_hash"], how="left")
    deduped["relevance_score_num"] = pd.to_numeric(deduped["relevance_score"], errors="coerce")

    membership = s3.load_membership()
    permnos = set(membership["permno"].astype(int))
    trading = s3.load_trading_days(permnos)
    mapped = s3.map_signal_dates(deduped, trading)
    daily = aggregate_daily_finbert(mapped)

    panel = s3.build_panel_skeleton(membership, trading)
    panel = panel.merge(daily, on=["permno", "date"], how="left")
    panel = s3.apply_no_news_defaults(panel)
    panel = s3.add_rolling_features(panel)

    panel.to_parquet(out_path, index=False)
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dedup_stats": stats,
        "rows": int(len(panel)),
        "finbert_score_path": str(FINBERT_SCORES),
    }
    (PANEL_ROOT / "S5R_finbert_daily_panel_manifest.json").write_text(
        json.dumps(json_safe(meta), indent=2), encoding="utf-8",
    )
    log.info("FinBERT daily panel: %s rows -> %s", len(panel), out_path)
    return out_path


def aggregate_event_sentiment_finbert(articles_enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (permno, evt), g in articles_enriched.groupby(["permno", "event_trade_date_v2"]):
        g = g.dropna(subset=["signed_score"])
        if g.empty:
            continue
        n = len(g)
        labels = g["predicted_label"].astype(str)
        pos = labels.isin(POSITIVE_FB).sum()
        neg = labels.isin(NEGATIVE_FB).sum()
        scores = g["signed_score"]
        rows.append(
            {
                "permno": int(permno),
                "event_trade_date_v2": pd.Timestamp(evt).normalize(),
                "news_count": n,
                "log_news_count": float(np.log1p(n)),
                "ticker_mean_sentiment": float(scores.mean()),
                "relevance_weighted_sentiment": weighted_finbert(g),
                "positive_news_share": float(pos / n),
                "negative_news_share": float(neg / n),
                "net_positive_share": float((pos - neg) / n),
                "sentiment_dispersion": float(scores.std(ddof=0)) if n >= 2 else np.nan,
                "relevance_mean": float(g["relevance_score_num"].mean()),
                "no_news_flag": 0,
            }
        )
    return pd.DataFrame(rows)


def build_e2_exante_panel(ex: Any) -> pd.DataFrame:
    v2 = pd.read_parquet(ex.V2_MAP)
    arts = pd.read_parquet(
        ARTICLES_PATH,
        columns=["permno", "canonical_url_hash", "relevance_score"],
    )
    arts["relevance_score_num"] = pd.to_numeric(arts["relevance_score"], errors="coerce")
    arts = arts.drop_duplicates(["permno", "canonical_url_hash"], keep="first")

    scores = pd.read_parquet(FINBERT_SCORES)
    scores = scores[scores["inference_status"] == "ok"][
        ["permno", "canonical_url_hash", "signed_score", "predicted_label"]
    ]

    eligible = ex.filter_eligible_articles(v2)
    eligible = eligible.merge(arts, on=["permno", "canonical_url_hash"], how="left", validate="many_to_one")
    eligible = eligible.merge(scores, on=["permno", "canonical_url_hash"], how="left", validate="many_to_one")
    eligible = eligible[eligible["relevance_score_num"] > 0]

    sent_daily = aggregate_event_sentiment_finbert(eligible)
    targets = (
        eligible.sort_values(["permno", "event_trade_date_v2", "time_published_utc"])
        .drop_duplicates(["permno", "event_trade_date_v2"], keep="first")[
            [
                "permno", "event_trade_date_v2", "split", "covid_post",
                "H_INTRADAY_T_v2", "H_SIGNAL_DAY_v2", "H_OVERNIGHT_v2",
            ]
        ]
        .copy()
    )
    targets["event_trade_date_v2"] = pd.to_datetime(targets["event_trade_date_v2"]).dt.normalize()
    panel = targets.merge(sent_daily, on=["permno", "event_trade_date_v2"], how="left")
    panel = ex.merge_lagged_alpha(panel)
    ex.assert_exante_timing(panel)
    panel = ex.add_event_rolling_sentiment(panel)
    panel["datetime"] = pd.to_datetime(panel["event_trade_date_v2"])
    panel["instrument"] = "P" + panel["permno"].astype(str)
    panel["split"] = panel["datetime"].map(ex.assign_split_ts)
    panel = panel[panel["datetime"] <= pd.Timestamp("2023-12-31")].copy()
    return panel.reset_index(drop=True)


def build_delayed_finbert_panel(av: Any) -> pd.DataFrame:
    from qlib.data import D

    av.init_qlib()
    sent = pd.read_parquet(PANEL_ROOT / "S5R_finbert_daily_sentiment_raw.parquet")
    sent["datetime"] = pd.to_datetime(sent["datetime"])
    for col in av.DIRECTIONAL:
        sent[col] = sent[col].where(sent["no_news_flag"] == 0)
    df = sent.merge(av.load_labels(), on=["datetime", "instrument"], how="inner")
    df = df.merge(av.load_alpha20(), on=["datetime", "instrument"], how="inner")
    df = df[df["datetime"] <= pd.Timestamp("2023-12-31")].copy()
    if (df["datetime"] >= pd.Timestamp("2024-01-01")).any():
        raise RuntimeError("2024 rows in delayed FinBERT panel")
    df["news_event_flag"] = (df["no_news_flag"] == 0).astype(np.int8)
    df["split"] = df["datetime"].map(av.assign_split_ts)
    return df.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def run_delayed_specs(av: Any) -> tuple[list[dict], dict]:
    df = build_delayed_finbert_panel(av)
    train = av.process_train_label(df[df["split"] == "train"])
    valid = av.process_train_label(df[df["split"] == "valid"])
    metrics: list[dict] = []
    results: dict = {}

    specs = {
        "D2": {"feats": av.ALPHA20 + av.SENTIMENT, "arch_note": "Alpha20+FinBERT full-panel"},
        "D4": {"feats": av.SENTIMENT, "arch_note": "FinBERT-only full-panel"},
    }

    for spec_id, cfg in specs.items():
        feats = cfg["feats"]
        log.info("%s full-panel delayed", spec_id)
        m_fp = av.lgb_train(train, valid, feats)
        pred_fp = df.copy()
        pred_fp["score"] = m_fp.predict(pred_fp[feats], num_iteration=m_fp.best_iteration)
        metrics.extend(av.eval_periods(pred_fp, "label", spec_id, "delayed", "full_panel", "LABEL0", "finbert_daily"))
        results[f"{spec_id}_full_panel"] = pred_fp

    # D3 two-stage
    log.info("D3 two-stage delayed")
    m_s1 = av.lgb_train(train, valid, av.ALPHA20)
    event_mask = df["news_event_flag"] == 1
    ev_train = train[event_mask.loc[train.index]].copy()
    ev_valid = valid[event_mask.loc[valid.index]].copy()
    ev_train["base"] = m_s1.predict(ev_train[av.ALPHA20], num_iteration=m_s1.best_iteration)
    ev_valid["base"] = m_s1.predict(ev_valid[av.ALPHA20], num_iteration=m_s1.best_iteration)
    ev_train["label_processed"] = ev_train["label"] - ev_train["base"]
    ev_valid["label_processed"] = ev_valid["label"] - ev_valid["base"]
    day_w = 1.0 / ev_train.groupby("datetime").size()
    ev_train = ev_train.join(day_w.rename("w"), on="datetime")
    pred_ts = df.copy()
    pred_ts["base"] = m_s1.predict(pred_ts[av.ALPHA20], num_iteration=m_s1.best_iteration)
    pred_ts["residual"] = 0.0
    m_s2 = av.lgb_train(ev_train, ev_valid, av.SENTIMENT, weights=ev_train["w"])
    pred_ts.loc[event_mask, "residual"] = m_s2.predict(
        df.loc[event_mask, av.SENTIMENT], num_iteration=m_s2.best_iteration,
    )
    pred_ts["residual"] = pred_ts.get("residual", 0.0)
    pred_ts["score"] = pred_ts["base"] + pred_ts["residual"]
    metrics.extend(av.eval_periods(pred_ts, "label", "D3", "delayed", "two_stage", "LABEL0", "finbert_daily"))
    results["D3_two_stage"] = pred_ts

    return metrics, results


def run_e2_exante(ex: Any) -> tuple[list[dict], dict]:
    panel = build_e2_exante_panel(ex)
    targets = {
        "H_INTRADAY_T_v2": "primary",
        "H_SIGNAL_DAY_v2": "robustness",
        "H_OVERNIGHT_v2": "robustness",
    }
    metrics: list[dict] = []
    results: dict = {}
    spec_id = "E2_exante"

    for target, role in targets.items():
        sub = panel.dropna(subset=[target]).copy()
        sub["label"] = sub[target]
        train = ex.process_train_label(sub[sub["split"] == "train"])
        valid_df = ex.process_train_label(sub[sub["split"] == "valid"])

        log.info("%s %s full-panel ex-ante FinBERT", spec_id, target)
        model = ex.lgb_train(train, valid_df, ex.ALPHA20 + ex.SENTIMENT, weights=None)
        pred = sub.copy()
        pred["score"] = model.predict(pred[ex.ALPHA20 + ex.SENTIMENT], num_iteration=model.best_iteration)
        ex.assert_prediction_variance(pred, spec_id, target)
        metrics.extend(
            ex.eval_periods(pred, target, spec_id, "full_panel", target, "v2_tradable_exante_finbert"),
        )
        pm = ex.event_pooled_metrics(pred, target)
        port_rows = []
        for period in ex.PERIODS:
            pf = ex.portfolio_eval(pred, target, period)
            port_rows.append({
                "spec": spec_id, "family": "immediate_exante_finbert", "architecture": "full_panel",
                "target": target, "sample": "v2_tradable_exante_finbert", **pm, **pf,
            })
        results[f"{target}_portfolio"] = port_rows
        if role == "primary":
            results[f"{target}_pred"] = pred

        log.info("%s %s two-stage ex-ante FinBERT", spec_id, target)
        delayed = ex.build_delayed_lagged_panel()
        tr = ex.process_train_label(delayed[delayed["split"] == "train"], "label")
        va = ex.process_train_label(delayed[delayed["split"] == "valid"], "label")
        m1 = ex.lgb_train(tr, va, ex.ALPHA20)
        pred_ts = sub.copy()
        pred_ts["base"] = m1.predict(pred_ts[ex.ALPHA20], num_iteration=m1.best_iteration)
        tr_ev = train.copy()
        va_ev = valid_df.copy()
        tr_ev["base"] = m1.predict(tr_ev[ex.ALPHA20], num_iteration=m1.best_iteration)
        va_ev["base"] = m1.predict(va_ev[ex.ALPHA20], num_iteration=m1.best_iteration)
        tr_ev["label_processed"] = tr_ev["label"] - tr_ev["base"]
        va_ev["label_processed"] = va_ev["label"] - va_ev["base"]
        m2 = ex.lgb_train(tr_ev, va_ev, ex.SENTIMENT, weights=None)
        pred_ts["residual"] = m2.predict(pred_ts[ex.SENTIMENT], num_iteration=m2.best_iteration)
        pred_ts["score"] = pred_ts["base"] + pred_ts["residual"]
        metrics.extend(
            ex.eval_periods(pred_ts, target, spec_id, "two_stage", target, "v2_tradable_exante_finbert"),
        )

    diag = {
        "feature_date_lt_event_all": bool(
            (pd.to_datetime(panel["feature_date"]) < pd.to_datetime(panel["event_trade_date_v2"])).all()
        ),
        "model_panel_permno_event_days": int(len(panel)),
        "finbert_scores_path": str(FINBERT_SCORES),
    }
    results["diagnostics"] = diag
    return metrics, results


def write_report(metrics_df: pd.DataFrame, out: Path) -> None:
    lines = [
        "# S5R FinBERT Experiments (D2/D3/D4 + E2 ex-ante)",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Validation Rank IC (threshold=25, valid period)",
        "",
        "| Spec | Family | Arch | Target | Rank IC | Valid days |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    sub = metrics_df[(metrics_df["ic_threshold"] == 25) & (metrics_df["period"] == "valid")]
    for _, r in sub.iterrows():
        lines.append(
            f"| {r.get('spec','')} | {r.get('family','')} | {r.get('architecture','')} | "
            f"{r.get('target','')} | {r.get('mean_daily_rank_ic', float('nan')):.4f} | "
            f"{int(r.get('valid_ic_days', 0))} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "delayed").mkdir(exist_ok=True)
    (OUT_ROOT / "immediate").mkdir(exist_ok=True)
    (OUT_ROOT / "reports").mkdir(exist_ok=True)
    (OUT_ROOT / "manifests").mkdir(exist_ok=True)
    (OUT_ROOT / "diagnostics").mkdir(exist_ok=True)

    val_path = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/finbert_full_history/S5R_finbert_final_validation.json"
    if not val_path.exists():
        log.error("Run S5R_finbert_final_validation.py first")
        return 1
    val = json.loads(val_path.read_text(encoding="utf-8"))
    if val.get("validation_status") != "COMPLETE":
        log.error("FinBERT validation not COMPLETE: %s", val.get("validation_status"))
        return 1

    build_finbert_daily_panel()
    av = load_module("av", PROJECT_ROOT / "03_fundamentals_news/S5R_corrected_av_only.py")
    ex = load_module("exante", PROJECT_ROOT / "03_fundamentals_news/S5R_corrected_av_only_exante_v2.py")

    all_metrics: list[dict] = []

    d_metrics, d_res = run_delayed_specs(av)
    all_metrics.extend(d_metrics)
    with (OUT_ROOT / "delayed/finbert_delayed_results.pkl").open("wb") as f:
        pickle.dump(d_res, f)

    e_metrics, e_res = run_e2_exante(ex)
    all_metrics.extend(e_metrics)
    with (OUT_ROOT / "immediate/E2_exante_results.pkl").open("wb") as f:
        pickle.dump(e_res, f)
    (OUT_ROOT / "diagnostics/E2_exante_diagnostics.json").write_text(
        json.dumps(json_safe(e_res.get("diagnostics", {})), indent=2), encoding="utf-8",
    )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_ROOT / "S5R_finbert_metrics.csv", index=False)
    write_report(metrics_df, OUT_ROOT / "reports/S5R_finbert_results.md")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "finbert_validation_sha256": val.get("sha256"),
        "experiments": ["D2", "D3", "D4", "E2_exante"],
        "delayed_target": "LABEL0",
        "immediate_primary_target": "H_INTRADAY_T_v2",
        "immediate_robustness_targets": ["H_SIGNAL_DAY_v2", "H_OVERNIGHT_v2"],
        "timing": {
            "delayed_alpha20": "same-day (S5R-D convention)",
            "immediate_alpha20": "prior CRSP session (ex-ante v2)",
            "immediate_finbert": "event-time v2 eligible articles before t open",
        },
        "holdout_2024_accessed": False,
    }
    (OUT_ROOT / "manifests/S5R_finbert_experiments_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2), encoding="utf-8",
    )
    log.info("S5R FinBERT experiments complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
