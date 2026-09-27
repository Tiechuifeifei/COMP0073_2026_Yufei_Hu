#!/usr/bin/env python3
"""S5R full-history FinBERT scoring with checkpointed shards and atomic commit.

Run in the sentiment-nlp env, e.g.:
  /opt/anaconda3/envs/sentiment-nlp/bin/python 03_fundamentals_news/S5R_finbert_full_history.py"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu  # noqa: E402

def _load_rel(_name, _rel):  # noqa: E402
    _p = PROJECT_ROOT / _rel
    _s = _ilu.spec_from_file_location(_name, _p)
    _m = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    return _m

_s3 = _load_rel("s3_panel", "01_data/sentiment_experiments/S3_daily_sentiment_panel.py")  # noqa: E402
PERIOD_END, PERIOD_START, dedupe_articles = _s3.PERIOD_END, _s3.PERIOD_START, _s3.dedupe_articles  # noqa: E402
_s6a_path = PROJECT_ROOT / "01_data/sentiment_experiments/S6A_sentiment_scorer_benchmark.py"
if _s6a_path.is_file():
    _s6a = _load_rel("s6a", "01_data/sentiment_experiments/S6A_sentiment_scorer_benchmark.py")
    FINBERT_CHECKPOINT, load_sector_map = _s6a.FINBERT_CHECKPOINT, _s6a.load_sector_map
else:
    FINBERT_CHECKPOINT, load_sector_map = None, None  # optional; not shipped in this pack

OUT_ROOT = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/finbert_full_history"
SHARD_DIR = OUT_ROOT / "shards"
STAGING_PATH = OUT_ROOT / "S5R_finbert_scores_full_history.staging.parquet"
FINAL_PATH = OUT_ROOT / "S5R_finbert_scores_full_history.parquet"
MANIFEST_PATH = OUT_ROOT / "S5R_finbert_full_history_manifest.json"
ARTICLES_PATH = PROJECT_ROOT / "data/sentiment_raw/alpha_vantage_full/normalized/articles.parquet"
SHARD_SIZE = 5000


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class TextLoader:
    def __init__(self) -> None:
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def _load_feed(self, raw_file_reference: str) -> list[dict[str, Any]]:
        if raw_file_reference not in self._cache:
            path = PROJECT_ROOT / raw_file_reference
            data = json.loads(path.read_text(encoding="utf-8"))
            feed = data.get("feed") if isinstance(data.get("feed"), list) else []
            self._cache[raw_file_reference] = feed
        return self._cache[raw_file_reference]

    def get_text(self, row: pd.Series) -> tuple[str, str] | None:
        feed = self._load_feed(str(row["raw_file_reference"]))
        target_id = str(row["article_id"])
        permno = int(row["permno"])
        query_ticker = str(row["queried_ticker"])
        for item in feed:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            tp_raw = str(item.get("time_published") or "")
            if sha1_text(f"{url}|{tp_raw}|{permno}|{query_ticker}") == target_id:
                headline = str(item.get("title") or "").strip()
                summary = str(item.get("summary") or "").strip()
                if headline and summary:
                    return headline, summary
                return None
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(payload: dict[str, Any]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_queue() -> pd.DataFrame:
    articles = pd.read_parquet(ARTICLES_PATH)
    articles = articles[
        (articles["publication_date"] >= str(PERIOD_START)) & (articles["publication_date"] <= str(PERIOD_END))
    ]
    deduped, stats = dedupe_articles(articles)
    deduped = deduped.copy()
    deduped["row_id"] = deduped["permno"].astype(str) + "|" + deduped["canonical_url_hash"].astype(str)
    deduped["dedup_stats"] = json.dumps(stats)
    return deduped[
        [
            "row_id",
            "permno",
            "returned_ticker",
            "queried_ticker",
            "canonical_url_hash",
            "article_id",
            "raw_file_reference",
            "time_published_utc",
        ]
    ]


def shard_path(idx: int) -> Path:
    return SHARD_DIR / f"shard_{idx:04d}.parquet"


def completed_shards() -> set[int]:
    if not SHARD_DIR.exists():
        return set()
    done = set()
    for p in SHARD_DIR.glob("shard_*.parquet"):
        try:
            done.add(int(p.stem.split("_")[1]))
        except ValueError:
            continue
    return done


def load_finbert_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_CHECKPOINT)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def run_finbert_batch(
    rows: pd.DataFrame,
    loader: TextLoader,
    sector_map: dict[int, str],
    model,
    tokenizer,
    device: str,
) -> pd.DataFrame:
    batch_size = 16
    max_length = 512

    out_rows: list[dict[str, Any]] = []
    texts: list[str] = []
    meta: list[pd.Series] = []

    for _, row in rows.iterrows():
        text = loader.get_text(row)
        ticker = str(row["returned_ticker"])
        company = sector_map.get(int(row["permno"]), ticker)
        if text is None:
            out_rows.append({
                "row_id": row["row_id"],
                "permno": int(row["permno"]),
                "canonical_url_hash": row["canonical_url_hash"],
                "signed_score": np.nan,
                "positive_probability": np.nan,
                "neutral_probability": np.nan,
                "negative_probability": np.nan,
                "predicted_label": "",
                "inference_status": "missing_text",
            })
            continue
        headline, summary = text
        texts.append(f"[TARGET {ticker} — {company}] {headline} {summary}")
        meta.append(row)

    import torch

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(batch_texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            for i, prob in enumerate(probs):
                row = meta[start + i]
                p_pos = float(prob[0]) if len(prob) > 0 else np.nan
                p_neg = float(prob[1]) if len(prob) > 1 else np.nan
                p_neu = float(prob[2]) if len(prob) > 2 else np.nan
                out_rows.append({
                    "row_id": row["row_id"],
                    "permno": int(row["permno"]),
                    "canonical_url_hash": row["canonical_url_hash"],
                    "signed_score": p_pos - p_neg,
                    "positive_probability": p_pos,
                    "neutral_probability": p_neu,
                    "negative_probability": p_neg,
                    "predicted_label": ["positive", "negative", "neutral"][int(np.argmax(prob))],
                    "inference_status": "ok",
                })
    return pd.DataFrame(out_rows)


def atomic_commit() -> None:
    shards = sorted(SHARD_DIR.glob("shard_*.parquet"))
    if not shards:
        raise RuntimeError("No shards to commit")
    parts = [pd.read_parquet(p) for p in shards]
    out = pd.concat(parts, ignore_index=True).drop_duplicates("row_id", keep="first")
    out.to_parquet(STAGING_PATH, index=False)
    os.replace(STAGING_PATH, FINAL_PATH)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    queue = load_queue()
    n = len(queue)
    n_shards = int(np.ceil(n / SHARD_SIZE))
    done = completed_shards()
    loader = TextLoader()
    sector_map = load_sector_map()
    model, tokenizer, device = load_finbert_model()

    manifest = {
        "generated_at_utc": utc_now(),
        "status": "in_progress",
        "checkpoint": FINBERT_CHECKPOINT,
        "total_rows": n,
        "shard_size": SHARD_SIZE,
        "total_shards": n_shards,
        "completed_shards": sorted(done),
        "python_executable": sys.executable,
        "holdout_2024_accessed": False,
        "frozen_s6a_modified": False,
    }
    write_manifest(manifest)

    t0 = time.time()
    for shard_idx in range(n_shards):
        if shard_idx in done:
            continue
        start = shard_idx * SHARD_SIZE
        end = min(start + SHARD_SIZE, n)
        batch = queue.iloc[start:end]
        print(f"FinBERT shard {shard_idx + 1}/{n_shards} rows {start}-{end}", flush=True)
        scored = run_finbert_batch(batch, loader, sector_map, model, tokenizer, device)
        tmp = shard_path(shard_idx).with_suffix(".tmp.parquet")
        scored.to_parquet(tmp, index=False)
        os.replace(tmp, shard_path(shard_idx))
        done.add(shard_idx)
        manifest["completed_shards"] = sorted(done)
        manifest["last_shard_completed"] = shard_idx
        manifest["elapsed_sec"] = round(time.time() - t0, 1)
        manifest["ok_rows"] = int(scored["inference_status"].eq("ok").sum())
        write_manifest(manifest)

    if len(done) == n_shards:
        atomic_commit()
        manifest["status"] = "complete"
        manifest["final_path"] = str(FINAL_PATH)
        manifest["completed_at_utc"] = utc_now()
        manifest["elapsed_sec"] = round(time.time() - t0, 1)
        write_manifest(manifest)
        print(f"FinBERT full history complete: {FINAL_PATH}", flush=True)
    else:
        print(f"Partial run: {len(done)}/{n_shards} shards", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
