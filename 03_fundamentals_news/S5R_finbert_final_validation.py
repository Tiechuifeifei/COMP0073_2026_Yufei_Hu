#!/usr/bin/env python3
"""Final validation for S5R FinBERT full-history scores (+ optional headline-only repair)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/finbert_full_history"
FINAL_PATH = OUT_ROOT / "S5R_finbert_scores_full_history.parquet"
STAGING_PATH = OUT_ROOT / "S5R_finbert_scores_full_history.staging.parquet"
MANIFEST_PATH = OUT_ROOT / "S5R_finbert_full_history_manifest.json"
SOURCE_MANIFEST = PROJECT_ROOT / "data/sentiment_experiments/S5R_corrected/manifests/S5R_scorer_availability.json"
VALID_LABELS = frozenset({"positive", "neutral", "negative"})
REQUIRED_OK = [
    "signed_score",
    "positive_probability",
    "neutral_probability",
    "negative_probability",
    "predicted_label",
]
PROB_TOL = 1e-4
SIGNED_TOL = 1e-9
EXPECTED_ROWS = 307_075


def load_finbert_module():
    path = PROJECT_ROOT / "03_fundamentals_news/S5R_finbert_full_history.py"
    spec = importlib.util.spec_from_file_location("s5r_finbert_hist", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    return obj


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose_missing_text(fb_mod, queue: pd.DataFrame, scores: pd.DataFrame) -> dict[str, int]:
    bad_ids = set(scores.loc[scores["inference_status"] != "ok", "row_id"])
    bad_q = queue[queue["row_id"].isin(bad_ids)]
    loader = fb_mod.TextLoader()
    counts = {"not_in_feed": 0, "empty_both": 0, "headline_only": 0, "summary_only": 0, "both_present": 0}
    for _, row in bad_q.iterrows():
        feed = loader._load_feed(str(row["raw_file_reference"]))
        target_id = str(row["article_id"])
        permno = int(row["permno"])
        query_ticker = str(row["queried_ticker"])
        found = False
        for item in feed:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            tp_raw = str(item.get("time_published") or "")
            if fb_mod.sha1_text(f"{url}|{tp_raw}|{permno}|{query_ticker}") == target_id:
                found = True
                headline = str(item.get("title") or "").strip()
                summary = str(item.get("summary") or "").strip()
                if headline and summary:
                    counts["both_present"] += 1
                elif headline and not summary:
                    counts["headline_only"] += 1
                elif summary and not headline:
                    counts["summary_only"] += 1
                else:
                    counts["empty_both"] += 1
                break
        if not found:
            counts["not_in_feed"] += 1
    counts["total_missing_text"] = int(len(bad_q))
    return counts


def repair_headline_only(fb_mod, queue: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predeclared sole repair: score headline-only when summary is empty."""
    bad = scores[scores["inference_status"] != "ok"].copy()
    if bad.empty:
        return scores, {"repaired_rows": 0, "policy": "headline_only_fallback"}

    bad_q = queue[queue["row_id"].isin(bad["row_id"])].copy()
    loader = fb_mod.TextLoader()
    _s6a_p = PROJECT_ROOT / "01_data/sentiment_experiments/S6A_sentiment_scorer_benchmark.py"
    if _s6a_p.is_file():
        import importlib.util as _ilu
        _sp = _ilu.spec_from_file_location("s6a", _s6a_p)
        _sm = _ilu.module_from_spec(_sp)
        _sp.loader.exec_module(_sm)
        sector_map = _sm.load_sector_map()
    else:
        sector_map = {}
    model, tokenizer, device = fb_mod.load_finbert_model()

    texts: list[str] = []
    row_ids: list[str] = []
    for _, row in bad_q.iterrows():
        feed = loader._load_feed(str(row["raw_file_reference"]))
        target_id = str(row["article_id"])
        permno = int(row["permno"])
        query_ticker = str(row["queried_ticker"])
        headline = None
        for item in feed:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            tp_raw = str(item.get("time_published") or "")
            if fb_mod.sha1_text(f"{url}|{tp_raw}|{permno}|{query_ticker}") == target_id:
                headline = str(item.get("title") or "").strip()
                break
        if not headline:
            continue
        ticker = str(row["returned_ticker"])
        company = sector_map.get(permno, ticker)
        texts.append(f"[TARGET {ticker} — {company}] {headline}")
        row_ids.append(row["row_id"])

    if not texts:
        return scores, {"repaired_rows": 0, "policy": "headline_only_fallback", "note": "no repairable rows"}

    import torch

    repaired_rows: list[dict[str, Any]] = []
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(batch_texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            for i, prob in enumerate(probs):
                p_pos = float(prob[0]) if len(prob) > 0 else np.nan
                p_neg = float(prob[1]) if len(prob) > 1 else np.nan
                p_neu = float(prob[2]) if len(prob) > 2 else np.nan
                repaired_rows.append({
                    "row_id": row_ids[start + i],
                    "signed_score": p_pos - p_neg,
                    "positive_probability": p_pos,
                    "neutral_probability": p_neu,
                    "negative_probability": p_neg,
                    "predicted_label": ["positive", "negative", "neutral"][int(np.argmax(prob))],
                    "inference_status": "ok",
                })

    rep = pd.DataFrame(repaired_rows)
    out = scores.set_index("row_id")
    rep = rep.set_index("row_id")
    out.update(rep)
    out = out.reset_index()
    meta = {"repaired_rows": int(len(rep)), "policy": "headline_only_fallback (summary empty)"}
    return out, meta


def validate_scores(scores: pd.DataFrame, queue: pd.DataFrame, source_manifest: dict) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("row_count_exact", len(scores) == EXPECTED_ROWS, {"rows": len(scores), "expected": EXPECTED_ROWS})
    add(
        "unique_article_ticker_keys",
        scores["row_id"].nunique() == len(scores) == EXPECTED_ROWS,
        {"unique_row_id": int(scores["row_id"].nunique())},
    )

    ok = scores[scores["inference_status"] == "ok"]
    add("all_inference_status_ok", scores["inference_status"].eq("ok").all(), scores["inference_status"].value_counts().to_dict())

    missing_fields = int(ok[REQUIRED_OK].isna().any(axis=1).sum()) if len(ok) else len(scores)
    add("no_missing_required_fields_on_ok", missing_fields == 0, {"bad_ok_rows": missing_fields})

    if len(ok):
        prob_sum = ok["positive_probability"] + ok["neutral_probability"] + ok["negative_probability"]
        add(
            "probabilities_sum_to_one",
            bool((prob_sum.sub(1.0).abs() <= PROB_TOL).all()),
            {"max_abs_deviation": float(prob_sum.sub(1.0).abs().max())},
        )
        signed_gap = (ok["signed_score"] - (ok["positive_probability"] - ok["negative_probability"])).abs()
        add(
            "signed_score_definition",
            bool((signed_gap <= SIGNED_TOL).all()),
            {"max_abs_gap": float(signed_gap.max())},
        )
        labels = set(ok["predicted_label"].astype(str).unique())
        add("valid_label_vocabulary", labels <= VALID_LABELS, {"labels": sorted(labels)})

    q_keys = set(queue["row_id"])
    s_keys = set(scores["row_id"])
    add(
        "coverage_matches_frozen_queue",
        q_keys == s_keys and len(queue) == EXPECTED_ROWS,
        {"queue_rows": len(queue), "missing_from_scores": len(q_keys - s_keys), "extra_in_scores": len(s_keys - q_keys)},
    )

    expected_pairs = source_manifest["deduplication"]["unique_article_permno_pairs"]
    add(
        "coverage_matches_source_manifest",
        expected_pairs == EXPECTED_ROWS,
        {"manifest_pairs": expected_pairs},
    )

    passed = all(c["passed"] for c in checks)
    return {"passed": passed, "checks": checks}


def write_validation_report(result: dict, sha256: str, repair_meta: dict | None) -> None:
    report = {
        "generated_at_utc": utc_now(),
        "validation_status": "COMPLETE" if result["passed"] else "FAILED",
        "final_path": str(FINAL_PATH),
        "sha256": sha256,
        "file_size_bytes": FINAL_PATH.stat().st_size,
        "checks": result["checks"],
        "repair": repair_meta,
    }
    (OUT_ROOT / "S5R_finbert_final_validation.json").write_text(
        json.dumps(json_safe(report), indent=2), encoding="utf-8",
    )
    lines = [
        "# S5R FinBERT Full-History Final Validation",
        "",
        f"**Generated (UTC):** {report['generated_at_utc']}",
        "",
        f"**Status:** `{report['validation_status']}`",
        "",
        f"**SHA-256:** `{sha256}`",
        "",
        "| Check | Pass | Detail |",
        "| --- | --- | --- |",
    ]
    for c in result["checks"]:
        lines.append(f"| {c['check']} | {'✓' if c['passed'] else '✗'} | `{json.dumps(c['detail'], default=str)[:120]}` |")
    if repair_meta:
        lines.extend(["", "## Repair applied", "", f"```json\n{json.dumps(json_safe(repair_meta), indent=2)}\n```"])
    (OUT_ROOT / "reports/S5R_finbert_final_validation.md").write_text("\n".join(lines), encoding="utf-8")


def update_manifest(validation_status: str, sha256: str, result: dict, repair_meta: dict | None) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["status"] = validation_status
    manifest["final_validation_status"] = validation_status
    manifest["validated_at_utc"] = utc_now()
    manifest["final_sha256"] = sha256
    manifest["validation_passed"] = result["passed"]
    manifest["validation_checks"] = result["checks"]
    if repair_meta:
        manifest["headline_only_repair"] = repair_meta
    manifest["ok_rows"] = int(pd.read_parquet(FINAL_PATH)["inference_status"].eq("ok").sum())
    MANIFEST_PATH.write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")


def main() -> int:
    fb_mod = load_finbert_module()
    queue = fb_mod.load_queue()
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    scores = pd.read_parquet(FINAL_PATH)

    repair_meta = None
    result = validate_scores(scores, queue, source_manifest)
    if not result["passed"]:
        miss = diagnose_missing_text(fb_mod, queue, scores)
        only_headline = (
            miss["total_missing_text"] > 0
            and miss["headline_only"] == miss["total_missing_text"]
            and miss["not_in_feed"] == 0
        )
        if only_headline:
            print(f"Applying headline-only repair to {miss['headline_only']} rows...", flush=True)
            scores, repair_meta = repair_headline_only(fb_mod, queue, scores)
            scores.to_parquet(STAGING_PATH, index=False)
            os.replace(STAGING_PATH, FINAL_PATH)
            result = validate_scores(scores, queue, source_manifest)
        else:
            print("Validation failed; missing_text diagnosis:", miss, flush=True)

    sha = sha256_file(FINAL_PATH)
    status = "COMPLETE" if result["passed"] else "VALIDATION_FAILED"
    write_validation_report(result, sha, repair_meta)
    update_manifest(status, sha, result, repair_meta)

    print(f"Validation: {status}")
    print(f"SHA-256: {sha}")
    for c in result["checks"]:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}: {c['check']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
