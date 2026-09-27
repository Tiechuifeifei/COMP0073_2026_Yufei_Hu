"""Phase H: Massive news coverage audit only. No strategy features."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from massive_audit.client import MassiveClient
from massive_audit.config import AUDIT_DIR, REPORT_DIR, ensure_dirs

YEARS = {
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026": ("2026-01-01", "2026-08-18"),
}
SAMPLE_TICKERS = ["AAPL", "NVDA", "JPM", "XOM", "JNJ"]


def run_news_coverage() -> dict:
    ensure_dirs()
    client = MassiveClient()
    rows = []
    field_presence = Counter()
    publishers = Counter()
    ids = []
    for year, (start, end) in YEARS.items():
        for ticker in SAMPLE_TICKERS:
            articles = client.paginate(
                "/v2/reference/news",
                {
                    "ticker": ticker,
                    "published_utc.gte": f"{start}T00:00:00Z",
                    "published_utc.lte": f"{end}T23:59:59Z",
                    "limit": 50,
                    "sort": "published_utc",
                    "order": "asc",
                },
                max_pages=2,
            )
            for art in articles:
                ids.append(art.get("id"))
                insights = art.get("insights") or []
                sent = None
                if insights:
                    hit = next((i for i in insights if str(i.get("ticker", "")).upper() == ticker), insights[0])
                    sent = hit.get("sentiment")
                rec = {
                    "year": year,
                    "query_ticker": ticker,
                    "id": art.get("id"),
                    "published_utc": art.get("published_utc"),
                    "title": art.get("title"),
                    "description": art.get("description"),
                    "article_url": art.get("article_url"),
                    "amp_url": art.get("amp_url"),
                    "author": art.get("author"),
                    "publisher": (art.get("publisher") or {}).get("name"),
                    "tickers": ",".join(art.get("tickers") or []),
                    "n_tickers": len(art.get("tickers") or []),
                    "has_title": bool(art.get("title")),
                    "has_summary": bool(art.get("description")),
                    "has_full_text": False,
                    "has_url": bool(art.get("article_url")),
                    "has_sentiment": sent is not None,
                    "sentiment": sent,
                    "has_publisher": bool((art.get("publisher") or {}).get("name")),
                }
                for k, v in rec.items():
                    if k.startswith("has_") and v:
                        field_presence[k] += 1
                publishers[rec["publisher"] or "UNKNOWN"] += 1
                rows.append(rec)
            print(f"H {year} {ticker}: {len(articles)} articles (capped)")

    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_DIR / "news_coverage_sample.csv", index=False)
    n = max(len(df), 1)
    dup_rate = 1.0 - (pd.Series(ids).nunique() / n) if ids else 0.0
    oldest = df["published_utc"].min() if not df.empty else ""
    newest = df["published_utc"].max() if not df.empty else ""

    # Global oldest probe
    st, payload, _ = client.get("/v2/reference/news", {"limit": 1, "sort": "published_utc", "order": "asc"})
    global_oldest = ""
    if st == 200 and isinstance(payload, dict) and payload.get("results"):
        global_oldest = payload["results"][0].get("published_utc", "")

    summary = {
        "n_sample_rows": int(len(df)),
        "duplicate_id_rate": float(dup_rate),
        "sample_oldest": oldest,
        "sample_newest": newest,
        "global_oldest_any_ticker": global_oldest,
        "full_text_in_api": False,
        "sentiment_in_insights": bool(field_presence.get("has_sentiment")),
        "ticker_association": True,
        "publication_timestamp": "published_utc RFC3339",
        "vs_alpha_vantage": {
            "av_in_this_project": "S1–S5R Alpha Vantage news; sparse 2025 coverage drift already documented",
            "massive_advantages_if_accessible": "native ticker tags, published_utc, optional insights sentiment, pagination",
            "massive_limitations": "no full article text in REST payload; plan history may be 2y on Basic; not used in H0",
        },
    }
    (AUDIT_DIR / "news_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# Massive news coverage report

Generated: {datetime.now(timezone.utc).isoformat()}

This is a **coverage audit**. News is not added to H0.

## Access

Sample rows: {len(df)}
Global oldest article (any ticker, 1-row probe): `{global_oldest}`
Sample range: `{oldest}` .. `{newest}`

## Fields present in REST payload

| field | present |
|---|---|
| ticker association | yes (`tickers` array) |
| publication timestamp | yes (`published_utc`) |
| headline | yes (`title`) |
| summary | yes (`description`) |
| full text | **no** |
| sentiment | sometimes (`insights[].sentiment`) |
| source/publisher | yes (`publisher.name`) |
| URL | yes (`article_url`) |

Duplicate id rate in sample: {dup_rate:.3f}

Top publishers in sample: {publishers.most_common(8)}

## Versus Alpha Vantage (this repo)

Alpha Vantage ticker news in S5R is the current sentiment source. Holdout work already found coverage drift, especially 2025, and S5B filters did not repair H0.

Massive is **better structured** (tickers + UTC timestamp + pagination + optional sentiment). It is **not automatically PIT-safer**: publisher timestamps can still be revised, and there is no full text for local re-scoring. Historical depth depends on plan (docs: Basic 2 years; Starter+ all history to 2016-06-22).

**Recommendation:** Massive news is a reasonable *future research* corpus if the plan includes sufficient history. It is not required for a 2026 H0 data extension, and it should not enter the frozen strategy in this phase.
"""
    (REPORT_DIR / "news_coverage_report.md").write_text(md, encoding="utf-8")
    print(f"H rows={len(df)} dup_rate={dup_rate:.3f}")
    return summary
