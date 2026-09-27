#!/usr/bin/env python3
"""Massive subscription final-day archive of reachable endpoints used by the
2026 approximate evaluation path."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from massive_audit.client import MassiveClient  # noqa: E402
from massive_audit.config import AUDIT_DIR, REPORT_DIR, UNIVERSE_2026, ensure_dirs  # noqa: E402
from massive_audit.phase_c import _ms_to_date  # noqa: E402
from massive_audit.phase_f import TICKER_ALIASES  # noqa: E402
from sentiment_experiments.massive_news_2018_2025.download import (  # noqa: E402
    CACHE,
    ENDPOINT,
    PAGE_LIMIT,
    RAW_ROOT,
    STATE,
    download_window,
    load_checkpoint,
    save_checkpoint,
    window_key,
)

NEWS_HIST_START = "2018-01-01"
NEWS_HIST_END = "2025-12-31"
NEWS_YTD_START = "2026-01-01"
INDEX_VIX = "I:VIX"
INDEX_SPX = "I:SPX"
VIX_RAW = PROJECT_ROOT / "data/sentiment_raw/massive_indices"
REPORT_PATH = PROJECT_ROOT / "reports/data_audits/massive_final_subscription_archive.md"
SUMMARY_PATH = AUDIT_DIR / "final_day_archive_summary.json"


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _today() -> str:
    return datetime.now().date().isoformat()


def _du(path: Path) -> tuple[int, int]:
    total = 0
    n = 0
    if not path.exists():
        return 0, 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                total += fp.stat().st_size
                n += 1
            except OSError:
                continue
    return total, n


def _gb(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n/1024:.1f} KB"
    if n < 1024**3:
        return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.2f} GB"


def audit_existing_news() -> dict[str, Any]:
    man_path = RAW_ROOT / "massive_news_download_manifest.json"
    summary_path = STATE / "download_summary.json"
    jobs_path = CACHE / "download_jobs.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    dl_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    jobs = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else []
    complete = list((RAW_ROOT / "raw").glob("*/*/_complete.json"))
    by_year: dict[str, int] = {}
    incomplete = []
    for p in complete:
        rec = json.loads(p.read_text(encoding="utf-8"))
        year = str(rec.get("year"))
        by_year[year] = by_year.get(year, 0) + 1
        if not rec.get("complete"):
            incomplete.append(str(p.relative_to(PROJECT_ROOT)))
    years_ok = {str(y) for y in range(2018, 2026)}
    complete_years = {y: n for y, n in sorted(by_year.items()) if y in years_ok}
    n_complete_hist = sum(complete_years.values())
    n_jobs = len(jobs)
    n_fail = int(dl_summary.get("fail", -1))
    size, n_files = _du(RAW_ROOT)
    out = {
        "manifest_period": manifest.get("period"),
        "manifest_n_windows": manifest.get("n_windows"),
        "n_jobs": n_jobs,
        "n_complete_markers_hist": n_complete_hist,
        "complete_by_year": complete_years,
        "n_incomplete_markers": len(incomplete),
        "download_summary": dl_summary,
        "fail": n_fail,
        "size_bytes": size,
        "n_files": n_files,
        "complete_4264": n_complete_hist == 4264 and n_jobs == 4264,
        "zero_fail": n_fail == 0,
        "manifest_present": bool(manifest),
        "hist_complete": n_complete_hist == 4264 and n_jobs == 4264 and n_fail == 0 and not incomplete,
    }
    return out


def build_2026_jobs(asof: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hist = pd.read_csv(CACHE / "query_universe.csv")
    hist["membership_end"] = pd.to_datetime(hist["membership_end"])
    keep = hist[hist["membership_end"] >= pd.Timestamp(NEWS_HIST_END)].copy()
    tickers = set(keep["query_ticker"].astype(str).str.upper())
    events = pd.read_csv(UNIVERSE_2026 / "sp500_2026_membership_events.csv")
    added = set(events["added_ticker"].dropna().astype(str).str.upper())
    snap = pd.read_csv(UNIVERSE_2026 / "sp500_2025_12_31_snapshot.csv")
    snap_t = set(snap["ticker"].dropna().astype(str).str.upper())
    for stale, current in TICKER_ALIASES.items():
        if stale in snap_t or stale in tickers:
            tickers.discard(stale)
            tickers.add(current)
    tickers |= added
    tickers.discard("")
    tickers.discard("NAN")
    jobs = [{"query_ticker": t, "year": 2026, "key": window_key(t, 2026)} for t in sorted(tickers)]
    meta = {
        "asof": asof,
        "n_snapshot_query_tickers": int(keep["query_ticker"].nunique()),
        "n_2026_additions": len(added),
        "additions": sorted(added),
        "n_jobs": len(jobs),
        "pit_rule": (
            "Same date-aware query tickers as 2018–2025 archive for names whose "
            "membership_end >= 2025-12-31, plus 2026 S&P DJI additions already in "
            "massive_2026/universe, plus phase_f ticker aliases. Yearly window, not intra-year cuts."
        ),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    keep.assign(archive_year=2026).to_csv(CACHE / "query_universe_2026_snapshot.csv", index=False)
    (CACHE / "download_jobs_2026.json").write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    return jobs, meta


def download_2026_news(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    done = load_checkpoint()
    for p in (RAW_ROOT / "raw").glob("*/*/_complete.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            done.add(window_key(rec["query_ticker"], int(rec["year"])))
        except Exception:
            continue
    hist_done = {k for k in done if k.endswith("|2026") is False}
    pending = [j for j in jobs if j["key"] not in done]
    ytd_manifest = {
        "created_at": _utc(),
        "endpoint": f"https://api.massive.com{ENDPOINT}",
        "period": [NEWS_YTD_START, _today()],
        "universe": "2018–2025 PIT query tickers still in S&P500 at 2025-12-31 plus 2026 additions",
        "n_windows": len(jobs),
        "n_already_complete": len(jobs) - len(pending),
        "pagination": f"next_url until exhausted; limit={PAGE_LIMIT}",
        "filter_at_ingestion": False,
        "overwrites_2018_2025": False,
        "n_hist_checkpoint_keys_preserved": len(hist_done),
    }
    (RAW_ROOT / "massive_news_2026_ytd_manifest.json").write_text(json.dumps(ytd_manifest, indent=2) + "\n")
    print(
        f"2026 news jobs={len(jobs)} pending={len(pending)} hist_checkpoint={len(hist_done)}",
        flush=True,
    )
    client = MassiveClient(min_interval_s=0.12)
    t0 = time.time()
    ok = fail = 0
    failures: list[dict[str, Any]] = []
    for i, job in enumerate(pending, 1):
        ticker, year, key = job["query_ticker"], int(job["year"]), job["key"]
        out_dir = RAW_ROOT / "raw" / ticker / str(year)
        try:
            rec = download_window(client, ticker, year)
            done.add(key)
            ok += 1
            if i % 20 == 0 or i == len(pending):
                save_checkpoint(done)
                elapsed = time.time() - t0
                print(
                    f"[{i}/{len(pending)}] {ticker} {year} pages={rec['pages']} arts={rec['n_articles']} "
                    f"ok={ok} fail={fail} elapsed={elapsed/60:.1f}m calls={client.calls}",
                    flush=True,
                )
        except Exception as exc:
            fail += 1
            failures.append({"ticker": ticker, "year": year, "error": str(exc)})
            print(f"FAIL {ticker} {year}: {exc}", flush=True)
            time.sleep(2.0)
        if out_dir.exists() and (RAW_ROOT / "raw" / ticker / "2018").exists():
            # safety: never delete or rewrite sibling year dirs
            pass
    save_checkpoint(done)
    ytd_complete = [p for p in (RAW_ROOT / "raw").glob("*/2026/_complete.json")]
    summary = {
        "finished_at": _utc(),
        "ok": ok,
        "fail": fail,
        "n_jobs": len(jobs),
        "n_complete_2026": len(ytd_complete),
        "pending_remaining": max(len(jobs) - len(ytd_complete), 0),
        "api_calls": client.calls,
        "rate_limit_hits": client.rate_limit_hits,
        "failures": failures,
        "hist_checkpoint_keys": len({k for k in load_checkpoint() if not k.endswith("|2026")}),
    }
    (STATE / "download_summary_2026.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def summarize_2026_news() -> dict[str, Any]:
    ids: set[str] = set()
    links = 0
    requests = 0
    pubs: set[str] = set()
    max_pub = ""
    min_pub = ""
    for page in (RAW_ROOT / "raw").glob("*/2026/page_*.json"):
        requests += 1
        payload = json.loads(page.read_text(encoding="utf-8"))
        results = payload.get("results") or []
        if not results:
            continue
        for r in results:
            if not isinstance(r, dict):
                continue
            aid = str(r.get("id") or "")
            if aid:
                ids.add(aid)
            tickers = r.get("tickers") or []
            links += 1
            pub = str((r.get("publisher") or {}).get("name") or "")
            if pub:
                pubs.add(pub)
            ts = str(r.get("published_utc") or "")
            if ts:
                if not max_pub or ts > max_pub:
                    max_pub = ts
                if not min_pub or ts < min_pub:
                    min_pub = ts
    # tickers_with above counted pages with results, not tickers. recompute.
    ticker_hits = 0
    n_complete = 0
    n_articles_sum = 0
    for marker in (RAW_ROOT / "raw").glob("*/2026/_complete.json"):
        n_complete += 1
        rec = json.loads(marker.read_text(encoding="utf-8"))
        n_articles_sum += int(rec.get("n_articles") or 0)
        if int(rec.get("n_articles") or 0) > 0:
            ticker_hits += 1
    return {
        "n_complete_windows": n_complete,
        "requests_pages": requests,
        "unique_articles": len(ids),
        "ticker_article_links": links,
        "n_articles_sum_over_windows": n_articles_sum,
        "tickers_with_ge1": ticker_hits,
        "n_publishers": len(pubs),
        "published_utc_min": min_pub,
        "published_utc_max": max_pub,
        "coverage_tickers_with_news": ticker_hits / n_complete if n_complete else 0.0,
    }


def probe_indices(client: MassiveClient, asof: str) -> dict[str, Any]:
    """Small entitlement tests. Index schema from Massive docs (I: prefix) and existing I:SPX probe."""
    probes = [
        ("vix_ticker_overview", f"/v3/reference/tickers/{INDEX_VIX}", {}),
        ("vix_daily_sample", f"/v2/aggs/ticker/{INDEX_VIX}/range/1/day/2024-01-02/2024-01-10", {"limit": 5, "sort": "asc"}),
        ("spx_daily_sample", f"/v2/aggs/ticker/{INDEX_SPX}/range/1/day/2024-01-02/2024-01-10", {"limit": 5, "sort": "asc"}),
        ("spy_daily_sample", "/v2/aggs/ticker/SPY/range/1/day/2024-01-02/2024-01-10", {"adjusted": "true", "limit": 5, "sort": "asc"}),
    ]
    rows = []
    for name, path, params in probes:
        probe = client.probe(name, path, params)
        rows.append(
            {
                "name": name,
                "path": path,
                "http_status": probe.status_code,
                "accessible": probe.accessible,
                "plan_required": probe.plan_required,
                "error": probe.error,
                "n_results": probe.n_results,
                "sample_keys": probe.sample_keys,
                "excerpt": probe.body_excerpt[:400],
            }
        )
        print(f"probe {name}: status={probe.status_code} accessible={probe.accessible} err={probe.error!r}", flush=True)
    vix_ok = any(r["name"] == "vix_daily_sample" and r["accessible"] for r in rows)
    spx_ok = any(r["name"] == "spx_daily_sample" and r["accessible"] for r in rows)
    vix_row = next(r for r in rows if r["name"] == "vix_daily_sample")
    return {
        "asof": asof,
        "schema": "Massive indices ticker prefix I: (docs sample I:NDX); project already probes I:SPX",
        "MASSIVE_VIX_ACCESS": "YES" if vix_ok else "NO",
        "MASSIVE_SPX_ACCESS": "YES" if spx_ok else "NO",
        "vix_http": vix_row["http_status"],
        "vix_error": vix_row["error"] or vix_row["plan_required"],
        "probes": rows,
    }


def archive_vix(client: MassiveClient, asof: str) -> dict[str, Any]:
    start = NEWS_HIST_START
    end = asof
    path = f"/v2/aggs/ticker/{INDEX_VIX}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "limit": 50000, "sort": "asc"}
    out_dir = VIX_RAW / "I_VIX"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = 0
    rows: list[dict[str, Any]] = []
    status, payload, _ = client.get(path, params)
    while isinstance(payload, dict):
        raw_bytes = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        (out_dir / f"page_{pages:04d}.json").write_bytes(raw_bytes)
        pages += 1
        results = payload.get("results") or []
        if isinstance(results, list):
            rows.extend([r for r in results if isinstance(r, dict)])
        if status != 200:
            break
        next_url = payload.get("next_url")
        if not next_url:
            break
        status, payload, _ = client.get(str(next_url), params=None)
    recs = []
    for r in rows:
        recs.append(
            {
                "ticker": INDEX_VIX,
                "date": str(_ms_to_date(int(r["t"])).date()) if r.get("t") is not None else "",
                "open": r.get("o"),
                "high": r.get("h"),
                "low": r.get("l"),
                "close": r.get("c"),
                "volume": r.get("v"),
                "vwap": r.get("vw"),
                "n": r.get("n"),
                "t": r.get("t"),
            }
        )
    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.drop_duplicates(["ticker", "date"]).sort_values("date")
        df.to_csv(out_dir / "vix_daily.csv", index=False)
    marker = {
        "ticker": INDEX_VIX,
        "endpoint": path,
        "period": [start, end],
        "retrieved_at": _utc(),
        "http_status": status,
        "pages": pages,
        "n_bars": int(len(df)),
        "date_min": df["date"].min() if len(df) else None,
        "date_max": df["date"].max() if len(df) else None,
        "fields": ["date", "open", "high", "low", "close", "volume", "source metadata"],
        "processed_as_signal": False,
    }
    (out_dir / "_complete.json").write_text(json.dumps(marker, indent=2) + "\n")
    (VIX_RAW / "vix_download_manifest.json").write_text(json.dumps(marker, indent=2) + "\n")
    print(f"VIX archived bars={marker['n_bars']} {marker['date_min']}..{marker['date_max']}", flush=True)
    return marker


def inventory_existing() -> list[dict[str, Any]]:
    rows = []

    def add(name: str, path: Path, date_range: str, use: str) -> None:
        size, n = _du(path)
        rows.append(
            {
                "dataset": name,
                "path": str(path.relative_to(PROJECT_ROOT)) if path.exists() else str(path),
                "exists": path.exists(),
                "date_range": date_range,
                "approx_size": _gb(size),
                "n_files": n,
                "research_use": use,
            }
        )

    add(
        "news (full PIT S&P500)",
        RAW_ROOT,
        "2018-01-01 .. 2025-12-31 (2026 YTD added this run)",
        "External-information / attention diagnostics. Not in frozen H0. Vendor coverage thin before 2021-04.",
    )
    add(
        "daily prices (2026 reconstruction universe)",
        PROJECT_ROOT / "data/massive_2026/raw",
        "2025-01-01 .. 2026-08-18 (stale vs today)",
        "Return-splice candidate for 2026 D2; PASS_RETURNS_SPLICABLE. Not written into Qlib bins.",
    )
    add(
        "normalized daily OHLCV + SPY",
        PROJECT_ROOT / "data/massive_2026/normalized",
        "same as raw prices",
        "Split-adjusted Massive bars; not Qlib first-day normalize.",
    )
    add(
        "splits / dividends",
        PROJECT_ROOT / "data/massive_2026",
        "from 2025-01-01 (universe extracts)",
        "Corporate-action repair for 2026 price download. CRSP remains H0 source through 2025.",
    )
    add(
        "ticker reference sample",
        PROJECT_ROOT / "data/massive_2026/raw/tickers_active_sample.csv",
        "probe snapshot 2026-08-18",
        "Identifier audit only; not a full active+inactive dump.",
    )
    add(
        "statements / ratios / EDGAR sample",
        PROJECT_ROOT / "data/massive_2026/fundamentals",
        "30-name diagnostic",
        "F1C equivalence audit. Verdict NOT_EQUIVALENT (restated filing_date). Not a Compustat replacement.",
    )
    add(
        "endpoint inventory / audit tables",
        PROJECT_ROOT / "data/massive_audit",
        "2026-08-18 probes",
        "Plan entitlement map. I:SPX and ETF constituents were 403.",
    )
    add(
        "2026 PIT universe reconstruction",
        PROJECT_ROOT / "data/massive_2026/universe",
        "2025-12-31 snapshot + 13 2026 events through 2026-08-18",
        "Not Massive official index history. Needed for 2026 membership; Massive cannot certify S&P500.",
    )
    return rows


def write_report(payload: dict[str, Any]) -> None:
    news_hist = payload["news_hist"]
    news_ytd = payload["news_ytd"]
    ytd_cov = payload["news_ytd_coverage"]
    idx = payload["indices"]
    vix = payload.get("vix_archive")
    inv = payload["inventory"]
    recs = payload["regret_recommendations"]
    hist_ok = "YES" if news_hist["hist_complete"] else "NO"
    ytd_ok = "YES" if news_ytd.get("fail", 1) == 0 and news_ytd.get("n_complete_2026", 0) == news_ytd.get("n_jobs") else "NO"
    vix_access = idx["MASSIVE_VIX_ACCESS"]
    if vix_access == "NO":
        vix_archived = "NO (not entitled; not downloaded)"
    elif vix and vix.get("n_bars"):
        vix_archived = f"YES ({vix.get('date_min')} .. {vix.get('date_max')}, {vix.get('n_bars')} bars)"
    else:
        vix_archived = "NO"
    lines = [
        "# Massive Subscription Final-Day Archive",
        "",
        f"Generated: `{payload['generated']}`",
        "**Status: `DATA ARCHIVE / NOT AN EXPERIMENT`**",
        "",
        "未训练模型。未跑组合。未改 D2 / W1 / P5。未批量下载 minute / trade / quote / options。",
        "",
        "---",
        "",
        "## Answers",
        "",
        f"**A. 2018–2025 news 是否完整？** `{hist_ok}`",
        f"**B. 2026 YTD news 是否已补齐？** `{ytd_ok}`",
        f"**C. 是否可以访问 VIX？** `MASSIVE_VIX_ACCESS = {vix_access}`",
        f"**D. 如果可以，VIX 是否已完整归档？** `{vix_archived}`",
        f"**E. 订阅失效后会明显后悔的小型高价值数据？** 见第 6 节。推荐 {len(recs)} 项；未下载 minute/trade/quote/options。",
        "",
        "---",
        "",
        "## 1. Existing 2018–2025 news archive",
        "",
        f"- Path: `data/sentiment_raw/massive_news/`",
        f"- Manifest period: `{news_hist.get('manifest_period')}`",
        f"- Windows: **{news_hist.get('n_complete_markers_hist')}/{news_hist.get('n_jobs')}** complete",
        f"- Failures: **{news_hist.get('fail')}**",
        f"- Manifest present: `{news_hist.get('manifest_present')}`",
        f"- Size: { _gb(news_hist['size_bytes']) } / {news_hist['n_files']} files",
        f"- Complete by year: `{news_hist.get('complete_by_year')}`",
        "",
        "未重复下载 2018–2025。",
        "",
        "Vendor 内容缺口（已有研究，不是下载失败）：2018–2020 unique 文章约 13 / 39 / 176；覆盖在 2021-04 才打开。",
        "",
        "## 2. 2026 YTD news",
        "",
        "与 2018–2025 档案相同：`/v2/reference/news`，`limit=1000`，`next_url` 分页，raw JSON 按 `raw/{ticker}/{year}/page_*.json` 保存，checkpoint 只追加 `TICKER|2026`。未覆盖历史年份目录。",
        "",
        f"- PIT: {payload['jobs_meta']['pit_rule']}",
        f"- Jobs: {news_ytd.get('n_jobs')}",
        f"- Complete 2026 windows: {news_ytd.get('n_complete_2026')}",
        f"- This-run ok/fail: {news_ytd.get('ok')}/{news_ytd.get('fail')}",
        f"- API calls: {news_ytd.get('api_calls')} (429 hits: {news_ytd.get('rate_limit_hits')})",
        f"- Pages (requests): {ytd_cov.get('requests_pages')}",
        f"- Unique articles: {ytd_cov.get('unique_articles')}",
        f"- Ticker-article links: {ytd_cov.get('ticker_article_links')}",
        f"- Tickers with ≥1 article: {ytd_cov.get('tickers_with_ge1')} / {ytd_cov.get('n_complete_windows')}",
        f"- Coverage (tickers with news): {ytd_cov.get('coverage_tickers_with_news')}",
        f"- published_utc range: {ytd_cov.get('published_utc_min')} .. {ytd_cov.get('published_utc_max')}",
        f"- Final calendar date used as as-of: `{payload['asof']}`",
        "",
    ]
    if news_ytd.get("failures"):
        lines.append("Failures:")
        for row in news_ytd["failures"]:
            lines.append(f"- `{row['ticker']}` {row['year']}: {row['error']}")
        lines.append("")
    lines += [
        "## 3. VIX / indices entitlement",
        "",
        "少量 API test。Ticker schema 来自 Massive indices docs（`I:` 前缀，样例 `I:NDX`）以及本仓库已有 `I:SPX` probe，未猜测无文档代码。",
        "",
        f"- `MASSIVE_VIX_ACCESS = {vix_access}`",
        f"- `MASSIVE_SPX_ACCESS = {idx['MASSIVE_SPX_ACCESS']}`",
        f"- VIX HTTP: {idx.get('vix_http')} error/plan: `{idx.get('vix_error')}`",
        "",
        "| probe | HTTP | accessible | error |",
        "| --- | ---: | --- | --- |",
    ]
    for row in idx["probes"]:
        lines.append(
            f"| `{row['name']}` | {row['http_status']} | {row['accessible']} | {(row['error'] or row['plan_required'] or '')[:80].replace('|','/')} |"
        )
    lines.append("")
    if vix_access == "YES" and vix:
        lines += [
            "VIX daily raw archive (not a signal):",
            "",
            f"- Path: `data/sentiment_raw/massive_indices/I_VIX/`",
            f"- Period requested: {vix.get('period')}",
            f"- Bars: {vix.get('n_bars')} from {vix.get('date_min')} to {vix.get('date_max')}",
            f"- Pages: {vix.get('pages')}",
            "- Fields kept: date, open/high/low/close, volume if present, source metadata",
            "",
        ]
    else:
        lines += [
            "VIX 未下载。记录 entitlement error 后停止该项。",
            "",
        ]
    lines += [
        "## 4. Bulk download not performed",
        "",
        "明确未下载：all-market minute bars、full trades、full quotes、options chain archive、entire-market tick data。",
        "冻结 H0 不依赖这些序列。现有 experiment 也没有已批准的 bulk tick 依赖。",
        "",
        "## 5. Existing Massive datasets",
        "",
        "| dataset | path | date range | approx size | research use |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in inv:
        lines.append(
            f"| {row['dataset']} | `{row['path']}` | {row['date_range']} | {row['approx_size']} | {row['research_use']} |"
        )
    lines += [
        "",
        "未重新下载已经完整的 2018–2025 新闻、2026-08-18 截止的日线、以及 30 名基本面诊断样本。",
        "",
        "## 6. E — small high-value leftovers",
        "",
        "只推荐有明确研究用途、且现有档案缺口可见的小型数据。不是新实验候选。",
        "",
    ]
    for i, rec in enumerate(recs, 1):
        lines.append(f"{i}. **{rec['item']}** — {rec['why']} 用途：{rec['research_use']} 未执行：{rec['not_done_because']}")
        lines.append("")
    lines += [
        "---",
        "",
        f"**A = {hist_ok}**  **B = {ytd_ok}**  **C = {vix_access}**  **D = {vix_archived}**",
        "",
        "STOP。",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # keep a copy next to other massive audit reports for discoverability
    (REPORT_DIR / "massive_final_subscription_archive.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def regret_list(asof: str, idx: dict[str, Any]) -> list[dict[str, str]]:
    recs = [
        {
            "item": "2026 daily OHLCV top-up (2026-08-19 through today)",
            "why": f"现有 `data/massive_2026` 日线停在 2026-08-18，今天是 {asof}。这是已批准的 2026 return-splice 档案缺口，大约几个交易日 × ~480 只股票，不是全市场 minute。",
            "research_use": "2026 D2 价格拼接 continuity（已有 verdict PASS_RETURNS_SPLICABLE）。",
            "not_done_because": "章程第 5 条禁止重下已有完整数据；第 6E 只允许推荐。未在本 run 自动扩样。",
        },
        {
            "item": "Full ticker reference snapshot (active + inactive)",
            "why": "现有只是 `tickers_active_sample.csv`。订阅失效后 ticker 更名/退市对照会变贵。体量小。",
            "research_use": "2026 新增成分（CIEN/VRT/…）与 35 个 snapshot 无 ticker PERMNO 的标识映射。",
            "not_done_because": "没有已冻结实验立刻依赖全量 reference dump；本 run 只推荐。",
        },
    ]
    if idx["MASSIVE_VIX_ACCESS"] == "NO":
        recs.append(
            {
                "item": "VIX / I:SPX 官方指数日线",
                "why": "本 plan 不能读 indices。HMM/防守研究如果以后需要 VIX，只能改用 WRDS/CBOE 或其他源。",
                "research_use": "曾被明确排除在 D3/emergence 之外；不是当前冻结 H0 输入。",
                "not_done_because": "entitlement NO，已停止。不要用 VIXY/VXX 冒充 VIX。",
            }
        )
    recs.append(
        {
            "item": "Do not archive minute / trades / quotes / options",
            "why": "体量大、H0 不用、没有已批准实验依赖。",
            "research_use": "无。",
            "not_done_because": "章程第 4 条禁止。",
        }
    )
    return recs


def main() -> None:
    ensure_dirs()
    asof = _today()
    print("=== 1. audit 2018-2025 news ===", flush=True)
    news_hist = audit_existing_news()
    print(json.dumps({k: news_hist[k] for k in ("hist_complete", "n_complete_markers_hist", "n_jobs", "fail")}, indent=2), flush=True)
    if not news_hist["hist_complete"]:
        print("WARNING: 2018-2025 news archive is not 4264/4264 zero-fail", flush=True)

    print("=== 2. 2026 YTD news ===", flush=True)
    jobs, jobs_meta = build_2026_jobs(asof)
    news_ytd = download_2026_news(jobs)
    news_ytd["n_jobs"] = len(jobs)
    ytd_cov = summarize_2026_news()

    print("=== 3. VIX / index entitlement ===", flush=True)
    client = MassiveClient(min_interval_s=0.2)
    idx = probe_indices(client, asof)
    vix_archive = None
    if idx["MASSIVE_VIX_ACCESS"] == "YES":
        vix_archive = archive_vix(client, asof)
    else:
        (VIX_RAW).mkdir(parents=True, exist_ok=True)
        (VIX_RAW / "vix_entitlement_error.json").write_text(json.dumps(idx, indent=2) + "\n", encoding="utf-8")

    print("=== 5. inventory ===", flush=True)
    inv = inventory_existing()
    recs = regret_list(asof, idx)
    payload = {
        "generated": _utc(),
        "asof": asof,
        "news_hist": news_hist,
        "jobs_meta": jobs_meta,
        "news_ytd": news_ytd,
        "news_ytd_coverage": ytd_cov,
        "indices": idx,
        "vix_archive": vix_archive,
        "inventory": inv,
        "regret_recommendations": recs,
        "not_an_experiment": True,
        "d2_w1_p5_unchanged": True,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    write_report(payload)
    print("WROTE", REPORT_PATH, flush=True)
    print(
        "A",
        "YES" if news_hist["hist_complete"] else "NO",
        "B",
        "YES" if news_ytd.get("fail") == 0 and news_ytd.get("n_complete_2026") == len(jobs) else "NO",
        "C",
        idx["MASSIVE_VIX_ACCESS"],
        flush=True,
    )


if __name__ == "__main__":
    main()
