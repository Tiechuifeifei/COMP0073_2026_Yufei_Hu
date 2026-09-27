#!/usr/bin/env python3
"""Massive final-day archive phase 2: 2026 daily top-up and full ticker reference."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from massive_audit.client import MassiveClient  # noqa: E402
from massive_audit.config import (  # noqa: E402
    AUDIT_DIR,
    MANIFEST_2026,
    NORM_2026,
    RAW_2026,
    UNIVERSE_2026,
    ensure_dirs,
)
from massive_audit.phase_c import _ms_to_date  # noqa: E402
from massive_audit.phase_f import TICKER_ALIASES, _bars_to_df, _universe_tickers  # noqa: E402
from sentiment_experiments.massive_news_2018_2025.download import FALLBACK_TICKER  # noqa: E402

EXTEND_FROM = pd.Timestamp("2026-08-19")
REF_ROOT = PROJECT_ROOT / "data/reference/massive_ticker_reference"
REPORT_PATH = PROJECT_ROOT / "reports/data_audits/massive_final_subscription_archive.md"
PHASE2_SUMMARY = AUDIT_DIR / "final_day_archive_phase2_summary.json"
YEARLY = PROJECT_ROOT / "data/portfolio_experiments/benchmark_leadership_capture/benchmark_contributors_yearly.csv"
MISSING_PERMNO = [
    11990, 12591, 13628, 14617, 15315, 15826, 16309, 16655, 16736, 18267,
    18726, 18911, 19788, 20189, 20391, 20583, 20892, 20894, 21723, 24294,
    24877, 24878, 25146, 26181, 27083, 27598, 79686, 82486, 82694, 83011,
    85059, 87034, 87356, 91907, 92043,
]


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def probe_latest_session(client: MassiveClient) -> dict[str, Any]:
    today = datetime.now().date().isoformat()
    start = EXTEND_FROM.date().isoformat()
    status, payload, _ = client.get(
        f"/v2/aggs/ticker/SPY/range/1/day/{start}/{today}",
        {"adjusted": "true", "limit": 50, "sort": "asc"},
    )
    dates = []
    if status == 200 and isinstance(payload, dict):
        for row in payload.get("results") or []:
            if isinstance(row, dict) and row.get("t") is not None:
                dates.append(_ms_to_date(int(row["t"])))
    latest = max(dates).date().isoformat() if dates else None
    out = {
        "http_status": status,
        "request_start": start,
        "request_end": today,
        "n_spy_new_sessions": len(dates),
        "new_session_dates": [d.date().isoformat() for d in dates],
        "latest_available_trading_date": latest,
    }
    print(f"latest SPY session={latest} n={len(dates)} http={status}", flush=True)
    if not latest:
        raise RuntimeError(f"could not determine latest Massive session: HTTP {status}")
    return out


def extend_2026_daily(client: MassiveClient, latest: str) -> dict[str, Any]:
    ensure_dirs()
    tickers = _universe_tickers()
    combined_path = NORM_2026 / "daily_ohlcv_split_adjusted.parquet"
    before = pd.read_parquet(combined_path)
    before["date"] = pd.to_datetime(before["date"])
    old_max = before["date"].max()
    old_n = int(len(before))
    old_tickers = int(before["ticker"].nunique())
    spy_old = before[(before["ticker"] == "SPY") & (before["date"] == pd.Timestamp("2026-08-18"))]
    spy_close_0818 = float(spy_old["close"].iloc[0]) if len(spy_old) else None
    cutoff_n = int((before["date"] <= pd.Timestamp("2026-08-18")).sum())

    ck_path = MANIFEST_2026 / "daily_extend_checkpoint.json"
    ck = _load_json(ck_path)
    if ck.get("extend_from") != EXTEND_FROM.date().isoformat() or ck.get("extend_to") != latest:
        ck = {
            "extend_from": EXTEND_FROM.date().isoformat(),
            "extend_to": latest,
            "completed": [],
            "empty": [],
            "failed": [],
            "updated_at": _utc(),
        }
    done = set(ck.get("completed") or [])
    empty = set(ck.get("empty") or [])
    failed: list[dict[str, Any]] = list(ck.get("failed") or [])
    log_path = MANIFEST_2026 / "daily_extend_request_log.jsonl"
    start = EXTEND_FROM.date().isoformat()
    added_rows = 0
    pending = [t for t in tickers if t not in done]
    print(f"daily extend {start}..{latest} tickers={len(tickers)} pending={len(pending)}", flush=True)

    for i, ticker in enumerate(pending, 1):
        path = RAW_2026 / f"{ticker}_daily.parquet"
        try:
            status, payload, _ = client.get(
                f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{latest}",
                {"adjusted": "true", "limit": 50000, "sort": "asc"},
            )
            rows = payload.get("results") if isinstance(payload, dict) else None
            n = len(rows) if isinstance(rows, list) else 0
            rec = {"ts": _utc(), "ticker": ticker, "http": status, "n_results": n}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec) + "\n")
            if status != 200:
                failed.append({"ticker": ticker, "http": status, "error": (payload or {}).get("error") if isinstance(payload, dict) else "bad"})
                print(f"FAIL {ticker} HTTP {status}", flush=True)
                continue
            new = _bars_to_df(rows or [], ticker)
            if path.exists():
                old = pd.read_parquet(path)
                old["date"] = pd.to_datetime(old["date"])
            else:
                old = pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "n_trades"])
            if new.empty:
                empty.add(ticker)
                done.add(ticker)
            else:
                new["date"] = pd.to_datetime(new["date"])
                new = new[new["date"] >= EXTEND_FROM]
                merged = pd.concat([old, new], ignore_index=True)
                before_n = len(merged)
                merged = merged.drop_duplicates(["ticker", "date"], keep="first").sort_values("date")
                added_rows += int(len(merged) - len(old))
                merged.to_parquet(path, index=False)
                rec["duplicates_dropped"] = int(before_n - len(merged))
                done.add(ticker)
            if i % 40 == 0 or i == len(pending):
                ck.update({"completed": sorted(done), "empty": sorted(empty), "failed": failed, "updated_at": _utc()})
                ck_path.write_text(json.dumps(ck, indent=2) + "\n", encoding="utf-8")
                print(f"[{i}/{len(pending)}] {ticker} new_rows={n} done={len(done)} empty={len(empty)} fail={len(failed)}", flush=True)
        except Exception as exc:
            failed.append({"ticker": ticker, "error": str(exc)})
            print(f"FAIL {ticker}: {exc}", flush=True)

    ck.update({"completed": sorted(done), "empty": sorted(empty), "failed": failed, "updated_at": _utc()})
    ck_path.write_text(json.dumps(ck, indent=2) + "\n", encoding="utf-8")

    frames = []
    for ticker in tickers:
        path = RAW_2026 / f"{ticker}_daily.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined["date"] = pd.to_datetime(combined["date"])
        n_before_dedup = len(combined)
        combined = combined.drop_duplicates(["ticker", "date"], keep="first").sort_values(["ticker", "date"])
        n_dup = int(n_before_dedup - len(combined))
        combined["ret"] = combined.groupby("ticker", sort=False)["close"].pct_change()
        combined.to_parquet(combined_path, index=False)
        spy = combined[combined["ticker"] == "SPY"].copy()
        spy.to_parquet(NORM_2026 / "spy_daily.parquet", index=False)
    else:
        n_dup = 0

    spy_after = combined[(combined["ticker"] == "SPY") & (combined["date"] == pd.Timestamp("2026-08-18"))]
    spy_close_after = float(spy_after["close"].iloc[0]) if len(spy_after) else None
    cutoff_after = int((combined["date"] <= pd.Timestamp("2026-08-18")).sum())
    new_max = combined["date"].max() if not combined.empty else None
    added = combined[combined["date"] >= EXTEND_FROM] if not combined.empty else combined
    spy_new = combined[(combined["ticker"] == "SPY") & (combined["date"] >= EXTEND_FROM)]
    spy_dates = set(spy_new["date"]) if len(spy_new) else set()
    missing_on_spy_dates = []
    if spy_dates:
        for ticker, g in combined.groupby("ticker"):
            if ticker == "SPY":
                continue
            have = set(g.loc[g["date"] >= EXTEND_FROM, "date"])
            miss = sorted(d.date().isoformat() for d in spy_dates - have)
            if miss:
                prior_max = g.loc[g["date"] < EXTEND_FROM, "date"].max() if (g["date"] < EXTEND_FROM).any() else None
                missing_on_spy_dates.append(
                    {
                        "ticker": ticker,
                        "missing_dates": miss,
                        "prior_max": None if prior_max is None else str(pd.Timestamp(prior_max).date()),
                    }
                )
    null_px = int(combined[["open", "high", "low", "close"]].isna().any(axis=1).sum()) if not combined.empty else 0
    aapl = combined[combined["ticker"] == "AAPL"]
    sanity = {
        "pre_cutoff_rowcount_unchanged": cutoff_after == cutoff_n,
        "spy_2026_08_18_close_unchanged": spy_close_0818 is not None and spy_close_after == spy_close_0818,
        "spy_2026_08_18_close": spy_close_after,
        "null_ohlc_rows": null_px,
        "aapl_new_dates": [d.date().isoformat() for d in aapl.loc[aapl["date"] >= EXTEND_FROM, "date"]],
        "spy_new_dates": [d.date().isoformat() for d in spy_new["date"]] if len(spy_new) else [],
        "volume_not_written_to_wrds": True,
        "wrds_crsp_untouched": True,
    }
    out = {
        "old_max_date": str(old_max.date()),
        "new_max_date": None if new_max is None else str(pd.Timestamp(new_max).date()),
        "old_n_rows": old_n,
        "new_n_rows": int(len(combined)),
        "rows_added": int(len(added)),
        "rows_added_vs_old": int(len(combined) - old_n),
        "tickers_requested": len(tickers),
        "tickers_covered": int(combined["ticker"].nunique()) if not combined.empty else 0,
        "old_tickers": old_tickers,
        "failed_tickers": failed,
        "empty_new_window": sorted(empty),
        "n_empty_new_window": len(empty),
        "duplicates_dropped_on_rebuild": n_dup,
        "missing_vs_spy_new_sessions": missing_on_spy_dates,
        "n_tickers_missing_some_new_spy_dates": len(missing_on_spy_dates),
        "spy_coverage": {
            "n_new_sessions": int(len(spy_new)),
            "dates": [d.date().isoformat() for d in spy_new["date"]] if len(spy_new) else [],
            "min": str(spy_new["date"].min().date()) if len(spy_new) else None,
            "max": str(spy_new["date"].max().date()) if len(spy_new) else None,
        },
        "sanity": sanity,
        "extend_from": start,
        "extend_to": latest,
        "api_calls": client.calls,
        "rate_limit_hits": client.rate_limit_hits,
    }
    (MANIFEST_2026 / "daily_extend_manifest.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(
        f"daily done old_max={out['old_max_date']} new_max={out['new_max_date']} "
        f"rows+={out['rows_added_vs_old']} fail={len(failed)}",
        flush=True,
    )
    return out


def download_ticker_reference(client: MassiveClient) -> dict[str, Any]:
    REF_ROOT.mkdir(parents=True, exist_ok=True)
    raw_dir = REF_ROOT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_path = REF_ROOT / "_state.json"
    state = _load_json(state_path)
    streams = [
        ("active", {"market": "stocks", "active": "true", "limit": 1000, "sort": "ticker", "order": "asc"}),
        ("inactive", {"market": "stocks", "active": "false", "limit": 1000, "sort": "ticker", "order": "asc"}),
    ]
    all_rows: list[dict[str, Any]] = []
    summary_streams = []
    for label, params in streams:
        st = state.get(label) or {}
        if st.get("complete"):
            pages = sorted((raw_dir / label).glob("page_*.json"))
            for p in pages:
                payload = json.loads(p.read_text(encoding="utf-8"))
                all_rows.extend([r for r in (payload.get("results") or []) if isinstance(r, dict)])
            summary_streams.append({"label": label, "resumed_complete": True, "pages": len(pages), "n_rows": st.get("n_rows")})
            print(f"reference {label}: skip, already complete pages={len(pages)}", flush=True)
            continue
        out_dir = raw_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        pages = int(st.get("pages") or 0)
        next_url = st.get("next_url")
        n_rows = int(st.get("n_rows") or 0)
        if pages > 0:
            for p in sorted(out_dir.glob("page_*.json")):
                prev = json.loads(p.read_text(encoding="utf-8"))
                all_rows.extend([r for r in (prev.get("results") or []) if isinstance(r, dict)])
        status = 200
        payload: dict[str, Any] | list[Any] | None
        if next_url:
            status, payload, _ = client.get(str(next_url), params=None)
        else:
            status, payload, _ = client.get("/v3/reference/tickers", params)
        while isinstance(payload, dict) and status == 200:
            (out_dir / f"page_{pages:04d}.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n")
            results = payload.get("results") or []
            batch = [r for r in results if isinstance(r, dict)]
            all_rows.extend(batch)
            n_rows += len(batch)
            pages += 1
            next_url = payload.get("next_url")
            st = {"pages": pages, "n_rows": n_rows, "next_url": next_url, "complete": not bool(next_url), "updated_at": _utc()}
            state[label] = st
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            if pages % 10 == 0:
                print(f"reference {label} pages={pages} rows={n_rows}", flush=True)
            if not next_url:
                break
            status, payload, _ = client.get(str(next_url), params=None)
        if status != 200:
            raise RuntimeError(f"ticker reference {label} HTTP {status}: {payload}")
        summary_streams.append({"label": label, "pages": pages, "n_rows": n_rows, "http_last": status})
        print(f"reference {label} DONE pages={pages} rows={n_rows}", flush=True)

    df = pd.DataFrame(all_rows)
    keep = [
        "ticker", "name", "market", "locale", "type", "active", "currency_name", "currency_symbol",
        "cik", "composite_figi", "share_class_figi", "primary_exchange", "last_updated_utc", "delisted_utc",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    n_raw = len(df)
    df = df.drop_duplicates(subset=[c for c in ["ticker", "cik", "composite_figi", "active", "last_updated_utc"] if c in df.columns])
    df = df.sort_values(["ticker", "active"], ascending=[True, False])
    df.to_parquet(REF_ROOT / "massive_ticker_reference.parquet", index=False)
    df.to_csv(REF_ROOT / "massive_ticker_reference.csv", index=False)
    manifest = {
        "created_at": _utc(),
        "endpoint": "https://api.massive.com/v3/reference/tickers",
        "filters": [{"market": "stocks", "active": True}, {"market": "stocks", "active": False}],
        "pagination": "next_url until exhausted; limit=1000",
        "n_raw_rows": n_raw,
        "n_deduped_rows": int(len(df)),
        "n_active": int(df["active"].fillna(False).astype(bool).sum()) if len(df) else 0,
        "n_inactive": int((~df["active"].fillna(False).astype(bool)).sum()) if len(df) else 0,
        "n_unique_tickers": int(df["ticker"].nunique()) if len(df) else 0,
        "fields": keep,
        "streams": summary_streams,
        "artifacts": {
            "csv": "data/reference/massive_ticker_reference/massive_ticker_reference.csv",
            "parquet": "data/reference/massive_ticker_reference/massive_ticker_reference.parquet",
            "raw_pages": "data/reference/massive_ticker_reference/raw/",
        },
        "api_calls": client.calls,
        "rate_limit_hits": client.rate_limit_hits,
    }
    (REF_ROOT / "massive_ticker_reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"reference artifact rows={len(df)} active={manifest['n_active']} inactive={manifest['n_inactive']}", flush=True)
    return {"manifest": manifest, "frame": df}


def _lookup_massive(ref: pd.DataFrame, ticker: str) -> pd.DataFrame:
    t = str(ticker).upper()
    hit = ref[ref["ticker"].astype(str).str.upper() == t].copy()
    return hit


def classify_match(hit: pd.DataFrame, ticker: str) -> dict[str, Any]:
    if hit.empty:
        return {"ticker": ticker, "status": "NO_MASSIVE_ROW", "n": 0}
    us = hit
    if "locale" in hit.columns:
        us_only = hit[hit["locale"].astype(str).str.lower() == "us"]
        if len(us_only):
            us = us_only
    active = us[us["active"].fillna(False).astype(bool)] if "active" in us.columns else us.iloc[0:0]
    cs = active[active["type"].astype(str).str.upper().isin(["CS", "ADRC", "AD"])] if len(active) and "type" in active.columns else active
    pool = cs if len(cs) else active
    if len(pool) == 1:
        row = pool.iloc[0]
        return {
            "ticker": ticker,
            "status": "UNIQUE_ACTIVE",
            "n": int(len(hit)),
            "massive_ticker": row.get("ticker"),
            "name": row.get("name"),
            "type": row.get("type"),
            "active": bool(row.get("active")),
            "primary_exchange": row.get("primary_exchange"),
            "cik": row.get("cik"),
            "composite_figi": row.get("composite_figi"),
            "delisted_utc": row.get("delisted_utc"),
        }
    if len(pool) > 1:
        return {
            "ticker": ticker,
            "status": "AMBIGUOUS_ACTIVE",
            "n": int(len(pool)),
            "names": pool["name"].astype(str).head(5).tolist() if "name" in pool.columns else [],
            "types": pool["type"].astype(str).head(5).tolist() if "type" in pool.columns else [],
        }
    inactive = us[~us["active"].fillna(False).astype(bool)] if "active" in us.columns else us
    if len(inactive) == 1:
        row = inactive.iloc[0]
        return {
            "ticker": ticker,
            "status": "UNIQUE_INACTIVE",
            "n": int(len(hit)),
            "massive_ticker": row.get("ticker"),
            "name": row.get("name"),
            "type": row.get("type"),
            "delisted_utc": row.get("delisted_utc"),
            "cik": row.get("cik"),
        }
    if len(inactive) > 1:
        return {"ticker": ticker, "status": "AMBIGUOUS_INACTIVE", "n": int(len(inactive))}
    return {"ticker": ticker, "status": "UNCLASSIFIED", "n": int(len(hit))}


def mapping_audit(ref: pd.DataFrame) -> dict[str, Any]:
    snap = pd.read_csv(UNIVERSE_2026 / "sp500_2025_12_31_snapshot.csv")
    miss = snap[snap["ticker"].isna() | (snap["ticker"].astype(str).str.upper().isin(["NAN", ""]))]
    miss_permnos = [int(p) for p in miss["permno"].tolist()]
    yearly = pd.read_csv(YEARLY)
    yearly["permno"] = pd.to_numeric(yearly["permno"], errors="coerce")
    yearly = yearly.dropna(subset=["permno", "ticker"]).copy()
    yearly["permno"] = yearly["permno"].astype(int)
    if "period" in yearly.columns:
        yearly = yearly.sort_values("period")
    ymap = yearly.drop_duplicates("permno", keep="last").set_index("permno")["ticker"].astype(str).str.upper().to_dict()
    rows = []
    for permno in miss_permnos:
        cand_y = ymap.get(int(permno))
        cand_f = FALLBACK_TICKER.get(int(permno))
        sources = []
        ticker = None
        if cand_y:
            sources.append("benchmark_contributors_yearly")
            ticker = str(cand_y).upper()
        if cand_f:
            sources.append("news_archive_FALLBACK_TICKER")
            if ticker and ticker != str(cand_f).upper():
                rows.append(
                    {
                        "permno": permno,
                        "instrument": f"P{permno}",
                        "status": "AMBIGUOUS_SOURCES",
                        "yearly_ticker": cand_y,
                        "fallback_ticker": cand_f,
                        "sources": sources,
                    }
                )
                continue
            ticker = str(cand_f).upper()
        if not ticker:
            rows.append(
                {
                    "permno": permno,
                    "instrument": f"P{permno}",
                    "status": "NO_CANDIDATE_TICKER",
                    "sources": sources,
                }
            )
            continue
        match = classify_match(_lookup_massive(ref, ticker), ticker)
        rec = {
            "permno": permno,
            "instrument": f"P{permno}",
            "candidate_ticker": ticker,
            "sources": "|".join(sources),
            "yearly_ticker": cand_y,
            "fallback_ticker": cand_f,
            **{k: v for k, v in match.items() if k != "ticker"},
            "match_status": match["status"],
        }
        rec["resolved_for_2026_map"] = match["status"] == "UNIQUE_ACTIVE"
        rows.append(rec)

    df = pd.DataFrame(rows)
    n_res = int(df["resolved_for_2026_map"].fillna(False).sum()) if "resolved_for_2026_map" in df.columns else 0
    unresolved = df[~df["resolved_for_2026_map"].fillna(False)] if "resolved_for_2026_map" in df.columns else df
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT_DIR / "permno_ticker_mapping_audit_35.csv", index=False)
    summary = {
        "n_missing_in_s5a_snapshot": len(miss_permnos),
        "n_unique_active_resolved": n_res,
        "n_still_unresolved": int(len(unresolved)),
        "unresolved_permnos": unresolved["permno"].tolist() if len(unresolved) else [],
        "status_counts": df["match_status"].value_counts(dropna=False).to_dict() if "match_status" in df.columns else {},
        "note": (
            "Audit only. Does not change frozen S5A map, membership_spells, or 2026 price universe. "
            "Candidates come from yearly contributor tickers and the news-archive FALLBACK_TICKER; "
            "accepted only if Massive reference has a unique US active match. "
            "CRSP names history in this repo does not cover these 35 PERMNOs."
        ),
        "rows": rows,
    }
    (AUDIT_DIR / "permno_ticker_mapping_audit_35.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"mapping 35: resolved={n_res} unresolved={len(unresolved)}", flush=True)
    return summary


def write_report(daily: dict[str, Any], ref_man: dict[str, Any], mapping: dict[str, Any], latest: dict[str, Any]) -> None:
    daily_ok = (
        daily.get("failed_tickers") == []
        and daily.get("sanity", {}).get("pre_cutoff_rowcount_unchanged")
        and daily.get("sanity", {}).get("spy_2026_08_18_close_unchanged")
        and daily.get("new_max_date") == latest.get("latest_available_trading_date")
        and daily.get("spy_coverage", {}).get("n_new_sessions", 0) > 0
    )
    ref_ok = bool(ref_man.get("n_active")) and bool(ref_man.get("n_inactive")) and ref_man.get("n_deduped_rows", 0) > 0
    flags = {
        "2026_DAILY_ARCHIVE_COMPLETE": "YES" if daily_ok else "NO",
        "FULL_TICKER_REFERENCE_ARCHIVED": "YES" if ref_ok else "NO",
    }
    flags["MASSIVE_FINAL_ARCHIVE_COMPLETE"] = (
        "YES" if flags["2026_DAILY_ARCHIVE_COMPLETE"] == "YES" and flags["FULL_TICKER_REFERENCE_ARCHIVED"] == "YES" else "NO"
    )
    miss_lines = []
    for row in (daily.get("missing_vs_spy_new_sessions") or [])[:25]:
        miss_lines.append(f"- `{row['ticker']}` missing {row['missing_dates']} (prior max {row.get('prior_max')})")
    map_rows = mapping.get("rows") or []
    resolved = [r for r in map_rows if r.get("resolved_for_2026_map")]
    unresolved = [r for r in map_rows if not r.get("resolved_for_2026_map")]
    phase2 = [
        "",
        "---",
        "",
        "## Phase 2 (2026-08-25)",
        "",
        "数据归档补全，不是新实验。未改 D2/W1/P5。未下 minute/trade/quote/options/VIX/新闻历史。",
        "",
        f"**`2026_DAILY_ARCHIVE_COMPLETE = {flags['2026_DAILY_ARCHIVE_COMPLETE']}`**",
        f"**`FULL_TICKER_REFERENCE_ARCHIVED = {flags['FULL_TICKER_REFERENCE_ARCHIVED']}`**",
        f"**`MASSIVE_FINAL_ARCHIVE_COMPLETE = {flags['MASSIVE_FINAL_ARCHIVE_COMPLETE']}`**",
        "",
        "### 2.1 2026 daily extension",
        "",
        f"- Universe: same `_universe_tickers()` as 2026 return-splice ({daily.get('tickers_requested')} tickers, aliases BF.B/DAY/CPAY/DOC, plus SPY). Did **not** add the 35 S5A-missing PERMNOs to the price book.",
        f"- Old max date: `{daily.get('old_max_date')}`",
        f"- New max date: `{daily.get('new_max_date')}` (Massive latest SPY session `{latest.get('latest_available_trading_date')}`)",
        f"- Rows added (rebuild vs old combined): **{daily.get('rows_added_vs_old')}**",
        f"- Tickers covered: {daily.get('tickers_covered')} (was {daily.get('old_tickers')})",
        f"- Failed tickers: {daily.get('failed_tickers') or 'none'}",
        f"- Duplicates dropped on rebuild: {daily.get('duplicates_dropped_on_rebuild')}",
        f"- Empty in new window: {daily.get('n_empty_new_window')} `{daily.get('empty_new_window')}`",
        f"- Tickers missing some new SPY dates: {daily.get('n_tickers_missing_some_new_spy_dates')}",
        f"- SPY new sessions: {daily.get('spy_coverage')}",
        f"- Sanity: pre-cutoff rows unchanged `{daily.get('sanity', {}).get('pre_cutoff_rowcount_unchanged')}`; SPY 2026-08-18 close unchanged `{daily.get('sanity', {}).get('spy_2026_08_18_close_unchanged')}` (`{daily.get('sanity', {}).get('spy_2026_08_18_close')}`); null OHLC `{daily.get('sanity', {}).get('null_ohlc_rows')}`; WRDS/CRSP untouched.",
        "",
    ]
    if miss_lines:
        phase2 += ["Missing vs SPY new dates (truncated/delisted series, not filled backwards):", ""] + miss_lines + [""]
    phase2 += [
        "### 2.2 Full ticker reference",
        "",
        f"- Path: `data/reference/massive_ticker_reference/`",
        f"- Active rows: {ref_man.get('n_active')}",
        f"- Inactive rows: {ref_man.get('n_inactive')}",
        f"- Unique tickers: {ref_man.get('n_unique_tickers')}",
        f"- Deduped rows: {ref_man.get('n_deduped_rows')}",
        "- Artifacts: `massive_ticker_reference.csv`, `.parquet`, `massive_ticker_reference_manifest.json`, raw pages under `raw/active` and `raw/inactive`.",
        "",
        "### 2.3 Mapping audit (35 S5A-missing PERMNOs)",
        "",
        "只审计，不改冻结 panel / S5A map / 2026 价格宇宙。本仓库 CRSP names history 不含这 35 个 PERMNO。候选来自 yearly contributor ticker 与 news-archive `FALLBACK_TICKER`；仅当 Massive reference 有唯一 US active 匹配才算 resolved。",
        "",
        f"- Previously unmapped: **{mapping.get('n_missing_in_s5a_snapshot')}**",
        f"- Now unique-active resolved: **{mapping.get('n_unique_active_resolved')}**",
        f"- Still unresolved: **{mapping.get('n_still_unresolved')}** `{mapping.get('unresolved_permnos')}`",
        f"- Status counts: `{mapping.get('status_counts')}`",
        "",
        "Resolved (unique active):",
        "",
        "| permno | candidate | name | type | source |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for r in resolved:
        phase2.append(
            f"| {r.get('permno')} | `{r.get('candidate_ticker')}` | {str(r.get('name') or '')[:40]} | {r.get('type')} | {r.get('sources')} |"
        )
    phase2 += ["", "Unresolved / not forced:", ""]
    for r in unresolved:
        phase2.append(
            f"- P{r.get('permno')} candidate=`{r.get('candidate_ticker')}` status=`{r.get('match_status') or r.get('status')}` sources=`{r.get('sources')}`"
        )
    phase2 += [
        "",
        "### 2.4 Saved Massive datasets (local after expiry)",
        "",
        "| dataset | path | range | still locally usable |",
        "| --- | --- | --- | --- |",
        "| News PIT S&P500 | `data/sentiment_raw/massive_news/` | 2018-01-01 .. 2026-08-24 | YES — attention/news research |",
        f"| 2026 daily OHLCV + SPY | `data/massive_2026/` | 2025-01-01 .. {daily.get('new_max_date')} | YES — return-splice / 2026 price continuity |",
        "| Splits/dividends | `data/massive_2026/raw/` | from 2025-01-01 | YES — corp-action repair |",
        "| Statements/ratios/EDGAR sample | `data/massive_2026/fundamentals/` | 30-name diagnostic | YES as audit only; NOT Compustat PIT |",
        "| Ticker reference active+inactive | `data/reference/massive_ticker_reference/` | snapshot at archive time | YES — identifier mapping |",
        "| 2026 membership reconstruction | `data/massive_2026/universe/` | 2025-12-31 snapshot + events to 2026-08-18 | PARTIAL — later S&P adds/drops need S&P DJI |",
        "",
        "### 2.5 Still needs an external source after expiry",
        "",
        "- **VIX / I:SPX official index history** — Massive 403 NOT_AUTHORIZED. Use WRDS/CBOE/other. Do not use VIXY/VXX.",
        "- **Official S&P500 membership after 2026-08-18 events** — Massive cannot certify the index.",
        "- **Unrestated Compustat PIT / rdqe** — Massive fundamentals remain NOT_EQUIVALENT.",
        "- **CRSP names for the 35 snapshot PERMNOs** — not in this dump; mapping above is Massive-side confirmation of candidate tickers only.",
        "- **Minute / trades / quotes / options** — not archived (by design).",
        "",
        "订阅失效后仍可本地完成：已有新闻诊断复现、2026 日线 splicing 研究（到本档案 max date）、identifier 对照、既有 massive_audit 报告。不能本地完成：新的 Massive API 拉取、VIX、官方指数、冻结 F1C 等价基本面。",
        "",
        "STOP。",
        "",
    ]
    existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    if "## Phase 2" in existing:
        existing = existing.split("## Phase 2")[0].rstrip() + "\n"
    # refresh top-level final flags after E
    flag_block = (
        "\n## Final archive status (Phase 2)\n\n"
        f"**`2026_DAILY_ARCHIVE_COMPLETE = {flags['2026_DAILY_ARCHIVE_COMPLETE']}`**  \n"
        f"**`FULL_TICKER_REFERENCE_ARCHIVED = {flags['FULL_TICKER_REFERENCE_ARCHIVED']}`**  \n"
        f"**`MASSIVE_FINAL_ARCHIVE_COMPLETE = {flags['MASSIVE_FINAL_ARCHIVE_COMPLETE']}`**\n"
    )
    if "## Final archive status (Phase 2)" in existing:
        pre, rest = existing.split("## Final archive status (Phase 2)", 1)
        # drop old flag block until next ## or --- after it
        rest_lines = rest.splitlines()
        cut = 1
        while cut < len(rest_lines) and not rest_lines[cut].startswith("## ") and rest_lines[cut] != "---":
            cut += 1
        existing = pre.rstrip() + "\n" + flag_block + "\n" + "\n".join(rest_lines[cut:]).lstrip()
    else:
        # insert after Answers section's STOP/--- following E
        needle = "**E. 订阅失效后会明显后悔的小型高价值数据？"
        if needle in existing:
            idx = existing.find(needle)
            # find the --- after answers
            after = existing.find("\n---\n", idx)
            if after != -1:
                existing = existing[: after + 5] + flag_block + existing[after + 5 :]
            else:
                existing = existing + flag_block
        else:
            existing = existing + flag_block
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = existing.rstrip() + "\n" + "\n".join(phase2)
    REPORT_PATH.write_text(text + "\n", encoding="utf-8")
    (PROJECT_ROOT / "reports/massive_audit/massive_final_subscription_archive.md").write_text(text + "\n", encoding="utf-8")
    return flags


def main() -> None:
    ensure_dirs()
    client = MassiveClient(min_interval_s=0.12)
    print("=== probe latest session ===", flush=True)
    latest = probe_latest_session(client)
    print("=== extend 2026 daily ===", flush=True)
    daily = extend_2026_daily(client, latest["latest_available_trading_date"])
    print("=== full ticker reference ===", flush=True)
    ref_pack = download_ticker_reference(client)
    print("=== mapping audit ===", flush=True)
    mapping = mapping_audit(ref_pack["frame"])
    flags = write_report(daily, ref_pack["manifest"], mapping, latest)
    payload = {
        "generated": _utc(),
        "latest_session": latest,
        "daily": daily,
        "ticker_reference": ref_pack["manifest"],
        "mapping_audit": {k: v for k, v in mapping.items() if k != "rows"},
        "flags": flags,
        "not_an_experiment": True,
        "d2_w1_p5_unchanged": True,
        "did_not_download": ["minute", "trades", "quotes", "options", "VIX", "VIXY", "VXX", "SPX", "news history"],
    }
    PHASE2_SUMMARY.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("WROTE", REPORT_PATH, flags, flush=True)


if __name__ == "__main__":
    main()
