#!/usr/bin/env python3
"""Archive Massive statements / ratios / EDGAR index for 2026 replay preflight.
Diagnostic archive; does not build F1C scores or replace Compustat PIT."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from massive_audit.client import MassiveClient
from massive_audit.config import FUND_2026, NORM_2026, ensure_dirs

OUT = FUND_2026 / "universe_archive"
RAW = OUT / "raw"
PERIOD_GTE = "2024-01-01"
FILING_GTE = "2024-01-01"
ENDPOINTS = (
    ("income", "/stocks/financials/v1/income-statements", {"timeframe": "quarterly", "period_end.gte": PERIOD_GTE}),
    ("balance", "/stocks/financials/v1/balance-sheets", {"timeframe": "quarterly", "period_end.gte": PERIOD_GTE}),
    ("cashflow", "/stocks/financials/v1/cash-flow-statements", {"timeframe": "quarterly", "period_end.gte": PERIOD_GTE}),
    ("ratios", "/stocks/financials/v1/ratios", {"limit": 10}),
    ("filings", "/stocks/filings/vX/index", {"form_type.any_of": "10-Q,10-K", "filing_date.gte": FILING_GTE}),
)


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def universe_tickers() -> list[str]:
    p = NORM_2026 / "daily_ohlcv_split_adjusted.parquet"
    df = pd.read_parquet(p, columns=["ticker"])
    tickers = sorted(t for t in df["ticker"].dropna().unique().tolist() if t and t != "SPY")
    return tickers


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in row.items():
        if isinstance(v, (list, dict)):
            out[k] = json.dumps(v, default=str)
        else:
            out[k] = v
    return out


def load_ckpt() -> dict[str, Any]:
    path = OUT / "checkpoint.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"tickers": {}, "started": _utc()}


def save_ckpt(ckpt: dict[str, Any]) -> None:
    ckpt["updated"] = _utc()
    (OUT / "checkpoint.json").write_text(json.dumps(ckpt, indent=2) + "\n", encoding="utf-8")


def fetch_endpoint(client: MassiveClient, ticker: str, name: str, path: str, extra: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    dest = RAW / name / ticker
    dest.mkdir(parents=True, exist_ok=True)
    params = dict(extra)
    if name == "filings":
        params["ticker"] = ticker
        params.setdefault("sort", "filing_date.asc")
    elif name == "ratios":
        params["ticker"] = ticker
    else:
        params["tickers.any_of"] = ticker
        params.setdefault("sort", "period_end.asc")
    params.setdefault("limit", 100)
    rows = client.paginate(path, params, max_pages=6)
    used = path
    if name == "filings" and not rows:
        used = "/stocks/filings/v1/index"
        rows = client.paginate(used, params, max_pages=6)
    payload = {"ticker": ticker, "endpoint": used, "retrieved_at": _utc(), "n": len(rows), "results": rows}
    (dest / "response.json").write_text(json.dumps(payload, default=str) + "\n", encoding="utf-8")
    out = []
    for row in rows:
        rec = _jsonable(dict(row))
        rec["query_ticker"] = ticker
        rec["archive_endpoint"] = used
        out.append(rec)
    return out, used


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    tickers = universe_tickers()
    ckpt = load_ckpt()
    client = MassiveClient(min_interval_s=0.25)
    buckets: dict[str, list[dict[str, Any]]] = {k: [] for k, *_ in ENDPOINTS}
    print(f"archiving {len(tickers)} tickers", flush=True)
    n_ok = n_fail = 0
    for i, ticker in enumerate(tickers, 1):
        state = ckpt["tickers"].get(ticker, {})
        row_state = dict(state)
        failed = False
        for name, path, extra in ENDPOINTS:
            if state.get(name) == "ok":
                cached = RAW / name / ticker / "response.json"
                if cached.exists():
                    payload = json.loads(cached.read_text(encoding="utf-8"))
                    for rec in payload.get("results") or []:
                        item = _jsonable(dict(rec))
                        item["query_ticker"] = ticker
                        item["archive_endpoint"] = payload.get("endpoint", path)
                        buckets[name].append(item)
                    continue
            try:
                rows, used = fetch_endpoint(client, ticker, name, path, extra)
                buckets[name].extend(rows)
                row_state[name] = "ok"
                row_state[f"{name}_n"] = len(rows)
                row_state[f"{name}_path"] = used
            except Exception as exc:  # noqa: BLE001 — archive must continue
                row_state[name] = "fail"
                row_state[f"{name}_error"] = str(exc)[:400]
                failed = True
                print(f"FAIL {ticker} {name}: {exc}", flush=True)
        ckpt["tickers"][ticker] = row_state
        if failed:
            n_fail += 1
        else:
            n_ok += 1
        if i % 20 == 0 or i == len(tickers):
            save_ckpt(ckpt)
            print(f"  {i}/{len(tickers)} ok={n_ok} fail={n_fail} calls={client.calls} 429={client.rate_limit_hits}", flush=True)
    save_ckpt(ckpt)

    counts = {}
    for name, rows in buckets.items():
        df = pd.DataFrame(rows)
        counts[name] = {"n_rows": int(len(df)), "n_tickers": int(df["query_ticker"].nunique()) if len(df) and "query_ticker" in df.columns else 0}
        if len(df):
            df.to_csv(OUT / f"{name}_quarterly_or_index.csv", index=False)
            df.to_parquet(OUT / f"{name}_quarterly_or_index.parquet", index=False)
        else:
            (OUT / f"{name}_quarterly_or_index.csv").write_text("query_ticker\n", encoding="utf-8")

    empty = {
        name: sorted(t for t in tickers if ckpt["tickers"].get(t, {}).get(f"{name}_n", 0) == 0)
        for name, *_ in ENDPOINTS
    }
    manifest = {
        "status": "DATA ARCHIVE / 2026 DEPLOYMENT REPLAY PREFLIGHT",
        "diagnostic_only": True,
        "usable_for_frozen_f1c": False,
        "usable_for_approximate_f1c_inputs": True,
        "period_end_gte": PERIOD_GTE,
        "filing_date_gte": FILING_GTE,
        "n_tickers_requested": len(tickers),
        "n_tickers_ok": n_ok,
        "n_tickers_fail": n_fail,
        "api_calls": client.calls,
        "rate_limit_hits": client.rate_limit_hits,
        "counts": counts,
        "empty_tickers": {k: v[:40] + (["..."] if len(v) > 40 else []) for k, v in empty.items()},
        "empty_counts": {k: len(v) for k, v in empty.items()},
        "note": "Massive financials.filing_date is last restating filing, not rdqe. Ratios are latest-day snapshots. Not Compustat PIT.",
        "generated": _utc(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
