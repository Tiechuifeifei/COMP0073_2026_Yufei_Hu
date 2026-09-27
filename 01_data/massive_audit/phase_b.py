"""Phase B: probe Massive endpoints. No bulk download."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from massive_audit.client import MassiveClient, ProbeResult
from massive_audit.config import AUDIT_DIR, REPORT_DIR, ensure_dirs


PROBES = [
    ("daily_ohlcv_adjusted", "/v2/aggs/ticker/AAPL/range/1/day/2024-01-02/2024-01-10", {"adjusted": "true", "limit": 50, "sort": "asc"}),
    ("daily_ohlcv_unadjusted", "/v2/aggs/ticker/AAPL/range/1/day/2024-01-02/2024-01-10", {"adjusted": "false", "limit": 50, "sort": "asc"}),
    ("daily_ohlcv_history_depth", "/v2/aggs/ticker/AAPL/range/1/day/2003-01-02/2004-01-15", {"adjusted": "true", "limit": 20, "sort": "asc"}),
    ("daily_ohlcv_2026", "/v2/aggs/ticker/AAPL/range/1/day/2026-01-02/2026-08-14", {"adjusted": "true", "limit": 20, "sort": "desc"}),
    ("grouped_daily", "/v2/aggs/grouped/locale/us/market/stocks/2026-08-14", {"adjusted": "true"}),
    ("minute_bars", "/v2/aggs/ticker/AAPL/range/1/minute/2026-08-14/2026-08-14", {"adjusted": "true", "limit": 5, "sort": "asc"}),
    ("previous_day", "/v2/aggs/ticker/AAPL/prev", {"adjusted": "true"}),
    ("trades", "/v3/trades/AAPL", {"timestamp": "2026-08-14", "limit": 1}),
    ("quotes", "/v3/quotes/AAPL", {"timestamp": "2026-08-14", "limit": 1}),
    ("splits", "/v3/reference/splits", {"ticker": "AAPL", "limit": 10}),
    ("dividends", "/v3/reference/dividends", {"ticker": "AAPL", "limit": 10}),
    ("ticker_overview", "/v3/reference/tickers/AAPL", {"date": "2025-12-31"}),
    ("tickers_active", "/v3/reference/tickers", {"ticker": "AAPL", "active": "true", "limit": 1, "market": "stocks"}),
    ("tickers_inactive", "/v3/reference/tickers", {"search": "SIVB", "active": "false", "limit": 5, "market": "stocks"}),
    ("ticker_types", "/v3/reference/tickers/types", {"asset_class": "stocks"}),
    ("ticker_events", "/vX/reference/tickers/AAPL/events", {}),
    ("market_holidays", "/v1/marketstatus/upcoming", {}),
    ("market_status", "/v1/marketstatus/now", {}),
    ("income_statements", "/stocks/financials/v1/income-statements", {"tickers.any_of": "AAPL", "timeframe": "quarterly", "limit": 5, "sort": "period_end.desc"}),
    ("balance_sheets", "/stocks/financials/v1/balance-sheets", {"tickers.any_of": "AAPL", "timeframe": "quarterly", "limit": 5, "sort": "period_end.desc"}),
    ("cash_flow_statements", "/stocks/financials/v1/cash-flow-statements", {"tickers.any_of": "AAPL", "timeframe": "quarterly", "limit": 5, "sort": "period_end.desc"}),
    ("financial_ratios", "/stocks/financials/v1/ratios", {"ticker": "AAPL", "limit": 5}),
    ("sec_edgar_index_v1", "/stocks/filings/v1/index", {"ticker": "AAPL", "form_type.any_of": "10-Q,10-K", "limit": 5, "sort": "filing_date.desc"}),
    ("sec_edgar_index_vx", "/stocks/filings/vX/index", {"ticker": "AAPL", "limit": 5}),
    ("news", "/v2/reference/news", {"ticker": "AAPL", "limit": 5, "sort": "published_utc", "order": "desc"}),
    ("news_oldest", "/v2/reference/news", {"ticker": "AAPL", "limit": 1, "sort": "published_utc", "order": "asc"}),
    ("spy_etf_daily", "/v2/aggs/ticker/SPY/range/1/day/2026-01-02/2026-08-14", {"adjusted": "true", "limit": 5, "sort": "desc"}),
    ("index_spx", "/v2/aggs/ticker/I:SPX/range/1/day/2026-01-02/2026-08-14", {"adjusted": "true", "limit": 5, "sort": "desc"}),
    ("etf_constituents_spy", "/etf-global/v1/constituents", {"ticker": "SPY", "limit": 5}),
    ("short_interest", "/stocks/short-interest/v1", {"ticker": "AAPL", "limit": 1}),
]


def _hist_depth(client: MassiveClient, probe: ProbeResult) -> str:
    if not probe.accessible:
        return ""
    if "history_depth" in probe.endpoint:
        status, payload, _ = client.get(
            "/v2/aggs/ticker/AAPL/range/1/day/2003-01-02/2004-01-15",
            {"adjusted": "true", "limit": 20, "sort": "asc"},
        )
        if status == 200 and isinstance(payload, dict):
            results = payload.get("results") or []
            if results:
                ts = results[0].get("t")
                if ts:
                    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat()
        return f"HTTP {status}"
    if probe.endpoint == "news_oldest":
        status, payload, _ = client.get(
            "/v2/reference/news",
            {"ticker": "AAPL", "limit": 1, "sort": "published_utc", "order": "asc"},
        )
        if status == 200 and isinstance(payload, dict):
            results = payload.get("results") or []
            if results:
                return str(results[0].get("published_utc", ""))[:10]
    return ""


def run_endpoint_inventory() -> list[dict[str, str]]:
    ensure_dirs()
    client = MassiveClient()
    rows: list[dict[str, str]] = []
    for name, path, params in PROBES:
        probe = client.probe(name, path, params)
        depth = _hist_depth(client, probe) if name in {"daily_ohlcv_history_depth", "news_oldest"} else ""
        if name == "daily_ohlcv_history_depth" and probe.accessible:
            # reuse first result timestamp
            status, payload, _ = 200, None, None
            _, payload, _ = client.get(path, params)
            if isinstance(payload, dict) and payload.get("results"):
                ts = payload["results"][0].get("t")
                if ts:
                    depth = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat()
        rows.append(
            {
                "endpoint": name,
                "url": path,
                "accessible": "TRUE" if probe.accessible else "FALSE",
                "http_status": "" if probe.status_code is None else str(probe.status_code),
                "plan_required": probe.plan_required,
                "historical_depth": depth,
                "date_filter_supported": "TRUE" if any(k in path or k in str(params) for k in ("from", "to", "published_utc", "period_end", "filing_date", "timestamp", "date")) or "range" in path else "PARTIAL",
                "pagination": "TRUE" if probe.extra.get("next_url") else "UNKNOWN",
                "rate_limit": f"limit={probe.extra.get('x-ratelimit-limit','')} remaining={probe.extra.get('x-ratelimit-remaining','')} hits_429={client.rate_limit_hits}",
                "timestamp_precision": _timestamp_guess(name, probe),
                "important_fields": probe.sample_keys,
                "n_results_sample": str(probe.n_results),
                "notes": (probe.error or probe.body_excerpt)[:400].replace("\n", " "),
            }
        )
        print(f"B {name}: accessible={probe.accessible} status={probe.status_code} n={probe.n_results}")

    out = AUDIT_DIR / "massive_endpoint_inventory.csv"
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md = [ "# Massive endpoint inventory", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", "", "Probe-only; no bulk download.", "", "| endpoint | accessible | HTTP | plan/error | fields | notes |", "|---|---|---|---|---|---|" ]
    for row in rows:
        md.append(
            f"| `{row['endpoint']}` | {row['accessible']} | {row['http_status']} | {row['plan_required'][:80].replace('|','/')} | `{row['important_fields'][:60]}` | {row['notes'][:80].replace('|','/')} |"
        )
    (REPORT_DIR / "massive_endpoint_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (AUDIT_DIR / "endpoint_inventory_raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return rows


def _timestamp_guess(name: str, probe: ProbeResult) -> str:
    if "minute" in name or "trades" in name or "quotes" in name:
        return "nanosecond or millisecond (tick/minute)"
    if "news" in name:
        return "RFC3339 UTC second"
    if "daily" in name or "grouped" in name or "spy" in name or "index" in name:
        return "Unix ms bar start (ET session)"
    if "filing" in name or "income" in name or "balance" in name or "cash" in name:
        return "calendar date YYYY-MM-DD"
    return "unknown" if not probe.accessible else "see fields"
