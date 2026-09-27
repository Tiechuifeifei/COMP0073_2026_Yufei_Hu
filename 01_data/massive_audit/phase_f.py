"""Phase F: download 2025-01-01 through current market data if price audit allows splicing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from massive_audit.client import MassiveClient
from massive_audit.config import (
    AUDIT_DIR,
    LOOKBACK_START,
    MANIFEST_2026,
    NORM_2026,
    RAW_2026,
    UNIVERSE_2026,
    ensure_dirs,
)
from massive_audit.phase_c import _ms_to_date


# Stale WRDS/S5A query tickers that Massive no longer serves under the old symbol.
TICKER_ALIASES = {
    "BF": "BF.B",      # Brown-Forman Class B (S&P 500 listing)
    "CDAY": "DAY",     # Ceridian -> Dayforce
    "FLT": "CPAY",     # FleetCor -> Corpay
    "PEAK": "DOC",     # Healthpeak
}


def _universe_tickers() -> list[str]:
    snap = pd.read_csv(UNIVERSE_2026 / "sp500_2025_12_31_snapshot.csv")
    events = pd.read_csv(UNIVERSE_2026 / "sp500_2026_membership_events.csv")
    tickers = set(snap["ticker"].dropna().astype(str).str.upper())
    tickers.update(events["added_ticker"].dropna().astype(str).str.upper())
    tickers.add("SPY")
    for stale, current in TICKER_ALIASES.items():
        if stale in tickers:
            tickers.discard(stale)
            tickers.add(current)
    tickers.discard("NAN")
    tickers.discard("")
    return sorted(tickers)


def _bars_to_df(rows: list[dict], ticker: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ticker"] = ticker
    df["date"] = df["t"].map(_ms_to_date)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "n_trades"})
    keep = [c for c in ["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "n_trades"] if c in df.columns]
    return df[keep].drop_duplicates(["ticker", "date"]).sort_values("date")


def run_market_download(price_verdict: str, full: bool | None = None) -> dict:
    ensure_dirs()
    client = MassiveClient(min_interval_s=0.2)
    today = datetime.now().date().isoformat()
    end = today
    start = LOOKBACK_START
    tickers = _universe_tickers()
    if full is None:
        full = str(price_verdict).startswith("PASS")
    if not full:
        tickers = [t for t in ["AAPL", "NVDA", "JPM", "XOM", "JNJ", "AMZN", "TSLA", "MSFT", "SPY"] if t in set(tickers) or t == "SPY"]
        tickers = sorted(set(tickers + ["SPY"]))

    failed = []
    frames = []
    for i, ticker in enumerate(tickers, 1):
        rows = client.paginate(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
            {"adjusted": "true", "limit": 50000, "sort": "asc"},
            max_pages=5,
        )
        df = _bars_to_df(rows, ticker)
        if df.empty:
            failed.append(ticker)
            print(f"F {i}/{len(tickers)} {ticker}: EMPTY")
            continue
        df.to_parquet(RAW_2026 / f"{ticker}_daily.parquet", index=False)
        frames.append(df)
        print(f"F {i}/{len(tickers)} {ticker}: {len(df)} bars {df['date'].min().date()}..{df['date'].max().date()}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined["ret"] = combined.groupby("ticker")["close"].pct_change()
        combined.to_parquet(NORM_2026 / "daily_ohlcv_split_adjusted.parquet", index=False)
        spy = combined[combined["ticker"] == "SPY"].copy()
        spy.to_parquet(NORM_2026 / "spy_daily.parquet", index=False)

    splits = client.paginate(
        "/v3/reference/splits",
        {"execution_date.gte": start, "limit": 1000, "sort": "execution_date", "order": "asc"},
        max_pages=50,
    )
    divs = client.paginate(
        "/v3/reference/dividends",
        {"ex_dividend_date.gte": start, "limit": 1000, "sort": "ex_dividend_date", "order": "asc"},
        max_pages=80,
    )
    pd.DataFrame(splits).to_csv(RAW_2026 / "splits_from_2025.csv", index=False)
    pd.DataFrame(divs).to_csv(RAW_2026 / "dividends_from_2025.csv", index=False)
    uni = set(tickers)
    if splits:
        pd.DataFrame(splits).assign(ticker=lambda d: d.get("ticker", d.get("ticker")))
        sdf = pd.DataFrame(splits)
        if "ticker" in sdf.columns:
            sdf[sdf["ticker"].astype(str).str.upper().isin(uni)].to_csv(NORM_2026 / "universe_splits.csv", index=False)
    if divs:
        ddf = pd.DataFrame(divs)
        tic_col = "ticker" if "ticker" in ddf.columns else None
        if tic_col:
            ddf[ddf[tic_col].astype(str).str.upper().isin(uni)].to_csv(NORM_2026 / "universe_dividends.csv", index=False)

    ref_rows = client.paginate("/v3/reference/tickers", {"market": "stocks", "active": "true", "limit": 1000}, max_pages=3)
    pd.DataFrame(ref_rows).to_csv(RAW_2026 / "tickers_active_sample.csv", index=False)
    pd.DataFrame(
        [{"stale_ticker": k, "massive_ticker": v, "reason": "WRDS/S5A query ticker no longer served by Massive"} for k, v in TICKER_ALIASES.items()]
    ).to_csv(MANIFEST_2026 / "ticker_aliases.csv", index=False)

    manifest = {
        "start": start,
        "end": end,
        "full_universe_download": full,
        "price_verdict": price_verdict,
        "n_tickers_requested": len(tickers),
        "n_tickers_ok": int(combined["ticker"].nunique()) if not combined.empty else 0,
        "failed_tickers": failed,
        "n_bars": int(len(combined)),
        "n_splits_raw": len(splits),
        "n_dividends_raw": len(divs),
        "lookback_reason": "Alpha20/HMM rolling windows (up to 60d; vol20) need 2025 history before 2026-01-02",
        "adjustment": "Massive adjusted=true split-adjusted daily bars; not Qlib first-day normalized",
        "generated": datetime.now(timezone.utc).isoformat(),
        "api_calls": client.calls,
        "rate_limit_hits": client.rate_limit_hits,
    }
    (MANIFEST_2026 / "market_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (AUDIT_DIR / "market_download_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"F done tickers_ok={manifest['n_tickers_ok']} failed={len(failed)}")
    return manifest


def repair_aliases_and_actions(existing_manifest: dict | None = None) -> dict:
    """Fill alias tickers + corporate actions without re-downloading the full universe."""
    ensure_dirs()
    client = MassiveClient(min_interval_s=0.2)
    today = datetime.now().date().isoformat()
    start = LOOKBACK_START
    end = today
    alias_ok = []
    alias_failed = []
    extra_frames = []
    for stale, current in TICKER_ALIASES.items():
        rows = client.paginate(
            f"/v2/aggs/ticker/{current}/range/1/day/{start}/{end}",
            {"adjusted": "true", "limit": 50000, "sort": "asc"},
            max_pages=5,
        )
        df = _bars_to_df(rows, current)
        if df.empty:
            alias_failed.append({"stale": stale, "current": current})
            print(f"F-repair {stale}->{current}: EMPTY")
            continue
        df.to_parquet(RAW_2026 / f"{current}_daily.parquet", index=False)
        extra_frames.append(df)
        alias_ok.append({"stale": stale, "current": current, "n_bars": int(len(df)), "end": str(df["date"].max().date())})
        print(f"F-repair {stale}->{current}: {len(df)} bars {df['date'].min().date()}..{df['date'].max().date()}")

    combined_path = NORM_2026 / "daily_ohlcv_split_adjusted.parquet"
    if combined_path.exists() and extra_frames:
        combined = pd.read_parquet(combined_path)
        add = pd.concat(extra_frames, ignore_index=True)
        combined = pd.concat([combined[~combined["ticker"].isin(add["ticker"].unique())], add], ignore_index=True)
        combined["ret"] = combined.groupby("ticker")["close"].pct_change()
        combined.to_parquet(combined_path, index=False)
    elif extra_frames:
        combined = pd.concat(extra_frames, ignore_index=True)
        combined["ret"] = combined.groupby("ticker")["close"].pct_change()
        combined.to_parquet(combined_path, index=False)

    splits = client.paginate(
        "/v3/reference/splits",
        {"execution_date.gte": start, "limit": 1000, "sort": "execution_date", "order": "asc"},
        max_pages=50,
    )
    divs = client.paginate(
        "/v3/reference/dividends",
        {"ex_dividend_date.gte": start, "limit": 1000, "sort": "ex_dividend_date", "order": "asc"},
        max_pages=80,
    )
    pd.DataFrame(splits).to_csv(RAW_2026 / "splits_from_2025.csv", index=False)
    pd.DataFrame(divs).to_csv(RAW_2026 / "dividends_from_2025.csv", index=False)
    uni = set(_universe_tickers())
    if splits:
        sdf = pd.DataFrame(splits)
        if "ticker" in sdf.columns:
            sdf[sdf["ticker"].astype(str).str.upper().isin(uni)].to_csv(NORM_2026 / "universe_splits.csv", index=False)
    if divs:
        ddf = pd.DataFrame(divs)
        if "ticker" in ddf.columns:
            ddf[ddf["ticker"].astype(str).str.upper().isin(uni)].to_csv(NORM_2026 / "universe_dividends.csv", index=False)
    pd.DataFrame(
        [{"stale_ticker": k, "massive_ticker": v, "reason": "WRDS/S5A query ticker no longer served by Massive"} for k, v in TICKER_ALIASES.items()]
    ).to_csv(MANIFEST_2026 / "ticker_aliases.csv", index=False)

    prev = existing_manifest or {}
    if (AUDIT_DIR / "market_download_summary.json").exists() and not prev:
        prev = json.loads((AUDIT_DIR / "market_download_summary.json").read_text(encoding="utf-8"))
    n_tickers = int(pd.read_parquet(combined_path)["ticker"].nunique()) if combined_path.exists() else prev.get("n_tickers_ok", 0)
    n_bars = int(len(pd.read_parquet(combined_path))) if combined_path.exists() else prev.get("n_bars", 0)
    manifest = dict(prev)
    manifest.update(
        {
            "end": end,
            "n_tickers_ok": n_tickers,
            "n_bars": n_bars,
            "failed_tickers_original": ["BF", "CDAY", "FLT", "PEAK"],
            "alias_recovered": alias_ok,
            "alias_failed": alias_failed,
            "n_splits_raw": len(splits),
            "n_dividends_raw": len(divs),
            "repair": "aliases_and_corporate_actions",
            "generated": datetime.now(timezone.utc).isoformat(),
            "api_calls": client.calls,
            "rate_limit_hits": client.rate_limit_hits,
        }
    )
    (MANIFEST_2026 / "market_download_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (AUDIT_DIR / "market_download_summary.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"F-repair splits={len(splits)} dividends={len(divs)} tickers_ok={n_tickers}")
    return manifest
