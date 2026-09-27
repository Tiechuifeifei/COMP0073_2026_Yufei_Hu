"""Per-ticker splits/dividends for the reconstructed 2026 universe."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from massive_audit.client import MassiveClient
from massive_audit.config import AUDIT_DIR, LOOKBACK_START, MANIFEST_2026, NORM_2026, RAW_2026
from massive_audit.phase_f import _universe_tickers


def main() -> None:
    client = MassiveClient(min_interval_s=0.2)
    start = LOOKBACK_START
    tickers = set(_universe_tickers())
    comb_path = NORM_2026 / "daily_ohlcv_split_adjusted.parquet"
    if comb_path.exists():
        tickers |= set(pd.read_parquet(comb_path)["ticker"].astype(str).str.upper())
    tickers = sorted(tickers)
    print(f"per-ticker corp actions n={len(tickers)}")

    div_rows: list[dict] = []
    empty_div: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        got = client.paginate(
            "/v3/reference/dividends",
            {
                "ticker": ticker,
                "ex_dividend_date.gte": start,
                "limit": 1000,
                "sort": "ex_dividend_date",
                "order": "asc",
            },
            max_pages=3,
        )
        if not got:
            empty_div.append(ticker)
        else:
            div_rows.extend(got)
        if i % 50 == 0:
            print(f"  div {i}/{len(tickers)} rows={len(div_rows)}")

    div_df = pd.DataFrame(div_rows)
    div_df.to_csv(RAW_2026 / "dividends_universe_by_ticker.csv", index=False)
    if not div_df.empty:
        div_df.to_csv(NORM_2026 / "universe_dividends.csv", index=False)
    aapl_n = int((div_df["ticker"] == "AAPL").sum()) if not div_df.empty else 0
    max_ex = str(div_df["ex_dividend_date"].max()) if not div_df.empty else None
    print(
        f"dividends={len(div_df)} tickers_with_div="
        f"{div_df['ticker'].nunique() if not div_df.empty else 0} empty={len(empty_div)} "
        f"AAPL={aapl_n} max_ex={max_ex}"
    )

    split_rows: list[dict] = []
    for i, ticker in enumerate(tickers, 1):
        got = client.paginate(
            "/v3/reference/splits",
            {
                "ticker": ticker,
                "execution_date.gte": start,
                "limit": 100,
                "sort": "execution_date",
                "order": "asc",
            },
            max_pages=2,
        )
        split_rows.extend(got)
        if i % 100 == 0:
            print(f"  splits {i}/{len(tickers)}")
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(RAW_2026 / "splits_universe_by_ticker.csv", index=False)
    if not split_df.empty:
        split_df.to_csv(NORM_2026 / "universe_splits.csv", index=False)
    print(f"splits_universe={len(split_df)}")

    man_path = AUDIT_DIR / "market_download_summary.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    man.update(
        {
            "n_universe_dividends": int(len(div_df)),
            "n_universe_dividend_tickers": int(div_df["ticker"].nunique()) if not div_df.empty else 0,
            "n_tickers_no_dividend_in_window_n": len(empty_div),
            "n_tickers_no_dividend_in_window_sample": empty_div[:40],
            "n_universe_splits": int(len(split_df)),
            "dividend_method": "per_ticker",
            "generated": datetime.now(timezone.utc).isoformat(),
        }
    )
    text = json.dumps(man, indent=2, default=str)
    man_path.write_text(text, encoding="utf-8")
    (MANIFEST_2026 / "market_download_manifest.json").write_text(text, encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
