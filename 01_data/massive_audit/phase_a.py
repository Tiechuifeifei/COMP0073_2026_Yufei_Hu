"""Phase A: inventory data required by frozen H0 / D2 / F1C / HMM. No API calls."""

from __future__ import annotations

import csv

from massive_audit.config import AUDIT_DIR, ensure_dirs

ROWS = [
    # D2 / Alpha20 / RD price-volume
    ["D2 / Alpha20", "open", "WRDS CRSP DlyOpen / DlyCumFacPr then Qlib first-day normalize", "Split-adjusted CRSP daily open; Qlib divides by first-sample close", "daily", "same-day bar available at t close for features used to predict t+1/t+2 label", "Must use historically available CRSP vintage; no restatement of past bars", "Alpha20 20 factors; D2 LightGBM; Qlib TopK", "CRITICAL"],
    ["D2 / Alpha20", "high", "WRDS CRSP DlyHigh / DlyCumFacPr", "Split-adjusted daily high", "daily", "same as open", "PIT market data", "Alpha20 KLEN/KLOW and range features", "CRITICAL"],
    ["D2 / Alpha20", "low", "WRDS CRSP DlyLow / DlyCumFacPr", "Split-adjusted daily low", "daily", "same as open", "PIT market data", "Alpha20 KLEN/KLOW", "CRITICAL"],
    ["D2 / Alpha20", "close", "WRDS CRSP DlyClose / DlyCumFacPr", "Split-adjusted daily close; Qlib $close", "daily", "Label uses Ref($close,-2)/Ref($close,-1)-1 so close t is used one day later", "PIT market data", "Alpha20/RD13/D2 features and LABEL0", "CRITICAL"],
    ["D2 / Alpha20", "volume", "WRDS CRSP DlyVol adjusted by 1/DlyCumFacPr inverse", "Split-adjusted volume so $close*$volume is roughly invariant", "daily", "same-day", "PIT market data", "Alpha20 CORR/WVMA/VSTD features", "CRITICAL"],
    ["D2 / Alpha20", "factor / split adjustment", "WRDS CRSP DlyCumFacPr", "factor=1/DlyCumFacPr; prices already / DlyCumFacPr in staging/csv", "event / daily", "split applied on CRSP effective date", "Must not apply later splits to earlier Qlib bins retroactively without consistent factor", "Qlib dump; RD13 price-volume", "CRITICAL"],
    ["D2 / RD13_v2", "RD13 unique price-volume vectors", "Qlib $open/$high/$low/$close/$volume plus RD-Agent implementations", "13 unique delayed factors from RD13_v2 workspace HDF5", "daily", "delayed / next-bar convention matching D2", "No future bars", "D2 feature stack", "CRITICAL"],
    ["D2 / Alpha20", "trading calendar", "CRSP DlyCalDt / Qlib calendars/day.txt", "US equity trading days in the dumped vintage", "daily", "n/a", "Must match CRSP session dates, not a generic NYSE calendar if they diverge", "Qlib instruments and backtest", "CRITICAL"],
    # F1C
    ["F1C", "roe", "Compustat PIT pitqtrdataus niq TTM / avg book equity", "ttm_niq / avg_book_equity; book equity prefers unrestated ceqq", "quarterly → daily ffill until next signal", "accounting_available_date = max(rdqe, datadate); signal_start = first CRSP date ≥ availability", "Use unrestated *r fields / rdqe; do not use later restated AFR snapshot", "F1C 11-feature LightGBM ensemble", "CRITICAL"],
    ["F1C", "roa", "Compustat PIT ttm_niq / avg_atq", "TTM net income over average total assets", "quarterly → daily", "same rdqe rule", "PIT unrestated assets and income", "F1C", "CRITICAL"],
    ["F1C", "gross_profitability", "Compustat PIT ttm_gpq / avg_atq", "TTM (saleq-cogsq) / average assets", "quarterly → daily", "rdqe", "PIT", "F1C", "CRITICAL"],
    ["F1C", "sales_growth_yoy", "Compustat PIT ttm_saleq / ttm_saleq_lag4 - 1", "Four-quarter TTM sales growth", "quarterly → daily", "rdqe", "Need four trailing PIT quarters", "F1C", "CRITICAL"],
    ["F1C", "asset_growth_yoy", "Compustat PIT atq / atq_lag4 - 1", "YoY total assets", "quarterly → daily", "rdqe", "PIT assets", "F1C", "CRITICAL"],
    ["F1C", "accruals", "Compustat PIT (ttm_niq - standalone TTM oancf) / avg_atq", "Sloan-style; YTD oancfq converted to quarterly then TTM; requires consecutive quarters", "quarterly → daily", "rdqe", "PIT cash-flow conversion flags", "F1C", "CRITICAL"],
    ["F1C", "leverage", "Compustat PIT total_debt / atq", "Debt / assets", "quarterly → daily", "rdqe", "PIT", "F1C", "HIGH"],
    ["F1C", "current_ratio", "Compustat PIT actq / lctq", "Current assets / current liabilities; missing if either missing", "quarterly → daily", "rdqe", "PIT actq/lctq supplement", "F1C", "HIGH"],
    ["F1C", "book_to_market", "Compustat PIT book_equity*1e6 / CRSP signal_market_cap", "Book equity (ceqq else seqq+txditcq-pstkq) over CRSP mcap on signal date", "quarterly mapped to daily mcap", "rdqe + CRSP mcap on signal_start", "PIT book and contemporaneous mcap", "F1C", "CRITICAL"],
    ["F1C", "earnings_yield", "ttm_niq*1e6 / signal_market_cap", "TTM earnings / mcap", "quarterly → daily", "rdqe", "PIT", "F1C", "CRITICAL"],
    ["F1C", "sales_to_price", "ttm_saleq*1e6 / signal_market_cap", "TTM sales / mcap", "quarterly → daily", "rdqe", "PIT", "F1C", "CRITICAL"],
    ["F1C", "cross-sectional z-score", "Phase 4B daily CS winsorize + z-score", "Each date, eligible stocks only; min cross-section 50", "daily", "same day as feature", "Only stocks with PIT-available fundamentals that day", "F1C inputs are *_z not raw ratios", "CRITICAL"],
    ["F1C", "filing / publication lag (rdqe)", "comp_pit.pitqtrdataus rdqe", "accounting_available_date=max(rdqe, datadate); NOT Compustat rdq and NOT later restated filing overlay", "quarterly event", "signal cannot start before rdqe", "True original availability; restated comparative filings must not replace original rdqe", "F1C PIT panel", "CRITICAL"],
    ["F1C", "gvkey-permno link", "CRSP/Compustat CCM linktable", "linkdt ≤ t ≤ linkenddt", "spell", "link must be valid on signal date", "PIT identifier link", "F1C and membership join", "CRITICAL"],
    # HMM / benchmark
    ["HMM 2-state", "spy_ret", "WRDS CRSP SPY PERMNO 84398 close pct_change", "staging/csv_benchmark/p84398.csv", "daily", "lag-1 state for trading", "No future SPY returns", "HMM train-only fit; H0 trade state", "CRITICAL"],
    ["HMM 2-state", "spy_vol20", "20-day rolling std of spy_ret", "min_periods=20", "daily", "needs 20-day lookback", "PIT rolling window", "HMM features", "CRITICAL"],
    ["Benchmark", "SPY OHLCV (P84398)", "WRDS CRSP SPY", "Same CRSP dump as equities; Qlib benchmark P84398", "daily", "same session", "PIT", "H0 excess ARR / costs", "CRITICAL"],
    # Universe / actions / mapping
    ["Universe", "S&P500 PIT membership spells", "CRSP S&P member extract → staging/instruments/sp500.txt and membership_spells.parquet", "PERMNO in index between MbrStartDt and MbrEndDt; Qlib instrument P{permno}", "daily spells", "member on date t iff start ≤ t ≤ end", "Must not backfill today's list", "H0 tradable universe; F1C eligibility", "CRITICAL"],
    ["Universe", "delistings / last trade date", "CRSP delist / last DlyCalDt in dump", "Instrument stops when membership or price series ends", "event", "cannot trade after last CRSP date", "PIT", "TopK dropout / survivorship", "CRITICAL"],
    ["Corporate actions", "splits", "CRSP DlyCumFacPr changes", "Encoded in factor; not a separate event table in Qlib dump", "event", "effective on CRSP date", "PIT factor path", "Price continuity for D2", "CRITICAL"],
    ["Corporate actions", "dividends", "CRSP returns include dividends in DlyRet; Qlib close is price not total return", "Primary features use price/volume not dividend-adjusted total return; cash dividends are not a separate Qlib field", "event", "ex-date in CRSP", "Do not silently switch to total-return prices", "Secondary; H0 uses close-to-close Qlib returns", "HIGH"],
    ["Identifiers", "PERMNO ↔ ticker mapping", "CRSP names / S5A ticker map / Compustat tic", "Qlib id is P{permno}; external APIs use ticker", "spell / as-of", "ticker as of date t, not current ticker", "PIT ticker changes (META/FB, GOOGL, etc.)", "Any Massive join", "CRITICAL"],
    ["Identifiers", "inactive / renamed tickers", "CRSP names history", "Historical ticker may differ from Massive current ticker", "spell", "as-of date", "PIT", "Delisted names in 2026 universe reconstruction", "HIGH"],
]


def write_required_inventory() -> None:
    ensure_dirs()
    path = AUDIT_DIR / "required_data_inventory.csv"
    fields = [
        "component",
        "required_field",
        "historical_source",
        "historical_definition",
        "frequency",
        "lag_requirement",
        "pit_requirement",
        "used_by",
        "criticality",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ROWS:
            writer.writerow(dict(zip(fields, row)))
    print(f"wrote {path} rows={len(ROWS)}")
