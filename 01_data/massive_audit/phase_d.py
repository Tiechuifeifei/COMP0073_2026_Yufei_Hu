"""Phase D: F1C feature mapping and PIT audit against Massive financials/filings."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import pandas as pd

from massive_audit.client import MassiveClient
from massive_audit.config import AUDIT_DIR, F1C_SAMPLE_TICKERS, REPORT_DIR, ensure_dirs

MAPPING = [
    {
        "f1c_feature": "roe_z",
        "historical_source_field": "ttm_niq / avg_book_equity then daily CS z",
        "historical_definition": "TTM Compustat niq / average book equity (ceqq else seqq+txditcq-pstkq)",
        "massive_endpoint": "/stocks/financials/v1/income-statements + balance-sheets",
        "massive_field": "net_income_loss_attributable_common_shareholders / equity (TTM constructed)",
        "definition_match": "PARTIAL",
        "unit_match": "YES_IF_BOTH_USD",
        "frequency_match": "YES_QUARTERLY",
        "notes": "Massive equity line items are not guaranteed to match Compustat ceqq unrestated PIT.",
    },
    {
        "f1c_feature": "roa_z",
        "historical_source_field": "ttm_niq / avg_atq",
        "historical_definition": "TTM NI / average total assets",
        "massive_endpoint": "income-statements + balance-sheets",
        "massive_field": "TTM NI / average assets",
        "definition_match": "PARTIAL",
        "unit_match": "YES_IF_BOTH_USD",
        "frequency_match": "YES_QUARTERLY",
        "notes": "Needs two-period assets; Massive filing_date is latest restating filing, not rdqe.",
    },
    {
        "f1c_feature": "gross_profitability_z",
        "historical_source_field": "ttm_gpq / avg_atq",
        "historical_definition": "TTM (saleq-cogsq)/avg assets",
        "massive_endpoint": "income-statements",
        "massive_field": "gross_profit TTM / avg assets",
        "definition_match": "PARTIAL",
        "unit_match": "YES_IF_BOTH_USD",
        "frequency_match": "YES_QUARTERLY",
        "notes": "Gross profit definition may differ from saleq-cogsq (shipping, D&A in COGS).",
    },
    {
        "f1c_feature": "sales_growth_yoy_z",
        "historical_source_field": "ttm_saleq / ttm_saleq_lag4 - 1",
        "historical_definition": "YoY TTM sales",
        "massive_endpoint": "income-statements timeframe=trailing_twelve_months",
        "massive_field": "revenue TTM vs lag4",
        "definition_match": "PARTIAL",
        "unit_match": "YES",
        "frequency_match": "YES",
        "notes": "Need original-period TTM, not restated comparatives from a later 10-K.",
    },
    {
        "f1c_feature": "asset_growth_yoy_z",
        "historical_source_field": "atq / atq_lag4 - 1",
        "historical_definition": "YoY total assets",
        "massive_endpoint": "balance-sheets",
        "massive_field": "assets vs lag4",
        "definition_match": "PARTIAL",
        "unit_match": "YES",
        "frequency_match": "YES",
        "notes": "Restated prior-year assets in later filings would leak.",
    },
    {
        "f1c_feature": "accruals_z",
        "historical_source_field": "(ttm_niq - standalone TTM oancf)/avg_atq",
        "historical_definition": "Sloan; YTD oancfq converted to quarterly then 4Q TTM with consecutive-quarter flags",
        "massive_endpoint": "cash-flow-statements",
        "massive_field": "operating cash flow TTM",
        "definition_match": "NO",
        "unit_match": "UNKNOWN",
        "frequency_match": "UNKNOWN",
        "notes": "Massive is unlikely to replicate Compustat YTD-to-quarter conversion and ttm_oancf_valid_4q_flag.",
    },
    {
        "f1c_feature": "leverage_z",
        "historical_source_field": "total_debt / atq",
        "historical_definition": "Debt/assets from PIT Compustat",
        "massive_endpoint": "balance-sheets",
        "massive_field": "debt / assets",
        "definition_match": "PARTIAL",
        "unit_match": "YES",
        "frequency_match": "YES",
        "notes": "Debt aggregation (ST+LT, leases) often differs across vendors.",
    },
    {
        "f1c_feature": "current_ratio_z",
        "historical_source_field": "actq / lctq",
        "historical_definition": "Current assets / current liabilities",
        "massive_endpoint": "balance-sheets",
        "massive_field": "current_assets / current_liabilities",
        "definition_match": "PARTIAL",
        "unit_match": "YES",
        "frequency_match": "YES",
        "notes": "",
    },
    {
        "f1c_feature": "book_to_market_z",
        "historical_source_field": "book_equity*1e6 / CRSP signal_market_cap",
        "historical_definition": "PIT book over CRSP mcap on signal_start_date",
        "massive_endpoint": "balance-sheets + daily close*shares",
        "massive_field": "equity / market cap",
        "definition_match": "NO",
        "unit_match": "NO",
        "frequency_match": "NO",
        "notes": "Frozen F1C uses CRSP mcap on the PIT signal date, not Massive current float.",
    },
    {
        "f1c_feature": "earnings_yield_z",
        "historical_source_field": "ttm_niq*1e6 / signal_market_cap",
        "historical_definition": "TTM NI / CRSP mcap",
        "massive_endpoint": "income-statements + price",
        "massive_field": "TTM EPS or NI / price",
        "definition_match": "PARTIAL",
        "unit_match": "PARTIAL",
        "frequency_match": "PARTIAL",
        "notes": "Massive /ratios is documented as latest-day TTM with latest price — snapshot, not PIT panel.",
    },
    {
        "f1c_feature": "sales_to_price_z",
        "historical_source_field": "ttm_saleq*1e6 / signal_market_cap",
        "historical_definition": "TTM sales / CRSP mcap",
        "massive_endpoint": "income-statements + price",
        "massive_field": "TTM revenue / mcap",
        "definition_match": "PARTIAL",
        "unit_match": "PARTIAL",
        "frequency_match": "PARTIAL",
        "notes": "Same snapshot risk as earnings_yield.",
    },
    {
        "f1c_feature": "operating_profitability_z (excluded from F1C)",
        "historical_source_field": "ttm_operating_profit / avg_book_equity",
        "historical_definition": "Not used in frozen F1C",
        "massive_endpoint": "income-statements operating_income",
        "massive_field": "operating_income",
        "definition_match": "N/A",
        "unit_match": "N/A",
        "frequency_match": "N/A",
        "notes": "Listed only to document exclusion.",
    },
    {
        "f1c_feature": "PIT availability date",
        "historical_source_field": "rdqe; accounting_available_date=max(rdqe, datadate)",
        "historical_definition": "Compustat PIT announcement/earnings date, then next CRSP session",
        "massive_endpoint": "financials.filing_date OR /stocks/filings/v1/index",
        "massive_field": "filing_date",
        "definition_match": "NO_FOR_FINANCIALS_YES_MAYBE_FOR_EDGAR",
        "unit_match": "DATE",
        "frequency_match": "EVENT",
        "notes": "Massive documents that financials.filing_date is the most recent filing that restates the period, not the original 10-Q/10-K date.",
    },
]


def _first_keys(obj: dict) -> str:
    return ",".join(sorted(obj.keys())[:30])


def run_f1c_audit() -> dict:
    ensure_dirs()
    client = MassiveClient()
    mapping_path = AUDIT_DIR / "f1c_massive_mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "f1c_feature",
            "historical_source_field",
            "historical_definition",
            "massive_endpoint",
            "massive_field",
            "definition_match",
            "unit_match",
            "frequency_match",
            "filing_date_available",
            "original_release_date_available",
            "restatement_risk",
            "pit_safe",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in MAPPING:
            out = dict(row)
            out.setdefault("filing_date_available", "UNKNOWN")
            out.setdefault("original_release_date_available", "UNKNOWN")
            out.setdefault("restatement_risk", "HIGH")
            out.setdefault("pit_safe", "NO")
            writer.writerow(out)

    pit_rows = []
    income_ok = False
    edgar_ok = False
    ratios_snapshot = False
    sample = F1C_SAMPLE_TICKERS[:30]
    for ticker in sample:
        inc_status, inc, _ = client.get(
            "/stocks/financials/v1/income-statements",
            {"tickers.any_of": ticker, "timeframe": "quarterly", "limit": 8, "sort": "period_end.desc"},
        )
        bs_status, bs, _ = client.get(
            "/stocks/financials/v1/balance-sheets",
            {"tickers.any_of": ticker, "timeframe": "quarterly", "limit": 4, "sort": "period_end.desc"},
        )
        cf_status, cf, _ = client.get(
            "/stocks/financials/v1/cash-flow-statements",
            {"tickers.any_of": ticker, "timeframe": "quarterly", "limit": 4, "sort": "period_end.desc"},
        )
        rat_status, rat, _ = client.get("/stocks/financials/v1/ratios", {"ticker": ticker, "limit": 3})
        ed_status, ed, _ = client.get(
            "/stocks/filings/vX/index",
            {"ticker": ticker, "form_type.any_of": "10-Q,10-K", "limit": 8, "sort": "filing_date.desc"},
        )
        if ed_status == 404:
            ed_status, ed, _ = client.get(
                "/stocks/filings/v1/index",
                {"ticker": ticker, "form_type.any_of": "10-Q,10-K", "limit": 8, "sort": "filing_date.desc"},
            )
        inc_rows = (inc or {}).get("results") or [] if isinstance(inc, dict) else []
        ed_rows = (ed or {}).get("results") or [] if isinstance(ed, dict) else []
        rat_rows = (rat or {}).get("results") or [] if isinstance(rat, dict) else []
        if inc_status == 200 and inc_rows:
            income_ok = True
        if ed_status == 200 and ed_rows:
            edgar_ok = True
        if rat_status == 200 and rat_rows:
            ratios_snapshot = True
        filing_dates = sorted({r.get("filing_date") for r in inc_rows if r.get("filing_date")})
        period_ends = [r.get("period_end") for r in inc_rows]
        dup_filing = len(filing_dates) < max(len(inc_rows) - 1, 0) if inc_rows else False
        # If several periods share one filing_date, that is restatement/comparative packing.
        from collections import Counter

        fc = Counter(r.get("filing_date") for r in inc_rows)
        shared = {k: v for k, v in fc.items() if k and v > 1}
        pit_rows.append(
            {
                "ticker": ticker,
                "income_http": inc_status,
                "balance_http": bs_status,
                "cashflow_http": cf_status,
                "ratios_http": rat_status,
                "edgar_http": ed_status,
                "n_income_rows": len(inc_rows),
                "n_edgar_10qk": len(ed_rows),
                "income_fields": _first_keys(inc_rows[0]) if inc_rows else "",
                "balance_fields": _first_keys(((bs or {}).get("results") or [{}])[0]) if bs_status == 200 else "",
                "cashflow_fields": _first_keys(((cf or {}).get("results") or [{}])[0]) if cf_status == 200 else "",
                "ratio_fields": _first_keys(rat_rows[0]) if rat_rows else "",
                "sample_period_end": period_ends[0] if period_ends else "",
                "sample_income_filing_date": inc_rows[0].get("filing_date") if inc_rows else "",
                "shared_filing_date_counts": json.dumps(shared),
                "edgar_latest_filing_date": ed_rows[0].get("filing_date") if ed_rows else "",
                "edgar_latest_form": ed_rows[0].get("form_type") if ed_rows else "",
                "financials_filing_date_is_original": "NO" if shared or inc_rows else "UNKNOWN",
                "pit_safe_using_financials_filing_date": "NO",
                "pit_recoverable_via_edgar_index": "YES" if ed_status == 200 and ed_rows else "NO",
            }
        )
        print(f"D {ticker}: inc={inc_status} n={len(inc_rows)} edgar={ed_status} n={len(ed_rows)}")

    pit_df = pd.DataFrame(pit_rows)
    pit_df.to_csv(AUDIT_DIR / "f1c_pit_audit.csv", index=False)

    if not income_ok:
        overall = "NOT_AVAILABLE"
    elif income_ok and not edgar_ok:
        overall = "NOT_EQUIVALENT"
    else:
        overall = "NOT_EQUIVALENT"
        # Even with EDGAR original dates, statement values may still be restated in the financials endpoint.
        # That is not EXACT_MATCH and not a documented-acceptable replacement for Compustat PIT unrestated *r fields.

    md = f"""# F1C equivalence / PIT report

Generated: {datetime.now(timezone.utc).isoformat()}

## Frozen F1C (what we must replicate)

F1C uses 11 daily cross-sectional z-scores, not raw Compustat items:

`roe, roa, gross_profitability, sales_growth_yoy, asset_growth_yoy, accruals, leverage, current_ratio, book_to_market, earnings_yield, sales_to_price`

constructed in `fundamental_pipeline/10_build_quarterly_fundamental_features.py` from Compustat PIT `pitqtrdataus` with availability `rdqe`, then Phase 4B daily CS z-score.

`operating_profitability` is **excluded**.

## Massive mapping status

See `data/massive_audit/f1c_massive_mapping.csv`.

## PIT test

Massive income-statement docs state:

> filing_date is the date of the **most recent SEC filing that included this period's data**. This is **not necessarily** the date this period was originally filed. Later 10-K/10-Q comparatives restate prior periods.

That is exactly the failure mode this audit forbids.

SEC EDGAR index (`/stocks/filings/v1/index`) returns a true submission `filing_date` per 10-Q/10-K. That can recover **timing**. It does **not** by itself recover unrestated line items equal to Compustat `*r` PIT fields.

Massive `/stocks/financials/v1/ratios` is documented as latest-day TTM with latest price — a **current snapshot**, unusable as a 2026 PIT panel.

## Sample probe ({len(sample)} tickers)

- income statements accessible: {income_ok}
- EDGAR index accessible: {edgar_ok}
- ratios accessible: {ratios_snapshot}

## Verdict

**{overall}**

Interpretation:

- `EXACT_MATCH`: not possible. Definitions, unrestated PIT values, CRSP mcap, and rdqe timing will not match bitwise.
- `ACCEPTABLE_WITH_DOCUMENTED_DIFFERENCE`: not accepted for frozen F1C. Accruals YTD conversion, book equity fallback, CRSP mcap, and unrestated PIT are material.
- `NOT_EQUIVALENT`: correct if financials are readable but cannot support the frozen PIT definition.
- `NOT_AVAILABLE`: correct if the current plan cannot read financials.

A 2026 F1C extension therefore still needs Compustat PIT (or an equivalent unrestated PIT vendor) plus original filing dates. Massive statements/EDGAR can at best **supplement** coverage checks, not replace F1C inputs.
"""
    (REPORT_DIR / "f1c_equivalence_report.md").write_text(md, encoding="utf-8")
    summary = {
        "verdict": overall,
        "income_ok": income_ok,
        "edgar_ok": edgar_ok,
        "ratios_ok": ratios_snapshot,
        "n_tickers": len(pit_df),
    }
    (AUDIT_DIR / "f1c_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"D verdict={overall}")
    return summary
