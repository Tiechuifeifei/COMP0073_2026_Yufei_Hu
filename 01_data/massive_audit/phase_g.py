"""Phase G: dump 2026 F1C-related fundamentals from Massive when financials are
reachable. Diagnostic panel; not claimed equivalent to frozen Compustat F1C."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from massive_audit.client import MassiveClient
from massive_audit.config import AUDIT_DIR, F1C_SAMPLE_TICKERS, FUND_2026, UNIVERSE_2026, ensure_dirs


def run_fundamentals_2026(f1c_verdict: str) -> dict:
    ensure_dirs()
    client = MassiveClient(min_interval_s=0.25)
    if f1c_verdict in {"NOT_AVAILABLE"}:
        summary = {
            "skipped": True,
            "reason": "Massive financials not on current plan",
            "f1c_verdict": f1c_verdict,
        }
        (FUND_2026 / "f1c_filing_manifest.csv").write_text("skipped,True\n", encoding="utf-8")
        (FUND_2026 / "fundamentals_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    tickers = list(F1C_SAMPLE_TICKERS)
    snap_path = UNIVERSE_2026 / "sp500_2025_12_31_snapshot.csv"
    if snap_path.exists() and f1c_verdict != "NOT_AVAILABLE":
        # Keep sample-sized panel; full 500-name statement history is not justified once PIT fails.
        pass

    income_all = []
    edgar_all = []
    for ticker in tickers:
        inc = client.paginate(
            "/stocks/financials/v1/income-statements",
            {
                "tickers.any_of": ticker,
                "timeframe": "quarterly",
                "period_end.gte": "2024-01-01",
                "limit": 100,
                "sort": "period_end.asc",
            },
            max_pages=3,
        )
        ed = client.paginate(
            "/stocks/filings/vX/index",
            {
                "ticker": ticker,
                "form_type.any_of": "10-Q,10-K",
                "filing_date.gte": "2024-01-01",
                "limit": 100,
                "sort": "filing_date.asc",
            },
            max_pages=3,
        )
        if not ed:
            ed = client.paginate(
                "/stocks/filings/v1/index",
                {
                    "ticker": ticker,
                    "form_type.any_of": "10-Q,10-K",
                    "filing_date.gte": "2024-01-01",
                    "limit": 100,
                    "sort": "filing_date.asc",
                },
                max_pages=3,
            )
        for row in inc:
            row = dict(row)
            row["query_ticker"] = ticker
            income_all.append(row)
        for row in ed:
            row = dict(row)
            row["query_ticker"] = ticker
            edgar_all.append(row)
        print(f"G {ticker}: income={len(inc)} edgar={len(ed)}")

    inc_df = pd.DataFrame(income_all)
    ed_df = pd.DataFrame(edgar_all)
    if not inc_df.empty:
        inc_df.to_parquet(FUND_2026 / "income_statements_sample.parquet", index=False)
    if not ed_df.empty:
        ed_df.to_parquet(FUND_2026 / "edgar_10qk_sample.parquet", index=False)

    # Diagnostic panel: one row per (ticker, period_end) using financials values,
    # availability_candidate = min EDGAR 10-Q/K filing_date on/after period_end (heuristic, not rdqe).
    panel_rows = []
    if not inc_df.empty:
        for ticker, g in inc_df.groupby("query_ticker"):
            filings = ed_df[ed_df["query_ticker"] == ticker].copy() if not ed_df.empty else pd.DataFrame()
            if not filings.empty:
                filings["filing_date"] = pd.to_datetime(filings["filing_date"], errors="coerce")
            for rec in g.to_dict("records"):
                period_end = pd.to_datetime(rec.get("period_end"), errors="coerce")
                fin_filing = rec.get("filing_date")
                orig = pd.NaT
                form = ""
                if not filings.empty and pd.notna(period_end):
                    later = filings[filings["filing_date"] >= period_end].sort_values("filing_date")
                    if len(later):
                        orig = later.iloc[0]["filing_date"]
                        form = later.iloc[0].get("form_type", "")
                panel_rows.append(
                    {
                        "ticker": ticker,
                        "period_end": rec.get("period_end"),
                        "fiscal_year": rec.get("fiscal_year"),
                        "fiscal_quarter": rec.get("fiscal_quarter"),
                        "massive_financials_filing_date": fin_filing,
                        "edgar_first_10qk_on_or_after_period_end": None if pd.isna(orig) else orig.date().isoformat(),
                        "edgar_form": form,
                        "revenue": rec.get("revenue"),
                        "gross_profit": rec.get("gross_profit"),
                        "net_income": rec.get("net_income_loss_attributable_common_shareholders"),
                        "operating_income": rec.get("operating_income"),
                        "pit_warning": "financial statement VALUES may already include later restatements; EDGAR date is timing-only heuristic, not Compustat rdqe",
                    }
                )
    panel = pd.DataFrame(panel_rows)
    if not panel.empty:
        panel.to_parquet(FUND_2026 / "f1c_pit_panel_2026.parquet", index=False)
        panel.to_csv(FUND_2026 / "f1c_pit_panel_2026.csv", index=False)
    manifest = panel[
        [
            "ticker",
            "period_end",
            "massive_financials_filing_date",
            "edgar_first_10qk_on_or_after_period_end",
            "edgar_form",
        ]
    ] if not panel.empty else pd.DataFrame()
    manifest_path = FUND_2026 / "f1c_filing_manifest.csv"
    if not manifest.empty:
        manifest.to_csv(manifest_path, index=False)
    else:
        manifest_path.write_text("ticker,note\n,no_rows\n", encoding="utf-8")

    summary = {
        "skipped": False,
        "diagnostic_only": True,
        "usable_for_frozen_f1c": False,
        "f1c_verdict": f1c_verdict,
        "n_tickers": len(tickers),
        "n_income_rows": int(len(inc_df)),
        "n_edgar_rows": int(len(ed_df)),
        "n_panel_rows": int(len(panel)),
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    (FUND_2026 / "fundamentals_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (AUDIT_DIR / "fundamentals_2026_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"G panel_rows={summary['n_panel_rows']} usable_for_frozen_f1c=False")
    return summary
