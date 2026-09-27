"""Phase I/J: source comparison and the 10 required answers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from massive_audit.config import AUDIT_DIR, REPORT_DIR, ensure_dirs


def _load(name: str) -> dict:
    path = AUDIT_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_final_reports(
    endpoints: list[dict] | None = None,
    price: dict | None = None,
    f1c: dict | None = None,
    universe: dict | None = None,
    market: dict | None = None,
    fund: dict | None = None,
    news: dict | None = None,
) -> dict:
    ensure_dirs()
    price = price or _load("price_equivalence_summary.json")
    f1c = f1c or _load("f1c_audit_summary.json")
    universe = universe or _load("universe_2026_summary.json")
    market = market or _load("market_download_summary.json")
    fund = fund or _load("fundamentals_2026_summary.json")
    news = news or _load("news_coverage_summary.json")

    accessible = []
    blocked = []
    if endpoints:
        for row in endpoints:
            if row.get("accessible") == "TRUE":
                accessible.append(row["endpoint"])
            else:
                blocked.append(f"{row['endpoint']} ({row.get('http_status')})")

    price_ok = str(price.get("verdict", "")).startswith("PASS")
    f1c_ok = f1c.get("verdict") in {"EXACT_MATCH", "ACCEPTABLE_WITH_DOCUMENTED_DIFFERENCE"}
    data_ready = bool(
        price_ok
        and market.get("n_tickers_ok", 0) >= 400
        and f1c_ok
        and universe.get("snapshot_n", 0) >= 490
    )
    # Frozen H0 also needs D2 scores / F1C scores / HMM states, not raw prices alone.
    gaps = []
    if not price_ok:
        gaps.append("Price return splice not PASS; cannot attach 2026 bars onto D2 Qlib path yet.")
    else:
        gaps.append("Even with splicable prices, Qlib dump / RD13 / D2 scores are not rebuilt in this phase (by design).")
    if not f1c_ok:
        gaps.append("Massive fundamentals are not a PIT-equivalent F1C source (filing_date restatement + definition gaps).")
    gaps.append("Official S&P500 membership is reconstructed from WRDS 2025-12-31 snapshot + S&P DJI/Wikipedia events, not from Massive.")
    gaps.append("PERMNO mapping is missing for 2026 additions that are absent from the WRDS dump.")
    gaps.append("HMM 20-day vol needs spliced SPY; not fitted here.")
    if market.get("n_tickers_ok", 0) < 400:
        gaps.append("Full-universe 2025–2026 OHLCV download incomplete or limited to sample.")
    if not news.get("n_sample_rows"):
        gaps.append("News coverage sample empty or inaccessible.")

    answers = {
        "1_accessible_data": accessible or "see massive_endpoint_inventory.csv",
        "2_better_than_existing": [
            "If plan includes 2026 daily bars: recency beyond WRDS 2025-12-31 cutoff.",
            "News payload is more structured than Alpha Vantage (tickers, published_utc, insights) if history is long enough.",
            "SEC EDGAR index is a convenient filing calendar; WRDS Compustat PIT remains superior for line items.",
        ],
        "3_supplement_only": [
            "Financial statements / ratios (restated filing_date; snapshot ratios).",
            "ETF constituents as a proxy for S&P500 membership.",
            "Minute/trades/quotes (not used by frozen H0).",
            "News (not in frozen H0).",
        ],
        "4_price_supports_2026_d2": price.get("verdict"),
        "5_fundamentals_support_2026_f1c": f1c.get("verdict"),
        "6_sp500_pit_rebuild": "PARTIAL: 2025-12-31 WRDS snapshot + public 2026 S&P DJI events. Massive cannot certify official index membership.",
        "7_news_vs_av": "Structurally better; not automatically PIT-safer; no full text; plan-dependent history. Suitable for later research, not this H0 extension.",
        "8_2026_ytd_h0_data_complete": False,
        "9_gaps": gaps,
        "10_DATA_READY_FOR_2026_EXTENSION": "NO" if not data_ready else "NO_STRATEGY_INPUTS_STILL_MISSING",
    }
    # Frozen H0 needs D2+F1C scores on the 2026 PIT universe. Raw Massive prices are not enough.
    answers["10_DATA_READY_FOR_2026_EXTENSION"] = "NO"
    answers["8_2026_ytd_h0_data_complete"] = False

    comparison = f"""# Data source comparison

Generated: {datetime.now(timezone.utc).isoformat()}

| Dimension | WRDS / existing pipeline | Alpha Vantage | Massive |
|---|---|---|---|
| Price quality | CRSP daily used in frozen Qlib dump; research-grade corporate actions | Not the H0 price source | SIP/exchange aggregates; 2026 available if plan allows |
| Adjustment | DlyCumFacPr split-adjust; Qlib first-day normalize | N/A | `adjusted=true` split-adjust; typically not dividend-total-return |
| Historical depth | Equities 2008-01-02..2025-12-31 in this dump | News history as downloaded in S1–S5R | Plan-dependent (docs: Basic 2y bars; Advanced all since 2003-09-10) |
| Fundamentals | Compustat PIT `pitqtrdataus` unrestated + rdqe | Not used for F1C | Statements/ratios on select plans; filing_date is last restating filing |
| PIT safety | Designed for this project | News timestamps only | Prices OK if split-consistent; fundamentals **not** PIT-safe as served |
| Filing dates | rdqe / signal_start_date | N/A | EDGAR index has true submission dates; financials.filing_date does not |
| News | Not in H0 | Current sentiment pipeline; 2025 coverage drift | Structured tags + optional sentiment; no full text |
| Timestamp precision | CRSP session date | Vendor article time | Bars: Unix ms; news: RFC3339 UTC |
| S&P500 universe | CRSP member spells / sp500.txt through 2025-12-31 | No | No official index history; ETF holdings ≠ index |
| 2026 availability | **Missing** (dump ends 2025-12-31) | Unknown / not used for prices | **Yes for prices** if endpoint inventory shows 2026 bars |
| Reproducibility | Frozen local dump + scripts | API + stored shards | API vintage will drift; must snapshot |
| API stability | WRDS pull already frozen | Rate/coverage issues already observed | Live API; 429 handled; plan gates financials |

## Answers required by the audit charter

1. **What can the current Massive plan access?** See `reports/massive_audit/massive_endpoint_inventory.md`. Accessible in this run: {accessible}. Blocked: {blocked}.

2. **Better than existing sources?** 2026 daily price recency. Possibly news structure vs Alpha Vantage. EDGAR filing calendar convenience.

3. **Supplement only, cannot replace?** Fundamentals/ratios, ETF constituents-as-index, tick/minute data, news for H0.

4. **Can Massive prices support 2026 D2?** `{answers['4_price_supports_2026_d2']}`. If PASS: splice **returns** onto last CRSP/Qlib close; do not paste raw dollar levels into existing bins; do not mix CRSP and Massive volume inside the same 60-day Alpha20 window.

5. **Can Massive fundamentals strictly support 2026 F1C?** `{answers['5_fundamentals_support_2026_f1c']}`. Frozen F1C needs unrestated Compustat PIT + rdqe + CRSP mcap + CS z-score. Massive financials.filing_date is restatement-prone by vendor documentation.

6. **Can 2026 PIT S&P500 be rebuilt?** Partial. Snapshot from WRDS 2025-12-31 plus S&P DJI/Wikipedia events. Massive is **not** the membership source. New tickers lack PERMNO.

7. **Is Massive news better than Alpha Vantage for later research?** Likely yes on structure/timestamps; no full text; history depends on plan. Out of scope for H0.

8. **Does 2026 YTD already have complete data to run frozen H0?** **No.** Missing rebuilt Qlib features, D2 scores, F1C PIT panel equivalent to Compustat, PERMNO map for additions, HMM feature update. This phase explicitly did not run H0.

9. **Remaining gaps?** {gaps}

10. **DATA_READY_FOR_2026_EXTENSION?** **NO.**

Price download summary: {json.dumps(market, default=str)[:800]}

F1C diagnostic panel is not production input: {json.dumps(fund, default=str)[:500]}
"""
    (REPORT_DIR / "data_source_comparison.md").write_text(comparison, encoding="utf-8")
    (AUDIT_DIR / "final_answers.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
    print("I/J wrote comparison report; DATA_READY_FOR_2026_EXTENSION=NO")
    return answers
