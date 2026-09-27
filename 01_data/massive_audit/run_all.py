#!/usr/bin/env python3
"""Run the Massive data audit and 2026 reconstruction stages (no H0 portfolio run)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from massive_audit.config import AUDIT_DIR, REPORT_DIR, ensure_dirs
from massive_audit.phase_a import write_required_inventory
from massive_audit.phase_b import run_endpoint_inventory
from massive_audit.phase_c import run_price_equivalence
from massive_audit.phase_d import run_f1c_audit
from massive_audit.phase_e import run_universe_2026
from massive_audit.phase_f import run_market_download
from massive_audit.phase_g import run_fundamentals_2026
from massive_audit.phase_h import run_news_coverage
from massive_audit.phase_i import write_final_reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--sample-download", action="store_true", help="Force sample tickers only for phase F")
    args = parser.parse_args()
    ensure_dirs()
    started = datetime.now(timezone.utc).isoformat()

    write_required_inventory()
    endpoints = run_endpoint_inventory()
    price = run_price_equivalence()
    f1c = run_f1c_audit()
    universe = run_universe_2026()
    if args.skip_download:
        market = {"skipped": True}
    else:
        full = None if not args.sample_download else False
        market = run_market_download(price.get("verdict", ""), full=full)
    fund = run_fundamentals_2026(f1c.get("verdict", "NOT_AVAILABLE"))
    news = run_news_coverage()
    answers = write_final_reports(endpoints, price, f1c, universe, market, fund, news)
    run_manifest = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "price": price,
        "f1c": f1c,
        "universe": universe,
        "market": {k: market.get(k) for k in ("n_tickers_ok", "failed_tickers", "full_universe_download", "skipped") if k in market or True},
        "fund": fund,
        "news_n": news.get("n_sample_rows"),
        "DATA_READY_FOR_2026_EXTENSION": answers.get("10_DATA_READY_FOR_2026_EXTENSION"),
    }
    (AUDIT_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"DATA_READY_FOR_2026_EXTENSION": answers.get("10_DATA_READY_FOR_2026_EXTENSION")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
