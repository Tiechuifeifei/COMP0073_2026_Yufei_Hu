"""Paths and constants for the Massive data audit. No strategy execution."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_BASE = "https://api.massive.com"
API_KEY_ENV = "MASSIVE_API_KEY"

AUDIT_DIR = PROJECT_ROOT / "data" / "massive_audit"
REPORT_DIR = PROJECT_ROOT / "reports" / "massive_audit"
RAW_2026 = PROJECT_ROOT / "data" / "massive_2026" / "raw"
NORM_2026 = PROJECT_ROOT / "data" / "massive_2026" / "normalized"
MANIFEST_2026 = PROJECT_ROOT / "data" / "massive_2026" / "manifests"
UNIVERSE_2026 = PROJECT_ROOT / "data" / "massive_2026" / "universe"
FUND_2026 = PROJECT_ROOT / "data" / "massive_2026" / "fundamentals"

CRSP_CSV_DIR = PROJECT_ROOT / "staging" / "csv"
SPY_CSV = PROJECT_ROOT / "staging" / "csv_benchmark" / "p84398.csv"
SP500_TXT = PROJECT_ROOT / "staging" / "instruments" / "sp500.txt"
TICKER_MAP = (
    PROJECT_ROOT
    / "data"
    / "portfolio_experiments"
    / "final_holdout"
    / "market_style_diagnostics"
    / "s5a"
    / "download"
    / "s5a_holdout_ticker_map.csv"
)
MEMBERSHIP_SPELLS = (
    PROJECT_ROOT / "fundamental_pipeline" / "data" / "sp500_pit_master" / "membership_spells.parquet"
)
F1_FEATURES_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "daily_fundamental_features"
    / "main"
    / "daily_fundamental_features.parquet"
)

PRICE_OVERLAP_START = "2023-01-01"
PRICE_OVERLAP_END = "2025-12-31"
LOOKBACK_START = "2025-01-01"
SNAPSHOT_DATE = "2025-12-31"

PRICE_SAMPLE = [
    ("AAPL", 14593),
    ("NVDA", 86580),
    ("JPM", 47896),
    ("XOM", 11850),
    ("JNJ", 22111),
    ("AMZN", 84788),
    ("TSLA", 93436),
    ("MSFT", 10107),
    ("GOOGL", 90319),
]

F1C_SAMPLE_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "JPM", "BAC", "WFC", "GS",
    "XOM", "CVX", "COP",
    "JNJ", "UNH", "PFE", "ABBV",
    "WMT", "PG", "KO", "COST",
    "CAT", "HON", "BA",
    "NEE", "DUK",
    "AMT", "PLD",
]

F1C_FEATURES = [
    "roe_z",
    "roa_z",
    "gross_profitability_z",
    "sales_growth_yoy_z",
    "asset_growth_yoy_z",
    "accruals_z",
    "leverage_z",
    "current_ratio_z",
    "book_to_market_z",
    "earnings_yield_z",
    "sales_to_price_z",
]


def api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith(f"{API_KEY_ENV}="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} is not set")
    return key


def ensure_dirs() -> None:
    for path in [
        AUDIT_DIR,
        REPORT_DIR,
        RAW_2026,
        NORM_2026,
        MANIFEST_2026,
        UNIVERSE_2026,
        FUND_2026,
    ]:
        path.mkdir(parents=True, exist_ok=True)
