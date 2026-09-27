#!/usr/bin/env python3
"""Phase S1-AV-PILOT: Alpha Vantage NEWS_SENTIMENT data-quality and PIT feasibility audit."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from calendar import monthrange
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import certifi
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
WRDS_CSV = PROJECT_ROOT / "data/extracted/dwklv3a23uaacd05_csv/dwklv3a23uaacd05.csv"
SP500_TXT = PROJECT_ROOT / "staging/instruments/sp500.txt"
AMBIGUOUS_CSV = PROJECT_ROOT / "fundamental_pipeline/data/sp500_pit_master/ambiguous_link_resolution.csv"

RAW_ROOT = PROJECT_ROOT / "data/sentiment_raw/alpha_vantage_pilot"
OUT_ROOT = PROJECT_ROOT / "data/sentiment_experiments/S1_alpha_vantage_pilot"

PILOT_YEARS = [2010, 2012, 2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
API_LIMIT = 1000
REQUEST_SLEEP_SEC = 0.35
ET = ZoneInfo("America/New_York")

SPLITS = {
    "train": (date(2010, 1, 1), date(2017, 12, 31)),
    "valid": (date(2018, 1, 1), date(2019, 12, 31)),
    "test": (date(2020, 1, 1), date(2023, 12, 31)),
}

# Frozen before download. av_ticker is the default query symbol; date-specific overrides come from CRSP.
FROZEN_PILOT_SYMBOLS: list[dict[str, Any]] = [
    {"pilot_id": "P01", "permno": 14593, "default_av_ticker": "AAPL", "criteria": "high_news_large_cap|technology"},
    {"pilot_id": "P02", "permno": 10107, "default_av_ticker": "MSFT", "criteria": "high_news_large_cap|technology"},
    {"pilot_id": "P03", "permno": 84788, "default_av_ticker": "AMZN", "criteria": "high_news_large_cap|consumer"},
    {"pilot_id": "P04", "permno": 90319, "default_av_ticker": "GOOGL", "criteria": "share_class|high_news_large_cap"},
    {"pilot_id": "P05", "permno": 14542, "default_av_ticker": "GOOG", "criteria": "share_class|high_news_large_cap"},
    {"pilot_id": "P06", "permno": 93436, "default_av_ticker": "TSLA", "criteria": "high_news_large_cap|late_index_add"},
    {"pilot_id": "P07", "permno": 13407, "default_av_ticker": "META", "criteria": "high_news_large_cap|ticker_change"},
    {"pilot_id": "P08", "permno": 47896, "default_av_ticker": "JPM", "criteria": "finance|high_news"},
    {"pilot_id": "P09", "permno": 11850, "default_av_ticker": "XOM", "criteria": "energy|large_cap"},
    {"pilot_id": "P10", "permno": 92655, "default_av_ticker": "UNH", "criteria": "healthcare|large_cap"},
    {"pilot_id": "P11", "permno": 55976, "default_av_ticker": "WMT", "criteria": "retail|moderate_news"},
    {"pilot_id": "P12", "permno": 18729, "default_av_ticker": "CL", "criteria": "consumer_staples|lower_news"},
    {"pilot_id": "P13", "permno": 24205, "default_av_ticker": "NEE", "criteria": "utility|lower_news"},
    {"pilot_id": "P14", "permno": 18163, "default_av_ticker": "PG", "criteria": "consumer_staples|lower_news"},
    {"pilot_id": "P15", "permno": 21186, "default_av_ticker": "WRK", "criteria": "ticker_change|merger_successor"},
    {"pilot_id": "P16", "permno": 24643, "default_av_ticker": "HWM", "criteria": "ticker_change|spinoff_chain"},
    {"pilot_id": "P17", "permno": 75034, "default_av_ticker": "BKR", "criteria": "ticker_change|energy"},
    {"pilot_id": "P18", "permno": 45356, "default_av_ticker": "JCI", "criteria": "ticker_change|index_reentry"},
    {"pilot_id": "P19", "permno": 11786, "default_av_ticker": "SIVB", "criteria": "delisted|bank_stress"},
    {"pilot_id": "P20", "permno": 12448, "default_av_ticker": "FRC", "criteria": "delisted|bank_stress"},
    {"pilot_id": "P21", "permno": 13688, "default_av_ticker": "PCG", "criteria": "multi_spell_index|utility"},
    {"pilot_id": "P22", "permno": 24328, "default_av_ticker": "EQT", "criteria": "multi_spell_index|energy"},
    {"pilot_id": "P23", "permno": 59328, "default_av_ticker": "INTC", "criteria": "technology|semiconductor"},
    {"pilot_id": "P24", "permno": 11081, "default_av_ticker": "DELL", "criteria": "delisted_relisted|technology"},
    {"pilot_id": "P25", "permno": 40416, "default_av_ticker": "AVP", "criteria": "acquired_delisted|consumer"},
    {"pilot_id": "P26", "permno": 83443, "default_av_ticker": "BRK", "criteria": "finance|conglomerate"},
    {"pilot_id": "P27", "permno": 11308, "default_av_ticker": "KO", "criteria": "consumer_staples|lower_news"},
    {"pilot_id": "P28", "permno": 10104, "default_av_ticker": "ORCL", "criteria": "technology|enterprise"},
]

FIELD_NAMES = [
    "time_published",
    "title",
    "summary",
    "source",
    "source_domain",
    "url",
    "topics",
    "overall_sentiment_score",
    "ticker_sentiment",
    "relevance_score",
    "ticker_sentiment_score",
    "ticker_sentiment_label",
]

AUDITED_FEED_FIELDS = [
    "time_published",
    "title",
    "summary",
    "source",
    "source_domain",
    "url",
    "topics",
    "overall_sentiment_score",
    "ticker_sentiment",
]

AUDITED_TICKER_FIELDS = [
    "ticker",
    "relevance_score",
    "ticker_sentiment_score",
    "ticker_sentiment_label",
]


@dataclass
class RequestRecord:
    pilot_id: str
    permno: int
    av_ticker: str
    window_start: datetime
    window_end: datetime
    slice_level: str
    http_status: int | None = None
    api_items: str | None = None
    articles_returned: int = 0
    truncated: bool = False
    retried: bool = False
    parent_request_id: str | None = None
    error: str | None = None
    raw_path: str | None = None
    elapsed_sec: float = 0.0

    @property
    def request_id(self) -> str:
        key = f"{self.pilot_id}|{self.av_ticker}|{self.window_start.isoformat()}|{self.window_end.isoformat()}|{self.slice_level}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]


@dataclass
class ArticleRecord:
    pilot_id: str
    permno: int
    av_ticker_queried: str
    av_ticker_matched: str | None
    url: str
    time_published_raw: str
    time_published_utc: datetime | None
    time_published_et: datetime | None
    title: str | None
    summary: str | None
    source: str | None
    source_domain: str | None
    topics: str | None
    overall_sentiment_score: float | None
    relevance_score: float | None
    ticker_sentiment_score: float | None
    ticker_sentiment_label: str | None
    field_presence: dict[str, bool] = field(default_factory=dict)
    crsp_ticker_on_date: str | None = None
    crsp_ticker_match: bool | None = None
    eligible_signal_date: date | None = None
    market_hours_class: str | None = None
    weekend_flag: bool | None = None
    duplicate_url: bool = False


def load_api_key() -> str:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Missing {ENV_PATH}")
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip() == "ALPHAVANTAGE_API_KEY":
            v = val.strip().strip('"').strip("'")
            if v:
                return v
    raise ValueError("ALPHAVANTAGE_API_KEY not found in .env")


def pilot_month(year: int) -> int:
    return 3 if year == 2020 else 1


def month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    last_day = monthrange(year, month)[1]
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return start, end


def av_time_str(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M")


def parse_time_published(raw: str) -> datetime | None:
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def load_membership() -> pd.DataFrame:
    rows = []
    for line in SP500_TXT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        inst, start, end = line.strip().split("\t")
        rows.append(
            {
                "qlib_instrument": inst,
                "permno": int(inst.lstrip("P")),
                "membership_start": pd.Timestamp(start),
                "membership_end": pd.Timestamp(end),
            }
        )
    return pd.DataFrame(rows)


def in_membership(membership: pd.DataFrame, permno: int, start: datetime, end: datetime) -> bool:
    spells = membership[membership["permno"] == permno]
    if spells.empty:
        return False
    s = pd.Timestamp(start.date())
    e = pd.Timestamp(end.date())
    for _, row in spells.iterrows():
        if row["membership_start"] <= e and row["membership_end"] >= s:
            return True
    return False


def load_ticker_history(permnos: set[int]) -> pd.DataFrame:
    cache = RAW_ROOT / "_cache_ticker_history.parquet"
    if cache.exists():
        hist = pd.read_parquet(cache)
        return hist[hist["permno"].isin(permnos)].copy()

    chunks = []
    for chunk in pd.read_csv(WRDS_CSV, usecols=["PERMNO", "Ticker", "DlyCalDt", "SICCD"], chunksize=750_000):
        sub = chunk[chunk["PERMNO"].isin(permnos)].dropna(subset=["Ticker"])
        if len(sub):
            chunks.append(sub)
    hist = pd.concat(chunks, ignore_index=True)
    hist = hist.rename(columns={"PERMNO": "permno", "Ticker": "ticker", "DlyCalDt": "date", "SICCD": "siccd"})
    hist["date"] = pd.to_datetime(hist["date"])
    hist["permno"] = hist["permno"].astype(int)
    hist["ticker"] = hist["ticker"].astype(str).str.strip()
    hist = hist.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"], keep="last")
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(cache, index=False)
    return hist


def load_trading_days(permnos: set[int], start: date, end: date) -> pd.DataFrame:
    cache = RAW_ROOT / "_cache_trading_days.parquet"
    if cache.exists():
        td = pd.read_parquet(cache)
        mask = (td["date"] >= pd.Timestamp(start)) & (td["date"] <= pd.Timestamp(end))
        return td[td["permno"].isin(permnos) & mask].copy()

    chunks = []
    for chunk in pd.read_csv(WRDS_CSV, usecols=["PERMNO", "DlyCalDt"], chunksize=750_000):
        sub = chunk[chunk["PERMNO"].isin(permnos)]
        if len(sub):
            chunks.append(sub)
    td = pd.concat(chunks, ignore_index=True)
    td = td.rename(columns={"PERMNO": "permno", "DlyCalDt": "date"})
    td["date"] = pd.to_datetime(td["date"]).dt.normalize()
    td["permno"] = td["permno"].astype(int)
    td = td.drop_duplicates(["permno", "date"]).sort_values(["permno", "date"])
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    td.to_parquet(cache, index=False)
    mask = (td["date"] >= pd.Timestamp(start)) & (td["date"] <= pd.Timestamp(end))
    return td[td["permno"].isin(permnos) & mask].copy()


def ticker_on_date(ticker_hist: pd.DataFrame, permno: int, dt: date) -> str | None:
    sub = ticker_hist[(ticker_hist["permno"] == permno) & (ticker_hist["date"] <= pd.Timestamp(dt))]
    if sub.empty:
        return None
    return str(sub.iloc[-1]["ticker"])


def ticker_changes(ticker_hist: pd.DataFrame, permno: int) -> list[tuple[date, str]]:
    sub = ticker_hist[ticker_hist["permno"] == permno].sort_values("date")
    if sub.empty:
        return []
    out = []
    prev = None
    for _, row in sub.iterrows():
        t = str(row["ticker"])
        d = row["date"].date()
        if t != prev:
            out.append((d, t))
            prev = t
    return out


class AlphaVantageClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.use_curl_fallback = False

    def fetch(self, ticker: str, time_from: datetime, time_to: datetime) -> tuple[int, dict[str, Any]]:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": av_time_str(time_from),
            "time_to": av_time_str(time_to),
            "limit": str(API_LIMIT),
            "apikey": self.api_key,
        }
        t0 = time.time()
        if not self.use_curl_fallback:
            try:
                resp = self.session.get(
                    "https://www.alphavantage.co/query",
                    params=params,
                    timeout=90,
                    verify=certifi.where(),
                )
                elapsed = time.time() - t0
                data = resp.json()
                return resp.status_code, data
            except requests.exceptions.SSLError:
                self.use_curl_fallback = True
            except Exception as exc:
                raise RuntimeError(f"requests failed: {type(exc).__name__}") from exc

        # curl fallback without exposing key in logs
        cmd = [
            "curl",
            "-sS",
            "-G",
            "https://www.alphavantage.co/query",
            "--data-urlencode",
            "function=NEWS_SENTIMENT",
            "--data-urlencode",
            f"tickers={ticker}",
            "--data-urlencode",
            f"time_from={av_time_str(time_from)}",
            "--data-urlencode",
            f"time_to={av_time_str(time_to)}",
            "--data-urlencode",
            f"limit={API_LIMIT}",
            "--data-urlencode",
            f"apikey={self.api_key}",
            "-w",
            "\n__HTTP__:%{http_code}",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"curl failed exit={proc.returncode}")
        raw = proc.stdout
        if "\n__HTTP__:" in raw:
            body, code = raw.rsplit("\n__HTTP__:", 1)
            status = int(code.strip())
        else:
            body, status = raw, 200
        return status, json.loads(body)


def is_api_error(data: dict[str, Any]) -> str | None:
    for key in ("Error Message", "Information", "Note"):
        if key in data:
            return f"{key}: {str(data[key])[:200]}"
    return None


def is_truncated(data: dict[str, Any]) -> bool:
    feed = data.get("feed")
    if not isinstance(feed, list):
        return False
    items = data.get("items")
    try:
        n_items = int(items)
    except (TypeError, ValueError):
        n_items = len(feed)
    if len(feed) >= API_LIMIT:
        return True
    if n_items > len(feed):
        return True
    if n_items == API_LIMIT:
        return True
    return False


def iter_slices(start: datetime, end: datetime, level: str):
    if level == "month":
        yield start, end, "month"
        return
    if level == "week":
        cur = start
        while cur <= end:
            w_end = min(cur + timedelta(days=6, hours=23, minutes=59, seconds=59), end)
            yield cur, w_end, "week"
            cur = w_end + timedelta(seconds=1)
        return
    if level == "day":
        cur = start
        while cur.date() <= end.date():
            d_end = min(datetime(cur.year, cur.month, cur.day, 23, 59, 59, tzinfo=timezone.utc), end)
            yield cur, d_end, "day"
            cur = d_end + timedelta(seconds=1)
        return
    raise ValueError(level)


def classify_market_hours(ts_et: datetime) -> str:
    if ts_et.weekday() >= 5:
        return "weekend"
    t = ts_et.time()
    if t < datetime.strptime("09:30", "%H:%M").time():
        return "pre_market"
    if t >= datetime.strptime("16:00", "%H:%M").time():
        return "after_market_close"
    return "regular_market_hours"


def eligible_signal_date(ts_et: datetime, trading_dates: list[date]) -> date | None:
    if not trading_dates:
        return None
    pub_date = ts_et.date()
    trading_set = set(trading_dates)
    if classify_market_hours(ts_et) in {"after_market_close", "weekend"}:
        candidates = [d for d in trading_dates if d > pub_date]
    else:
        candidates = [d for d in trading_dates if d >= pub_date]
    return candidates[0] if candidates else None


def split_for_year(year: int) -> str:
    for name, (s, e) in SPLITS.items():
        if s <= date(year, 1, 1) <= e or s <= date(year, 12, 31) <= e:
            if date(year, 6, 1) >= s and date(year, 6, 1) <= e:
                return name
    if year <= 2017:
        return "train"
    if year <= 2019:
        return "valid"
    return "test"


def write_pilot_spec(path: Path) -> pd.DataFrame:
    rows = []
    membership = load_membership()
    for spec in FROZEN_PILOT_SYMBOLS:
        permno = int(spec["permno"])
        spells = membership[membership["permno"] == permno]
        rows.append(
            {
                "pilot_id": spec["pilot_id"],
                "permno": permno,
                "qlib_instrument": f"P{permno}",
                "default_av_ticker": spec["default_av_ticker"],
                "selection_criteria": spec["criteria"],
                "membership_spell_count": int(len(spells)),
                "membership_start_min": spells["membership_start"].min() if len(spells) else None,
                "membership_end_max": spells["membership_end"].max() if len(spells) else None,
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def adaptive_download(
    client: AlphaVantageClient,
    pilot_id: str,
    permno: int,
    av_ticker: str,
    start: datetime,
    end: datetime,
    raw_dir: Path,
    request_records: list[RequestRecord],
    parent_id: str | None = None,
) -> list[dict[str, Any]]:
    rec = RequestRecord(
        pilot_id=pilot_id,
        permno=permno,
        av_ticker=av_ticker,
        window_start=start,
        window_end=end,
        slice_level="month" if parent_id is None else ("week" if (end - start).days > 1 else "day"),
        retried=parent_id is not None,
        parent_request_id=parent_id,
    )
    t0 = time.time()
    try:
        status, data = client.fetch(av_ticker, start, end)
        rec.http_status = status
        rec.elapsed_sec = time.time() - t0
        err = is_api_error(data)
        if err:
            rec.error = err
            request_records.append(rec)
            time.sleep(REQUEST_SLEEP_SEC)
            return []
        feed = data.get("feed") if isinstance(data.get("feed"), list) else []
        rec.api_items = str(data.get("items", len(feed)))
        rec.articles_returned = len(feed)
        rec.truncated = is_truncated(data)
        raw_path = raw_dir / f"{rec.request_id}.json"
        redacted = {k: v for k, v in data.items() if k != "feed"}
        redacted["feed_count"] = len(feed)
        redacted["feed"] = feed
        raw_path.write_text(json.dumps(redacted, ensure_ascii=False), encoding="utf-8")
        rec.raw_path = str(raw_path.relative_to(PROJECT_ROOT))
        request_records.append(rec)
        time.sleep(REQUEST_SLEEP_SEC)

        if rec.truncated and rec.slice_level == "month":
            articles: list[dict[str, Any]] = []
            for ws, we, lvl in iter_slices(start, end, "week"):
                articles.extend(
                    adaptive_download(client, pilot_id, permno, av_ticker, ws, we, raw_dir, request_records, rec.request_id)
                )
            return articles
        if rec.truncated and rec.slice_level == "week":
            articles = []
            for ws, we, lvl in iter_slices(start, end, "day"):
                articles.extend(
                    adaptive_download(client, pilot_id, permno, av_ticker, ws, we, raw_dir, request_records, rec.request_id)
                )
            return articles
        return feed
    except Exception as exc:
        rec.error = f"{type(exc).__name__}"
        rec.elapsed_sec = time.time() - t0
        request_records.append(rec)
        time.sleep(REQUEST_SLEEP_SEC)
        return []


def parse_article(
    item: dict[str, Any],
    pilot_id: str,
    permno: int,
    av_ticker_queried: str,
) -> ArticleRecord:
    url = str(item.get("url") or "")
    tp_raw = str(item.get("time_published") or "")
    tp_utc = parse_time_published(tp_raw)
    tp_et = tp_utc.astimezone(ET) if tp_utc else None
    matched = None
    rel = tscore = tlabel = None
    tsent = item.get("ticker_sentiment")
    if isinstance(tsent, list):
        for entry in tsent:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("ticker", "")).upper() == av_ticker_queried.upper():
                matched = str(entry.get("ticker"))
                rel = _to_float(entry.get("relevance_score"))
                tscore = _to_float(entry.get("ticker_sentiment_score"))
                tlabel = entry.get("ticker_sentiment_label")
                break
        if matched is None and tsent:
            e0 = tsent[0]
            if isinstance(e0, dict):
                matched = str(e0.get("ticker"))
                rel = _to_float(e0.get("relevance_score"))
                tscore = _to_float(e0.get("ticker_sentiment_score"))
                tlabel = e0.get("ticker_sentiment_label")

    presence = {
        "time_published": bool(tp_raw),
        "title": bool(item.get("title")),
        "summary": bool(item.get("summary")),
        "source": bool(item.get("source")),
        "source_domain": bool(item.get("source_domain")),
        "url": bool(url),
        "topics": bool(item.get("topics")),
        "overall_sentiment_score": item.get("overall_sentiment_score") not in (None, ""),
        "ticker_sentiment": isinstance(tsent, list) and len(tsent) > 0,
        "relevance_score": rel is not None,
        "ticker_sentiment_score": tscore is not None,
        "ticker_sentiment_label": bool(tlabel),
    }
    return ArticleRecord(
        pilot_id=pilot_id,
        permno=permno,
        av_ticker_queried=av_ticker_queried,
        av_ticker_matched=matched,
        url=url,
        time_published_raw=tp_raw,
        time_published_utc=tp_utc,
        time_published_et=tp_et,
        title=item.get("title"),
        summary=item.get("summary"),
        source=item.get("source"),
        source_domain=item.get("source_domain"),
        topics=json.dumps(item.get("topics")) if item.get("topics") is not None else None,
        overall_sentiment_score=_to_float(item.get("overall_sentiment_score")),
        relevance_score=rel,
        ticker_sentiment_score=tscore,
        ticker_sentiment_label=tlabel,
        field_presence=presence,
    )


def _to_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def decide_feasibility(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    if (
        checks["mapping_success_rate"] >= 0.85
        and checks["ticker_sentiment_field_rate"] >= 0.90
        and checks["timestamp_parse_rate"] >= 0.99
        and checks["api_error_rate"] <= 0.05
        and checks["median_news_day_coverage"] >= 0.05
    ):
        return "READY_FOR_2010_2023_SENTIMENT_PIPELINE"
    if checks["api_error_rate"] <= 0.15 and checks["ticker_sentiment_field_rate"] >= 0.75:
        return "READY_WITH_LIMITATIONS"
    return "NOT_SUITABLE"


def main() -> int:
    started = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    spec_path = OUT_ROOT / "S1AV_pilot_symbol_specification.csv"
    pilot_spec = write_pilot_spec(spec_path)

    api_key = load_api_key()
    client = AlphaVantageClient(api_key)
    membership = load_membership()
    permnos = {int(x["permno"]) for x in FROZEN_PILOT_SYMBOLS}
    ticker_hist = load_ticker_history(permnos)
    trading_days = load_trading_days(permnos, date(2010, 1, 1), date(2023, 12, 31))
    trading_by_permno: dict[int, list[date]] = {
        p: sorted(g["date"].dt.date.unique().tolist()) for p, g in trading_days.groupby("permno")
    }

    ambiguous = pd.read_csv(AMBIGUOUS_CSV) if AMBIGUOUS_CSV.exists() else pd.DataFrame()

    request_records: list[RequestRecord] = []
    all_articles: list[ArticleRecord] = []

    for spec in FROZEN_PILOT_SYMBOLS:
        pilot_id = spec["pilot_id"]
        permno = int(spec["permno"])
        raw_dir = RAW_ROOT / pilot_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        for year in PILOT_YEARS:
            month = pilot_month(year)
            start, end = month_bounds(year, month)
            if not in_membership(membership, permno, start, end):
                continue
            mid = date(year, month, 15)
            av_ticker = ticker_on_date(ticker_hist, permno, mid) or spec["default_av_ticker"]
            feeds = adaptive_download(client, pilot_id, permno, av_ticker, start, end, raw_dir, request_records)
            for item in feeds:
                if isinstance(item, dict):
                    all_articles.append(parse_article(item, pilot_id, permno, av_ticker))

    # dedupe by URL globally and within symbol
    url_counts = Counter(a.url for a in all_articles if a.url)
    seen_symbol_url: set[tuple[str, str]] = set()
    for art in all_articles:
        if art.url and url_counts[art.url] > 1:
            art.duplicate_url = True
        key = (art.pilot_id, art.url)
        if key in seen_symbol_url:
            art.duplicate_url = True
        seen_symbol_url.add(key)

        if art.time_published_et:
            art.market_hours_class = classify_market_hours(art.time_published_et)
            art.weekend_flag = art.time_published_et.weekday() >= 5
            td = trading_by_permno.get(art.permno, [])
            art.eligible_signal_date = eligible_signal_date(art.time_published_et, td)
        pub_date = art.time_published_utc.date() if art.time_published_utc else None
        if pub_date:
            crsp_t = ticker_on_date(ticker_hist, art.permno, pub_date)
            art.crsp_ticker_on_date = crsp_t
            art.crsp_ticker_match = (
                crsp_t is not None
                and art.av_ticker_matched is not None
                and crsp_t.upper() == art.av_ticker_matched.upper()
            )

    # --- output tables ---
    req_df = pd.DataFrame([r.__dict__ for r in request_records])
    req_df.to_csv(OUT_ROOT / "S1AV_request_inventory.csv", index=False)

    if all_articles:
        art_df = pd.DataFrame([a.__dict__ for a in all_articles])
        art_df["field_presence"] = art_df["field_presence"].apply(json.dumps)
    else:
        art_df = pd.DataFrame()

    # field coverage
    field_rows = []
    n = max(len(all_articles), 1)
    for fname in FIELD_NAMES:
        if fname in {"relevance_score", "ticker_sentiment_score", "ticker_sentiment_label"}:
            rate = sum(1 for a in all_articles if a.field_presence.get(fname)) / n
        elif fname == "ticker_sentiment":
            rate = sum(1 for a in all_articles if a.field_presence.get(fname)) / n
        else:
            rate = sum(1 for a in all_articles if a.field_presence.get(fname)) / n
        field_rows.append({"field_name": fname, "coverage_rate": rate, "articles_with_field": int(rate * len(all_articles))})
    pd.DataFrame(field_rows).to_csv(OUT_ROOT / "S1AV_field_coverage.csv", index=False)

    # coverage by symbol-year
    cov_sy_rows = []
    for spec in FROZEN_PILOT_SYMBOLS:
        pid = spec["pilot_id"]
        permno = int(spec["permno"])
        sub = [a for a in all_articles if a.pilot_id == pid]
        td_list = trading_by_permno.get(permno, [])
        for year in PILOT_YEARS:
            month = pilot_month(year)
            if not in_membership(membership, permno, *month_bounds(year, month)):
                continue
            y_start, y_end = month_bounds(year, month)
            ys = [a for a in sub if a.time_published_utc and y_start <= a.time_published_utc <= y_end]
            unique_urls = len({a.url for a in ys if a.url})
            days_with_news = len({a.eligible_signal_date for a in ys if a.eligible_signal_date})
            month_td = [d for d in td_list if date(year, month, 1) <= d <= date(year, month, monthrange(year, month)[1])]
            cov_sy_rows.append(
                {
                    "pilot_id": pid,
                    "permno": permno,
                    "year": year,
                    "pilot_month": month,
                    "split": split_for_year(year),
                    "article_count": len(ys),
                    "unique_article_count": unique_urls,
                    "trading_days_in_month": len(month_td),
                    "trading_days_with_news": days_with_news,
                    "pct_trading_days_with_news": (days_with_news / len(month_td)) if month_td else None,
                    "ticker_sentiment_coverage": sum(1 for a in ys if a.field_presence.get("ticker_sentiment_score")) / max(len(ys), 1),
                    "relevance_coverage": sum(1 for a in ys if a.field_presence.get("relevance_score")) / max(len(ys), 1),
                    "duplicate_rate": sum(1 for a in ys if a.duplicate_url) / max(len(ys), 1),
                    "earliest_time_published_utc": min((a.time_published_utc for a in ys if a.time_published_utc), default=None),
                    "latest_time_published_utc": max((a.time_published_utc for a in ys if a.time_published_utc), default=None),
                }
            )
    pd.DataFrame(cov_sy_rows).to_csv(OUT_ROOT / "S1AV_coverage_by_symbol_year.csv", index=False)

    # coverage by split
    split_rows = []
    cov_sy = pd.DataFrame(cov_sy_rows)
    for split_name in ["train", "valid", "test"]:
        s = cov_sy[cov_sy["split"] == split_name]
        split_rows.append(
            {
                "split": split_name,
                "symbol_months": len(s),
                "total_articles": int(s["article_count"].sum()) if len(s) else 0,
                "unique_articles": int(s["unique_article_count"].sum()) if len(s) else 0,
                "mean_pct_trading_days_with_news": float(s["pct_trading_days_with_news"].mean()) if len(s) else None,
                "median_pct_trading_days_with_news": float(s["pct_trading_days_with_news"].median()) if len(s) else None,
                "mean_ticker_sentiment_coverage": float(s["ticker_sentiment_coverage"].mean()) if len(s) else None,
                "mean_relevance_coverage": float(s["relevance_coverage"].mean()) if len(s) else None,
                "mean_duplicate_rate": float(s["duplicate_rate"].mean()) if len(s) else None,
            }
        )
    pd.DataFrame(split_rows).to_csv(OUT_ROOT / "S1AV_coverage_by_split.csv", index=False)

    # timestamp audit
    ts_rows = []
    parsed = [a for a in all_articles if a.time_published_utc]
    ts_rows.append({"metric": "articles_total", "value": len(all_articles)})
    ts_rows.append({"metric": "timestamp_parse_success_rate", "value": len(parsed) / max(len(all_articles), 1)})
    ts_rows.append({"metric": "timestamp_format", "value": "YYYYMMDDTHHMMSS assumed UTC per Alpha Vantage docs"})
    ts_rows.append({"metric": "timezone_assumption", "value": "UTC input converted to America/New_York for market-hours audit"})
    ts_rows.append({"metric": "precision", "value": "minute-level (seconds present in sample)"})
    for cls, cnt in Counter(a.market_hours_class for a in parsed if a.market_hours_class).items():
        ts_rows.append({"metric": f"market_hours_{cls}", "value": cnt})
    ts_rows.append({"metric": "weekend_publications", "value": sum(1 for a in parsed if a.weekend_flag)})
    ts_rows.append({"metric": "duplicate_or_syndicated_urls", "value": sum(1 for a in all_articles if a.duplicate_url)})
    pd.DataFrame(ts_rows).to_csv(OUT_ROOT / "S1AV_timestamp_audit.csv", index=False)

    # identifier mapping audit
    id_rows = []
    for spec in FROZEN_PILOT_SYMBOLS:
        permno = int(spec["permno"])
        changes = ticker_changes(ticker_hist, permno)
        sub = [a for a in all_articles if a.permno == permno]
        id_rows.append(
            {
                "pilot_id": spec["pilot_id"],
                "permno": permno,
                "default_av_ticker": spec["default_av_ticker"],
                "crsp_ticker_change_count": max(len(changes) - 1, 0),
                "crsp_ticker_history": "; ".join(f"{d}:{t}" for d, t in changes[:6]),
                "articles_with_crsp_match": sum(1 for a in sub if a.crsp_ticker_match),
                "articles_with_crsp_mismatch": sum(1 for a in sub if a.crsp_ticker_match is False),
                "articles_unmapped": sum(1 for a in sub if a.crsp_ticker_match is None),
                "current_ticker_survivorship_risk": spec["default_av_ticker"] != (changes[-1][1] if changes else ""),
            }
        )
    pd.DataFrame(id_rows).to_csv(OUT_ROOT / "S1AV_identifier_mapping_audit.csv", index=False)

    issue_rows = []
    for spec in FROZEN_PILOT_SYMBOLS:
        permno = int(spec["permno"])
        changes = ticker_changes(ticker_hist, permno)
        if len(changes) > 1:
            issue_rows.append({"pilot_id": spec["pilot_id"], "permno": permno, "issue_type": "ticker_change", "detail": "; ".join(f"{d}:{t}" for d, t in changes)})
        if spec["criteria"].find("share_class") >= 0:
            issue_rows.append({"pilot_id": spec["pilot_id"], "permno": permno, "issue_type": "share_class", "detail": "Separate CRSP PERMNO for GOOG vs GOOGL; query ticker must match spell/date"})
        if spec["criteria"].find("delisted") >= 0:
            issue_rows.append({"pilot_id": spec["pilot_id"], "permno": permno, "issue_type": "delisted", "detail": "Historical queries must use delisted ticker, not successor"})
        if spec["criteria"].find("multi_spell") >= 0:
            issue_rows.append({"pilot_id": spec["pilot_id"], "permno": permno, "issue_type": "index_reentry", "detail": "Multiple S&P membership spells; skip non-member pilot months"})
    if not ambiguous.empty:
        for permno in permnos:
            sub = ambiguous[ambiguous["permno"] == permno]
            if len(sub):
                issue_rows.append({"pilot_id": next(s["pilot_id"] for s in FROZEN_PILOT_SYMBOLS if s["permno"] == permno), "permno": permno, "issue_type": "ccm_ambiguous_link", "detail": sub["evidence"].iloc[0][:200]})
    pd.DataFrame(issue_rows).to_csv(OUT_ROOT / "S1AV_identifier_issues.csv", index=False)

    dup_rows = [
        {"metric": "duplicate_url_articles", "value": sum(1 for a in all_articles if a.duplicate_url)},
        {"metric": "duplicate_rate", "value": sum(1 for a in all_articles if a.duplicate_url) / max(len(all_articles), 1)},
        {"metric": "unique_urls", "value": len({a.url for a in all_articles if a.url})},
    ]
    pd.DataFrame(dup_rows).to_csv(OUT_ROOT / "S1AV_duplicate_audit.csv", index=False)

    src_counter = Counter(a.source_domain for a in all_articles if a.source_domain)
    total = sum(src_counter.values()) or 1
    src_rows = [{"source_domain": dom, "article_count": cnt, "share": cnt / total} for dom, cnt in src_counter.most_common(25)]
    pd.DataFrame(src_rows).to_csv(OUT_ROOT / "S1AV_source_concentration.csv", index=False)

    trunc_df = req_df[req_df["truncated"] == True]  # noqa: E712
    trunc_df.to_csv(OUT_ROOT / "S1AV_truncation_audit.csv", index=False)

    api_errors = req_df[req_df["error"].notna()]
    mapping_success = sum(1 for a in all_articles if a.crsp_ticker_match) / max(sum(1 for a in all_articles if a.crsp_ticker_match is not None), 1)
    ts_rate = len([a for a in all_articles if a.time_published_utc]) / max(len(all_articles), 1)
    tscore_rate = sum(1 for a in all_articles if a.field_presence.get("ticker_sentiment_score")) / max(len(all_articles), 1)
    med_cov = float(pd.DataFrame(cov_sy_rows)["pct_trading_days_with_news"].median()) if cov_sy_rows else 0.0

    summary_checks = {
        "mapping_success_rate": mapping_success,
        "timestamp_parse_rate": ts_rate,
        "ticker_sentiment_field_rate": tscore_rate,
        "api_error_rate": len(api_errors) / max(len(req_df), 1),
        "median_news_day_coverage": med_cov,
        "truncation_events": int(len(trunc_df)),
        "queried_2024": False,
    }
    decision = decide_feasibility({"checks": summary_checks})

    timing_md = OUT_ROOT / "S1AV_point_in_time_timing_specification.md"
    timing_md.write_text(
        "\n".join(
            [
                "# S1AV Point-in-Time Timing Specification (Draft)",
                "",
                "## Label anchor (frozen fundamental stream)",
                "",
                "Target label: `Ref($close, -2) / Ref($close, -1) - 1`",
                "",
                "Interpretation: signal dated *t* must use only information available before the return window used by the label.",
                "",
                "## Alpha Vantage publication timestamp",
                "",
                "- Field: `time_published`",
                "- Parsed as UTC (`YYYYMMDDTHHMMSS` or `YYYYMMDDTHHMM`)",
                "- Converted to `America/New_York` for session classification",
                "",
                "## Strict eligibility rule (pilot)",
                "",
                "1. Parse `time_published` to UTC, convert to US/Eastern.",
                "2. Classify publication session:",
                "   - `pre_market` (< 09:30 ET)",
                "   - `regular_market_hours` (09:30–16:00 ET)",
                "   - `after_market_close` (>= 16:00 ET)",
                "   - `weekend`",
                "3. Assign `eligible_signal_date` = first CRSP trading day for the PERMNO such that:",
                "   - if `after_market_close` or `weekend`: first trading day **strictly after** publication calendar date;",
                "   - otherwise: first trading day **on or after** publication calendar date.",
                "4. **No news item may affect a signal dated before `time_published`.**",
                "",
                "## Primary company signal",
                "",
                "Use ticker-specific fields from `ticker_sentiment` (`ticker_sentiment_score`, `relevance_score`, `ticker_sentiment_label`),",
                "not `overall_sentiment_score`, for issuer-level features.",
                "",
                "## Pilot feasibility notes",
                "",
                f"- Articles parsed: {len(all_articles)}",
                f"- Timestamp parse rate: {ts_rate:.1%}",
                f"- CRSP ticker match rate (where comparable): {mapping_success:.1%}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report_md = OUT_ROOT / "S1AV_feasibility_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# S1AV Alpha Vantage Pilot Feasibility Report",
                "",
                f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
                f"**Decision:** `{decision}`",
                "",
                "## Pilot universe",
                "",
                f"- Frozen symbols: **{len(FROZEN_PILOT_SYMBOLS)}**",
                f"- Pilot years/months: {PILOT_YEARS} (January each year; **March 2020**)",
                "- S&P membership enforced per month before querying",
                "",
                "## API execution",
                "",
                f"- Requests: **{len(req_df)}**",
                f"- API errors: **{len(api_errors)}**",
                f"- Truncation events: **{len(trunc_df)}**",
                f"- HTTP client: `requests` + `certifi` ({'curl fallback used' if client.use_curl_fallback else 'no fallback needed'})",
                f"- Runtime seconds: **{time.time() - started:.1f}**",
                "",
                "## Coverage highlights",
                "",
                "| split | symbol_months | total_articles | mean_pct_trading_days_with_news | mean_ticker_sentiment_coverage |",
                "|---|---:|---:|---:|---:|",
            ]
            + [
                f"| {r['split']} | {r['symbol_months']} | {r['total_articles']} | {r['mean_pct_trading_days_with_news']:.1%} | {r['mean_ticker_sentiment_coverage']:.1%} |"
                if r["mean_pct_trading_days_with_news"] is not None
                else f"| {r['split']} | {r['symbol_months']} | {r['total_articles']} | n/a | n/a |"
                for r in split_rows
            ]
            + [
                "## Identifier / PIT highlights",
                "",
                f"- CRSP ticker match rate: **{mapping_success:.1%}**",
                f"- Ticker-specific sentiment field rate: **{tscore_rate:.1%}**",
                f"- Median trading-day news coverage (pilot months): **{med_cov:.1%}**",
                "",
                "## Exclusions",
                "",
                "- No 2024 queries",
                "- No model training",
                "- No IC / Rank IC / ARR / portfolio metrics",
                "- Raw article JSON stored only under `data/sentiment_raw/alpha_vantage_pilot/` (gitignored)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    meta = {
        "experiment": "S1_alpha_vantage_pilot",
        "status": "COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feasibility_decision": decision,
        "pilot_symbol_count": len(FROZEN_PILOT_SYMBOLS),
        "pilot_years": PILOT_YEARS,
        "request_count": int(len(req_df)),
        "runtime_seconds": round(time.time() - started, 2),
        "article_records_parsed": len(all_articles),
        "api_error_count": int(len(api_errors)),
        "truncation_event_count": int(len(trunc_df)),
        "checks": summary_checks,
        "http_client": "requests+certifi" if not client.use_curl_fallback else "curl_fallback",
        "raw_data_dir": str(RAW_ROOT.relative_to(PROJECT_ROOT)),
        "holdout_2024_queried": False,
    }
    (OUT_ROOT / "S1AV_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Decision: {decision}")
    print(f"Requests: {len(req_df)} Articles: {len(all_articles)} Runtime: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
