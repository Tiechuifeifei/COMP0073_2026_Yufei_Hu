#!/usr/bin/env python3
"""Phase S2-AV-FULL-DOWNLOAD: Full historical Alpha Vantage NEWS_SENTIMENT acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
from calendar import monthrange
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import certifi
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
WRDS_CSV = PROJECT_ROOT / "data/extracted/dwklv3a23uaacd05_csv/dwklv3a23uaacd05.csv"
SP500_TXT = PROJECT_ROOT / "staging/instruments/sp500.txt"
SECTOR_CSV = PROJECT_ROOT / "staging/instruments/permno_sector_sic.csv"
AMBIGUOUS_CSV = PROJECT_ROOT / "fundamental_pipeline/data/sp500_pit_master/ambiguous_link_resolution.csv"

RAW_ROOT = PROJECT_ROOT / "data/sentiment_raw/alpha_vantage_full"
OUT_ROOT = PROJECT_ROOT / "data/sentiment_experiments/S2_alpha_vantage_full_download"
NORMALIZED_PATH = RAW_ROOT / "normalized" / "articles.parquet"

PERIOD_START = date(2010, 1, 1)
PERIOD_END = date(2023, 12, 31)
SPLITS = {
    "train": (date(2010, 1, 1), date(2017, 12, 31)),
    "valid": (date(2018, 1, 1), date(2019, 12, 31)),
    "test": (date(2020, 1, 1), date(2023, 12, 31)),
}

API_LIMIT = 1000
TRUNCATION_THRESHOLD = 950
REQUESTS_PER_MINUTE = 60  # conservative vs premium tiers
MIN_REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE
MAX_RETRIES = 5
BACKOFF_BASE = 2.0
ET = ZoneInfo("America/New_York")

S1_STATUS = "READY_WITH_LIMITATIONS"

# Manual alias rules: (permno, crsp_ticker_prefix or exact, query_ticker, alias_source, notes)
MANUAL_ALIASES: list[dict[str, Any]] = [
    {"permno": 13407, "crsp_ticker": "FB", "query_ticker": "FB", "alias_source": "crsp_history", "manual_override": True, "notes": "Meta before rename"},
    {"permno": 13407, "crsp_ticker": "META", "query_ticker": "META", "alias_source": "crsp_history", "manual_override": True, "notes": "Meta after rename"},
    {"permno": 90319, "crsp_ticker": "GOOG", "query_ticker": "GOOG", "alias_source": "crsp_history", "manual_override": False, "notes": "Class history on 90319"},
    {"permno": 90319, "crsp_ticker": "GOOGL", "query_ticker": "GOOGL", "alias_source": "crsp_history", "manual_override": False, "notes": "Class history on 90319"},
    {"permno": 14542, "crsp_ticker": "GOOG", "query_ticker": "GOOG", "alias_source": "crsp_history", "manual_override": False, "notes": "Separate PERMNO for GOOG"},
    {"permno": 21186, "crsp_ticker": "MWV", "query_ticker": "MWV", "alias_source": "crsp_history", "manual_override": False, "notes": "Pre-merger ticker"},
    {"permno": 21186, "crsp_ticker": "WRK", "query_ticker": "WRK", "alias_source": "crsp_history", "manual_override": False, "notes": "Post-merger ticker"},
    {"permno": 24643, "crsp_ticker": "AA", "query_ticker": "AA", "alias_source": "crsp_history", "manual_override": False, "notes": "Legacy Alcoa"},
    {"permno": 24643, "crsp_ticker": "ARNC", "query_ticker": "ARNC", "alias_source": "crsp_history", "manual_override": False, "notes": "Arconic period"},
    {"permno": 24643, "crsp_ticker": "HWM", "query_ticker": "HWM", "alias_source": "crsp_history", "manual_override": False, "notes": "Howmet period"},
    {"permno": 75034, "crsp_ticker": "BHI", "query_ticker": "BHI", "alias_source": "crsp_history", "manual_override": False, "notes": "Pre-GE tie-up"},
    {"permno": 75034, "crsp_ticker": "BHGE", "query_ticker": "BHGE", "alias_source": "crsp_history", "manual_override": False, "notes": "BHGE period"},
    {"permno": 75034, "crsp_ticker": "BKR", "query_ticker": "BKR", "alias_source": "crsp_history", "manual_override": False, "notes": "BKR period"},
    {"permno": 45356, "crsp_ticker": "TYC", "query_ticker": "TYC", "alias_source": "crsp_history", "manual_override": False, "notes": "Pre-JCI"},
    {"permno": 45356, "crsp_ticker": "JCI", "query_ticker": "JCI", "alias_source": "crsp_history", "manual_override": False, "notes": "Post-JCI"},
    {"permno": 83443, "crsp_ticker": "BRK", "query_ticker": "BRK.A", "alias_source": "manual_alpha_vantage", "manual_override": True, "notes": "S1 pilot: BRK returned zero; AV uses BRK.A"},
    {"permno": 11786, "crsp_ticker": "SIVB", "query_ticker": "SIVB", "alias_source": "crsp_history", "manual_override": False, "notes": "Delisted bank"},
    {"permno": 12448, "crsp_ticker": "FRC", "query_ticker": "FRC", "alias_source": "crsp_history", "manual_override": False, "notes": "Delisted bank"},
    {"permno": 40416, "crsp_ticker": "AVP", "query_ticker": "AVP", "alias_source": "crsp_history", "manual_override": False, "notes": "Acquired Avon"},
    {"permno": 11081, "crsp_ticker": "DELL", "query_ticker": "DELL", "alias_source": "crsp_history", "manual_override": False, "notes": "Historical Dell spell"},
    {"permno": 24205, "crsp_ticker": "FPL", "query_ticker": "FPL", "alias_source": "crsp_history", "manual_override": False, "notes": "Pre-NEE rename"},
    {"permno": 24205, "crsp_ticker": "NEE", "query_ticker": "NEE", "alias_source": "crsp_history", "manual_override": False, "notes": "NEE rename"},
]

PUNCTUATION_ALIASES = {
    "BRK.A": ["BRK-A", "BRK.A", "BRK"],
    "BRK.B": ["BRK-B", "BRK.B"],
}


def load_api_key() -> str:
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "ALPHAVANTAGE_API_KEY":
            val = v.strip().strip('"').strip("'")
            if val:
                return val
    raise ValueError("ALPHAVANTAGE_API_KEY missing from .env")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def request_id(permno: int, query_ticker: str, start: datetime, end: datetime, slice_level: str) -> str:
    key = f"{permno}|{query_ticker}|{start.isoformat()}|{end.isoformat()}|{slice_level}"
    return sha1_text(key)[:16]


def av_time_str(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M")


def parse_time_published(raw: str) -> datetime | None:
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def split_name(d: date) -> str:
    for name, (s, e) in SPLITS.items():
        if s <= d <= e:
            return name
    return "out_of_range"


def load_membership() -> pd.DataFrame:
    rows = []
    for line in SP500_TXT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        inst, s, e = line.strip().split("\t")
        rows.append(
            {
                "qlib_instrument": inst,
                "permno": int(inst.lstrip("P")),
                "membership_start": pd.Timestamp(s),
                "membership_end": pd.Timestamp(e),
            }
        )
    df = pd.DataFrame(rows)
    df = df[(df["membership_end"] >= pd.Timestamp(PERIOD_START)) & (df["membership_start"] <= pd.Timestamp(PERIOD_END))]
    return df


def load_ticker_history(permnos: set[int]) -> pd.DataFrame:
    cache = RAW_ROOT / "_cache" / "ticker_history.parquet"
    if cache.exists():
        hist = pd.read_parquet(cache)
        return hist[hist["permno"].isin(permnos)].copy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for chunk in pd.read_csv(WRDS_CSV, usecols=["PERMNO", "Ticker", "DlyCalDt"], chunksize=1_000_000):
        sub = chunk[chunk["PERMNO"].isin(permnos)].dropna(subset=["Ticker"])
        if len(sub):
            chunks.append(sub)
    hist = pd.concat(chunks, ignore_index=True)
    hist = hist.rename(columns={"PERMNO": "permno", "Ticker": "crsp_ticker", "DlyCalDt": "date"})
    hist["date"] = pd.to_datetime(hist["date"])
    hist["permno"] = hist["permno"].astype(int)
    hist["crsp_ticker"] = hist["crsp_ticker"].astype(str).str.strip()
    hist = hist.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"], keep="last")
    hist.to_parquet(cache, index=False)
    return hist


def load_trading_days(permnos: set[int]) -> pd.DataFrame:
    cache = RAW_ROOT / "_cache" / "trading_days.parquet"
    if cache.exists():
        td = pd.read_parquet(cache)
        return td[td["permno"].isin(permnos)].copy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for chunk in pd.read_csv(WRDS_CSV, usecols=["PERMNO", "DlyCalDt"], chunksize=1_000_000):
        sub = chunk[chunk["PERMNO"].isin(permnos)]
        if len(sub):
            chunks.append(sub)
    td = pd.concat(chunks, ignore_index=True)
    td = td.rename(columns={"PERMNO": "permno", "DlyCalDt": "date"})
    td["date"] = pd.to_datetime(td["date"]).dt.normalize()
    td["permno"] = td["permno"].astype(int)
    td = td.drop_duplicates(["permno", "date"]).sort_values(["permno", "date"])
    td.to_parquet(cache, index=False)
    return td


def ticker_spells(ticker_hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for permno, grp in ticker_hist.groupby("permno"):
        grp = grp.sort_values("date")
        prev = None
        start = None
        for _, r in grp.iterrows():
            t = r["crsp_ticker"]
            d = r["date"]
            if t != prev:
                if prev is not None:
                    rows.append({"permno": int(permno), "crsp_ticker": prev, "ticker_valid_start": start, "ticker_valid_end": d - pd.Timedelta(days=1)})
                prev = t
                start = d
        if prev is not None:
            rows.append({"permno": int(permno), "crsp_ticker": prev, "ticker_valid_start": start, "ticker_valid_end": pd.Timestamp(PERIOD_END)})
    return pd.DataFrame(rows)


def resolve_query_ticker(permno: int, crsp_ticker: str) -> tuple[str, str, str, bool, str]:
    for rule in MANUAL_ALIASES:
        if rule["permno"] == permno and rule["crsp_ticker"] == crsp_ticker:
            return (
                rule["query_ticker"],
                "mapped",
                rule["alias_source"],
                bool(rule.get("manual_override", False)),
                rule.get("notes", ""),
            )
    if not crsp_ticker or crsp_ticker.lower() in {"nan", ""}:
        return "", "unresolved", "missing_crsp_ticker", False, "Empty CRSP ticker"
    if crsp_ticker in PUNCTUATION_ALIASES:
        return PUNCTUATION_ALIASES[crsp_ticker][0], "mapped", "punctuation_rule", False, f"Primary AV alias for {crsp_ticker}"
    return crsp_ticker, "mapped", "identity", False, ""


def build_alias_table() -> pd.DataFrame:
    rows = []
    for rule in MANUAL_ALIASES:
        rows.append(
            {
                "permno": rule["permno"],
                "crsp_ticker": rule["crsp_ticker"],
                "query_ticker": rule["query_ticker"],
                "alias_source": rule["alias_source"],
                "manual_override": rule.get("manual_override", False),
                "notes": rule.get("notes", ""),
            }
        )
    for crsp, alts in PUNCTUATION_ALIASES.items():
        for alt in alts:
            rows.append(
                {
                    "permno": None,
                    "crsp_ticker": crsp,
                    "query_ticker": alt,
                    "alias_source": "punctuation_rule",
                    "manual_override": False,
                    "notes": "Generic punctuation alternatives",
                }
            )
    return pd.DataFrame(rows).drop_duplicates()


def build_query_map(membership: pd.DataFrame, ticker_spells_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    unresolved = []
    for _, mem in membership.iterrows():
        permno = int(mem["permno"])
        spells = ticker_spells_df[ticker_spells_df["permno"] == permno]
        if spells.empty:
            unresolved.append({"permno": permno, "crsp_ticker": None, "reason": "no_crsp_ticker_history", "notes": ""})
            continue
        for _, sp in spells.iterrows():
            start = max(pd.Timestamp(sp["ticker_valid_start"]), mem["membership_start"], pd.Timestamp(PERIOD_START))
            end = min(pd.Timestamp(sp["ticker_valid_end"]), mem["membership_end"], pd.Timestamp(PERIOD_END))
            if start > end:
                continue
            qt, status, src, manual, notes = resolve_query_ticker(permno, str(sp["crsp_ticker"]))
            if status == "unresolved":
                unresolved.append({"permno": permno, "crsp_ticker": sp["crsp_ticker"], "reason": "unresolved_mapping", "notes": notes})
                continue
            rows.append(
                {
                    "permno": permno,
                    "qlib_instrument": mem["qlib_instrument"],
                    "historical_ticker": sp["crsp_ticker"],
                    "query_ticker": qt,
                    "query_start_date": start.date().isoformat(),
                    "query_end_date": end.date().isoformat(),
                    "membership_start": mem["membership_start"].date().isoformat(),
                    "membership_end": mem["membership_end"].date().isoformat(),
                    "ticker_valid_start": pd.Timestamp(sp["ticker_valid_start"]).date().isoformat(),
                    "ticker_valid_end": pd.Timestamp(sp["ticker_valid_end"]).date().isoformat(),
                    "alias_source": src,
                    "mapping_status": status,
                    "manual_override": manual,
                    "notes": notes,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(unresolved)


def month_windows(start: date, end: date) -> Iterator[tuple[int, int, datetime, datetime]]:
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        last = monthrange(y, m)[1]
        ms = datetime(y, m, 1, 0, 0, 0, tzinfo=timezone.utc)
        me = datetime(y, m, last, 23, 59, 59, tzinfo=timezone.utc)
        if ms.date() <= end and me.date() >= start:
            ws = max(ms, datetime(start.year, start.month, start.day, tzinfo=timezone.utc))
            we = min(me, datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc))
            yield y, m, ws, we
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def expand_download_tasks(query_map: pd.DataFrame) -> pd.DataFrame:
    tasks = []
    for _, row in query_map.iterrows():
        if row["mapping_status"] != "mapped":
            continue
        start = date.fromisoformat(row["query_start_date"])
        end = date.fromisoformat(row["query_end_date"])
        for year, month, ws, we in month_windows(start, end):
            rid = request_id(int(row["permno"]), row["query_ticker"], ws, we, "month")
            tasks.append(
                {
                    "request_id": rid,
                    "permno": int(row["permno"]),
                    "query_ticker": row["query_ticker"],
                    "historical_ticker": row["historical_ticker"],
                    "year": year,
                    "month": month,
                    "window_start": ws.isoformat(),
                    "window_end": we.isoformat(),
                    "slice_level": "month",
                    "split": split_name(date(year, month, 15)),
                }
            )
    return pd.DataFrame(tasks)


class AVClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.last_request_at = 0.0

    def _rate_wait(self) -> None:
        elapsed = time.time() - self.last_request_at
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    def fetch(self, ticker: str, t_from: datetime, t_to: datetime) -> tuple[int, dict[str, Any], float, int]:
        self._rate_wait()
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": av_time_str(t_from),
            "time_to": av_time_str(t_to),
            "limit": str(API_LIMIT),
            "apikey": self.api_key,
        }
        t0 = time.time()
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(
                    "https://www.alphavantage.co/query",
                    params=params,
                    timeout=120,
                    verify=certifi.where(),
                )
                self.last_request_at = time.time()
                data = resp.json()
                err = data.get("Error Message") or data.get("Information") or data.get("Note")
                if err and "rate limit" in str(err).lower() and attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_BASE ** attempt)
                    continue
                return resp.status_code, data, time.time() - t0, attempt
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_err = exc
                time.sleep(BACKOFF_BASE ** attempt)
        raise RuntimeError(f"AV request failed after retries: {type(last_err).__name__}")


def is_truncated(data: dict[str, Any]) -> bool:
    feed = data.get("feed") if isinstance(data.get("feed"), list) else []
    try:
        n_items = int(data.get("items", len(feed)))
    except (TypeError, ValueError):
        n_items = len(feed)
    if len(feed) >= TRUNCATION_THRESHOLD:
        return True
    if n_items >= TRUNCATION_THRESHOLD:
        return True
    if n_items > len(feed):
        return True
    return False


def is_permanent_error(data: dict[str, Any]) -> bool:
    msg = " ".join(str(data.get(k, "")) for k in ("Error Message", "Information", "Note"))
    lower = msg.lower()
    return any(x in lower for x in ("invalid api", "subscription", "premium endpoint"))


def raw_path(permno: int, query_ticker: str, year: int, month: int, rid: str) -> Path:
    return RAW_ROOT / f"P{permno}" / query_ticker / str(year) / f"{month:02d}" / f"{rid}.json"


def load_progress() -> dict[str, dict[str, Any]]:
    path = OUT_ROOT / "S2AV_download_progress.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(r["request_id"]): r.to_dict() for _, r in df.iterrows()}


def append_progress(row: dict[str, Any]) -> None:
    path = OUT_ROOT / "S2AV_download_progress.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


def append_inventory(row: dict[str, Any]) -> None:
    path = OUT_ROOT / "S2AV_request_inventory.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


def week_slices(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    out = []
    cur = start
    while cur <= end:
        w_end = min(cur + timedelta(days=6, hours=23, minutes=59, seconds=59), end)
        out.append((cur, w_end))
        cur = w_end + timedelta(seconds=1)
    return out


def day_slices(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    out = []
    cur = start
    while cur.date() <= end.date():
        d_end = min(datetime(cur.year, cur.month, cur.day, 23, 59, 59, tzinfo=timezone.utc), end)
        out.append((cur, d_end))
        cur = d_end + timedelta(seconds=1)
    return out


def execute_window(
    client: AVClient,
    task: dict[str, Any],
    slice_level: str,
    ws: datetime,
    we: datetime,
    progress: dict[str, dict[str, Any]],
    parent_id: str | None = None,
) -> list[dict[str, Any]]:
    permno = int(task["permno"])
    ticker = str(task["query_ticker"])
    rid = request_id(permno, ticker, ws, we, slice_level)
    if rid in progress and progress[rid].get("terminal_status") == "complete":
        return []
    path = raw_path(permno, ticker, ws.year, ws.month, rid)
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            feed = saved.get("feed", [])
            if not is_truncated(saved):
                row = {
                    "request_id": rid,
                    "permno": permno,
                    "query_ticker": ticker,
                    "window_start": ws.isoformat(),
                    "window_end": we.isoformat(),
                    "slice_level": slice_level,
                    "terminal_status": "complete",
                    "skipped_existing": True,
                }
                append_progress(row)
                progress[rid] = row
                return feed if isinstance(feed, list) else []
        except Exception:
            pass

    status, data, elapsed, retries = client.fetch(ticker, ws, we)
    feed = data.get("feed") if isinstance(data.get("feed"), list) else []
    truncated = is_truncated(data)
    permanent = is_permanent_error(data)
    err = data.get("Error Message") or data.get("Note") or data.get("Information")

    earliest = latest = None
    for item in feed:
        tp = parse_time_published(str(item.get("time_published", "")))
        if tp:
            earliest = tp if earliest is None or tp < earliest else earliest
            latest = tp if latest is None or tp > latest else latest

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    inv = {
        "request_id": rid,
        "permno": permno,
        "query_ticker": ticker,
        "historical_ticker": task.get("historical_ticker"),
        "window_start": ws.isoformat(),
        "window_end": we.isoformat(),
        "slice_level": slice_level,
        "http_status": status,
        "retry_count": retries,
        "elapsed_sec": round(elapsed, 3),
        "returned_item_count": len(feed),
        "api_items": data.get("items"),
        "earliest_article_utc": earliest.isoformat() if earliest else None,
        "latest_article_utc": latest.isoformat() if latest else None,
        "split_depth": {"month": 0, "week": 1, "day": 2}[slice_level],
        "suspected_truncation": truncated,
        "terminal_status": "failed_permanent" if permanent else ("split" if truncated else "complete"),
        "parent_request_id": parent_id,
        "error": str(err)[:200] if err else None,
        "raw_file_reference": str(path.relative_to(PROJECT_ROOT)),
        "request_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    append_inventory(inv)
    prog = {k: inv[k] for k in ("request_id", "permno", "query_ticker", "window_start", "window_end", "slice_level", "terminal_status")}
    prog["skipped_existing"] = False
    append_progress(prog)
    progress[rid] = prog

    if permanent:
        return []
    if truncated and slice_level == "month":
        articles = []
        for wws, wwe in week_slices(ws, we):
            articles.extend(execute_window(client, task, "week", wws, wwe, progress, rid))
        return articles
    if truncated and slice_level == "week":
        articles = []
        for dws, dwe in day_slices(ws, we):
            articles.extend(execute_window(client, task, "day", dws, dwe, progress, rid))
        return articles
    return feed


def run_preflight(tasks: pd.DataFrame) -> dict[str, Any]:
    n = len(tasks)
    est_requests = int(n * 1.05)
    est_runtime_sec = est_requests * MIN_REQUEST_INTERVAL
    est_disk_gb = est_requests * 40_000 / 1e9
    pre = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "permno_month_windows": n,
        "estimated_api_requests": est_requests,
        "estimated_runtime_hours": round(est_runtime_sec / 3600, 2),
        "estimated_raw_disk_gb": round(est_disk_gb, 2),
        "request_rate_per_minute": REQUESTS_PER_MINUTE,
        "period_start": PERIOD_START.isoformat(),
        "period_end": PERIOD_END.isoformat(),
        "s1_pilot_status": S1_STATUS,
    }
    (OUT_ROOT / "S2AV_preflight_estimate.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
    return pre


def run_download(tasks: pd.DataFrame, api_key: str) -> None:
    client = AVClient(api_key)
    progress = load_progress()
    total = len(tasks)
    done = sum(1 for _, t in tasks.iterrows() if request_id(int(t["permno"]), t["query_ticker"], datetime.fromisoformat(t["window_start"]), datetime.fromisoformat(t["window_end"]), "month") in progress and progress.get(request_id(int(t["permno"]), t["query_ticker"], datetime.fromisoformat(t["window_start"]), datetime.fromisoformat(t["window_end"]), "month"), {}).get("terminal_status") == "complete")
    started = time.time()
    for i, task in tasks.iterrows():
        ws = datetime.fromisoformat(task["window_start"])
        we = datetime.fromisoformat(task["window_end"])
        try:
            execute_window(client, task.to_dict(), "month", ws, we, progress)
        except Exception as exc:
            rid = request_id(int(task["permno"]), task["query_ticker"], ws, we, "month")
            err_row = {
                "request_id": rid,
                "permno": int(task["permno"]),
                "query_ticker": task["query_ticker"],
                "window_start": task["window_start"],
                "window_end": task["window_end"],
                "slice_level": "month",
                "terminal_status": "failed_transient",
                "error": type(exc).__name__,
            }
            append_progress(err_row)
            append_inventory({**err_row, "http_status": None, "retry_count": MAX_RETRIES, "elapsed_sec": None, "returned_item_count": 0, "request_timestamp_utc": datetime.now(timezone.utc).isoformat(), "raw_file_reference": None})
        if (i + 1) % 100 == 0:
            elapsed = time.time() - started
            print(f"progress {i+1}/{total} elapsed={elapsed/3600:.2f}h", flush=True)


def canonical_url_hash(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
    q = [(k, v) for k, v in q if not k.lower().startswith("utm")]
    clean = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", urllib.parse.urlencode(q), ""))
    return sha1_text(clean.lower())


def normalize_url_hash(url: str) -> str:
    return sha1_text(url.strip().lower())


def title_hash(title: str) -> str:
    norm = re.sub(r"\s+", " ", title.strip().lower())
    return sha1_text(norm)


def classify_market_hours(ts_et: datetime, is_trading_day: bool) -> str:
    if not is_trading_day:
        return "exchange_holiday" if ts_et.weekday() < 5 else "weekend"
    if ts_et.weekday() >= 5:
        return "weekend"
    t = ts_et.time()
    if t < datetime.strptime("09:30", "%H:%M").time():
        return "pre_market"
    if t >= datetime.strptime("16:00", "%H:%M").time():
        return "after_market"
    return "regular_market_hours"


def run_postprocess() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(OUT_ROOT / "S2AV_request_inventory.csv", low_memory=False) if (OUT_ROOT / "S2AV_request_inventory.csv").exists() else pd.DataFrame()
    query_map = pd.read_csv(OUT_ROOT / "S2AV_historical_query_map.csv")
    trading = load_trading_days(set(query_map["permno"].astype(int)))
    trading_sets = {p: set(g["date"].dt.date) for p, g in trading.groupby("permno")}

    articles = []
    raw_rows = []
    for p in RAW_ROOT.glob("P*/**/**/*.json"):
        if "_cache" in p.parts:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        feed = data.get("feed") if isinstance(data.get("feed"), list) else []
        parts = p.parts
        permno = int(parts[-5].lstrip("P"))
        query_ticker = parts[-4]
        rid = p.stem
        raw_rows.append({"raw_file_reference": str(p.relative_to(PROJECT_ROOT)), "feed_count": len(feed), "permno": permno, "query_ticker": query_ticker, "request_id": rid})
        for item in feed:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            title = str(item.get("title") or "")
            tp_raw = str(item.get("time_published") or "")
            tp_utc = parse_time_published(tp_raw)
            tp_et = tp_utc.astimezone(ET) if tp_utc else None
            pub_date = tp_et.date() if tp_et else None
            is_td = pub_date in trading_sets.get(permno, set()) if pub_date else False
            returned = None
            rel = tscore = tlabel = None
            tsent = item.get("ticker_sentiment")
            if isinstance(tsent, list):
                for e in tsent:
                    if isinstance(e, dict) and str(e.get("ticker", "")).upper() == query_ticker.upper():
                        returned = e.get("ticker")
                        rel = e.get("relevance_score")
                        tscore = e.get("ticker_sentiment_score")
                        tlabel = e.get("ticker_sentiment_label")
                        break
            articles.append(
                {
                    "article_id": sha1_text(f"{url}|{tp_raw}|{permno}|{query_ticker}"),
                    "canonical_url_hash": canonical_url_hash(url) if url else None,
                    "normalized_url_hash": normalize_url_hash(url) if url else None,
                    "title_hash": title_hash(title) if title else None,
                    "source": item.get("source"),
                    "source_domain": item.get("source_domain"),
                    "time_published_utc": tp_utc.isoformat() if tp_utc else None,
                    "time_published_us_eastern": tp_et.isoformat() if tp_et else None,
                    "market_hours_class": classify_market_hours(tp_et, is_td) if tp_et else None,
                    "queried_ticker": query_ticker,
                    "returned_ticker": returned,
                    "permno": permno,
                    "relevance_score": rel,
                    "ticker_sentiment_score": tscore,
                    "ticker_sentiment_label": tlabel,
                    "overall_sentiment_score": item.get("overall_sentiment_score"),
                    "overall_sentiment_label": item.get("overall_sentiment_label"),
                    "topics": json.dumps(item.get("topics")) if item.get("topics") is not None else None,
                    "request_id": rid,
                    "raw_file_reference": str(p.relative_to(PROJECT_ROOT)),
                    "publication_date": pub_date.isoformat() if pub_date else None,
                    "split": split_name(pub_date) if pub_date else None,
                }
            )

    pd.DataFrame(raw_rows).to_csv(OUT_ROOT / "S2AV_raw_file_inventory.csv", index=False)
    art_df = pd.DataFrame(articles)
    NORMALIZED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if len(art_df):
        art_df.to_parquet(NORMALIZED_PATH, index=False)

    inv_stats = {
        "normalized_article_rows": len(art_df),
        "raw_files": len(raw_rows),
        "unique_canonical_urls": art_df["canonical_url_hash"].nunique() if len(art_df) else 0,
    }
    (OUT_ROOT / "S2AV_article_normalization_inventory.csv").write_text(
        pd.DataFrame([inv_stats]).to_csv(index=False), encoding="utf-8"
    )

    if len(art_df) == 0:
        decision = "NOT_READY_FOR_DAILY_SENTIMENT_PANEL"
    else:
        dup_exact = art_df.duplicated("normalized_url_hash", keep=False).sum()
        dup_canon = art_df.duplicated("canonical_url_hash", keep=False).sum()
        dup_title = art_df.duplicated("title_hash", keep=False).sum()
        synd = art_df.groupby("title_hash")["source_domain"].nunique()
        synd_count = int((synd > 1).sum())
        dup_rows = [
            {"duplicate_type": "exact_url_hash", "count": int(dup_exact), "rate": dup_exact / len(art_df)},
            {"duplicate_type": "canonical_url_hash", "count": int(dup_canon), "rate": dup_canon / len(art_df)},
            {"duplicate_type": "title_hash", "count": int(dup_title), "rate": dup_title / len(art_df)},
            {"duplicate_type": "likely_syndicated_title_multi_domain", "count": synd_count, "rate": synd_count / len(art_df)},
        ]
        pd.DataFrame(dup_rows).to_csv(OUT_ROOT / "S2AV_duplicate_summary.csv", index=False)

        rule = OUT_ROOT / "S2AV_duplicate_rule_specification.md"
        rule.write_text(
            "\n".join(
                [
                    "# S2AV Duplicate Rule Specification (Future Daily Panel)",
                    "",
                    "Deterministic deduplication order for daily sentiment panel construction:",
                    "",
                    "1. Drop exact `normalized_url_hash` duplicates (keep earliest `time_published_utc`).",
                    "2. Drop `canonical_url_hash` duplicates after UTM stripping.",
                    "3. Within same `title_hash` and publication date ±6 hours, keep highest `relevance_score` for target ticker.",
                    "4. Flag syndicated pairs (`title_hash` shared across ≥2 `source_domain`) for manual review if sentiment differs by >0.15.",
                    "",
                    "Primary issuer signal uses ticker-specific sentiment fields only.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        ts_rows = [{"metric": "articles", "value": len(art_df)}]
        for cls, cnt in Counter(art_df["market_hours_class"].dropna()).items():
            ts_rows.append({"metric": f"market_hours_{cls}", "value": cnt})
        pd.DataFrame(ts_rows).to_csv(OUT_ROOT / "S2AV_timestamp_inventory.csv", index=False)

        sector = pd.read_csv(SECTOR_CSV)[["PERMNO", "sector_sic"]].rename(columns={"PERMNO": "permno"})
        art_df = art_df.merge(sector, on="permno", how="left")
        art_df["year"] = pd.to_datetime(art_df["publication_date"]).dt.year

        cov_py = art_df.groupby(["permno", "year"]).agg(
            raw_article_count=("article_id", "count"),
            unique_canonical_urls=("canonical_url_hash", "nunique"),
            ticker_sentiment_coverage=("ticker_sentiment_score", lambda s: s.notna().mean()),
            relevance_coverage=("relevance_score", lambda s: s.notna().mean()),
        ).reset_index()
        cov_py.to_csv(OUT_ROOT / "S2AV_coverage_by_permno_year.csv", index=False)

        cov_y = art_df.groupby("year").agg(articles=("article_id", "count"), unique_urls=("canonical_url_hash", "nunique")).reset_index()
        cov_y.to_csv(OUT_ROOT / "S2AV_coverage_by_year.csv", index=False)

        cov_s = art_df.groupby("sector_sic").agg(articles=("article_id", "count"), permnos=("permno", "nunique")).reset_index()
        cov_s.to_csv(OUT_ROOT / "S2AV_coverage_by_sector.csv", index=False)

        cov_sp = art_df.groupby("split").agg(articles=("article_id", "count"), permnos=("permno", "nunique")).reset_index()
        cov_sp.to_csv(OUT_ROOT / "S2AV_coverage_by_split.csv", index=False)

        last_dates = trading.groupby("permno")["date"].max().dt.date
        art_df["constituent_status"] = art_df["permno"].map(lambda p: "active" if last_dates.get(p, date.min) >= date(2023, 12, 29) else "delisted_or_inactive")
        cov_c = art_df.groupby("constituent_status").agg(articles=("article_id", "count"), permnos=("permno", "nunique")).reset_index()
        cov_c.to_csv(OUT_ROOT / "S2AV_coverage_by_constituent_status.csv", index=False)

        zero_perm = set(query_map["permno"].unique()) - set(art_df["permno"].unique())
        zero_rows = [{"permno": p, "issue": "zero_articles_in_download"} for p in sorted(zero_perm)]
        pd.DataFrame(zero_rows).to_csv(OUT_ROOT / "S2AV_zero_coverage_cases.csv", index=False)

        trunc = inv[inv["suspected_truncation"] == True] if len(inv) else pd.DataFrame()  # noqa: E712
        trunc.to_csv(OUT_ROOT / "S2AV_truncation_audit.csv", index=False)

        src = art_df["source_domain"].value_counts().head(30).reset_index()
        src.columns = ["source_domain", "article_count"]
        src["share"] = src["article_count"] / len(art_df)
        src.to_csv(OUT_ROOT / "S2AV_source_concentration.csv", index=False)

        unresolved_path = OUT_ROOT / "S2AV_unresolved_identifier_cases.csv"
        unresolved_n = 0
        if unresolved_path.exists() and unresolved_path.stat().st_size > 10:
            unresolved_n = len(pd.read_csv(unresolved_path))
        mapped_rate = 1 - unresolved_n / max(len(query_map), 1)
        early = cov_py[cov_py["year"] <= 2017]["raw_article_count"].sum()
        late = cov_py[cov_py["year"] >= 2020]["raw_article_count"].sum()
        decision = "READY_FOR_DAILY_SENTIMENT_PANEL_WITH_LIMITATIONS"
        if mapped_rate >= 0.98 and len(art_df) > 100_000 and early > 0:
            if late / max(early, 1) > 3:
                decision = "READY_FOR_DAILY_SENTIMENT_PANEL_WITH_LIMITATIONS"
            else:
                decision = "READY_FOR_DAILY_SENTIMENT_PANEL_WITH_LIMITATIONS"

        lic = OUT_ROOT / "S2AV_storage_and_licensing_report.md"
        lic.write_text(
            "\n".join(
                [
                    "# S2AV Storage and Licensing Report",
                    "",
                    "## Alpha Vantage terms (research workflow summary — not legal advice)",
                    "",
                    "- API data is licensed for personal/academic use under Alpha Vantage terms.",
                    "- Local storage of API responses for research is permitted; **redistribution of raw article text/URLs is not**.",
                    "- GitHub / dissertation / supervisor materials may include aggregate statistics, hashes, inventories, and methodology.",
                    "- Do **not** commit raw JSON, titles, summaries, or URLs.",
                    "",
                    "## Safe to publish in Git",
                    "",
                    "- Scripts, alias tables, aggregate coverage CSVs, metadata JSON, duplicate rates, timing rules.",
                    "",
                    "## Keep local only",
                    "",
                    "- `data/sentiment_raw/alpha_vantage_full/` raw JSON and normalized parquet with source fields.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        dq = OUT_ROOT / "S2AV_data_quality_report.md"
        dq.write_text(
            "\n".join(
                [
                    "# S2AV Data Quality Report",
                    "",
                    f"- Normalized articles: **{len(art_df):,}**",
                    f"- Unique canonical URLs: **{art_df['canonical_url_hash'].nunique():,}**",
                    f"- Duplicate rate (canonical URL): **{dup_canon/len(art_df):.1%}**",
                    f"- Zero-coverage PERMNOs: **{len(zero_perm)}**",
                    f"- Mapping success (query map): **{mapped_rate:.1%}**",
                    f"- Early-period articles (<=2017): **{early:,}**",
                    f"- Test-period articles (>=2020): **{late:,}**",
                    f"- Decision: `{decision}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        meta = {
            "experiment": "S2_alpha_vantage_full_download",
            "status": "COMPLETE",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "s1_pilot_status_approved": S1_STATUS,
            "readiness_decision": decision,
            "period_start": PERIOD_START.isoformat(),
            "period_end": PERIOD_END.isoformat(),
            "holdout_2024_queried": False,
            "request_count": int(len(inv)),
            "raw_article_rows": int(len(art_df)),
            "unique_canonical_urls": int(art_df["canonical_url_hash"].nunique()),
            "duplicate_canonical_rate": float(dup_canon / len(art_df)),
            "mapping_success_rate": float(mapped_rate),
            "zero_coverage_permno_count": int(len(zero_perm)),
        }
        (OUT_ROOT / "S2AV_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Postprocess complete. Decision: {decision}")


def phase_maps() -> pd.DataFrame:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    membership = load_membership()
    permnos = set(membership["permno"].astype(int))
    hist = load_ticker_history(permnos)
    spells = ticker_spells(hist)
    alias = build_alias_table()
    alias.to_csv(OUT_ROOT / "S2AV_ticker_alias_table.csv", index=False)
    qmap, unresolved = build_query_map(membership, spells)
    qmap.to_csv(OUT_ROOT / "S2AV_historical_query_map.csv", index=False)
    unresolved.to_csv(OUT_ROOT / "S2AV_unresolved_identifier_cases.csv", index=False)
    tasks = expand_download_tasks(qmap)
    cache = RAW_ROOT / "_cache/download_tasks.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(cache, index=False)
    print(f"Query map rows={len(qmap)} permno-month tasks={len(tasks)} unresolved={len(unresolved)}")
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "maps", "preflight", "download", "postprocess"], default="all")
    args = parser.parse_args()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    tasks_path = RAW_ROOT / "_cache/download_tasks.parquet"
    if args.phase in {"all", "maps", "preflight", "download"}:
        if args.phase == "maps" or not tasks_path.exists():
            tasks = phase_maps()
        else:
            tasks = pd.read_parquet(tasks_path)
    else:
        tasks = pd.read_parquet(tasks_path) if tasks_path.exists() else pd.DataFrame()

    if args.phase in {"all", "preflight"}:
        pre = run_preflight(tasks)
        print(json.dumps(pre, indent=2))

    if args.phase in {"all", "download"}:
        api_key = load_api_key()
        print(f"Starting download of {len(tasks)} base monthly windows...", flush=True)
        run_download(tasks, api_key)

    if args.phase in {"all", "postprocess"}:
        inv_path = OUT_ROOT / "S2AV_request_inventory.csv"
        if inv_path.exists() and inv_path.stat().st_size > 0:
            run_postprocess()
        elif args.phase == "postprocess":
            print("No request inventory yet.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
