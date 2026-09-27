"""Phase E: reconstruct 2026 S&P 500 PIT membership without backfilling today's list."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from massive_audit.client import MassiveClient
from massive_audit.config import (
    AUDIT_DIR,
    SNAPSHOT_DATE,
    SP500_TXT,
    TICKER_MAP,
    UNIVERSE_2026,
    ensure_dirs,
)

# Compiled from S&P DJI press releases as collated by Wikipedia (retrieved 2026-08-18).
# effective_date = first trading date the change is in the index ("prior to the open").
# Massive does not provide official S&P 500 constituent history.
SP500_2026_EVENTS = [
    {"effective_date": "2026-02-09", "announcement_date": "2026-02-04", "added_ticker": "CIEN", "removed_ticker": "DAY", "reason": "Thoma Bravo acquired Dayforce", "source": "S&P DJI PR 2026-02-04 / Wikipedia"},
    {"effective_date": "2026-03-23", "announcement_date": "2026-03-06", "added_ticker": "VRT", "removed_ticker": "MTCH", "reason": "Market capitalization changes", "source": "S&P DJI PR 2026-03-06 / Wikipedia"},
    {"effective_date": "2026-03-23", "announcement_date": "2026-03-06", "added_ticker": "LITE", "removed_ticker": "MOH", "reason": "Market capitalization changes", "source": "S&P DJI PR 2026-03-06 / Wikipedia"},
    {"effective_date": "2026-03-23", "announcement_date": "2026-03-06", "added_ticker": "COHR", "removed_ticker": "LW", "reason": "Market capitalization changes", "source": "S&P DJI PR 2026-03-06 / Wikipedia"},
    {"effective_date": "2026-03-23", "announcement_date": "2026-03-06", "added_ticker": "SATS", "removed_ticker": "PAYC", "reason": "Market capitalization changes", "source": "S&P DJI PR 2026-03-06 / Wikipedia"},
    {"effective_date": "2026-04-09", "announcement_date": "2026-04-06", "added_ticker": "CASY", "removed_ticker": "HOLX", "reason": "Blackstone/TPG acquired Hologic", "source": "S&P DJI PR 2026-04-06 / Wikipedia"},
    {"effective_date": "2026-05-07", "announcement_date": "2026-04-30", "added_ticker": "VEEV", "removed_ticker": "CTRA", "reason": "Devon Energy acquired Coterra", "source": "S&P DJI PR 2026-04-30 / Wikipedia"},
    {"effective_date": "2026-06-01", "announcement_date": "2026-05-27", "added_ticker": "FDXF", "removed_ticker": "EPAM", "reason": "FedEx Freight spin-off; EPAM removed on market-cap change (Wikipedia splits 06-01/06-02)", "source": "S&P DJI PR 2026-05-27 / Wikipedia; pairing documented as uncertain"},
    {"effective_date": "2026-06-22", "announcement_date": "2026-06-05", "added_ticker": "MRVL", "removed_ticker": "POOL", "reason": "Market capitalization changes", "source": "S&P DJI PR 2026-06-05 / Wikipedia"},
    {"effective_date": "2026-06-22", "announcement_date": "2026-06-05", "added_ticker": "FLEX", "removed_ticker": "CPB", "reason": "Market capitalization changes", "source": "S&P DJI PR 2026-06-05 / Wikipedia"},
    {"effective_date": "2026-06-29", "announcement_date": "2026-06-23", "added_ticker": "HONA", "removed_ticker": "CAG", "reason": "Honeywell Aerospace spin-off; CAG removed (Wikipedia lists CAG on 2026-06-30)", "source": "S&P DJI PR 2026-06-23 / Wikipedia; pairing documented as uncertain"},
    {"effective_date": "2026-08-05", "announcement_date": "2026-07-31", "added_ticker": "FERG", "removed_ticker": "EA", "reason": "PIF/Silver Lake/Affinity acquired Electronic Arts", "source": "S&P DJI PR 2026-07-31 / Wikipedia"},
    {"effective_date": "2026-08-18", "announcement_date": "2026-08-13", "added_ticker": "RDDT", "removed_ticker": "AVB", "reason": "Equity Residential acquired AvalonBay; combined firm remains", "source": "S&P DJI PR 2026-08-13 / PR Newswire"},
]


def _load_ticker_map() -> pd.DataFrame:
    m = pd.read_csv(TICKER_MAP)
    m["permno"] = m["permno"].astype(int)
    m["query_ticker"] = m["query_ticker"].astype(str).str.upper()
    return m


def run_universe_2026() -> dict:
    ensure_dirs()
    client = MassiveClient()
    status, payload, _ = client.get("/etf-global/v1/constituents", {"ticker": "SPY", "limit": 5})
    massive_has_index_members = status == 200
    note_massive = (
        "Massive returned SPY ETF constituents"
        if massive_has_index_members
        else f"Massive ETF constituents inaccessible (HTTP {status}); not used as S&P500 PIT"
    )

    spells = pd.read_csv(SP500_TXT, sep="\t", header=None, names=["instrument", "start", "end"])
    spells["permno"] = spells["instrument"].str.replace("P", "", regex=False).astype(int)
    spells["start"] = pd.to_datetime(spells["start"])
    spells["end"] = pd.to_datetime(spells["end"])
    snap_day = pd.Timestamp(SNAPSHOT_DATE)
    snap = spells[(spells["start"] <= snap_day) & (spells["end"] >= snap_day)].copy()
    tmap = _load_ticker_map()
    snap = snap.merge(tmap[["permno", "query_ticker"]], on="permno", how="left")
    snap["ticker"] = snap["query_ticker"]
    snap["asof"] = SNAPSHOT_DATE
    snap["source"] = "WRDS/Qlib staging/instruments/sp500.txt membership containing 2025-12-31"
    snap_out = snap[["asof", "permno", "instrument", "ticker", "start", "end", "source"]].rename(
        columns={"start": "spell_start", "end": "spell_end"}
    )
    snap_out.to_csv(UNIVERSE_2026 / "sp500_2025_12_31_snapshot.csv", index=False)

    events = pd.DataFrame(SP500_2026_EVENTS)
    events.to_csv(UNIVERSE_2026 / "sp500_2026_membership_events.csv", index=False)

    # Daily membership 2026-01-02 .. 2026-08-18 using snapshot tickers + events.
    cal_status, cal_payload, _ = client.get(
        "/v2/aggs/ticker/SPY/range/1/day/2026-01-02/2026-08-18",
        {"adjusted": "true", "limit": 50000, "sort": "asc"},
    )
    if cal_status == 200 and isinstance(cal_payload, dict) and cal_payload.get("results"):
        dates = [
            pd.to_datetime(r["t"], unit="ms", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
            for r in cal_payload["results"]
        ]
    else:
        dates = list(pd.bdate_range("2026-01-02", "2026-08-18"))

    members = set(snap_out["ticker"].dropna().astype(str).str.upper())
    members.discard("NAN")
    daily_rows = []
    events["effective_date"] = pd.to_datetime(events["effective_date"])
    for d in dates:
        todays = events[events["effective_date"] == pd.Timestamp(d)]
        for rec in todays.itertuples(index=False):
            if rec.removed_ticker:
                members.discard(str(rec.removed_ticker).upper())
            if rec.added_ticker:
                members.add(str(rec.added_ticker).upper())
        for tic in sorted(members):
            daily_rows.append({"date": pd.Timestamp(d).date().isoformat(), "ticker": tic, "in_index": True})
    daily = pd.DataFrame(daily_rows)
    daily.to_parquet(UNIVERSE_2026 / "sp500_2026_daily_membership.parquet", index=False)
    daily.to_csv(UNIVERSE_2026 / "sp500_2026_daily_membership.csv", index=False)

    summary = {
        "snapshot_n": int(len(snap_out)),
        "snapshot_missing_ticker": int(snap_out["ticker"].isna().sum()),
        "n_events": int(len(events)),
        "n_daily_rows": int(len(daily)),
        "massive_official_sp500_history": False,
        "massive_etf_constituents_accessible": massive_has_index_members,
        "note": note_massive
        + ". Official PIT membership still requires S&P DJI announcements; ETF holdings are not the index.",
        "open_issues": [
            "PERMNO is missing for 2026 additions that never appear in the WRDS dump.",
            "HONA/CAG and FDXF/EPAM Wikipedia dates are split across adjacent sessions; recorded as paired with a caveat.",
            "Ticker map uses S5A query tickers (as-of holdout), not a full CRSP names history.",
        ],
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    (UNIVERSE_2026 / "universe_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (AUDIT_DIR / "universe_2026_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"E snapshot={summary['snapshot_n']} events={summary['n_events']} daily_rows={summary['n_daily_rows']}")
    return summary
