#!/usr/bin/env python3
"""IBKR Paper ledger preflight and P5 quantity proposal from Massive reference
prices and account ExchangeRate:USD. Read-only on the broker (no orders).
VT0 is not sized for execution."""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ibapi.account_summary_tags import AccountSummaryTags
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IBKR_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(IBKR_DIR) not in sys.path:
    sys.path.insert(0, str(IBKR_DIR))

from p5_paper_execute_massive_sizing import (  # noqa: E402
    EXPECTED,
    WEIGHT,
    classify_account,
    load_target,
    massive_latest_prices,
    pick_nav,
    pick_usd_rate,
)

HOST = "127.0.0.1"
PORT = 7497
CLIENT_IDS = (17801, 17802, 17803)
OUT_MD = PROJECT_ROOT / "reports/ibkr/p5_paper_ledger_preflight.md"
OUT_JSON = PROJECT_ROOT / "reports/ibkr/p5_paper_ledger_preflight.json"
EXPECTED_ACCOUNT = os.environ["EXPECTED_ACCOUNT"]
SUMMARY_TAGS = ",".join(
    [
        AccountSummaryTags.AccountType,
        AccountSummaryTags.NetLiquidation,
        AccountSummaryTags.TotalCashValue,
        AccountSummaryTags.AvailableFunds,
    ]
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def usd_cash_ledger(values: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in values.items():
        ku = key.upper()
        if ":USD" not in ku and not ku.endswith(":USD") and not ku.endswith("USD"):
            continue
        if any(
            tok in ku
            for tok in (
                "CASH",
                "SETTLED",
                "FX",
                "TOTALCASH",
                "NETLIQUIDATION",
                "EXCHANGERATE",
                "CURRENCY",
            )
        ):
            out[key] = val
    return dict(sorted(out.items()))


class LedgerProbe(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.lock = threading.Lock()
        self.errors: list[dict[str, Any]] = []
        self.managed_accounts: list[str] = []
        self.account_kind = "UNKNOWN"
        self.live_stop = False
        self.summary: dict[str, dict[str, str]] = {}
        self.values: dict[str, dict[str, str]] = {}
        self.positions: list[dict[str, Any]] = []
        self.connected_event = threading.Event()
        self.managed_event = threading.Event()
        self.summary_event = threading.Event()
        self.ledger_usd_event = threading.Event()
        self.positions_event = threading.Event()
        self.acct_end_event = threading.Event()
        self.farms_ok = threading.Event()

    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):  # noqa: N802
        rec = {"req_id": reqId, "code": int(errorCode), "message": str(errorString), "ts": utc_now()}
        with self.lock:
            self.errors.append(rec)
        if int(errorCode) in {2104, 2106, 2158}:
            self.farms_ok.set()
        if int(errorCode) not in {2104, 2106, 2108, 2158, 2100}:
            print(f"ERR {reqId} {errorCode} {errorString}", flush=True)

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        print(f"nextValidId={orderId} negotiated={self.serverVersion()}", flush=True)
        self.connected_event.set()
        self.reqManagedAccts()

    def managedAccounts(self, accountsList: str) -> None:  # noqa: N802
        accts = [a.strip() for a in (accountsList or "").split(",") if a.strip()]
        kinds = {classify_account(a) for a in accts}
        with self.lock:
            self.managed_accounts = accts
            if "LIVE" in kinds:
                self.account_kind = "LIVE"
                self.live_stop = True
            elif accts and kinds <= {"PAPER"}:
                self.account_kind = "PAPER"
            else:
                self.account_kind = "UNKNOWN"
        print(f"managedAccounts={accts} kind={self.account_kind}", flush=True)
        self.managed_event.set()

    def accountSummary(self, reqId, account, tag, value, currency) -> None:  # noqa: N802
        acct = account or ""
        label = f"{tag}:{currency}" if currency else str(tag)
        with self.lock:
            self.summary.setdefault(acct, {})[str(tag)] = str(value)
            if currency:
                self.summary[acct][f"{tag}_currency"] = str(currency)
                self.summary[acct][label] = str(value)
            self.values.setdefault(acct, {})[label] = str(value)
            if acct != EXPECTED_ACCOUNT:
                self.values.setdefault(EXPECTED_ACCOUNT, {})[label] = str(value)

    def accountSummaryEnd(self, reqId) -> None:  # noqa: N802
        if int(reqId) == 9203:
            self.ledger_usd_event.set()
        else:
            self.summary_event.set()

    def updateAccountValue(self, key, val, currency, accountName) -> None:  # noqa: N802
        label = f"{key}:{currency}" if currency else str(key)
        with self.lock:
            self.values.setdefault(accountName, {})[label] = str(val)

    def accountDownloadEnd(self, accountName) -> None:  # noqa: N802
        self.acct_end_event.set()

    def position(self, account, contract, position, avgCost) -> None:
        rec = {
            "account": account,
            "symbol": getattr(contract, "symbol", None),
            "sec_type": getattr(contract, "secType", None),
            "position": float(position),
            "avg_cost": float(avgCost),
        }
        with self.lock:
            self.positions.append(rec)

    def positionEnd(self) -> None:  # noqa: N802
        self.positions_event.set()


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    plan = payload.get("proposed") or []
    plan_lines = ["(none; preflight did not pass)"]
    if plan:
        plan_lines = [
            (
                f"- `{r['ticker']}` qty=**{r['quantity']}** "
                f"massive_usd={r['reference_price_massive']} "
                f"target_gbp={r['target_value_gbp']:.2f} "
                f"target_usd={r['target_value_usd']:.2f} "
                f"ref_notional_usd={r['sized_usd']:.2f}"
            )
            for r in plan
        ]
    lines = [
        "# P5 Paper Ledger Preflight (no orders)",
        "",
        f"Generated: `{payload.get('generated')}`",
        f"Status: **{payload.get('status')}**",
        "",
        "No orders submitted. VT0 shadow-only. Frozen P5 parameters unchanged.",
        "",
        f"- Account `{payload.get('account_id')}` `{payload.get('account_kind')}`",
        f"- Expected `{EXPECTED_ACCOUNT}` / PAPER: `{payload.get('account_match')}`",
        f"- NAV GBP `{payload.get('nav_gbp')}`",
        f"- ExchangeRate:USD `{payload.get('exchange_rate_usd')}` ({payload.get('exchange_rate_key')})",
        f"- USD cash ledger: `{payload.get('usd_cash_ledger')}`",
        f"- Stock positions: `{payload.get('stock_positions')}`",
        f"- Massive session `{payload.get('massive_session')}`",
        "",
        "## Proposed P5 quantities (not submitted)",
        "",
        *plan_lines,
        "",
        "Fatal: " + "; ".join(payload.get("fatal") or ["none"]),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    names, terr = load_target()
    fatal: list[str] = []
    if terr:
        write_outputs({"generated": utc_now(), "status": "FAIL_TARGET", "fatal": terr, "orders_submitted": False})
        return 1

    try:
        massive = massive_latest_prices(names)
    except Exception as exc:  # noqa: BLE001
        write_outputs({"generated": utc_now(), "status": "FAIL_MASSIVE", "fatal": [str(exc)], "orders_submitted": False})
        print(f"Massive failed: {exc}", flush=True)
        return 1

    app = LedgerProbe()
    last_err = None
    for cid in CLIENT_IDS:
        try:
            print(f"connecting {HOST}:{PORT} clientId={cid}", flush=True)
            app = LedgerProbe()
            app.connect(HOST, PORT, cid)
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            continue
        if not app.isConnected():
            last_err = "not connected"
            continue
        threading.Thread(target=app.run, daemon=True).start()
        if app.connected_event.wait(25):
            break
        last_err = "timeout nextValidId"
        try:
            app.disconnect()
        except Exception:
            pass
    if not app.connected_event.is_set():
        write_outputs({"generated": utc_now(), "status": "FAIL_CONNECT", "fatal": [last_err or "connect"], "orders_submitted": False})
        return 1
    app.managed_event.wait(8)
    acct = app.managed_accounts[0] if app.managed_accounts else ""
    if not app.farms_ok.wait(90):
        write_outputs(
            {
                "generated": utc_now(),
                "status": "FAIL_TWS_SERVER",
                "fatal": [
                    "TWS not connected to IBKR servers (no 2104; saw 2110/2103). "
                    "Account ledger not refreshed. No orders submitted."
                ],
                "account_id": acct,
                "account_kind": app.account_kind,
                "orders_submitted": False,
                "errors": app.errors,
            }
        )
        print("FAIL_TWS_SERVER: no farm-ok 2104", flush=True)
        try:
            app.disconnect()
        except Exception:
            pass
        return 1
    time.sleep(2)

    account_match = acct.upper() == EXPECTED_ACCOUNT and app.account_kind == "PAPER" and not app.live_stop
    if app.live_stop or app.account_kind != "PAPER":
        fatal.append(f"not PAPER: {app.managed_accounts} {app.account_kind}")
    if acct.upper() != EXPECTED_ACCOUNT:
        fatal.append(f"account {acct} != {EXPECTED_ACCOUNT}")

    app.reqAccountSummary(9201, "All", SUMMARY_TAGS)
    app.reqAccountSummary(9203, "All", "$LEDGER:USD")
    app.reqPositions()
    app.reqAccountUpdates(True, acct)
    app.summary_event.wait(10)
    app.ledger_usd_event.wait(10)
    app.positions_event.wait(10)
    app.acct_end_event.wait(8)

    deadline = time.time() + 20
    usd_rate = None
    usd_key = None
    merged: dict[str, str] = {}
    while time.time() < deadline:
        merged = {}
        with app.lock:
            for src in (app.values.get(acct) or {}, app.values.get(EXPECTED_ACCOUNT) or {}, app.values.get("All") or {}):
                merged.update(src)
        usd_rate, usd_key = pick_usd_rate(merged)
        if usd_rate:
            break
        time.sleep(0.2)

    nav, nav_src = pick_nav(app.summary.get(acct, {}) or app.summary.get("All", {}), merged)
    usd_cash = usd_cash_ledger(merged)
    fx_keys = [k for k in merged if "exchange" in k.lower() or k.upper().endswith(":USD")]
    nonzero = [p for p in app.positions if abs(p["position"]) > 1e-9]
    stock_nonzero = [p for p in nonzero if str(p.get("sec_type") or "").upper() == "STK"]

    if nav is None or nav <= 0:
        fatal.append(f"bad NAV {nav}")
    if usd_rate is None:
        fatal.append(f"missing ExchangeRate:USD; exchange/USD keys={fx_keys}")
    if stock_nonzero:
        fatal.append(f"existing stock positions {stock_nonzero}")

    proposed: list[dict[str, Any]] = []
    if not fatal and usd_rate and nav:
        for ticker in names:
            ref = massive[ticker]["reference_price_massive"]
            target_gbp = WEIGHT * nav
            target_usd = target_gbp / usd_rate
            qty = int(math.floor(target_usd / ref))
            proposed.append(
                {
                    "ticker": ticker,
                    "target_weight": WEIGHT,
                    "target_value_gbp": target_gbp,
                    "target_value_usd": target_usd,
                    "reference_price_massive": ref,
                    "massive_session": massive[ticker]["massive_session"],
                    "quantity": qty,
                    "sized_usd": qty * ref,
                    "round_down_residual_usd": target_usd - qty * ref,
                }
            )
            print(
                f"PROPOSED {ticker} qty={qty} massive={ref} "
                f"target_gbp={target_gbp:.2f} target_usd={target_usd:.2f}",
                flush=True,
            )

    status = "PASS_LEDGER_DRY_RUN" if not fatal and proposed and all(p["quantity"] >= 1 for p in proposed) else "FAIL_LEDGER"
    if not fatal and any(p["quantity"] < 1 for p in proposed):
        fatal.append("one or more proposed qty < 1")
        status = "FAIL_SIZING"

    payload = {
        "generated": utc_now(),
        "status": status,
        "orders_submitted": False,
        "whatif_used": False,
        "ibkr_quotes_used": False,
        "vt0_orders": False,
        "p5_parameters_changed": False,
        "account_id": acct,
        "account_kind": app.account_kind,
        "account_match": account_match,
        "nav_gbp": nav,
        "nav_source": nav_src,
        "exchange_rate_usd": usd_rate,
        "exchange_rate_key": usd_key,
        "exchange_rate_meaning": "IBKR account ExchangeRate for USD = GBP per 1 USD; USD = GBP / rate",
        "usd_cash_ledger": usd_cash,
        "all_usd_keys": {k: merged[k] for k in sorted(merged) if "USD" in k.upper()},
        "exchange_or_usd_keys": sorted(fx_keys),
        "stock_positions": stock_nonzero,
        "all_nonzero_positions": nonzero,
        "massive_session": next(iter(massive.values()))["massive_session"],
        "massive_reference_prices": {t: massive[t]["reference_price_massive"] for t in names},
        "proposed": proposed,
        "fatal": fatal,
        "errors": [e for e in app.errors if e.get("code") not in {2104, 2106, 2108, 2158, 2100}],
    }
    try:
        app.cancelAccountSummary(9201)
        app.cancelAccountSummary(9203)
        app.cancelPositions()
        app.reqAccountUpdates(False, acct)
        app.disconnect()
    except Exception:
        pass
    write_outputs(payload)
    print(f"status={status} usd_rate={usd_rate} usd_cash={usd_cash} wrote {OUT_MD}", flush=True)
    return 0 if status == "PASS_LEDGER_DRY_RUN" else 1


if __name__ == "__main__":
    sys.exit(main())
