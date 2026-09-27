#!/usr/bin/env python3
"""Submit first P5 IBKR Paper MOC orders. VT0 remains shadow-only.
Preflight failure results in zero orders. Sizing uses NetLiquidation and
Massive reference prices (not BuyingPower)."""

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
from ibapi.commission_report import CommissionReport
from ibapi.contract import Contract, ContractDetails
from ibapi.execution import Execution
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.wrapper import EWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 7497
CLIENT_IDS = (17501, 17502, 17503)
TARGET_CSV = PROJECT_ROOT / "data/portfolio_experiments/market_risk/live_target/order_target.csv"
OUT_MD = PROJECT_ROOT / "reports/ibkr/p5_paper_first_orders.md"
OUT_JSON = PROJECT_ROOT / "reports/ibkr/p5_paper_first_orders.json"
OUT_FILLS = PROJECT_ROOT / "reports/ibkr/p5_paper_first_orders_fills.csv"

EXPECTED_TICKERS = ["AXON", "CNC", "CSGP", "INTU", "MCHP"]
EXPECTED_WEIGHT = 0.19
EXPECTED_CASH = 0.05
EQUITY_CAP = 0.95
PRIMARY = {
    "AXON": "NASDAQ",
    "CNC": "NYSE",
    "CSGP": "NASDAQ",
    "INTU": "NASDAQ",
    "MCHP": "NASDAQ",
}
BID, ASK, LAST = 1, 2, 4
DELAYED_BID, DELAYED_ASK, DELAYED_LAST = 66, 67, 68
SUMMARY_TAGS = ",".join(
    [
        AccountSummaryTags.AccountType,
        AccountSummaryTags.NetLiquidation,
        AccountSummaryTags.AvailableFunds,
        AccountSummaryTags.BuyingPower,
        AccountSummaryTags.TotalCashValue,
        AccountSummaryTags.ExcessLiquidity,
    ]
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def classify_account(account_id: str) -> str:
    acct = (account_id or "").strip().upper()
    if acct.startswith("DU") or acct.startswith("DF"):
        return "PAPER"
    if acct.startswith(("U", "F", "I")):
        return "LIVE"
    return "UNKNOWN"


def load_and_validate_target() -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not TARGET_CSV.exists():
        return [], [f"missing {TARGET_CSV}"]
    rows = []
    with TARGET_CSV.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        for line in handle:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "ticker": parts[0].upper(),
                    "execution_arm": parts[1],
                    "execution_weight": float(parts[2]),
                }
            )
    names = [r for r in rows if r["ticker"] != "CASH"]
    cash = [r for r in rows if r["ticker"] == "CASH"]
    got = [r["ticker"] for r in names]
    if got != EXPECTED_TICKERS:
        errors.append(f"ticker mismatch csv={got} expected={EXPECTED_TICKERS}")
    for r in names:
        if r["execution_arm"] != "P5":
            errors.append(f"{r['ticker']} execution_arm={r['execution_arm']} not P5")
        if abs(r["execution_weight"] - EXPECTED_WEIGHT) > 1e-12:
            errors.append(f"{r['ticker']} weight={r['execution_weight']} not {EXPECTED_WEIGHT}")
    if not cash or abs(cash[0]["execution_weight"] - EXPECTED_CASH) > 1e-9:
        errors.append(f"cash weight not {EXPECTED_CASH}")
    if any("VT0" in r["execution_arm"].upper() for r in rows):
        errors.append("VT0 present in execution csv")
    return names, errors


def stock_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.currency = "USD"
    c.exchange = "SMART"
    c.primaryExchange = PRIMARY[symbol]
    return c


def fx_contract() -> Contract:
    c = Contract()
    c.symbol = "GBP"
    c.secType = "CASH"
    c.currency = "USD"
    c.exchange = "IDEALPRO"
    return c


def make_order(action: str, qty: int, order_ref: str, account: str, what_if: bool = False) -> Order:
    o = Order()
    o.action = action
    o.totalQuantity = int(qty)
    o.orderType = "MKT"
    o.tif = "DAY"
    o.transmit = True
    o.eTradeOnly = False
    o.firmQuoteOnly = False
    o.outsideRth = False
    o.whatIf = bool(what_if)
    o.orderRef = order_ref
    o.account = account
    return o


class P5PaperTrader(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.lock = threading.Lock()
        self.errors: list[dict[str, Any]] = []
        self.fatal_preflight: list[str] = []
        self.managed_accounts: list[str] = []
        self.account_kind = "UNKNOWN"
        self.live_stop = False
        self.summary: dict[str, dict[str, str]] = {}
        self.account_values: dict[str, dict[str, str]] = {}
        self.positions: list[dict[str, Any]] = []
        self.contract_hits: dict[int, list[ContractDetails]] = {}
        self.ticks: dict[int, dict[str, float]] = {}
        self.orders: dict[int, dict[str, Any]] = {}
        self.execs: dict[str, dict[str, Any]] = {}
        self.next_order_id: int | None = None
        self.connected_event = threading.Event()
        self.managed_event = threading.Event()
        self.summary_event = threading.Event()
        self.positions_event = threading.Event()
        self.account_end_event = threading.Event()
        self.contract_events: dict[int, threading.Event] = {}
        self.whatif_event = threading.Event()
        self.post_summary_event = threading.Event()
        self.post_positions_event = threading.Event()
        self.hist: dict[int, list[dict[str, float]]] = {}
        self.hist_events: dict[int, threading.Event] = {}
        self.phase = "preflight"
        self.readonly_blocked = False

    def error(self, reqId: int, errorCode: int, errorString: str) -> None:  # noqa: N802
        rec = {"req_id": int(reqId), "code": int(errorCode), "message": str(errorString), "ts": utc_now()}
        with self.lock:
            self.errors.append(rec)
        msg = str(errorString).lower()
        if "read-only" in msg or int(errorCode) == 10349:
            self.readonly_blocked = True
            self.fatal_preflight.append(f"read-only/API block code={errorCode} {errorString}")
            print(f"READONLY/BLOCK {errorCode}: {errorString}", flush=True)
        if int(errorCode) in {201, 202, 203, 321, 354, 502, 504, 1100, 10147, 10349} or "read-only" in msg:
            print(f"IBKR error {errorCode}: {errorString}", flush=True)

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        self.next_order_id = int(orderId)
        self.connected_event.set()
        print(f"nextValidId={orderId} serverVersion={self.serverVersion()}", flush=True)
        self.reqMarketDataType(3)
        self.reqManagedAccts()

    def managedAccounts(self, accountsList: str) -> None:  # noqa: N802
        accts = [a.strip() for a in (accountsList or "").split(",") if a.strip()]
        kinds = {classify_account(a) for a in accts}
        with self.lock:
            self.managed_accounts = accts
            if "LIVE" in kinds:
                self.account_kind = "LIVE"
                self.live_stop = True
                self.fatal_preflight.append(f"LIVE account(s) {accts}")
            elif accts and kinds <= {"PAPER"}:
                self.account_kind = "PAPER"
            else:
                self.account_kind = "UNKNOWN"
                self.fatal_preflight.append(f"account kind not PAPER: {accts}")
        print(f"managedAccounts={accts} kind={self.account_kind}", flush=True)
        self.managed_event.set()

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str) -> None:  # noqa: N802
        with self.lock:
            self.summary.setdefault(account, {})[tag] = value
            if currency:
                self.summary[account][f"{tag}_currency"] = currency

    def accountSummaryEnd(self, reqId: int) -> None:  # noqa: N802
        if self.phase == "postfill":
            self.post_summary_event.set()
        else:
            self.summary_event.set()

    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str) -> None:  # noqa: N802
        label = f"{key}:{currency}" if currency else key
        with self.lock:
            self.account_values.setdefault(accountName, {})[label] = val

    def accountDownloadEnd(self, accountName: str) -> None:  # noqa: N802
        self.account_end_event.set()

    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:
        rec = {
            "account": account,
            "symbol": getattr(contract, "symbol", None),
            "sec_type": getattr(contract, "secType", None),
            "currency": getattr(contract, "currency", None),
            "position": float(position),
            "avg_cost": float(avgCost),
            "con_id": getattr(contract, "conId", None),
        }
        with self.lock:
            self.positions.append(rec)

    def positionEnd(self) -> None:  # noqa: N802
        if self.phase == "postfill":
            self.post_positions_event.set()
        else:
            self.positions_event.set()

    def contractDetails(self, reqId: int, contractDetails: ContractDetails) -> None:  # noqa: N802
        with self.lock:
            self.contract_hits.setdefault(int(reqId), []).append(contractDetails)

    def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
        ev = self.contract_events.get(int(reqId))
        if ev is not None:
            ev.set()

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib) -> None:  # noqa: N802
        if price is None or float(price) <= 0:
            return
        key = {
            BID: "bid",
            ASK: "ask",
            LAST: "last",
            DELAYED_BID: "bid",
            DELAYED_ASK: "ask",
            DELAYED_LAST: "last",
        }.get(int(tickType))
        if not key:
            return
        with self.lock:
            slot = self.ticks.setdefault(int(reqId), {})
            slot[key] = float(price)
            slot[f"{key}_tick"] = int(tickType)
        print(f"tickPrice id={reqId} type={tickType} {key}={price}", flush=True)

    def tickByTickBidAsk(self, reqId, time_, bidPrice, askPrice, bidSize, askSize, tickAttribBidAsk) -> None:  # noqa: N802
        if bidPrice and bidPrice > 0:
            with self.lock:
                self.ticks.setdefault(int(reqId), {})["bid"] = float(bidPrice)
        if askPrice and askPrice > 0:
            with self.lock:
                self.ticks.setdefault(int(reqId), {})["ask"] = float(askPrice)
        print(f"tickByTickBidAsk id={reqId} bid={bidPrice} ask={askPrice}", flush=True)

    def historicalData(self, reqId: int, bar) -> None:  # noqa: N802
        rec = {
            "close": float(getattr(bar, "close", 0) or 0),
            "high": float(getattr(bar, "high", 0) or 0),
            "low": float(getattr(bar, "low", 0) or 0),
        }
        with self.lock:
            self.hist.setdefault(int(reqId), []).append(rec)

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
        ev = self.hist_events.get(int(reqId))
        if ev is not None:
            ev.set()
        print(f"historicalDataEnd id={reqId} n={len(self.hist.get(int(reqId), []))}", flush=True)

    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState: OrderState) -> None:  # noqa: N802
        with self.lock:
            rec = self.orders.setdefault(int(orderId), {})
            rec["open_status"] = getattr(orderState, "status", None)
            rec["whatIf"] = bool(getattr(order, "whatIf", False))
            rec["warning"] = getattr(orderState, "warningText", None)
            rec["init_margin"] = getattr(orderState, "initMarginChange", None)
        if getattr(order, "whatIf", False):
            self.whatif_event.set()

    def orderStatus(
        self,
        orderId: int,
        status: str,
        filled: float,
        remaining: float,
        avgFillPrice: float,
        permId: int,
        parentId: int,
        lastFillPrice: float,
        clientId: int,
        whyHeld: str,
        mktCapPrice: float,
    ) -> None:  # noqa: N802
        with self.lock:
            rec = self.orders.setdefault(int(orderId), {})
            rec["status"] = status
            rec["filled"] = float(filled)
            rec["remaining"] = float(remaining)
            rec["avg_fill_price"] = float(avgFillPrice)
            rec["perm_id"] = int(permId)
            rec["last_fill_price"] = float(lastFillPrice)
            rec["why_held"] = whyHeld
            rec["status_ts"] = utc_now()
        print(f"orderStatus id={orderId} {status} filled={filled} avg={avgFillPrice}", flush=True)

    def execDetails(self, reqId: int, contract: Contract, execution: Execution) -> None:  # noqa: N802
        with self.lock:
            self.execs[execution.execId] = {
                "exec_id": execution.execId,
                "time": execution.time,
                "account": execution.acctNumber,
                "exchange": execution.exchange,
                "side": execution.side,
                "shares": float(execution.shares),
                "price": float(execution.price),
                "order_id": int(execution.orderId),
                "avg_price": float(execution.avgPrice),
                "cum_qty": float(execution.cumQty),
            }
            rec = self.orders.setdefault(int(execution.orderId), {})
            rec.setdefault("executions", []).append(execution.execId)

    def commissionReport(self, commissionReport: CommissionReport) -> None:  # noqa: N802
        with self.lock:
            ex = self.execs.get(commissionReport.execId)
            if ex is None:
                self.execs[commissionReport.execId] = {}
                ex = self.execs[commissionReport.execId]
            ex["commission"] = float(commissionReport.commission)
            ex["commission_currency"] = commissionReport.currency
            oid = ex.get("order_id")
            if oid is not None:
                rec = self.orders.setdefault(int(oid), {})
                rec["commission"] = rec.get("commission", 0.0) + float(commissionReport.commission)
                rec["commission_currency"] = commissionReport.currency

    def alloc_order_id(self) -> int:
        assert self.next_order_id is not None
        oid = self.next_order_id
        self.next_order_id += 1
        return oid


def pick_contract(hits: list[ContractDetails], symbol: str) -> Contract | None:
    stks = []
    for h in hits:
        c = h.contract
        if c.secType == "STK" and c.currency == "USD":
            stks.append(c)
    if not stks:
        return None
    primary = PRIMARY[symbol]
    preferred = [c for c in stks if (c.primaryExchange or c.exchange or "").upper() in {primary, "NASDAQ", "NYSE", "ARCA", "AMEX", "SMART"}]
    pool = preferred or stks
    # unique conId
    by_id = {}
    for c in pool:
        by_id[int(c.conId)] = c
    if len(by_id) != 1 and len({c.symbol for c in pool}) == 1:
        # same name multiple venues: pick SMART-capable / matching primary
        for c in pool:
            if (c.primaryExchange or "").upper() == primary:
                out = Contract()
                out.conId = c.conId
                out.symbol = c.symbol
                out.secType = "STK"
                out.currency = "USD"
                out.exchange = "SMART"
                out.primaryExchange = c.primaryExchange or primary
                return out
    c = next(iter(by_id.values()))
    out = Contract()
    out.conId = c.conId
    out.symbol = c.symbol
    out.secType = "STK"
    out.currency = "USD"
    out.exchange = "SMART"
    out.primaryExchange = c.primaryExchange or primary
    return out


def executable_px(ticks: dict[str, float]) -> float | None:
    ask = ticks.get("ask")
    if ask and ask > 0:
        return float(ask)
    return None


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    fills = payload.get("fills") or []
    fill_lines = ["(none)"]
    if fills:
        fill_lines = [
            (
                f"- `{f['ticker']}` qty={f['quantity']} fill={f['fill_quantity']} "
                f"avg={f['avg_fill_price']} comm={f['commission']} "
                f"w_tgt={f['target_weight']} w_act={f['actual_post_fill_weight']} "
                f"slip_bps={f['slippage_bps']}"
            )
            for f in fills
        ]
    lines = [
        "# P5 Paper First Orders",
        "",
        f"Generated: `{payload['generated']}`",
        f"Status: **{payload['status']}**",
        "",
        "Execution arm: **P5**. VT0 shadow-only; no VT0 orders.",
        "BuyingPower/leverage not used. Frozen P5 parameters unchanged.",
        "",
        "## Preflight",
        "",
        f"- Account: `{payload.get('account_id')}` kind=**{payload.get('account_kind')}** live_stop=`{payload.get('live_stop')}`",
        f"- NetLiquidation GBP: `{payload.get('nav_gbp')}`",
        f"- GBPUSD used for sizing: `{payload.get('gbp_usd')}`",
        f"- NAV USD equivalent: `{payload.get('nav_usd')}`",
        f"- Positions empty: `{payload.get('positions_empty')}`",
        f"- Target CSV match: `{payload.get('target_ok')}`",
        f"- Orders submitted: `{payload.get('orders_submitted')}`",
        "",
        "## Fills",
        "",
        *fill_lines,
        "",
        "## Post-fill",
        "",
        f"- NetLiquidation GBP: `{payload.get('post_nav_gbp')}`",
        f"- USD stock market value: `{payload.get('post_stock_usd')}`",
        f"- Strategy USD vs starting NAV USD: `{payload.get('strategy_usd_return')}`",
        "",
        "Fatal: " + "; ".join(payload.get("fatal") or ["none"]),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if fills:
        keys = list(fills[0].keys())
        with OUT_FILLS.open("w", encoding="utf-8") as handle:
            handle.write(",".join(keys) + "\n")
            for f in fills:
                handle.write(",".join("" if f[k] is None else str(f[k]) for k in keys) + "\n")


def fail_payload(app: P5PaperTrader | None, status: str, fatal: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "generated": utc_now(),
        "status": status,
        "execution_arm": "P5",
        "vt0_orders": False,
        "orders_submitted": False,
        "live_stop": bool(app.live_stop) if app else False,
        "account_kind": app.account_kind if app else None,
        "account_id": (app.managed_accounts or [None])[0] if app else None,
        "fatal": fatal,
        "errors": app.errors if app else [],
        "fills": [],
    }
    if extra:
        payload.update(extra)
    return payload


def main() -> int:
    names, csv_errors = load_and_validate_target()
    if csv_errors:
        payload = fail_payload(None, "NO_ORDERS_TARGET_VALIDATION_FAILED", csv_errors, {"target_ok": False})
        write_outputs(payload)
        print("STOP: target csv validation failed", csv_errors, flush=True)
        return 1

    app: P5PaperTrader | None = None
    last_err = None
    for client_id in CLIENT_IDS:
        app = P5PaperTrader()
        try:
            print(f"connecting {HOST}:{PORT} clientId={client_id}", flush=True)
            app.connect(HOST, PORT, clientId=client_id)
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            continue
        if not app.isConnected():
            last_err = "not connected"
            continue
        threading.Thread(target=app.run, name="ibkr-p5", daemon=True).start()
        if not app.connected_event.wait(25):
            last_err = "timeout nextValidId"
            try:
                app.disconnect()
            except Exception:
                pass
            continue
        break
    assert app is not None
    if not app.connected_event.is_set():
        payload = fail_payload(app, "NO_ORDERS_CONNECT_FAILED", [last_err or "connect failed"])
        write_outputs(payload)
        return 1

    if app.live_stop:
        try:
            app.disconnect()
        except Exception:
            pass
        payload = fail_payload(app, "NO_ORDERS_LIVE_ACCOUNT", app.fatal_preflight)
        write_outputs(payload)
        return 1

    app.managed_event.wait(8)
    if app.live_stop or app.account_kind != "PAPER":
        try:
            app.disconnect()
        except Exception:
            pass
        payload = fail_payload(app, "NO_ORDERS_NOT_PAPER", app.fatal_preflight or ["not PAPER"])
        write_outputs(payload)
        return 1
    acct = app.managed_accounts[0]
    if not acct.upper().startswith("DU"):
        payload = fail_payload(app, "NO_ORDERS_NOT_PAPER", [f"account {acct} does not start with DU"])
        write_outputs(payload)
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    app.reqAccountSummary(9101, "All", SUMMARY_TAGS)
    app.reqPositions()
    app.reqAccountUpdates(True, acct)
    app.summary_event.wait(10)
    app.positions_event.wait(10)
    app.account_end_event.wait(5)

    nonzero = [p for p in app.positions if abs(p["position"]) > 1e-9]
    if nonzero:
        payload = fail_payload(
            app,
            "NO_ORDERS_POSITIONS_NOT_EMPTY",
            [f"nonzero positions {nonzero}"],
            {"positions_empty": False, "positions": nonzero},
        )
        write_outputs(payload)
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    nav_s = app.summary.get(acct, {}).get("NetLiquidation")
    nav_ccy = app.summary.get(acct, {}).get("NetLiquidation_currency", "GBP")
    try:
        nav_gbp = float(nav_s)
    except (TypeError, ValueError):
        nav_gbp = float("nan")
    if not math.isfinite(nav_gbp) or nav_gbp <= 0:
        payload = fail_payload(app, "NO_ORDERS_NAV_FAILED", [f"bad NetLiquidation {nav_s}"])
        write_outputs(payload)
        try:
            app.disconnect()
        except Exception:
            pass
        return 1
    if nav_ccy and nav_ccy.upper() != "GBP":
        payload = fail_payload(app, "NO_ORDERS_NAV_CURRENCY", [f"NetLiquidation currency {nav_ccy} not GBP"])
        write_outputs(payload)
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    # Qualify contracts
    app.reqMarketDataType(3)
    time.sleep(0.3)
    req_map: dict[str, int] = {}
    rid = 8000
    for name in EXPECTED_TICKERS:
        rid += 1
        ev = threading.Event()
        app.contract_events[rid] = ev
        req_map[name] = rid
        app.reqContractDetails(rid, stock_contract(name))
    fx_rid = 8099
    app.contract_events[fx_rid] = threading.Event()
    app.reqContractDetails(fx_rid, fx_contract())

    qualified: dict[str, Contract] = {}
    for name, r in req_map.items():
        if not app.contract_events[r].wait(12):
            app.fatal_preflight.append(f"contract details timeout {name}")
            continue
        hits = app.contract_hits.get(r, [])
        c = pick_contract(hits, name)
        if c is None:
            app.fatal_preflight.append(f"no unique USD STK contract for {name} n={len(hits)}")
        else:
            qualified[name] = c
            print(f"qualified {name} conId={c.conId} primary={c.primaryExchange}", flush=True)
    if not app.contract_events[fx_rid].wait(12) or not app.contract_hits.get(fx_rid):
        app.fatal_preflight.append("GBP.USD IDEALPRO contract details failed")
    fx_c = fx_contract()
    if app.contract_hits.get(fx_rid):
        fx_c = app.contract_hits[fx_rid][0].contract
        fx_c.exchange = "IDEALPRO"

    if len(qualified) != 5 or app.fatal_preflight:
        payload = fail_payload(
            app,
            "NO_ORDERS_CONTRACT_QUALIFICATION_FAILED",
            app.fatal_preflight,
            {"target_ok": True, "positions_empty": True, "nav_gbp": nav_gbp},
        )
        write_outputs(payload)
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    # Market data: delayed streaming + tick-by-tick BidAsk; historical BID_ASK fallback.
    # Do not reqMktData on FX: ibapi 9.81 / serverVer 157 hits error 10285 (needs API v163).
    md_ids: dict[str, int] = {}
    tbt_ids: dict[str, int] = {}
    hist_ids: dict[str, int] = {}
    md = 8200
    tbt = 8300
    hid = 8400
    for name, c in qualified.items():
        md += 1
        tbt += 1
        hid += 1
        md_ids[name] = md
        tbt_ids[name] = tbt
        hist_ids[name] = hid
        app.reqMktData(md, c, "", False, False, [])
        app.reqTickByTickData(tbt, c, "BidAsk", 0, True)
        app.hist_events[hid] = threading.Event()
        app.reqHistoricalData(hid, c, "", "300 S", "1 min", "BID_ASK", 0, 1, False, [])
    fx_hid = 8499
    app.hist_events[fx_hid] = threading.Event()
    app.reqHistoricalData(fx_hid, fx_c, "", "300 S", "1 min", "MIDPOINT", 0, 1, False, [])

    deadline = time.time() + 25
    while time.time() < deadline:
        ok = True
        for name in EXPECTED_TICKERS:
            merged = {}
            merged.update(app.ticks.get(md_ids[name], {}))
            merged.update(app.ticks.get(tbt_ids[name], {}))
            if not executable_px(merged):
                bars = app.hist.get(hist_ids[name], [])
                if bars and bars[-1].get("high", 0) > 0:
                    continue
                ok = False
        fx_bars = app.hist.get(fx_hid, [])
        fx_ok = bool(fx_bars and fx_bars[-1].get("close", 0) > 0)
        if ok and fx_ok:
            break
        time.sleep(0.2)

    # Merge executable ask: live/delayed tick, else last BID_ASK bar high (IB: high=ask)
    missing_px = []
    for name in EXPECTED_TICKERS:
        merged = {}
        merged.update(app.ticks.get(md_ids[name], {}))
        merged.update(app.ticks.get(tbt_ids[name], {}))
        bars = app.hist.get(hist_ids[name], [])
        if bars:
            last_bar = bars[-1]
            if last_bar.get("high", 0) > 0:
                merged.setdefault("ask", last_bar["high"])
            if last_bar.get("low", 0) > 0:
                merged.setdefault("bid", last_bar["low"])
            if last_bar.get("close", 0) > 0:
                merged.setdefault("last", last_bar["close"])
        app.ticks[md_ids[name]] = merged
        if not executable_px(merged):
            missing_px.append(name)

    fx_ask = None
    fx_bid = None
    gbp_usd = None
    fx_bars = app.hist.get(fx_hid, [])
    if fx_bars and fx_bars[-1].get("close", 0) > 0:
        gbp_usd = float(fx_bars[-1]["close"])
        fx_ask = gbp_usd
        fx_bid = gbp_usd
    if gbp_usd is None:
        er = app.account_values.get(acct, {}).get("ExchangeRate:USD")
        try:
            usd_to_gbp = float(er) if er else float("nan")
            if usd_to_gbp > 0:
                gbp_usd = 1.0 / usd_to_gbp
        except (TypeError, ValueError):
            gbp_usd = None

    if missing_px or not gbp_usd or gbp_usd <= 0:
        payload = fail_payload(
            app,
            "NO_ORDERS_MARKET_DATA_FAILED",
            [f"missing ask {missing_px}", f"gbp_usd={gbp_usd}", f"ticks={ {k: app.ticks.get(v) for k,v in md_ids.items()} }"],
            {"target_ok": True, "positions_empty": True, "nav_gbp": nav_gbp},
        )
        write_outputs(payload)
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    nav_usd = nav_gbp * gbp_usd
    planned = []
    total_usd = 0.0
    for row in names:
        ticker = row["ticker"]
        px = executable_px(app.ticks[md_ids[ticker]])
        assert px is not None
        target_gbp = EXPECTED_WEIGHT * nav_gbp
        target_usd = target_gbp * gbp_usd
        qty = int(math.floor(target_usd / px))
        if qty < 1:
            app.fatal_preflight.append(f"{ticker} qty=0 at px={px} target_usd={target_usd}")
            continue
        usd_val = qty * px
        if usd_val - target_usd > 1e-6:
            app.fatal_preflight.append(f"{ticker} rounding exceeded target")
            continue
        gbp_val = usd_val / gbp_usd
        total_usd += usd_val
        planned.append(
            {
                "ticker": ticker,
                "target_weight": EXPECTED_WEIGHT,
                "target_value_gbp": target_gbp,
                "target_value_usd": target_usd,
                "size_price_usd": px,
                "bid": app.ticks[md_ids[ticker]].get("bid"),
                "ask": app.ticks[md_ids[ticker]].get("ask"),
                "last": app.ticks[md_ids[ticker]].get("last"),
                "quantity": qty,
                "planned_usd": usd_val,
                "planned_gbp": gbp_val,
                "contract": qualified[ticker],
            }
        )
    equity_usd = total_usd
    if equity_usd / nav_usd - EQUITY_CAP > 1e-8:
        app.fatal_preflight.append(f"planned equity {equity_usd/nav_usd:.6f} exceeds {EQUITY_CAP}")
    if len(planned) != 5 or app.fatal_preflight:
        payload = fail_payload(
            app,
            "NO_ORDERS_SIZING_FAILED",
            app.fatal_preflight,
            {"nav_gbp": nav_gbp, "gbp_usd": gbp_usd, "planned": [{k: v for k, v in p.items() if k != "contract"} for p in planned]},
        )
        write_outputs(payload)
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    print(
        f"NAV_GBP={nav_gbp:.2f} GBPUSD={gbp_usd:.6f} NAV_USD={nav_usd:.2f} planned_equity={equity_usd/nav_usd:.4f}",
        flush=True,
    )
    for p in planned:
        print(f"  {p['ticker']} qty={p['quantity']} ask={p['size_price_usd']} usd={p['planned_usd']:.2f}", flush=True)

    # what-if permission probe (not a live/working order)
    probe = planned[0]
    oid_if = app.alloc_order_id()
    app.orders[oid_if] = {"ticker": probe["ticker"], "whatIf": True}
    app.placeOrder(
        oid_if,
        probe["contract"],
        make_order("BUY", probe["quantity"], "P5_PAPER_WHATIF", acct, what_if=True),
    )
    if not app.whatif_event.wait(12) or app.readonly_blocked or app.fatal_preflight:
        payload = fail_payload(
            app,
            "NO_ORDERS_WHATIF_OR_READONLY_FAILED",
            app.fatal_preflight or ["whatIf did not confirm; not submitting real orders"],
            {
                "nav_gbp": nav_gbp,
                "gbp_usd": gbp_usd,
                "readonly_blocked": app.readonly_blocked,
                "target_ok": True,
                "positions_empty": True,
            },
        )
        write_outputs(payload)
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    # Real P5 orders only
    app.phase = "submit"
    submitted_at = utc_now()
    for p in planned:
        oid = app.alloc_order_id()
        p["order_id"] = oid
        p["submitted_ts"] = utc_now()
        p["order_type"] = "MKT"
        app.orders[oid] = {
            "ticker": p["ticker"],
            "target_weight": p["target_weight"],
            "target_value_gbp": p["target_value_gbp"],
            "target_value_usd": p["target_value_usd"],
            "quantity": p["quantity"],
            "size_price_usd": p["size_price_usd"],
            "order_type": "MKT",
            "submitted_ts": p["submitted_ts"],
        }
        app.placeOrder(
            oid,
            p["contract"],
            make_order("BUY", p["quantity"], f"P5_PAPER_{p['ticker']}", acct, what_if=False),
        )
        print(f"SUBMIT P5 BUY {p['ticker']} qty={p['quantity']} id={oid}", flush=True)

    fill_deadline = time.time() + 90
    while time.time() < fill_deadline:
        states = [app.orders.get(p["order_id"], {}).get("status") for p in planned]
        if all(s in {"Filled", "Cancelled", "Inactive", "ApiCancelled"} for s in states if s) and all(s for s in states):
            if all(s == "Filled" for s in states):
                break
        time.sleep(0.3)
    time.sleep(2.0)  # commissions

    app.phase = "postfill"
    app.positions = []
    app.post_summary_event.clear()
    app.post_positions_event.clear()
    app.reqAccountSummary(9102, "All", SUMMARY_TAGS)
    app.reqPositions()
    app.post_summary_event.wait(10)
    app.post_positions_event.wait(10)

    post_nav = float(app.summary.get(acct, {}).get("NetLiquidation") or nav_gbp)
    fills = []
    stock_usd = 0.0
    for p in planned:
        rec = app.orders.get(p["order_id"], {})
        fill_qty = float(rec.get("filled") or 0.0)
        avg = float(rec.get("avg_fill_price") or 0.0)
        usd_mkt = fill_qty * avg
        stock_usd += usd_mkt
        gbp_mkt = usd_mkt / gbp_usd if gbp_usd else float("nan")
        actual_w = gbp_mkt / post_nav if post_nav else float("nan")
        slip = None
        if avg > 0 and p["size_price_usd"] > 0:
            slip = (avg - p["size_price_usd"]) / p["size_price_usd"] * 1e4
        fills.append(
            {
                "ticker": p["ticker"],
                "order_id": p["order_id"],
                "submitted_ts": p["submitted_ts"],
                "order_type": "MKT",
                "target_weight": p["target_weight"],
                "target_value_gbp": round(p["target_value_gbp"], 4),
                "target_value_usd": round(p["target_value_usd"], 4),
                "size_price_usd_ask": p["size_price_usd"],
                "quantity": p["quantity"],
                "status": rec.get("status"),
                "fill_quantity": fill_qty,
                "avg_fill_price": avg,
                "commission": rec.get("commission"),
                "commission_currency": rec.get("commission_currency"),
                "usd_market_value": round(usd_mkt, 4),
                "gbp_market_value": round(gbp_mkt, 4) if math.isfinite(gbp_mkt) else None,
                "actual_post_fill_weight": round(actual_w, 8) if math.isfinite(actual_w) else None,
                "slippage_bps_vs_ask": round(slip, 4) if slip is not None else None,
                "slippage_bps": round(slip, 4) if slip is not None else None,
            }
        )

    strategy_usd_ret = None
    if nav_usd:
        # cash remaining in USD terms ≈ post_nav * fx - stock_usd; PnL vs start
        strategy_usd_ret = (post_nav * gbp_usd - nav_usd) / nav_usd

    any_unfilled = any((f["status"] != "Filled") or (f["fill_quantity"] + 1e-9 < f["quantity"]) for f in fills)

    payload = {
        "generated": utc_now(),
        "status": "P5_PAPER_ORDERS_COMPLETE" if not any_unfilled else "P5_PAPER_ORDERS_PARTIAL_OR_UNFILLED",
        "execution_arm": "P5",
        "vt0_orders": False,
        "orders_submitted": True,
        "submitted_batch_ts": submitted_at,
        "account_id": acct,
        "account_kind": "PAPER",
        "live_stop": False,
        "target_ok": True,
        "positions_empty": True,
        "nav_gbp": nav_gbp,
        "nav_gbp_currency": "GBP",
        "gbp_usd": gbp_usd,
        "fx_bid": fx_bid,
        "fx_ask": fx_ask,
        "nav_usd": nav_usd,
        "planned_equity_weight": equity_usd / nav_usd,
        "buying_power_used": False,
        "leverage_used": False,
        "post_nav_gbp": post_nav,
        "post_stock_usd": stock_usd,
        "post_stock_gbp": stock_usd / gbp_usd,
        "strategy_usd_return": strategy_usd_ret,
        "post_positions": [p for p in app.positions if abs(p["position"]) > 1e-9],
        "fills": fills,
        "errors": app.errors,
        "fatal": app.fatal_preflight,
    }

    for i in list(md_ids.values()):
        try:
            app.cancelMktData(i)
        except Exception:
            pass
    for i in list(tbt_ids.values()):
        try:
            app.cancelTickByTickData(i)
        except Exception:
            pass
    for i in list(hist_ids.values()) + [fx_hid]:
        try:
            app.cancelHistoricalData(i)
        except Exception:
            pass
    try:
        app.cancelAccountSummary(9101)
        app.cancelAccountSummary(9102)
        app.cancelPositions()
        app.reqAccountUpdates(False, acct)
        app.disconnect()
    except Exception:
        pass
    write_outputs(payload)
    print(f"status={payload['status']} wrote {OUT_MD}", flush=True)
    return 0 if payload["status"] == "P5_PAPER_ORDERS_COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
