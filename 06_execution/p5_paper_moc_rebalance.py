#!/usr/bin/env python3
"""P5 Paper n_drop=1 rebalance via MOC for the next session close: sell the
dropped name, buy the new name (no clip of remaining names back to 19%).
VT0 is not ordered; no IBKR market-data requests."""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from ibapi.account_summary_tags import AccountSummaryTags
from ibapi.client import EClient
from ibapi.contract import Contract, ContractDetails
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.wrapper import EWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IBKR_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(IBKR_DIR) not in sys.path:
    sys.path.insert(0, str(IBKR_DIR))

from p5_paper_execute_massive_sizing import (  # noqa: E402
    WEIGHT,
    classify_account,
    pick_nav,
    pick_usd_rate,
)
from p5_paper_ledger_preflight import LedgerProbe, EXPECTED_ACCOUNT  # noqa: E402

HOST = "127.0.0.1"
PORT = 7497
CLIENT_IDS = (17901, 17902, 17903)
TARGET_JSON = PROJECT_ROOT / "data/portfolio_experiments/market_risk/live_target/order_target.json"
PRICE_PARQUET = PROJECT_ROOT / "data/massive_2026/normalized/daily_ohlcv_split_adjusted.parquet"
OUT_MD = PROJECT_ROOT / "reports/ibkr/p5_paper_moc_rebalance.md"
OUT_JSON = PROJECT_ROOT / "reports/ibkr/p5_paper_moc_rebalance.json"
PRIMARY = {
    "CIEN": "NYSE",
    "CAH": "NYSE",
    "PANW": "NASDAQ",
    "AAPL": "NASDAQ",
    "AXON": "NASDAQ",
    "CNC": "NYSE",
    "CSGP": "NASDAQ",
    "INTU": "NASDAQ",
    "IT": "NYSE",
    "MCHP": "NASDAQ",
    "TTWO": "NASDAQ",
    "KLAC": "NASDAQ",
    "FICO": "NYSE",
    "EXPE": "NASDAQ",
}
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


def massive_close(ticker: str, session: str) -> float:
    import pandas as pd

    px = pd.read_parquet(PRICE_PARQUET)
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px["ticker"] = px["ticker"].astype(str).str.upper()
    hit = px[(px["ticker"] == ticker) & (px["date"] == pd.Timestamp(session))]
    if hit.empty:
        raise RuntimeError(f"Massive close missing {ticker} {session}")
    close = float(hit["close"].iloc[0])
    if close <= 0:
        raise RuntimeError(f"non-positive Massive close {ticker}")
    return close


def stk_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.currency = "USD"
    c.exchange = "SMART"
    c.primaryExchange = PRIMARY[symbol]
    return c


def make_moc(action: str, qty: int, ref: str, account: str) -> Order:
    o = Order()
    o.action = action
    o.totalQuantity = Decimal(int(qty))
    o.orderType = "MOC"
    o.tif = "DAY"
    o.transmit = True
    o.outsideRth = False
    o.whatIf = False
    o.orderRef = ref
    o.account = account
    return o


class MocApp(LedgerProbe):
    def __init__(self) -> None:
        super().__init__()
        self.next_oid: int | None = None
        self.readonly_blocked = False
        self.orders: dict[int, dict[str, Any]] = {}
        self.contracts: dict[int, list[ContractDetails]] = {}
        self.contract_events: dict[int, threading.Event] = {}

    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):  # noqa: N802
        super().error(reqId, errorTime, errorCode, errorString, advancedOrderRejectJson)
        msg = str(errorString).lower()
        if int(errorCode) == 10349 or "read-only" in msg:
            self.readonly_blocked = True

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        self.next_oid = int(orderId)
        print(f"nextValidId={orderId} negotiated={self.serverVersion()}", flush=True)
        self.connected_event.set()
        self.reqManagedAccts()

    def contractDetails(self, reqId, contractDetails) -> None:  # noqa: N802
        with self.lock:
            self.contracts.setdefault(int(reqId), []).append(contractDetails)

    def contractDetailsEnd(self, reqId) -> None:  # noqa: N802
        ev = self.contract_events.get(int(reqId))
        if ev:
            ev.set()

    def openOrder(self, orderId, contract, order, orderState: OrderState) -> None:  # noqa: N802
        print(
            f"openOrder {orderId} {getattr(order, 'action', None)} "
            f"{getattr(order, 'orderType', None)} status={getattr(orderState, 'status', None)}",
            flush=True,
        )
        with self.lock:
            rec = self.orders.setdefault(int(orderId), {})
            rec["status"] = getattr(orderState, "status", None)
            rec["order_type"] = getattr(order, "orderType", None)

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):  # noqa: N802
        with self.lock:
            rec = self.orders.setdefault(int(orderId), {})
            rec["status"] = status
            rec["filled"] = float(filled)
            rec["remaining"] = float(remaining)
            rec["avg_fill_price"] = float(avgFillPrice)
        print(f"orderStatus {orderId} {status} filled={filled} avg={avgFillPrice}", flush=True)

    def alloc_oid(self) -> int:
        assert self.next_oid is not None
        oid = self.next_oid
        self.next_oid += 1
        return oid

    def still_paper(self) -> bool:
        return (
            not self.live_stop
            and self.account_kind == "PAPER"
            and bool(self.managed_accounts)
            and self.managed_accounts[0].upper().startswith("DU")
        )


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    planned = payload.get("planned") or []
    lines = [
        "# P5 Paper MOC Rebalance",
        "",
        f"Generated: `{payload.get('generated')}`",
        f"Status: **{payload.get('status')}**",
        "",
        f"- Account `{payload.get('account_id')}` `{payload.get('account_kind')}`",
        f"- Decision `{payload.get('decision_session')}` effective `{payload.get('order_session')}`",
        f"- Sell `{payload.get('sell')}` Buy `{payload.get('buy')}`",
        f"- Orders submitted `{payload.get('orders_submitted')}`",
        "",
        "## Planned",
        "",
    ]
    if planned:
        for p in planned:
            lines.append(
                f"- {p['action']} `{p['ticker']}` qty={p['quantity']} "
                f"type={p['order_type']} massive={p.get('reference_price_massive')}"
            )
    else:
        lines.append("(none)")
    lines.extend(["", "Fatal: " + "; ".join(payload.get("fatal") or ["none"]), ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def connect_app() -> tuple[MocApp | None, str | None]:
    last_err = None
    for cid in CLIENT_IDS:
        app = MocApp()
        try:
            print(f"connecting {HOST}:{PORT} clientId={cid}", flush=True)
            app.connect(HOST, PORT, cid)
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            continue
        if not app.isConnected():
            last_err = "not connected"
            continue
        threading.Thread(target=app.run, daemon=True).start()
        if app.connected_event.wait(20):
            return app, None
        last_err = "timeout nextValidId"
        try:
            app.disconnect()
        except Exception:
            pass
    return None, last_err or "connect failed"


def qualify(app: MocApp, ticker: str) -> Contract | None:
    rid = 8100 + (hash(ticker) % 99)
    ev = threading.Event()
    app.contract_events[rid] = ev
    app.reqContractDetails(rid, stk_contract(ticker))
    if not ev.wait(15):
        return None
    hits = app.contracts.get(rid, [])
    stks = [h.contract for h in hits if h.contract.secType == "STK" and h.contract.currency == "USD"]
    if not stks:
        return None
    primary = PRIMARY[ticker]
    c = next((x for x in stks if (x.primaryExchange or "").upper() == primary), stks[0])
    out = Contract()
    out.conId = c.conId
    out.symbol = c.symbol
    out.secType = "STK"
    out.currency = "USD"
    out.exchange = "SMART"
    out.primaryExchange = c.primaryExchange or primary
    return out


def main() -> int:
    target = json.loads(TARGET_JSON.read_text(encoding="utf-8"))
    wanted = {str(t).upper() for t in target["tickers"]}
    decision = str(target.get("last_complete_session") or "")
    order_session = str(target.get("order_session") or "")
    fatal: list[str] = []
    if target.get("execution_arm") != "P5":
        fatal.append(f"execution_arm={target.get('execution_arm')}")
    if len(wanted) != 5:
        fatal.append(f"wanted {sorted(wanted)} is not Top5")
    if fatal:
        write_outputs({"generated": utc_now(), "status": "NO_ORDERS_TARGET", "fatal": fatal, "orders_submitted": False, "planned": []})
        return 1

    app, err = connect_app()
    if app is None:
        write_outputs(
            {
                "generated": utc_now(),
                "status": "NO_ORDERS_CONNECT",
                "fatal": [err or "TWS not listening on 127.0.0.1:7497"],
                "decision_session": decision,
                "order_session": order_session,
                "orders_submitted": False,
                "planned": [],
                "note": "TWS Paper API is down. Open TWS Simulated Trading + API 7497, then rerun.",
            }
        )
        print(f"TWS not available: {err}", flush=True)
        return 1

    app.managed_event.wait(8)
    if not app.farms_ok.wait(90):
        write_outputs(
            {
                "generated": utc_now(),
                "status": "FAIL_TWS_SERVER",
                "fatal": ["TWS not connected to IBKR servers (no 2104). No orders submitted."],
                "orders_submitted": False,
                "planned": [],
                "errors": app.errors,
            }
        )
        try:
            app.disconnect()
        except Exception:
            pass
        return 1
    if not app.still_paper() or (app.managed_accounts[0] if app.managed_accounts else "") != EXPECTED_ACCOUNT:
        write_outputs({"generated": utc_now(), "status": "NO_ORDERS_NOT_PAPER", "fatal": [str(app.managed_accounts)], "orders_submitted": False, "planned": []})
        try:
            app.disconnect()
        except Exception:
            pass
        return 1
    acct = app.managed_accounts[0]
    app.reqAccountSummary(9301, "All", SUMMARY_TAGS)
    app.reqPositions()
    app.reqAccountUpdates(True, acct)
    app.summary_event.wait(10)
    app.positions_event.wait(10)
    app.acct_end_event.wait(8)
    deadline = time.time() + 12
    usd_rate = None
    usd_key = None
    merged: dict[str, str] = {}
    while time.time() < deadline:
        with app.lock:
            merged = dict(app.values.get(acct) or {})
        usd_rate, usd_key = pick_usd_rate(merged)
        if usd_rate:
            break
        time.sleep(0.2)
    nav, nav_src = pick_nav(app.summary.get(acct, {}), merged)
    stock = [p for p in app.positions if str(p.get("sec_type") or "").upper() == "STK" and abs(p["position"]) > 1e-9]
    by_sym = {str(p["symbol"]).upper(): p for p in stock}
    held = set(by_sym)
    sell_names = sorted(held - wanted)
    buy_names = sorted(wanted - held)
    hold_names = sorted(held & wanted)
    print(f"rotation HOLD={hold_names} SELL={sell_names} BUY={buy_names}", flush=True)
    if nav is None or nav <= 0:
        fatal.append(f"bad NAV {nav}")
    if usd_rate is None:
        fatal.append("missing ExchangeRate:USD")
    if not (5 <= len(held) <= 6):
        fatal.append(f"expected 5 names or 5+residual, have {sorted(held)}")
    if len(buy_names) != 1:
        fatal.append(f"n_drop=1 requires exactly one buy; sell={sell_names} buy={buy_names}")
    if not sell_names:
        fatal.append(f"nothing to sell; sell={sell_names} buy={buy_names}")
    buy_tkr = buy_names[0] if len(buy_names) == 1 else None
    sell_qty_by = {}
    for tkr in sell_names:
        qty = int(round(abs(float(by_sym.get(tkr, {}).get("position") or 0))))
        sell_qty_by[tkr] = qty
        if qty < 1:
            fatal.append(f"no {tkr} position to sell: {stock}")
    if buy_tkr and buy_tkr in by_sym:
        fatal.append(f"{buy_tkr} already held; refusing a second sleeve")
    if fatal:
        write_outputs(
            {
                "generated": utc_now(),
                "status": "NO_ORDERS_ACCOUNT",
                "fatal": fatal,
                "orders_submitted": False,
                "planned": [],
                "account_id": acct,
                "nav_gbp": nav,
                "sell": sell_names,
                "buy": buy_tkr,
                "hold": hold_names,
            }
        )
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    buy_px = massive_close(buy_tkr, decision)
    target_usd = WEIGHT * nav / usd_rate
    buy_qty = int(math.floor(target_usd / buy_px))
    if buy_qty < 1:
        write_outputs({"generated": utc_now(), "status": "NO_ORDERS_SIZING", "fatal": [f"{buy_tkr} qty=0 px={buy_px}"], "orders_submitted": False, "planned": []})
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    planned = []
    for tkr in sell_names:
        planned.append(
            {
                "action": "SELL",
                "ticker": tkr,
                "quantity": sell_qty_by[tkr],
                "order_type": "MOC",
                "target_weight": 0.0,
                "reference_price_massive": None,
            }
        )
    planned.append(
        {
            "action": "BUY",
            "ticker": buy_tkr,
            "quantity": buy_qty,
            "order_type": "MOC",
            "target_weight": WEIGHT,
            "reference_price_massive": buy_px,
            "target_value_gbp": WEIGHT * nav,
            "target_value_usd": target_usd,
        }
    )
    print(
        f"NAV={nav:.2f} FX={usd_rate} SELL {sell_qty_by} BUY {buy_tkr} {buy_qty} @{buy_px}",
        flush=True,
    )

    cmap = {}
    for tkr in sell_names + [buy_tkr]:
        c = qualify(app, tkr)
        if c is None:
            fatal.append(f"contract {tkr}")
        else:
            cmap[tkr] = c
            print(f"qualified {tkr} conId={c.conId}", flush=True)
    if fatal:
        write_outputs({"generated": utc_now(), "status": "NO_ORDERS_CONTRACT", "fatal": fatal, "orders_submitted": False, "planned": planned})
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    submitted = []
    for p in planned:
        if not app.still_paper() or app.readonly_blocked:
            fatal.append("lost PAPER or read-only")
            break
        oid = app.alloc_oid()
        p["order_id"] = oid
        p["submitted_ts"] = utc_now()
        app.placeOrder(oid, cmap[p["ticker"]], make_moc(p["action"], p["quantity"], f"P5_MOC_{p['action']}_{p['ticker']}", acct))
        print(f"SUBMIT MOC {p['action']} {p['ticker']} qty={p['quantity']} id={oid}", flush=True)
        submitted.append(p)
        time.sleep(0.4)
        if app.readonly_blocked:
            fatal.append("read-only after submit")
            break
    time.sleep(3)
    payload = {
        "generated": utc_now(),
        "status": "P5_MOC_QUEUED" if submitted and not fatal else "P5_MOC_PARTIAL_OR_FAIL",
        "execution_arm": "P5",
        "vt0_orders": False,
        "order_type": "MOC",
        "orders_submitted": bool(submitted),
        "account_id": acct,
        "account_kind": "PAPER",
        "nav_gbp": nav,
        "nav_source": nav_src,
        "exchange_rate_usd": usd_rate,
        "exchange_rate_key": usd_key,
        "decision_session": decision,
        "order_session": order_session,
        "sell": sell_names,
        "buy": buy_tkr,
        "hold": hold_names,
        "planned": planned,
        "submitted": submitted,
        "orders": app.orders,
        "errors": [e for e in app.errors if e.get("code") not in {2104, 2106, 2108, 2158, 2100}],
        "fatal": fatal,
        "p5_parameters_changed": False,
    }
    try:
        app.cancelAccountSummary(9301)
        app.cancelPositions()
        app.reqAccountUpdates(False, acct)
        app.disconnect()
    except Exception:
        pass
    write_outputs(payload)
    print(f"status={payload['status']} wrote {OUT_MD}", flush=True)
    return 0 if payload["status"] == "P5_MOC_QUEUED" else 2


if __name__ == "__main__":
    sys.exit(main())
