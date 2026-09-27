#!/usr/bin/env python3
"""P5 Paper sizing helper: Massive reference closes + IBKR account ExchangeRate:USD.
Massive price is for sizing only; IBKR AvgPrice is the recorded fill.
No IBKR market-data subscription; no VT0 orders."""

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
from ibapi.commission_and_fees_report import CommissionAndFeesReport
from ibapi.contract import Contract, ContractDetails
from ibapi.execution import Execution
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.server_versions import MAX_CLIENT_VER, MIN_SERVER_VER_FRACTIONAL_SIZE_SUPPORT
from ibapi.wrapper import EWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_data = str(PROJECT_ROOT / "01_data")
if _data not in sys.path:
    sys.path.insert(0, _data)

from massive_audit.client import MassiveClient  # noqa: E402
from massive_audit.phase_c import _ms_to_date  # noqa: E402

HOST = "127.0.0.1"
PORT = 7497
CLIENT_IDS = (17731, 17732, 17733)
TARGET_CSV = PROJECT_ROOT / "data/portfolio_experiments/market_risk/live_target/order_target.csv"
OUT_MD = PROJECT_ROOT / "reports/ibkr/p5_paper_first_orders.md"
OUT_JSON = PROJECT_ROOT / "reports/ibkr/p5_paper_first_orders.json"
OUT_FILLS = PROJECT_ROOT / "reports/ibkr/p5_paper_first_orders_fills.csv"

EXPECTED = ["AXON", "CNC", "CSGP", "INTU", "MCHP"]
WEIGHT = 0.19
CASH_W = 0.05
PRIMARY = {
    "AXON": "NASDAQ",
    "CNC": "NYSE",
    "CSGP": "NASDAQ",
    "INTU": "NASDAQ",
    "MCHP": "NASDAQ",
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


def classify_account(account_id: str) -> str:
    acct = (account_id or "").strip().upper()
    if acct.startswith("DU") or acct.startswith("DF"):
        return "PAPER"
    if acct.startswith(("U", "F", "I")):
        return "LIVE"
    return "UNKNOWN"


def load_target() -> tuple[list[str], list[str]]:
    err: list[str] = []
    rows = TARGET_CSV.read_text(encoding="utf-8").strip().splitlines()[1:]
    names = []
    cash = None
    for line in rows:
        ticker, arm, w, *_ = line.split(",")
        ticker, arm = ticker.strip().upper(), arm.strip()
        w = float(w)
        if ticker == "CASH":
            cash = w
            continue
        if arm != "P5":
            err.append(f"{ticker} arm={arm}")
        if abs(w - WEIGHT) > 1e-12:
            err.append(f"{ticker} weight={w}")
        names.append(ticker)
    if names != EXPECTED:
        err.append(f"tickers {names} != {EXPECTED}")
    if cash is None or abs(cash - CASH_W) > 1e-9:
        err.append(f"cash={cash}")
    return names, err


def massive_latest_prices(tickers: list[str]) -> dict[str, dict[str, Any]]:
    client = MassiveClient(min_interval_s=12.5)
    today = datetime.now().date().isoformat()
    status, payload, _ = client.get(
        f"/v2/aggs/ticker/SPY/range/1/day/2026-08-20/{today}",
        {"adjusted": "true", "limit": 20, "sort": "desc"},
    )
    if status != 200 or not isinstance(payload, dict) or not payload.get("results"):
        raise RuntimeError(f"Massive SPY latest failed HTTP {status}")
    latest = _ms_to_date(int(payload["results"][0]["t"])).date().isoformat()
    st, body, _ = client.get(
        f"/v2/aggs/grouped/locale/us/market/stocks/{latest}",
        {"adjusted": "true"},
    )
    if st != 200 or not isinstance(body, dict):
        raise RuntimeError(f"Massive grouped {latest} HTTP {st}")
    by_t = {str(r.get("T", "")).upper(): r for r in (body.get("results") or []) if isinstance(r, dict)}
    out: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        bar = by_t.get(ticker)
        if not bar:
            raise RuntimeError(f"Massive grouped {latest} missing {ticker}")
        px = float(bar.get("c") or 0)
        if px <= 0:
            raise RuntimeError(f"Massive {ticker} non-positive close")
        out[ticker] = {
            "reference_price_massive": px,
            "massive_session": latest,
            "massive_close": px,
            "source": f"Massive split-adjusted grouped daily close {latest}",
        }
        print(f"Massive {ticker} {latest} close={px}", flush=True)
    return out


def stk_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.currency = "USD"
    c.exchange = "SMART"
    c.primaryExchange = PRIMARY[symbol]
    return c


def make_buy(qty: int, ref: str, account: str, what_if: bool = False) -> Order:
    o = Order()
    o.action = "BUY"
    o.totalQuantity = Decimal(int(qty))
    o.orderType = "MKT"
    o.tif = "DAY"
    o.transmit = True
    o.outsideRth = False
    o.whatIf = bool(what_if)
    o.orderRef = ref
    o.account = account
    return o


class PaperP5(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.lock = threading.Lock()
        self.errors: list[dict[str, Any]] = []
        self.managed_accounts: list[str] = []
        self.account_kind = "UNKNOWN"
        self.live_stop = False
        self.readonly_blocked = False
        self.summary: dict[str, dict[str, str]] = {}
        self.values: dict[str, dict[str, str]] = {}
        self.positions: list[dict[str, Any]] = []
        self.contracts: dict[int, list[ContractDetails]] = {}
        self.contract_events: dict[int, threading.Event] = {}
        self.orders: dict[int, dict[str, Any]] = {}
        self.execs: dict[str, dict[str, Any]] = {}
        self.next_oid: int | None = None
        self.connected_event = threading.Event()
        self.managed_event = threading.Event()
        self.summary_event = threading.Event()
        self.positions_event = threading.Event()
        self.acct_end_event = threading.Event()
        self.whatif_event = threading.Event()
        self.post_summary_event = threading.Event()
        self.post_positions_event = threading.Event()
        self.phase = "pre"
        self.whatif_oid: int | None = None

    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):  # noqa: N802
        rec = {
            "req_id": reqId,
            "code": int(errorCode),
            "message": str(errorString),
            "ts": utc_now(),
        }
        with self.lock:
            self.errors.append(rec)
        msg = str(errorString).lower()
        if int(errorCode) == 10349 or "read-only" in msg or "readonly" in msg.replace(" ", ""):
            self.readonly_blocked = True
        if self.whatif_oid is not None and int(reqId) == int(self.whatif_oid):
            if int(errorCode) not in {2104, 2106, 2108, 2158, 2100}:
                self.whatif_event.set()
        if int(errorCode) not in {2104, 2106, 2108, 2158}:
            print(f"ERR {reqId} {errorCode} {errorString}", flush=True)

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        self.next_oid = int(orderId)
        print(f"nextValidId={orderId} negotiated={self.serverVersion()} max={MAX_CLIENT_VER}", flush=True)
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
        with self.lock:
            self.summary.setdefault(account, {})[tag] = str(value)
            if currency:
                self.summary[account][f"{tag}_currency"] = str(currency)

    def accountSummaryEnd(self, reqId) -> None:  # noqa: N802
        if self.phase == "post":
            self.post_summary_event.set()
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
        if self.phase == "post":
            self.post_positions_event.set()
        else:
            self.positions_event.set()

    def contractDetails(self, reqId, contractDetails) -> None:  # noqa: N802
        with self.lock:
            self.contracts.setdefault(int(reqId), []).append(contractDetails)

    def contractDetailsEnd(self, reqId) -> None:  # noqa: N802
        ev = self.contract_events.get(int(reqId))
        if ev:
            ev.set()

    def openOrder(self, orderId, contract, order, orderState: OrderState) -> None:  # noqa: N802
        print(
            f"openOrder {orderId} whatIf={getattr(order, 'whatIf', None)} "
            f"status={getattr(orderState, 'status', None)} "
            f"warning={getattr(orderState, 'warningText', None)}",
            flush=True,
        )
        if self.whatif_oid is not None and int(orderId) == int(self.whatif_oid):
            self.whatif_event.set()

    def orderStatus(
        self,
        orderId,
        status,
        filled,
        remaining,
        avgFillPrice,
        permId,
        parentId,
        lastFillPrice,
        clientId,
        whyHeld,
        mktCapPrice,
    ) -> None:  # noqa: N802
        with self.lock:
            rec = self.orders.setdefault(int(orderId), {})
            rec["status"] = status
            rec["filled"] = float(filled)
            rec["remaining"] = float(remaining)
            rec["avg_fill_price"] = float(avgFillPrice)
            rec["status_ts"] = utc_now()
        if self.whatif_oid is not None and int(orderId) == int(self.whatif_oid):
            self.whatif_event.set()
        print(f"orderStatus {orderId} {status} filled={filled} avg={avgFillPrice}", flush=True)

    def execDetails(self, reqId, contract, execution: Execution) -> None:  # noqa: N802
        with self.lock:
            oid = int(execution.orderId)
            rec_ex = {
                "exec_id": execution.execId,
                "time": execution.time,
                "price": float(execution.price),
                "shares": float(execution.shares),
                "avg_price": float(execution.avgPrice),
                "order_id": oid,
                "exchange": execution.exchange,
            }
            self.execs[execution.execId] = rec_ex
            rec = self.orders.setdefault(oid, {})
            rec.setdefault("executions", []).append(execution.execId)
            rec["ibkr_exec_price"] = float(execution.price)
            rec["ibkr_avg_price"] = float(execution.avgPrice)
            rec["execution_time"] = execution.time

    def commissionAndFeesReport(self, commissionAndFeesReport: CommissionAndFeesReport) -> None:  # noqa: N802
        with self.lock:
            ex = self.execs.setdefault(commissionAndFeesReport.execId, {})
            fee = float(commissionAndFeesReport.commissionAndFees)
            ex["commission"] = fee
            ex["commission_currency"] = commissionAndFeesReport.currency
            oid = ex.get("order_id")
            if oid is not None:
                rec = self.orders.setdefault(int(oid), {})
                rec["commission"] = rec.get("commission", 0.0) + fee
                rec["commission_currency"] = commissionAndFeesReport.currency

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


def pick_usd_rate(values: dict[str, str]) -> tuple[float | None, str | None]:
    """IBKR account ExchangeRate for USD = base currency (GBP) per 1 USD.

    Only accept an explicit USD tag. Do not fall back to GBP/BASE (=1).
    """
    wanted = {
        "EXCHANGERATE:USD",
        "$LEDGER-EXCHANGERATE:USD",
        "EXCHANGERATE:USD.IDEALPRO",
    }
    for key, raw in values.items():
        if key.upper().replace(" ", "") not in wanted:
            continue
        try:
            rate = float(raw)
        except (TypeError, ValueError):
            continue
        if rate > 0:
            return rate, key
    return None, None


def pick_nav(summary: dict[str, str], values: dict[str, str]) -> tuple[float | None, str]:
    if summary.get("NetLiquidation"):
        try:
            return float(summary["NetLiquidation"]), "accountSummary.NetLiquidation"
        except ValueError:
            pass
    for key in ("NetLiquidation:GBP", "NetLiquidation:BASE"):
        if key in values:
            try:
                return float(values[key]), key
            except ValueError:
                continue
    return None, ""


def qualified_contract(hits: list[ContractDetails], symbol: str) -> Contract | None:
    stks = [h.contract for h in hits if h.contract.secType == "STK" and h.contract.currency == "USD"]
    if not stks:
        return None
    primary = PRIMARY[symbol]
    for c in stks:
        if (c.primaryExchange or "").upper() == primary:
            out = Contract()
            out.conId = c.conId
            out.symbol = c.symbol
            out.secType = "STK"
            out.currency = "USD"
            out.exchange = "SMART"
            out.primaryExchange = c.primaryExchange
            return out
    c = stks[0]
    out = Contract()
    out.conId = c.conId
    out.symbol = c.symbol
    out.secType = "STK"
    out.currency = "USD"
    out.exchange = "SMART"
    out.primaryExchange = c.primaryExchange or primary
    return out


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    fills = payload.get("fills") or []
    fill_lines = ["(none)"]
    if fills:
        fill_lines = [
            (
                f"- `{f['ticker']}` qty={f['quantity']} massive={f['reference_price_massive']} "
                f"fill={f['ibkr_fill_price']} tgt_w={f['target_weight']} act_w={f['actual_post_fill_weight']} "
                f"ts={f['execution_timestamp']}"
            )
            for f in fills
        ]
        keys = list(fills[0].keys())
        with OUT_FILLS.open("w", encoding="utf-8") as handle:
            handle.write(",".join(keys) + "\n")
            for f in fills:
                handle.write(",".join("" if f[k] is None else str(f[k]) for k in keys) + "\n")
    lines = [
        "# P5 Paper First Orders",
        "",
        f"Generated: `{payload.get('generated')}`",
        f"Status: **{payload.get('status')}**",
        "",
        "P5 executed (or attempted). VT0 shadow-only. No IBKR market-data subscription.",
        "Sizing: Massive daily close + IBKR account ExchangeRate:USD. Fill price = IBKR AvgPrice.",
        "",
        f"- Account `{payload.get('account_id')}` `{payload.get('account_kind')}`",
        f"- NAV GBP `{payload.get('nav_gbp')}`",
        f"- ExchangeRate USD `{payload.get('exchange_rate_usd')}` ({payload.get('exchange_rate_key')})",
        f"- GBP per 1 USD; USD = GBP / ExchangeRate",
        f"- Massive session `{payload.get('massive_session')}`",
        f"- Orders submitted `{payload.get('orders_submitted')}`",
        "",
        "## Fills",
        "",
        *fill_lines,
        "",
        "Fatal: " + "; ".join(payload.get("fatal") or ["none"]),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    names, terr = load_target()
    if terr:
        write_outputs({"generated": utc_now(), "status": "NO_ORDERS_TARGET", "fatal": terr, "orders_submitted": False, "fills": []})
        return 1
    try:
        massive = massive_latest_prices(names)
    except Exception as exc:  # noqa: BLE001
        write_outputs({"generated": utc_now(), "status": "NO_ORDERS_MASSIVE", "fatal": [str(exc)], "orders_submitted": False, "fills": []})
        print(f"Massive failed: {exc}", flush=True)
        return 1

    app = PaperP5()
    last_err = None
    for cid in CLIENT_IDS:
        try:
            print(f"connecting {HOST}:{PORT} clientId={cid}", flush=True)
            app = PaperP5()
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
        write_outputs({"generated": utc_now(), "status": "NO_ORDERS_CONNECT", "fatal": [last_err or "connect"], "orders_submitted": False, "fills": []})
        return 1
    if int(app.serverVersion() or 0) < MIN_SERVER_VER_FRACTIONAL_SIZE_SUPPORT:
        write_outputs({"generated": utc_now(), "status": "NO_ORDERS_API_VERSION", "fatal": [f"negotiated {app.serverVersion()}"], "orders_submitted": False, "fills": []})
        try:
            app.disconnect()
        except Exception:
            pass
        return 1
    app.managed_event.wait(8)
    if not app.still_paper():
        write_outputs(
            {
                "generated": utc_now(),
                "status": "NO_ORDERS_NOT_PAPER",
                "fatal": [f"{app.managed_accounts} {app.account_kind}"],
                "orders_submitted": False,
                "fills": [],
                "live_stop": app.live_stop,
            }
        )
        try:
            app.disconnect()
        except Exception:
            pass
        return 1
    acct = app.managed_accounts[0]

    app.reqAccountSummary(9201, "All", SUMMARY_TAGS)
    app.reqPositions()
    app.reqAccountUpdates(True, acct)
    app.summary_event.wait(10)
    app.positions_event.wait(10)
    app.acct_end_event.wait(8)
    # ExchangeRate:USD can arrive after accountDownloadEnd; GBP-only ledgers
    # may never emit it. Wait, then fail closed if still missing.
    deadline = time.time() + 15
    usd_rate = None
    usd_key = None
    while time.time() < deadline:
        usd_rate, usd_key = pick_usd_rate(app.values.get(acct, {}))
        if usd_rate:
            break
        time.sleep(0.2)
    print(
        "ExchangeRate keys="
        + str([k for k in (app.values.get(acct) or {}) if "exchange" in k.lower() or "usd" in k.lower()]),
        flush=True,
    )

    nav, nav_src = pick_nav(app.summary.get(acct, {}), app.values.get(acct, {}))
    nonzero = [p for p in app.positions if abs(p["position"]) > 1e-9]
    stock_nonzero = [p for p in nonzero if str(p.get("sec_type") or "").upper() == "STK"]
    fatal: list[str] = []
    if nav is None or nav <= 0:
        fatal.append(f"bad NAV {nav}")
    if usd_rate is None:
        usd_keys = [k for k in (app.values.get(acct) or {}) if "exchange" in k.lower() or k.startswith("ExchangeRate")]
        fatal.append(f"missing ExchangeRate:USD; have {usd_keys}")
    if stock_nonzero:
        fatal.append(f"existing stock positions {stock_nonzero}; refusing to add a full P5 sleeve")
    if fatal:
        write_outputs(
            {
                "generated": utc_now(),
                "status": "NO_ORDERS_ACCOUNT_DATA",
                "fatal": fatal,
                "account_id": acct,
                "account_kind": app.account_kind,
                "nav_gbp": nav,
                "exchange_rate_usd": usd_rate,
                "exchange_rate_key": usd_key,
                "massive_session": next(iter(massive.values()))["massive_session"] if massive else None,
                "massive_reference_prices": {
                    t: massive[t]["reference_price_massive"] for t in names
                },
                "account_value_keys": sorted((app.values.get(acct) or {}).keys()),
                "orders_submitted": False,
                "fills": [],
                "ibkr_quotes_used": False,
                "vt0_orders": False,
                "note": (
                    "IBKR only streams ExchangeRate for currencies that appear on the "
                    "account ledger. This GBP Paper account currently has no USD cash "
                    "line, so ExchangeRate:USD is absent. No orders submitted."
                ),
            }
        )
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    # IBKR: ExchangeRate for USD = base(GBP) per 1 USD
    planned = []
    for ticker in names:
        ref = massive[ticker]["reference_price_massive"]
        target_gbp = WEIGHT * nav
        target_usd = target_gbp / usd_rate
        qty = int(math.floor(target_usd / ref))
        if qty < 1:
            fatal.append(f"{ticker} qty=0 ref={ref} target_usd={target_usd}")
            continue
        planned.append(
            {
                "ticker": ticker,
                "target_weight": WEIGHT,
                "target_value_gbp": target_gbp,
                "target_value_usd": target_usd,
                "reference_price_massive": ref,
                "massive_session": massive[ticker]["massive_session"],
                "quantity": qty,
            }
        )
    if fatal or len(planned) != 5:
        write_outputs({"generated": utc_now(), "status": "NO_ORDERS_SIZING", "fatal": fatal, "orders_submitted": False, "fills": [], "planned": planned})
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    print(
        f"NAV_GBP={nav:.2f} ExchangeRate:USD={usd_rate} ({usd_key}) "
        f"USD=GBP/rate 19%={WEIGHT * nav / usd_rate:.2f}",
        flush=True,
    )
    for p in planned:
        print(
            f"  {p['ticker']} massive={p['reference_price_massive']} qty={p['quantity']} "
            f"ref_usd={p['quantity'] * p['reference_price_massive']:.2f}",
            flush=True,
        )

    # Qualify contracts (no IBKR quotes)
    qmap: dict[str, Contract] = {}
    rid = 7000
    for ticker in names:
        rid += 1
        ev = threading.Event()
        app.contract_events[rid] = ev
        app.reqContractDetails(rid, stk_contract(ticker))
        if not ev.wait(12):
            fatal.append(f"contract timeout {ticker}")
            continue
        c = qualified_contract(app.contracts.get(rid, []), ticker)
        if c is None:
            fatal.append(f"no USD STK {ticker}")
        else:
            qmap[ticker] = c
            print(f"qualified {ticker} conId={c.conId}", flush=True)
    if len(qmap) != 5:
        write_outputs({"generated": utc_now(), "status": "NO_ORDERS_CONTRACT", "fatal": fatal, "orders_submitted": False, "fills": []})
        try:
            app.disconnect()
        except Exception:
            pass
        return 1

    # what-if permission probe (not a working order)
    probe = planned[0]
    if not app.still_paper():
        write_outputs({"generated": utc_now(), "status": "NO_ORDERS_NOT_PAPER", "fatal": ["lost PAPER before whatIf"], "orders_submitted": False, "fills": []})
        try:
            app.disconnect()
        except Exception:
            pass
        return 1
    oid_if = app.alloc_oid()
    app.whatif_oid = oid_if
    app.placeOrder(oid_if, qmap[probe["ticker"]], make_buy(probe["quantity"], "P5_PAPER_WHATIF", acct, True))
    got_whatif = app.whatif_event.wait(12)
    if app.readonly_blocked:
        write_outputs(
            {
                "generated": utc_now(),
                "status": "NO_ORDERS_READONLY_OR_WHATIF",
                "fatal": ["TWS API is Read-Only; no live orders"],
                "readonly_blocked": True,
                "whatif_oid": oid_if,
                "account_id": acct,
                "account_kind": app.account_kind,
                "nav_gbp": nav,
                "exchange_rate_usd": usd_rate,
                "planned": planned,
                "errors": app.errors,
                "orders_submitted": False,
                "fills": [],
            }
        )
        try:
            app.disconnect()
        except Exception:
            pass
        return 1
    if not got_whatif:
        print("whatIf produced no callback; API not read-only; proceeding to live P5 orders", flush=True)
    app.whatif_oid = None

    submitted_at = utc_now()
    for p in planned:
        if not app.still_paper():
            fatal.append(f"lost PAPER before {p['ticker']}")
            break
        if app.readonly_blocked:
            fatal.append(f"read-only before {p['ticker']}")
            break
        oid = app.alloc_oid()
        p["order_id"] = oid
        p["submitted_ts"] = utc_now()
        app.orders[oid] = dict(p)
        app.placeOrder(
            oid,
            qmap[p["ticker"]],
            make_buy(p["quantity"], f"P5_PAPER_{p['ticker']}", acct, False),
        )
        print(f"SUBMIT P5 BUY {p['ticker']} qty={p['quantity']} id={oid}", flush=True)
        time.sleep(0.35)
        if app.readonly_blocked:
            fatal.append(f"read-only after {p['ticker']}")
            break
    if fatal:
        write_outputs({"generated": utc_now(), "status": "STOPPED_MID_SUBMIT", "fatal": fatal, "orders_submitted": True, "fills": [], "errors": app.errors})
        try:
            app.disconnect()
        except Exception:
            pass
        return 2

    deadline = time.time() + 90
    while time.time() < deadline:
        states = [app.orders.get(p["order_id"], {}).get("status") for p in planned]
        if states and all(s == "Filled" for s in states):
            break
        time.sleep(0.3)
    time.sleep(2.5)

    app.phase = "post"
    app.positions = []
    app.post_summary_event.clear()
    app.post_positions_event.clear()
    app.reqAccountSummary(9202, "All", SUMMARY_TAGS)
    app.reqPositions()
    app.post_summary_event.wait(10)
    app.post_positions_event.wait(10)
    post_nav, _ = pick_nav(app.summary.get(acct, {}), app.values.get(acct, {}))
    post_nav = post_nav or nav

    fills = []
    for p in planned:
        rec = app.orders.get(p["order_id"], {})
        fill_qty = float(rec.get("filled") or 0)
        ibkr_px = float(rec.get("avg_fill_price") or rec.get("ibkr_avg_price") or rec.get("ibkr_exec_price") or 0)
        usd_mkt = fill_qty * ibkr_px
        gbp_mkt = usd_mkt * usd_rate  # USD * (GBP per USD)
        actual_w = gbp_mkt / post_nav if post_nav else None
        fills.append(
            {
                "ticker": p["ticker"],
                "order_id": p["order_id"],
                "execution_timestamp": rec.get("execution_time") or rec.get("status_ts") or p.get("submitted_ts"),
                "ibkr_exec_price": rec.get("ibkr_exec_price"),
                "submitted_ts": p.get("submitted_ts"),
                "order_type": "MKT",
                "quantity": p["quantity"],
                "fill_quantity": fill_qty,
                "reference_price_massive": p["reference_price_massive"],
                "massive_session": p["massive_session"],
                "ibkr_fill_price": ibkr_px,
                "target_weight": p["target_weight"],
                "target_value_gbp": round(p["target_value_gbp"], 4),
                "target_value_usd": round(p["target_value_usd"], 4),
                "actual_post_fill_weight": round(actual_w, 8) if actual_w is not None else None,
                "usd_market_value_at_fill": round(usd_mkt, 4),
                "gbp_market_value_at_fill": round(gbp_mkt, 4),
                "commission": rec.get("commission"),
                "commission_currency": rec.get("commission_currency"),
                "status": rec.get("status"),
                "weight_gap_vs_19pct": round(actual_w - WEIGHT, 8) if actual_w is not None else None,
            }
        )

    unfilled = any(f["status"] != "Filled" or f["fill_quantity"] + 1e-9 < f["quantity"] for f in fills)
    payload = {
        "generated": utc_now(),
        "status": "P5_PAPER_ORDERS_COMPLETE" if not unfilled else "P5_PAPER_ORDERS_PARTIAL",
        "execution_arm": "P5",
        "vt0_orders": False,
        "ibkr_quotes_used": False,
        "massive_sizing": True,
        "orders_submitted": True,
        "submitted_batch_ts": submitted_at,
        "account_id": acct,
        "account_kind": "PAPER",
        "nav_gbp": nav,
        "nav_source": nav_src,
        "exchange_rate_usd": usd_rate,
        "exchange_rate_key": usd_key,
        "exchange_rate_meaning": "IBKR account ExchangeRate for USD = GBP per 1 USD",
        "massive_session": planned[0]["massive_session"],
        "post_nav_gbp": post_nav,
        "post_positions": [p for p in app.positions if abs(p["position"]) > 1e-9],
        "fills": fills,
        "errors": app.errors,
        "fatal": [],
        "p5_parameters_changed": False,
    }
    try:
        app.cancelAccountSummary(9201)
        app.cancelAccountSummary(9202)
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
