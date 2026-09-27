#!/usr/bin/env python3
"""Assemble Chapter 7 paper-execution summary tables from local frozen evidence
(IBKR ledger / Massive sizing logs). Read-only; does not submit orders."""

from __future__ import annotations
import os

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(os.environ["PROJECT_ROOT"]) if os.environ.get("PROJECT_ROOT") else Path(__file__).resolve().parents[1]
OUT = PROJECT / "reports/ibkr/ch7_paper_execution_readonly_20260924"
OUT.mkdir(parents=True, exist_ok=True)

AUDIT = PROJECT / "reports/ibkr/dissertation_final_paper_audit_20260831_20260910"
IBKR_REP = PROJECT / "reports/ibkr"
IBKR_DIR = PROJECT / "ibkr"
LIVE = PROJECT / "data/portfolio_experiments/market_risk/live_target"
SPY_PATH = PROJECT / "data/massive_2026/raw/SPY_daily.parquet"

SESSIONS = [
    "2026-08-31",
    "2026-09-01",
    "2026-09-02",
    "2026-09-03",
    "2026-09-04",
    "2026-09-08",
    "2026-09-09",
    "2026-09-10",
]


def parse_pos(s: str) -> dict[str, float]:
    out = {}
    if not isinstance(s, str) or not s.strip():
        return out
    for part in s.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        t, q = part.split(":", 1)
        out[t.strip()] = float(q)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    # ---------- load sources ----------
    path_c = AUDIT / "C_daily_account_path.csv"
    path_d = AUDIT / "D_rebalance_reconciliation.csv"
    path_e = AUDIT / "E_residual_root_cause.csv"
    path_f = AUDIT / "F_execution_quality.json"
    path_g = AUDIT / "G_fill_vs_massive.csv"
    path_ab = AUDIT / "A_B_window_and_performance.json"
    path_h = AUDIT / "H_pnl_attribution.json"
    path_fills = IBKR_DIR / "audit_actual_fills_20260831_20260910.csv"
    path_pos = IBKR_DIR / "audit_actual_positions_20260831_20260910.csv"
    path_pnl = IBKR_DIR / "audit_actual_paper_pnl_20260831_20260910.csv"
    path_sum = IBKR_DIR / "audit_actual_paper_pnl_20260831_20260910_summary.json"
    path_live = IBKR_DIR / "audit_ibkr_live_pull_20260910.json"
    path_weights = LIVE / "weights_full.csv"
    path_daily_th = LIVE / "daily_full.csv"
    path_run = LIVE / "run_summary.json"
    path_0831 = IBKR_REP / "p5_paper_20260831_status.json"
    path_0901 = IBKR_REP / "p5_paper_20260901_status.json"
    path_moc_fills = IBKR_REP / "p5_paper_moc_fills.json"
    path_stage = IBKR_REP / "p5_moc_staged_20260908.json"
    path_first = IBKR_REP / "p5_paper_first_orders.json"
    path_rebal = IBKR_REP / "p5_paper_moc_rebalance.json"
    path_rule = PROJECT / ".cursor/rules/p5-paper-execution.mdc"

    C = pd.read_csv(path_c, parse_dates=["date"])
    D = pd.read_csv(path_d)
    E = pd.read_csv(path_e)
    G = pd.read_csv(path_g, parse_dates=["date"])
    F = json.loads(path_f.read_text())
    AB = json.loads(path_ab.read_text())
    H = json.loads(path_h.read_text())
    fills = pd.read_csv(path_fills, parse_dates=["date"])
    summary = json.loads(path_sum.read_text())
    st0831 = json.loads(path_0831.read_text())
    stage = json.loads(path_stage.read_text())
    first = json.loads(path_first.read_text())
    rebal_now = json.loads(path_rebal.read_text())
    run_sum = json.loads(path_run.read_text()) if path_run.exists() else {}

    weights = pd.read_csv(path_weights, parse_dates=["date"])
    theory = pd.read_csv(path_daily_th, parse_dates=["trade_date", "decision_date"])

    # ---------- 0 inventory ----------
    sources = {
        "formal_window": {
            "first_trading_day": SESSIONS[0],
            "last_trading_day": SESSIONS[-1],
            "n_trading_days": len(SESSIONS),
            "sessions": SESSIONS,
            "account_id": os.environ["EXPECTED_ACCOUNT"],
            "account_kind": "PAPER",
            "base_currency": "GBP",
            "note": "Formal MOC window; 2026-08-28 MKT smoke excluded",
        },
        "strategy_identity": {
            "signal": "W1 (RD13 frozen fusion)",
            "portfolio_rule": "Top5 / n_drop=1 / hold_thresh=1 / risk_degree=0.95 / equal-within-TopK (P5)",
            "execution": "IBKR MOC (Market-on-Close); Massive adj-close sizing; ExchangeRate:USD for GBP↔USD",
            "vt0": "shadow-only (never submitted)",
            "freeze_docs": [
                str(path_rule),
                "reports/final/canonical_experiment_registry.csv [PAPER-P5-IBKR]",
                "data/portfolio_experiments/market_risk/live_target/ (weights_full / daily_full / order_target)",
            ],
            "code_entrypoints": [
                "ibkr/p5_paper_moc_rebalance.py",
                "ibkr/p5_paper_execute_massive_sizing.py",
                "ibkr/p5_paper_ledger_preflight.py",
            ],
            "freeze_commit_note": "Operational freeze documented in .cursor/rules/p5-paper-execution.mdc and PAPER-P5-IBKR registry; script evolution via dated bak_* / p5_paper_moc_rebalance_YYYYMMDD.py snapshots (not a single dissertation git SHA for live orders).",
        },
        "source_files": {
            "daily_nav_path": str(path_c),
            "rebalance_recon": str(path_d),
            "residuals": str(path_e),
            "exec_quality": str(path_f),
            "fill_vs_massive": str(path_g),
            "window_perf": str(path_ab),
            "pnl_attribution": str(path_h),
            "fills_csv": str(path_fills),
            "positions_csv": str(path_pos),
            "pnl_csv": str(path_pnl),
            "pnl_summary": str(path_sum),
            "live_pull_0910": str(path_live),
            "status_0831": str(path_0831),
            "status_0901": str(path_0901),
            "moc_fills_0831": str(path_moc_fills),
            "stage_0908": str(path_stage),
            "first_orders_smoke": str(path_first),
            "rebalance_json_current": str(path_rebal),
            "intended_weights": str(path_weights),
            "theory_daily": str(path_daily_th),
            "prior_narrative_audit": str(AUDIT / "dissertation_paper_trading_final_audit.md"),
        },
    }
    (OUT / "00_data_sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")

    # ---------- 1 daily session rows ----------
    # Map order_session -> decision intended names from D
    sess_meta = []
    for osess, g in D.groupby("order_session", sort=True):
        row0 = g.iloc[0]
        intended = [x.strip() for x in str(row0["intended_P5_holdings"]).split(",") if x.strip()]
        hold = [x.strip() for x in str(row0["HOLD"]).split(",") if x.strip()]
        sell = [x.strip() for x in str(row0["SELL"]).replace('"', "").split(",") if x.strip()]
        buy = [x.strip() for x in str(row0["BUY"]).replace('"', "").split(",") if x.strip()]
        orders = []
        for _, r in g.iterrows():
            orders.append(
                {
                    "ticker": r["ticker"],
                    "side": r["side"],
                    "order_type": "MOC",
                    "submitted_qty": None if pd.isna(r["submitted_qty"]) else float(r["submitted_qty"]),
                    "filled_qty": None if pd.isna(r["filled_qty"]) else float(r["filled_qty"]),
                    "ending_qty": None if pd.isna(r["ending_broker_position"]) else float(r["ending_broker_position"]),
                    "residual_qty": None if pd.isna(r["residual_qty"]) else float(r["residual_qty"]),
                    "status": r["final_order_status"],
                    "evidence": r["evidence_note"],
                }
            )
        # enrich fills with price/time/commission where available
        fday = fills[fills["date"] == pd.Timestamp(osess)]
        enrich = {}
        for _, fr in fday.iterrows():
            enrich[(fr["ticker"], fr["side"])] = fr
        gday = G[G["date"] == pd.Timestamp(osess)]
        gmap = {(r.ticker, r.side): r for _, r in gday.iterrows()}
        for o in orders:
            key = (o["ticker"], o["side"])
            if key in enrich:
                fr = enrich[key]
                o["avg_fill_price"] = float(fr["avg_fill_price"]) if pd.notna(fr["avg_fill_price"]) else None
                o["commission"] = float(fr["commission"]) if pd.notna(fr["commission"]) else None
                o["commission_currency"] = fr.get("commission_currency", "USD")
                o["execution_timestamp"] = fr.get("execution_timestamp")
                o["fill_source"] = fr.get("source")
            if key in gmap:
                gr = gmap[key]
                o["massive_close"] = (
                    float(gr["massive_close_same_session"])
                    if pd.notna(gr["massive_close_same_session"])
                    else None
                )
                o["fill_minus_massive_bps"] = (
                    float(gr["fill_minus_massive_bps"]) if pd.notna(gr["fill_minus_massive_bps"]) else None
                )
                o["price_source_note"] = gr.get("price_source_note")

        # ending broker book from C
        c_end = C[C["date"] == pd.Timestamp(osess)]
        if len(c_end):
            ce = c_end.iloc[0]
            actual_pos = parse_pos(ce["actual_positions"])
            nav = float(ce["NAV_GBP"])
            cash = float(ce["cash_GBP"])
            fx = float(ce["fx_gbp_per_usd"])
            nav_anchor = bool(ce["nav_is_ibkr_anchor"])
        else:
            actual_pos, nav, cash, fx, nav_anchor = {}, np.nan, np.nan, np.nan, False

        # decision-date target weights (equal 0.19)
        dsess = str(row0["decision_session"])
        wsub = weights[weights["date"] == pd.Timestamp(dsess)]
        target_w = {str(t): 0.19 for t in intended}
        if len(wsub):
            target_w = {str(r.ticker): float(r.weight) for _, r in wsub.iterrows()}

        # actual weights approx from qty * (no price in C) — use equity share of nav when possible
        # Use membership overlap + cash share as primary fidelity metrics
        actual_names = set(actual_pos)
        intended_set = set(intended)
        overlap = jaccard(intended_set, actual_names)
        only_actual = sorted(actual_names - intended_set)
        missing = sorted(intended_set - actual_names)

        # anomalies
        anomalies = []
        for o in orders:
            if o["status"] == "PARTIAL_FILL":
                anomalies.append(
                    f"PARTIAL {o['side']} {o['ticker']}: filled {o['filled_qty']} / submitted {o['submitted_qty']}"
                )
            if o.get("residual_qty") and o["residual_qty"] and o["side"] == "SELL" and o["residual_qty"] > 0:
                anomalies.append(f"RESIDUAL_LONG {o['ticker']} qty={o['residual_qty']}")
        if only_actual:
            anomalies.append(f"EXTRA_HOLDINGS {only_actual}")
        if missing:
            anomalies.append(f"MISSING_VS_TARGET {missing}")
        if cash == cash and nav == nav and nav > 0:
            cash_share = cash / nav
            if cash_share > 0.10:
                anomalies.append(f"CASH_SHARE={cash_share:.1%} (design cash≈5%)")

        # theory r_net that day
        th = theory[theory["trade_date"] == pd.Timestamp(osess)]
        theory_r = float(th["r_net"].iloc[0]) if len(th) else np.nan

        sess_meta.append(
            {
                "order_session": osess,
                "decision_session": dsess,
                "theory_target_tickers": ",".join(intended),
                "theory_target_weights": ";".join(f"{k}:{v:.2f}" for k, v in sorted(target_w.items())),
                "planned_orders_json": json.dumps(orders, ensure_ascii=False),
                "n_planned_orders": len(orders),
                "n_full_fills": sum(1 for o in orders if o["status"] == "Filled"),
                "n_partial_fills": sum(1 for o in orders if o["status"] == "PARTIAL_FILL"),
                "HOLD": ",".join(hold),
                "SELL": ",".join(sell),
                "BUY": ",".join(buy),
                "broker_positions_eod": "; ".join(f"{k}:{int(v)}" for k, v in sorted(actual_pos.items())),
                "broker_cash_GBP": cash,
                "broker_NAV_GBP": nav,
                "nav_is_ibkr_anchor": nav_anchor,
                "fx_GBP_per_USD": fx,
                "name_overlap_jaccard": overlap,
                "extra_holdings": ",".join(only_actual),
                "missing_vs_target": ",".join(missing),
                "anomalies": " | ".join(anomalies) if anomalies else "",
                "theory_r_net_live_target": theory_r,
                "account_daily_return": float(c_end["daily_return"].iloc[0])
                if len(c_end) and pd.notna(c_end["daily_return"].iloc[0])
                else np.nan,
            }
        )

    daily = pd.DataFrame(sess_meta)
    daily["order_session"] = pd.to_datetime(daily["order_session"])
    # attach C path fields
    cleft = C.rename(columns={"date": "order_session"})[
        [
            "order_session",
            "daily_PnL_GBP",
            "cumulative_return",
            "gross_equity_GBP",
            "n_positions",
        ]
    ].copy()
    cleft["order_session"] = pd.to_datetime(cleft["order_session"])
    daily = daily.merge(cleft, on="order_session", how="left")
    daily_path = OUT / "ch7_paper_daily_detail.csv"
    daily.to_csv(daily_path, index=False)

    # also explode order-level detail
    order_rows = []
    for s in sess_meta:
        for o in json.loads(s["planned_orders_json"]):
            order_rows.append(
                {
                    "order_session": s["order_session"],
                    "decision_session": s["decision_session"],
                    **o,
                }
            )
    orders_df = pd.DataFrame(order_rows)
    orders_df.to_csv(OUT / "ch7_paper_order_fill_detail.csv", index=False)

    # ---------- 2 aggregate metrics ----------
    broker_g = G[G["price_source_note"].fillna("").str.contains("INFERRED") == False].copy()
    # also exclude rows whose note says inferred
    if "price_source_note" in G.columns:
        broker_g = G[~G["price_source_note"].astype(str).str.contains("INFERRED", case=False, na=False)].copy()
    buy_bps = broker_g.loc[broker_g["side"] == "BUY", "fill_minus_massive_bps"].dropna()
    sell_bps = broker_g.loc[broker_g["side"] == "SELL", "fill_minus_massive_bps"].dropna()

    comm_usd = float(fills["commission"].fillna(0).sum())
    # Prefer summary commission which matched 0831 archive sum
    comm_usd = float(summary.get("commission_usd", comm_usd))
    comm_gbp = float(summary.get("commission_gbp_approx", comm_usd * 0.74))
    # day-average cost in bp of NAV: total commission / start NAV / n days * 1e4
    start_nav = float(AB["performance"]["starting_NAV_GBP"])
    end_nav = float(AB["performance"]["ending_NAV_GBP"])
    daily_cost_bp = (comm_gbp / start_nav / len(SESSIONS)) * 1e4
    # one-way 1bp backtest assumption on equity notional ~0.95*NAV: compare
    backtest_1bp_day_bp = 1.0  # convention: each trade side charged 1bp of traded notional in Qlib; not directly comparable

    mean_overlap = float(daily["name_overlap_jaccard"].mean())
    # weight abs deviation: when we have missing/extra, count as 0.19 each missing + extras; rough
    weight_mae = []
    for _, r in daily.iterrows():
        intended = set(str(r["theory_target_tickers"]).split(","))
        actual = set()
        if isinstance(r["broker_positions_eod"], str) and r["broker_positions_eod"]:
            actual = set(parse_pos(r["broker_positions_eod"].replace("; ", ";")).keys())
            # parse_pos expects "A:1; B:2"
            actual = set(parse_pos(r["broker_positions_eod"].replace("; ", ";")).keys())
        # rebuild parse
        actual = set(parse_pos(str(r["broker_positions_eod"]).replace("; ", ";")).keys())
        # weight vector on union; target 0.19 or 0; actual equal among holdings * (1-cash_share)*...
        # simpler: membership MAE = (missing+extra)*0.19 / 5
        miss = len(intended - actual)
        extra = len(actual - intended)
        weight_mae.append((miss + extra) * 0.19 / 5.0)
    mean_w_mae = float(np.mean(weight_mae)) if weight_mae else np.nan

    # ---------- 3 fail-closed ----------
    fail_closed = [
        {
            "when": "2026-08-28 (OUTSIDE formal window)",
            "trigger": "Intraday MKT first deployment (methodology breach vs MOC rule)",
            "handling": "Recorded as smoke-test; formal evaluation starts 2026-08-31 MOC",
            "recovery": "2026-08-31 first compliant MOC",
            "source": str(path_first),
        },
        {
            "when": "2026-09-07 stage → intended 2026-09-08 MOC",
            "trigger": f"NO_ORDERS_CONTRACT fatal={stage.get('fatal')}",
            "handling": "fail-closed: no orders submitted that evening; planned SELL INTU 741 / BUY CIEN 776 not sent",
            "recovery": "2026-09-08 session later executed MOC CIEN buy + partial INTU sell (460/741)",
            "source": str(path_stage),
        },
        {
            "when": "2026-09-14 (OUTSIDE formal window)",
            "trigger": "Audit pack recorded NO_ORDERS_CONTRACT (KLAC, EXPE) on contemporaneous rebalance JSON",
            "handling": "fail-closed / no formal-window impact",
            "recovery": f"Current {path_rebal.name} status={rebal_now.get('status')} (file may have been overwritten after audit)",
            "source": f"{path_f} + current {path_rebal}",
        },
    ]

    # ---------- 4 returns ----------
    spy = pd.read_parquet(SPY_PATH)
    spy["date"] = pd.to_datetime(spy["date"])
    sclose = spy.set_index("date")["close"]
    fx0 = float(C.loc[C["date"] == "2026-08-31", "fx_gbp_per_usd"].iloc[0])
    fx1 = float(C.loc[C["date"] == "2026-09-10", "fx_gbp_per_usd"].iloc[0])
    spy_usd = float(sclose.loc["2026-09-10"] / sclose.loc["2026-08-31"] - 1)
    spy_gbp = float((sclose.loc["2026-09-10"] * fx1) / (sclose.loc["2026-08-31"] * fx0) - 1)
    fx_contrib = spy_gbp - spy_usd

    # theory: compound live_target r_net from 09-01..09-10 (post 08-31 EOD NAV anchor)
    th_w = theory[(theory["trade_date"] > "2026-08-31") & (theory["trade_date"] <= "2026-09-10")]
    theory_cum = float((1 + th_w["r_net"].astype(float)).prod() - 1)
    actual_cum = float(AB["performance"]["cumulative_paper_return"])
    actual_minus_theory = actual_cum - theory_cum

    perf = {
        "start_NAV_GBP": start_nav,
        "end_NAV_GBP": end_nav,
        "actual_cum_return": actual_cum,
        "spy_usd_return": spy_usd,
        "spy_gbp_return": spy_gbp,
        "fx_contribution_to_spy_gbp": fx_contrib,
        "excess_vs_spy_gbp": actual_cum - spy_gbp,
        "theory_cum_return_live_target_r_net_0901_0910": theory_cum,
        "actual_minus_theory": actual_minus_theory,
        "attribution_notes": {
            "mtm_primary": H.get("conclusion") if isinstance(H, dict) else None,
            "partial_timing_pnl_gbp": summary.get("execution_vs_full_fill", {}).get(
                "partial_sell_timing_pnl_vs_full_fill_gbp"
            ),
            "commission_gbp": comm_gbp,
            "end_cash_GBP": float(C.loc[C["date"] == "2026-09-10", "cash_GBP"].iloc[0]),
            "end_cash_share": float(C.loc[C["date"] == "2026-09-10", "cash_GBP"].iloc[0] / end_nav),
            "interpretation": (
                "Actual (−6.47%) less negative than theory (−7.64%): consistent with cash under-deployment "
                "reducing equity exposure in a falling window; not evidence of better stock picking. "
                "No statistical inference — window has 8 sessions."
            ),
        },
    }

    summary_rows = [
        {"section": "window", "metric": "first_day", "value": SESSIONS[0], "source": str(path_ab)},
        {"section": "window", "metric": "last_day", "value": SESSIONS[-1], "source": str(path_ab)},
        {"section": "window", "metric": "n_trading_days", "value": len(SESSIONS), "source": str(path_ab)},
        {"section": "window", "metric": "account", "value": "${EXPECTED_ACCOUNT} PAPER GBP", "source": str(path_0831)},
        {"section": "orders", "metric": "n_order_rows", "value": int(len(orders_df)), "source": str(path_d)},
        {
            "section": "orders",
            "metric": "n_full_fills",
            "value": int((orders_df["status"] == "Filled").sum()),
            "source": str(path_d),
        },
        {
            "section": "orders",
            "metric": "n_partial_fills",
            "value": int((orders_df["status"] == "PARTIAL_FILL").sum()),
            "source": str(path_d),
        },
        {
            "section": "orders",
            "metric": "fill_rate_full_over_rows",
            "value": float((orders_df["status"] == "Filled").mean()),
            "source": str(path_d),
        },
        {
            "section": "slippage",
            "metric": "broker_fill_vs_massive_mean_bps",
            "value": float(broker_g["fill_minus_massive_bps"].mean()),
            "source": str(path_g),
        },
        {
            "section": "slippage",
            "metric": "broker_fill_vs_massive_median_bps",
            "value": float(broker_g["fill_minus_massive_bps"].median()),
            "source": str(path_g),
        },
        {
            "section": "slippage",
            "metric": "BUY_mean_bps",
            "value": float(buy_bps.mean()) if len(buy_bps) else np.nan,
            "source": str(path_g),
        },
        {
            "section": "slippage",
            "metric": "SELL_mean_bps",
            "value": float(sell_bps.mean()) if len(sell_bps) else np.nan,
            "source": str(path_g),
        },
        {"section": "cost", "metric": "commission_USD", "value": comm_usd, "source": str(path_sum)},
        {"section": "cost", "metric": "commission_GBP_approx", "value": comm_gbp, "source": str(path_sum)},
        {
            "section": "cost",
            "metric": "approx_daily_commission_bp_of_start_NAV",
            "value": daily_cost_bp,
            "source": f"{path_sum}; vs backtest one-way 1bp on traded notional (not identical basis)",
        },
        {
            "section": "fidelity",
            "metric": "mean_name_jaccard_intended_vs_actual",
            "value": mean_overlap,
            "source": f"{path_d}+{path_c}",
        },
        {
            "section": "fidelity",
            "metric": "mean_rough_weight_MAE",
            "value": mean_w_mae,
            "source": "membership diff ×0.19 / 5 (proxy; not mark-to-market weights)",
        },
        {"section": "returns", "metric": "actual_cum_return", "value": actual_cum, "source": str(path_ab)},
        {"section": "returns", "metric": "spy_usd_return", "value": spy_usd, "source": str(SPY_PATH)},
        {"section": "returns", "metric": "spy_gbp_return", "value": spy_gbp, "source": f"{SPY_PATH}+{path_c}"},
        {
            "section": "returns",
            "metric": "fx_contribution_spy_gbp_minus_usd",
            "value": fx_contrib,
            "source": path_c.name,
        },
        {
            "section": "returns",
            "metric": "theory_cum_r_net_0901_0910",
            "value": theory_cum,
            "source": str(path_daily_th),
        },
        {
            "section": "returns",
            "metric": "actual_minus_theory",
            "value": actual_minus_theory,
            "source": "account NAV path − live_target r_net compound",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT / "ch7_paper_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    # shortcuts
    daily.to_csv(PROJECT / "reports/ibkr/ch7_paper_daily_detail.csv", index=False)
    summary_df.to_csv(PROJECT / "reports/ibkr/ch7_paper_summary.csv", index=False)

    (OUT / "fail_closed_events.json").write_text(json.dumps(fail_closed, indent=2), encoding="utf-8")
    (OUT / "performance_block.json").write_text(json.dumps(perf, indent=2), encoding="utf-8")

    # ---------- 5 markdown ----------
    def pct(x, d=2):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{100 * float(x):.{d}f}%"

    def pp(x, d=2):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{100 * float(x):+.{d}f} pp"

    lines = [
        "# 第7章 · 实际执行：IBKR Paper Trading 只读汇总",
        "",
        "**只读审计。** 未下单、未改部署配置或冻结规格。",
        "",
        "## 结论（先读）",
        "",
        "正式评价窗 **2026-08-31 → 2026-09-10**（8 个美股交易日）内，Paper 账户按冻结 **P5（W1×Top5/drop1）+ MOC** 规则运行；"
        "执行大体符合设计（收盘 MOC、DUT PAPER、VT0 不下单），但 **并非无摩擦对齐理论目标**："
        "存在 **3 笔部分成交**（MCHP 卖残、INTU 卖残、KLAC 买残）、**1 次窗内 fail-closed**（09-07 合约映射失败推迟 09-04 信号）、"
        "以及 **现金长期偏高**（09-10 收盘现金约 40% NAV，远高于设计 ~5%）。",
        "",
        "同期账户收益 **"
        + pct(actual_cum)
        + "**，SPY（GBP）**"
        + pct(spy_gbp)
        + "**，超额 **"
        + pp(actual_cum - spy_gbp)
        + "**。"
        "相对 `live_target` 理论回放（同窗 r_net 复利 **"
        + pct(theory_cum)
        + "**），实际账户少亏约 **"
        + pp(actual_minus_theory)
        + "**，与 **低权益暴露/高现金** 在下跌市中的机械效应一致；"
        "**不对短窗收益做任何统计推断或年化**。",
        "",
        "盘中 **MKT** 首次部署（2026-08-28）在正式窗外，仅记执行偏差。",
        "",
        "## 0. 数据来源与规格",
        "",
        f"- **账户**：`${EXPECTED_ACCOUNT}` PAPER，基础货币 **GBP**",
        f"- **窗口**：{SESSIONS[0]} → {SESSIONS[-1]}（{len(SESSIONS)} 日）",
        "- **策略**：信号 W1；组合 Top5/`n_drop=1`/`hold_thresh=1`/`risk_degree=0.95`；执行 **MOC**；sizing=Massive 复权收盘；汇率=`ExchangeRate:USD`",
        f"- **冻结说明**：`{path_rule}`；登记项 PAPER-P5-IBKR",
        f"- **权威叙事审计包**：`{AUDIT}/`",
        f"- **来源清单 JSON**：`{OUT / '00_data_sources.json'}`",
        "",
        "## 关键数字",
        "",
        "| 项目 | 数值 | 来源 |",
        "|------|------|------|",
        f"| 期初 / 期末 NAV | £{start_nav:,.2f} → £{end_nav:,.2f} | `{path_ab.name}` / live pull |",
        f"| 账户累计收益 | {pct(actual_cum)} | `{path_ab.name}` |",
        f"| SPY USD / GBP | {pct(spy_usd)} / {pct(spy_gbp)} | Massive SPY + C 表汇率 |",
        f"| 汇率对 SPY(GBP) 贡献 | {pp(fx_contrib)} | FX {fx0:.6f}→{fx1:.6f} |",
        f"| 理论回放累计（09-01…09-10） | {pct(theory_cum)} | `{path_daily_th.name}` r_net |",
        f"| 实际 − 理论 | {pp(actual_minus_theory)} | 上两行 |",
        f"| 订单行 / 全成 / 部分成 | {len(orders_df)} / {(orders_df.status=='Filled').sum()} / {(orders_df.status=='PARTIAL_FILL').sum()} | D 表 |",
        f"| 成交价相对 Massive 收盘（经纪商成交） | 均值 {broker_g['fill_minus_massive_bps'].mean():+.1f} bp；中位 {broker_g['fill_minus_massive_bps'].median():+.1f} bp | G 表 |",
        f"| 买 / 卖侧均值偏离 | {buy_bps.mean():+.1f} / {sell_bps.mean():+.1f} bp | G 表 |",
        f"| 佣金合计 | ${comm_usd:.2f} ≈ £{comm_gbp:.2f}（约 {daily_cost_bp:.3f} bp/日·期初NAV） | summary JSON |",
        f"| 回测假设 | 单边 1 bp（成交名义） | Qlib 冻结；与 IBKR 佣金口径不同 |",
        f"| 日均名称重合（Jaccard） | {mean_overlap:.3f} | 意图 Top5 vs 收盘持仓 |",
        "",
        "## 1. 逐日对账（摘要）",
        "",
        "完整字段见 CSV；下表为再平衡日撮要。",
        "",
        "| 成交日 | 决策日 | 理论 Top5 | 异常要点 | NAV £ | 现金占比 |",
        "|--------|--------|-----------|----------|------:|---------:|",
    ]
    for _, r in daily.iterrows():
        cash_share = r["broker_cash_GBP"] / r["broker_NAV_GBP"] if r["broker_NAV_GBP"] else np.nan
        anom = (r["anomalies"] or "—")[:80]
        lines.append(
            f"| {r.order_session.date() if hasattr(r.order_session,'date') else r.order_session} | {r.decision_session} | "
            f"{r.theory_target_tickers} | {anom} | {r.broker_NAV_GBP:,.0f} | {100*cash_share:.1f}% |"
        )

    lines += [
        "",
        f"逐日明细：`{daily_path}`",
        f"订单/成交明细：`{OUT / 'ch7_paper_order_fill_detail.csv'}`",
        "",
        "## 2. 部分成交与残仓",
        "",
    ]
    for _, r in E.iterrows():
        lines.append(
            f"- **{r['order_session']} {r['ticker']}**：{r['cause']}；"
            f"start→filled→end = {r['start_qty']}→{r['filled_qty']}→{r['end_qty']}；"
            f"清除于 {r['cleared_on']}。证据：{r['evidence']}"
        )
    lines += [
        "",
        "- **KLAC 2026-09-10**：买入提交 1307、成交 36（部分建仓，非卖出残仓）",
        "- **CAH**：成交量=提交量，但相对 ~19% 袖口 **系统性偏小**（实现缺口，非拒单）",
        "",
        "## 3. fail-closed / 人工干预",
        "",
    ]
    for ev in fail_closed:
        lines.append(
            f"- **{ev['when']}**：触发 `{ev['trigger']}`；处理：{ev['handling']}；恢复：{ev['recovery']}；来源 `{ev['source']}`"
        )
    lines += [
        "",
        "未发现其他书面「人工改单/强制平仓」记录；09-01…09-03 缺少 TWS tick 归档，数量由持仓路径重建（价格可能为推断）。",
        "",
        "## 4. 收益分解（描述性，非推断）",
        "",
        f"- 账户路径：£{start_nav:,.2f} → £{end_nav:,.2f}（{pct(actual_cum)}）",
        f"- SPY USD {pct(spy_usd)}；SPY GBP {pct(spy_gbp)}（其中汇率贡献 {pp(fx_contrib)}）",
        f"- 理论 P5（`live_target` r_net，09-01…09-10）{pct(theory_cum)}",
        f"- 实际 − 理论 = {pp(actual_minus_theory)}",
        f"- 已知执行残差时点 PnL 量级：约 £{summary.get('execution_vs_full_fill',{}).get('partial_sell_timing_pnl_vs_full_fill_gbp', float('nan')):,.0f}（相对「当日全成」反事实）",
        f"- 佣金约 £{comm_gbp:.0f}（可忽略相对 £64k NAV 变动）",
        "- **主要叙事**：亏损主因是实际持仓的市值波动；执行残差与佣金解释不了大部分回撤；高现金是暴露保真问题，并在本窗使实际路径略好于满仓理论回放。",
        "",
        "## 5. 数据缺口 / 无法核实",
        "",
        "1. **2026-09-01…09-03**：无可用 TWS `reqExecutions` tick；成交量来自持仓快照差分，部分价格标记为推断（G 表 `INFERRED_*`），**不计入**滑点统计。",
        "2. **`p5_paper_moc_rebalance.json` 曾被覆盖**：正式窗内 09-10 提交量以审计 captures / fills CSV 为准；09-14 fail-closed 以 `F_execution_quality.json` 为准，当前 JSON 可能已是后续状态。",
        "3. **理论回放**用 `live_target/daily_full.csv` 的近似日收益（APPROXIMATE Massive 路径），**不是**用 IBKR 成交价重放的 Qlib 引擎；与账户 NAV 的差不能精细拆成「每一 bp 滑点」。",
        "4. **权重偏差**缺逐日券商 mark-to-market 权重序列；报告用名称 Jaccard + 粗 MAE 代理。",
        "5. **正式窗之后**无 reconcile-PASS 的完整持仓链路，故不延长评价窗。",
        "6. Activity Statement 用户报 £−66,268 与 NAV 差分 £−64,240 差约 £2k，未强制对齐（summary 已注）。",
        "",
        f"汇总 CSV：`{summary_path}`",
        f"本报告：`{OUT / 'ch7_paper_execution_report.md'}`",
    ]
    md_path = OUT / "ch7_paper_execution_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    (PROJECT / "reports/ibkr/ch7_paper_execution_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"out": str(OUT), "actual": actual_cum, "theory": theory_cum, "spy_gbp": spy_gbp}, indent=2))


if __name__ == "__main__":
    main()
