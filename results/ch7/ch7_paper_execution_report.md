# Chapter 7 · Live execution: IBKR Paper Trading read-only summary

**Read-only audit.** No orders were submitted and no deployment or freeze specifications were changed.

## Takeaways

In the formal evaluation window **2026-08-31 → 2026-09-10** (8 US equity sessions), the Paper account ran the frozen **P5 (W1×Top5/drop1) + MOC** rule. Execution broadly matched the design (close MOC, DUT PAPER, VT0 not ordered), but was **not frictionless relative to the theoretical target**: there were **3 partial fills** (residual MCHP sell, residual INTU sell, residual KLAC buy), **1 in-window fail-closed event** (09-07 contract-mapping failure deferred the 09-04 signal), and **persistently high cash** (about 40% of NAV at the 09-10 close, versus a design cash share of ~5%).

Over the same window the account return was **-6.47%**, SPY (GBP) **-0.91%**, excess **-5.55 pp**. Relative to the `live_target` theoretical replay (same window, compounded r_net **-7.64%**), the live account lost about **+1.17 pp less**, consistent with the mechanical effect of **lower equity exposure / higher cash** in a down market. **No statistical inference or annualisation is applied to this short window.**

The intraday **MKT** first deployment (2026-08-28) lies outside the formal window and is recorded only as an execution deviation.

## 0. Data sources and specification

- **Account**: `${EXPECTED_ACCOUNT}` PAPER, base currency **GBP**
- **Window**: 2026-08-31 → 2026-09-10 (8 days)
- **Strategy**: signal W1; portfolio Top5 / `n_drop=1` / `hold_thresh=1` / `risk_degree=0.95`; execution **MOC**; sizing = Massive adjusted close; FX = `ExchangeRate:USD`
- **Freeze note**: `${PROJECT_ROOT}/.cursor/rules/p5-paper-execution.mdc`; registry entry PAPER-P5-IBKR
- **Authoritative narrative audit pack**: `${PROJECT_ROOT}/reports/ibkr/dissertation_final_paper_audit_20260831_20260910/`
- **Source inventory JSON**: `${PROJECT_ROOT}/reports/ibkr/ch7_paper_execution_readonly_20260924/00_data_sources.json`

## Key numbers

| Item | Value | Source |
|------|------|------|
| Start / end NAV | £993,256.26 → £929,016.25 | `A_B_window_and_performance.json` / live pull |
| Account cumulative return | -6.47% | `A_B_window_and_performance.json` |
| SPY USD / GBP | -1.20% / -0.91% | Massive SPY + table C FX |
| FX contribution to SPY(GBP) | +0.29 pp | FX 0.738045→0.740200 |
| Theoretical replay cumulative (09-01…09-10) | -7.64% | `daily_full.csv` r_net |
| Live − theoretical | +1.17 pp | previous two rows |
| Order rows / full fills / partial fills | 18 / 15 / 3 | table D |
| Fill vs Massive close (broker fills) | mean -4.3 bp; median -5.2 bp | table G |
| Buy / sell mean deviation | -5.3 / -3.4 bp | table G |
| Commission total | $52.24 ≈ £38.56 (≈ 0.049 bp/day·start NAV) | summary JSON |
| Backtest cost assumption | 1 bp one-way (notional) | frozen Qlib; not the same as IBKR commissions |
| Mean daily name overlap (Jaccard) | 0.958 | intended Top5 vs EOD holdings |

## 1. Daily reconciliation (summary)

Full fields are in the CSV; the table below summarises rebalance days.

| Fill date | Decision date | Theoretical Top5 | Exception notes | NAV £ | Cash share |
|--------|--------|-----------|----------|------:|---------:|
| 2026-08-31 | 2026-08-28 | AAPL,AXON,CNC,INTU,MCHP | — | 993,256 | 6.0% |
| 2026-09-01 | 2026-08-31 | AAPL,AXON,CNC,INTU,IT | PARTIAL SELL MCHP: filled 2861.0 / submitted 3416.0 | RESIDUAL_LONG MCHP qty=555 | 974,890 | 3.0% |
| 2026-09-02 | 2026-09-01 | AAPL,AXON,CSGP,INTU,IT | — | 973,889 | 7.7% |
| 2026-09-03 | 2026-09-02 | AXON,CSGP,INTU,IT,PANW | — | 997,011 | 8.1% |
| 2026-09-04 | 2026-09-03 | AXON,CAH,CSGP,INTU,IT | CASH_SHARE=22.3% (design cash≈5%) | 971,304 | 22.3% |
| 2026-09-08 | 2026-09-04 | AXON,CAH,CIEN,CSGP,IT | PARTIAL SELL INTU: filled 460.0 / submitted 741.0 | RESIDUAL_LONG INTU qty=281.0 | 943,095 | 13.8% |
| 2026-09-09 | 2026-09-08 | AXON,CAH,CIEN,CSGP,TTWO | CASH_SHARE=19.3% (design cash≈5%) | 927,988 | 19.3% |
| 2026-09-10 | 2026-09-09 | AXON,CAH,CSGP,KLAC,TTWO | PARTIAL BUY KLAC: filled 36.0 / submitted 1307.0 | CASH_SHARE=39.7% (design cash | 929,016 | 39.7% |

Daily detail: `${PROJECT_ROOT}/reports/ibkr/ch7_paper_execution_readonly_20260924/ch7_paper_daily_detail.csv`  
Order/fill detail: `${PROJECT_ROOT}/reports/ibkr/ch7_paper_execution_readonly_20260924/ch7_paper_order_fill_detail.csv`

## 2. Partial fills and residual positions

- **2026-09-01 MCHP**: UNKNOWN; start→filled→end = 3416→2861→555; cleared on 2026-09-02. Evidence: Only IBKR position snapshots (08-31 EOD vs 09-01 status) document the qty change. fills_agg in p5_paper_20260901_status.json is empty; no orderStatus/submitted qty archived. Cannot distinguish undersized submit vs partial MOC fill.
- **2026-09-08 INTU**: correct quantity but partial/unfilled execution; start→filled→end = 741→460→281; cleared on 2026-09-09. Evidence: TWS executions orderId=13 sold 460; EOD residual 281 (=741-460). Prior 2026-09-07 stage planned SELL 741 but status NO_ORDERS_CONTRACT (not that day's fill). Residual fully sold next session 2026-09-09 (orderId=15, 281 @ 314.12).

- **KLAC 2026-09-10**: buy submitted 1307, filled 36 (partial entry, not a sell residual)
- **CAH**: filled quantity equals submitted quantity, but size is **systematically small** relative to the ~19% sleeve (implementation gap, not a reject)

## 3. Fail-closed / manual intervention

- **2026-08-28 (OUTSIDE formal window)**: trigger `Intraday MKT first deployment (methodology breach vs MOC rule)`; handling: Recorded as smoke-test; formal evaluation starts 2026-08-31 MOC; recovery: 2026-08-31 first compliant MOC; source `${PROJECT_ROOT}/reports/ibkr/p5_paper_first_orders.json`
- **2026-09-07 stage → intended 2026-09-08 MOC**: trigger `NO_ORDERS_CONTRACT fatal=['contract INTU', 'contract CIEN']`; handling: fail-closed: no orders submitted that evening; planned SELL INTU 741 / BUY CIEN 776 not sent; recovery: 2026-09-08 session later executed MOC CIEN buy + partial INTU sell (460/741); source `${PROJECT_ROOT}/reports/ibkr/p5_moc_staged_20260908.json`
- **2026-09-14 (OUTSIDE formal window)**: trigger `Audit pack recorded NO_ORDERS_CONTRACT (KLAC, EXPE) on contemporaneous rebalance JSON`; handling: fail-closed / no formal-window impact; recovery: Current p5_paper_moc_rebalance.json status=P5_MOC_QUEUED (file may have been overwritten after audit); source `${PROJECT_ROOT}/reports/ibkr/dissertation_final_paper_audit_20260831_20260910/F_execution_quality.json + current ${PROJECT_ROOT}/reports/ibkr/p5_paper_moc_rebalance.json`

No other written “manual reprice / forced flat” records were found; 09-01…09-03 lack archived TWS ticks, so quantities were rebuilt from the position path (prices may be inferred).

## 4. Return decomposition (descriptive, not inferential)

- Account path: £993,256.26 → £929,016.25 (−6.47%)
- SPY USD −1.20%; SPY GBP −0.91% (of which FX +0.29 pp)
- Theoretical P5 (`live_target` r_net, 09-01…09-10) −7.64%
- Live − theoretical = +1.17 pp
- Known execution-residual PnL scale: about £−373 (versus a same-day full-fill counterfactual)
- Commissions about £39 (negligible versus a £64k NAV move)
- **Main narrative**: losses are driven by mark-to-market on the live book; execution residuals and commissions do not explain most of the drawdown; high cash is an exposure-fidelity issue and, in this window, made the live path slightly better than a full-invested theoretical replay.

## 5. Data gaps / not verifiable

1. **2026-09-01…09-03**: no usable TWS `reqExecutions` ticks; fill quantities come from position-snapshot differences; some prices are marked inferred (table G `INFERRED_*`) and are **excluded** from slippage statistics.
2. **`p5_paper_moc_rebalance.json` was overwritten**: in-window 09-10 submitted quantities follow audit captures / fills CSV; the 09-14 fail-closed event follows `F_execution_quality.json`; the current JSON may already be a later state.
3. **Theoretical replay** uses approximate daily returns from `live_target/daily_full.csv` (APPROXIMATE Massive path), **not** a Qlib engine replayed on IBKR fills; the gap versus account NAV cannot be attributed bp-by-bp to slippage.
4. **Weight error** lacks a daily broker mark-to-market weight series; the report uses name Jaccard + a coarse MAE proxy.
5. **After the formal window** there is no reconcile-PASS full holdings chain, so the evaluation window is not extended.
6. The Activity Statement user figure £−66,268 versus the NAV difference £−64,240 differs by about £2k and was not force-aligned (noted in the summary).

Summary CSV: `${PROJECT_ROOT}/reports/ibkr/ch7_paper_execution_readonly_20260924/ch7_paper_summary.csv`  
This report: `${PROJECT_ROOT}/reports/ibkr/ch7_paper_execution_readonly_20260924/ch7_paper_execution_report.md`
