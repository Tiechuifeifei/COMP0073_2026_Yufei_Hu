# Data sources

Licensed / credentialed data are **not** redistributed with this repository. Obtain each source independently, then rebuild local panels with the scripts listed below. Table and field names are taken from SQL / API calls in `01_data/` (and related runners).

---

## 1. WRDS — CRSP daily prices and S&P 500 membership (Qlib universe)

| Item | Detail |
|---|---|
| Provider / access | Wharton Research Data Services (WRDS); institutional subscription (`WRDS_USERNAME` / `WRDS_PASSWORD`). |
| Dataset / table | Local **CRSP Daily Stock File in CIZ CSV format** (zip → CSV). Downloaded as CSV through the WRDS web query interface (CRSP Daily Stock File, CIZ format) with S&P 500 index membership fields; the ingest scripts read the local file and issue no SQL. S&P 500 membership spells are built from the same CSV using `MbrStartDt` / `MbrEndDt`. |
| Fields used | `PERMNO`, `MbrStartDt`, `MbrEndDt`, `DlyCalDt`, `DlyOpen`, `DlyHigh`, `DlyLow`, `DlyClose`, `DlyPrc`, `DlyPrcFlg`, `DlyVol`, `DlyCumFacPr` (`01_data/convert_wrds_stage1.py` `READ_COLUMNS`). Mapped outputs: `date`, `open`, `high`, `low`, `close`, `volume`, `factor`. |
| Sample period | Qlib / RD-Factor windows in configs: train 2008-01-01–2017-12-31, valid 2018–2019, test 2020–2023 (`experiments/conf_alpha20_sp500_transfer.yaml`). The CIZ extract covers 2008-01-02 to 2025-12-31. |
| Scripts | `01_data/convert_wrds_stage1.py`, `01_data/convert_wrds_stage2.py` (PIT `sp500.txt`), `01_data/convert_wrds_stage3.py` (dump_bin). |
| Special | Benchmark **SPY = CRSP PERMNO 84398**, Qlib instrument `P84398`. |
| Not redistributed | CRSP proprietary; WRDS / CRSP licence. |

---

## 2. WRDS — CRSP daily market data (fundamentals / market-cap path)

| Item | Detail |
|---|---|
| Provider / access | WRDS institutional subscription. |
| Dataset / table | `crsp.dsf` (`CRSP_SOURCE_TABLE` in `01_data/fundamental_pipeline/07_download_crsp_daily_market_data.py`). Name audit also queries `crsp.msenames` (`08_build_quarterly_signal_events.py`). |
| Fields used | `permno`, `date`, `prc`, `shrout`, `ret`, `retx`, `vol`, `bid`, `ask`, `openprc`, `numtrd`; derived `market_cap = abs(prc) * shrout * 1000`. |
| Sample period | `DATE_START = 2008-01-01`; download end through at least `QLIB_RESEARCH_END = 2025-12-31`. |
| Scripts | `01_data/fundamental_pipeline/07_download_crsp_daily_market_data.py` (+ consumers `08_*`, daily expanders). |
| Not redistributed | Same CRSP licence. |

---

## 3. WRDS — CRSP–Compustat link table

| Item | Detail |
|---|---|
| Provider / access | WRDS institutional subscription. |
| Dataset / table | `crsp_a_ccm.ccmxpf_linktable`. |
| Fields used | Production select (`04_build_sp500_pit_master.py` `CCM_COLS`): `gvkey`, `linkprim`, `linktype`, `lpermno`, `lpermco`, `usedflag`, `linkdt`, `linkenddt`. Filters: `linkprim = 'P'`, `linktype IN ('LU','LC')`, `usedflag = 1`. Mag5 pilot uses `SELECT *` (`02_download_mag5_pit_panel.py`). |
| Sample period | Link intervals covering the S&P 500 PERMNO universe (membership research start 2008-01-02). |
| Scripts | `02_download_mag5_pit_panel.py`, `04_build_sp500_pit_master.py`, `06_build_final_quarterly_master.py`. |
| Not redistributed | CCM / linked product; WRDS licence. |

---

## 4. WRDS — Compustat Point-in-Time quarterly fundamentals

| Item | Detail |
|---|---|
| Provider / access | WRDS institutional subscription (Compustat PIT). |
| Dataset / table | `comp_pit.pitqtrdataus`; `comp_pit.pit_hist_date_tableus` (point-date history; Mag5 / Phase-1.5 audits). |
| Fields used | Identifiers / timing (`04_build_sp500_pit_master.py` `ID_AND_PIT_COLS`): `gvkey`, `conm`, `tic`, `datadate`, `datacqtr`, `fqtr`, `fyrq`, `qtryr`, `qtrend`, `rdqe`, `prelimqprd`, `finalqprd`, `compstq`, `compstqr`. Accounting (`ACCOUNTING_COLS`): `atq`/`atqr`, `ceqq`/`ceqqr`, `seqq`/`seqqr`, `txditcq`/`txditcqr`, `pstkq`/`pstkqr`, `niq`/`niqr`, `ibq`/`ibqr`, `saleq`/`saleqr`, `cogsq`/`cogsqr`, `oibdpq`/`oibdpqr`, `dpq`/`dpqr`, `oancfq`/`oancfqr`, `dlcq`/`dlcqr`, `dlttq`/`dlttqr`, `ltq`/`ltqr`, `xintq`/`xintqr`. Supplement (`10_build_quarterly_fundamental_features.py`): `actq`, `lctq`. |
| Derived characteristics | Twelve quarterly features in `FEATURES` (`10_build_quarterly_fundamental_features.py`): `roe`, `roa`, `gross_profitability`, `operating_profitability`, `sales_growth_yoy`, `asset_growth_yoy`, `accruals`, `leverage`, `current_ratio`, `book_to_market`, `earnings_yield`, `sales_to_price`. The F1C model path uses the eleven z-scored names in `F1C_FEATURES` (`01_data/massive_audit/config.py`), excluding `operating_profitability`. |
| Sample period | PIT pull `datadate >= 2006-01-01` (`ACCOUNTING_START`); S&P research window from `2008-01-02` (`MEMBERSHIP_RESEARCH_START`). |
| Scripts | `02_download_mag5_pit_panel.py`, `04_build_sp500_pit_master.py`, `05_phase15_audit_resolution.py`, `06_build_final_quarterly_master.py`, `10_build_quarterly_fundamental_features.py`, `12_expand_quarterly_features_to_daily.py`, `13_preprocess_daily_fundamental_features.py`. |
| Special | **Accounting available date** = `max(rdqe, datadate)` (`06_build_final_quarterly_master.py`, `availability_rule = max_rdqe_datadate`). |
| Not redistributed | Compustat PIT licence via WRDS. |

---

## 5. Alpha Vantage — News & Sentiment

| Item | Detail |
|---|---|
| Provider / access | Alpha Vantage News & Sentiment API; env var in code: `ALPHAVANTAGE_API_KEY`. |
| Dataset / endpoint | HTTP function `NEWS_SENTIMENT` (`S1_alpha_vantage_pilot.py`, `S2_alpha_vantage_full_download.py`). |
| Fields used | Feed: `time_published`, `title`, `summary`, `source`, `source_domain`, `url`, `topics`, `overall_sentiment_score`, `ticker_sentiment`. Per-ticker: `ticker`, `relevance_score`, `ticker_sentiment_score`, `ticker_sentiment_label`. |
| Sample period | Full download / panel: **2010-01-01 – 2023-12-31** (`PERIOD_START` / `PERIOD_END` in S2/S3). |
| Scripts | `01_data/sentiment_experiments/S1_alpha_vantage_pilot.py`, `S2_alpha_vantage_full_download.py`, `S3_daily_sentiment_panel.py`; downstream under `03_fundamentals_news/`. |
| Special | Articles aligned by **publication timestamp**: parse `time_published` → UTC → US/Eastern; map to eligible trading / signal dates. |
| Not redistributed | Third-party news text and scores; API terms. |

---

## 6. Massive API — 2026 prices, corporate actions, financials

| Item | Detail |
|---|---|
| Provider / access | Massive (`https://api.massive.com`); `MASSIVE_API_KEY`. |
| Endpoints used (code) | Daily aggregates `/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}`; splits `/v3/reference/splits`; dividends `/v3/reference/dividends`; ticker reference; quarterly income / balance / cash-flow under `/stocks/financials/v1/...`; filings index; news probe `/v2/reference/news` (`01_data/massive_audit/phase_b.py`). Exact response field lists are plan-dependent. |
| Sample period | Overlap audit vs CRSP: `2023-01-01`–`2025-12-31`; 2026 evaluation window in runners: **2026-01-02 – 2026-08-18** (`phase_e.py` / `phase_h.py`); lookback from `2025-01-01`. |
| Scripts | `01_data/massive_audit/` (`config.py`, `phase_b.py` …); 2026 evaluation under `05_later_evaluation/`; paper sizing reference closes in `06_execution/`. |
| Special | Approximate 2026 extension and paper **reference** prices; not a substitute for WRDS CRSP in the main 2008–2025 research sample. |
| Not redistributed | Vendor market data; API key + licence. |

---

## 7. Interactive Brokers (IBKR) — Paper execution

| Item | Detail |
|---|---|
| Provider / access | IBKR **Paper** account via TWS/Gateway (`IBKR_HOST` / `IBKR_PORT` / `EXPECTED_ACCOUNT`). |
| Dataset / table | Not a research table; account summary / executions via `ibapi`. |
| Fields / tags used | Account kind (PAPER), `NetLiquidation`, `ExchangeRate` for USD (GBP per 1 USD), fill `Price` / `AvgPrice`, quantities. |
| Sample period | Prospective paper trading after strategy freeze (ledger timestamps). |
| Scripts | `06_execution/p5_paper_ledger_preflight.py`, `p5_paper_submit_orders.py`, related helpers. |
| Special | P5 arm only; MOC at US close; VT0 shadow-only. |
| Not redistributed | Broker account dumps and credentials. |

---

## 8. OpenAI — RD-Factor LLM backend

| Item | Detail |
|---|---|
| Provider / access | OpenAI API (`OPENAI_API_KEY`; optional `OPENAI_API_BASE`); LiteLLM in RD-Agent. |
| Dataset / table | N/A (hypothesis generation). |
| Model | Chat model `o4-mini` (`02_rdagent/matched_rerun/CONFIG_AUDIT.md`). |
| Sample period | RD8 search run window 2026-09-16–2026-09-18; RD13 earlier (Appendix B.1). |
| Scripts | External RD-Agent checkout + `02_rdagent/` matched-rerun overlay / Loop-2 injection. |
| Special | RD-Factor search is **not bit-reproducible** across runs; see `docs/REPRODUCIBILITY.md`. |
| Not redistributed | API keys; intermediate LLM transcripts are trimmed in this pack. |

---

## Cross-cutting conventions (from code)

| Convention | Where |
|---|---|
| SPY = PERMNO **84398** / Qlib `P84398` | `experiments/conf_alpha20_sp500_transfer.yaml`; portfolio / holdout runners |
| `accounting_available_date = max(rdqe, datadate)` | `06_build_final_quarterly_master.py` |
| News aligned on `time_published` (UTC → US/Eastern) | `S1_*`, `S2_*`, `S3_daily_sentiment_panel.py` |

---

## Why raw data are absent from the repo

WRDS/CRSP/Compustat, Alpha Vantage news text, Massive bars/statements, and IBKR account artefacts are licensed or account-bound. This pack ships **result tables** (`results/manifest.csv`) and rebuild scripts so reported figures can be checked (Tier A) or rebuilt after obtaining credentials (Tier B).
