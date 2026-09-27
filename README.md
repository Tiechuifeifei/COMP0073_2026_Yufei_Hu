# Automated Factor Discovery and Tradability Evaluation in US Equities

Code, configurations and result tables for the UCL MSc Computer Science dissertation (COMP0073), *Automated Factor Discovery and Tradability Evaluation in US Equities: An Evaluation of RD-Agent(Q) with Fundamental and News Sentiment Signals*. The dissertation PDF is in `paper/`.

This repository allows the tables and figures in the report to be checked against the shipped result files (`results/manifest.csv`). The underlying data are licensed and not redistributed; `docs/DATA_SOURCES.md` explains how to obtain each source. RD-Factor search is not bit-reproducible, because the language model generates different hypotheses on each run; `docs/REPRODUCIBILITY.md` describes what can be rerun and what each step requires.

## Repository structure

| Path | Contents |
|---|---|
| `paper/` | Dissertation PDF |
| `docs/` | Data sources, reproducibility notes, lineage and experiment registries |
| `patches/` | Changes to RD-Agent for the US market (`rdagent_us_rd8.patch`) |
| `01_data/` | CRSP universe construction, point-in-time fundamentals, news alignment, Massive 2026 audit |
| `02_rdagent/` | RD8 trajectory launch configuration and the Loop 2 correction |
| `03_fundamentals_news/` | Price–volume, fundamental and news models; signal fusion |
| `04_portfolio/` | Portfolio construction, replacement rules and market-state experiments |
| `05_later_evaluation/` | 2024–2025 and 2026 evaluation |
| `06_execution/` | IBKR paper trading (market-on-close) |
| `results/` | Result tables and figures by chapter, indexed in `results/manifest.csv` |

## Names used in the report and in the code

| Report | Code identifier |
|---|---|
| PV-RD13 | D2 |
| PV-RD8 | B1_R50 / B1-R50 |
| FUND | F1C |
| Fused-RD13 | W1 |
| Fused-RD8 | F_BASE_W010 |
| RD13-Top20 | H0 / P20D2 |
| RD13-Top5 | P5 |
| RD8-Top20 | K20_D2 |
| RD8-Top5 | K5D1_WD |

The full mapping is in Appendix I of the report.

## Periods and evidence status

| Period | Role |
|---|---|
| 2008–2017 | Training |
| 2018–2019 | Validation |
| 2020–2023 | Development (search feedback); not an independent test |
| 2024–2025 | Independent test for specifications fixed before it was opened (RD13-Top20, Fused-RD13); first evaluation for the RD8 chain; post hoc for RD13-Top5 |
| 2026-01-05 to 2026-08-24 | Holdout for RD13-Top5, kept closed until the strategy was formed; approximate data from Massive |
| From 2026-08-31 | Prospective IBKR paper trading |

## Environment

```bash
conda env create -f environment_qlib.yml
conda env create -f environment_rdagent.yml
cp .env.example .env   # add your own credentials; never commit .env
source set_env.sh      # sets PROJECT_ROOT, RDAGENT_ROOT, QLIB_ROOT
```

RD-Agent is an external project: clone https://github.com/microsoft/RD-Agent at commit `4f9ecb00` and apply `patches/rdagent_us_rd8.patch`.

## Reproduction path

1. Obtain the data listed in `docs/DATA_SOURCES.md` and build the historical S&P 500 Qlib universe (`01_data/`).
2. Run the RD-Factor trajectory with the patched RD-Agent (`02_rdagent/`).
3. Build the downstream models and portfolios from the frozen factor sets (`03_fundamentals_news/`, `04_portfolio/`) and regenerate the evaluation tables (`05_later_evaluation/`).

Scripts that depend on files outside this repository are listed in `docs/REPRODUCIBILITY.md`.
