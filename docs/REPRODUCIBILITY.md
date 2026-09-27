# Reproducibility

This repository ships experiment code, configurations and selected result files. Licensed datasets and large intermediate prediction panels are not redistributed.

## 1. Environment

```bash
cd /path/to/this/repo
conda env create -f environment_qlib.yml
conda env create -f environment_rdagent.yml
cp .env.example .env   # fill credentials locally; never commit .env
source ./set_env.sh
export RDAGENT_ROOT=/path/to/RD-Agent   # patched checkout
export QLIB_ROOT=/path/to/qlib
export QLIB_DATA_DIR=/path/to/qlib_data # S&P 500 Qlib provider used by configs
```

RD-Agent: clone https://github.com/microsoft/RD-Agent at commit `4f9ecb00`, apply `patches/rdagent_us_rd8.patch`, then use `02_rdagent/` (see `02_rdagent/README.md`).

Data sources and extraction scripts: `docs/DATA_SOURCES.md`.

## 2. Tier A — check reported numbers against shipped results

No WRDS / API keys required.

1. Open `results/manifest.csv` for the path and evidence status of each table/figure.
2. Compare the dissertation tables and figures to the corresponding files under `results/` (for example `results/ch4/`, `results/ch5/`, `results/ch6/`, `results/ch7/`, `results/appE/` …).
3. Treat rows marked static / report-only in the manifest as authoritative relative to the PDF; do not expect a local rebuild for those cells.

This is the intended examiner check path.

## 3. Tier B — rebuild (requires licensed data and external checkouts)

Full rebuild needs:

- WRDS access (CRSP daily, CCM link, Compustat PIT) and a rebuilt Qlib S&P 500 universe (`01_data/`, provider at `QLIB_DATA_DIR`).
- Alpha Vantage key for news/sentiment; Massive key for the 2026 approximate chain; OpenAI key for RD-Factor search; IBKR Paper for execution scripts.
- Patched RD-Agent (`RDAGENT_ROOT`) and a Qlib install (`QLIB_ROOT`).

### What can and cannot be bit-reproduced

- **RD-Factor search is not bit-reproducible.** The language model proposes different hypotheses on each run. The dissertation RD8 trajectory is the regenerated run with one Loop-2 correction injection (Appendix B.1); `02_rdagent/` documents how that trajectory was launched, not a guarantee of identical new loops.
- **Downstream models and portfolios** from frozen factor sets / prediction artefacts can be re-run when those inputs are present locally.
- **2026 approximate evaluation** depends on Massive snapshots and local archives that are not fully shipped. Prefer the tables under `results/` (and `results/manifest.csv`) as the report’s numerical source for 2026.

### High-level rebuild path

1. Obtain data in `docs/DATA_SOURCES.md` and build the historical S&P 500 Qlib universe (`01_data/`).
2. Run RD-Factor with the patched RD-Agent and `02_rdagent/` overlays as needed.
3. Rebuild downstream panels and portfolios (`03_fundamentals_news/`, `04_portfolio/`) and evaluation tables (`05_later_evaluation/`).

### Scripts that depend on paths outside this repository

Many runners still resolve artefacts under a private working tree (prediction panels, RD-Agent session dumps, IBKR ledgers, Massive archives). Typical patterns:

| Area | Examples | Outside dependency |
|---|---|---|
| RD8 launch / Loop-2 | `02_rdagent/matched_rerun/*.sh`, `02_rdagent/loop2_injection/run_corrected_20loop.sh` | `RDAGENT_ROOT`; session / pickle paths under a local reports tree |
| Fundamentals / news | `03_fundamentals_news/*.py` | Local `data/` prediction panels, `daily_pv.h5`, frozen experiment directories |
| Holdout / 2026 | `05_later_evaluation/**` | Extended CRSP / Massive archives; prior `reports/matched_2x2_2026_*` |
| Paper execution | `06_execution/*.py`, `03_fundamentals_news/ch7_paper_execution_readonly_summary.py` | Live IBKR Paper + local `reports/ibkr/` ledgers |
| Config | `experiments/conf_alpha20_sp500_transfer.yaml` | `QLIB_DATA_DIR` provider |

If a required input is missing, the script will fail at load time; that is expected in Tier A. For Tier B, rebuild or point environment variables at a local data tree first.
