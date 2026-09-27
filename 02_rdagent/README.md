# RD-Agent US market overlay (RD8)

External project: [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) at commit `4f9ecb00`, then apply `patches/rdagent_us_rd8.patch` (14 files under `rdagent/` only; `prompts.yaml` is excluded).

## What the patch changes

- **US market and calendar windows** in `rdagent/app/qlib_rd_loop/conf.py` and the factor/model YAML templates (`sp500`, train 2008–2017 / valid 2018–2019 / test 2020–2023, TopK=20 / `n_drop=2`, benchmark wiring in templates).
- **US data template** helpers (`factor_data_template/generate.py`, `generate_us_template.py`); set `QLIB_DATA_DIR` to your Qlib provider (the patch uses `${QLIB_DATA_DIR}` instead of a machine-local path).
  The Qlib data path in the patched templates is shown as `${QLIB_DATA_DIR}`; replace it with the local path to the Qlib data directory before running.
- **`factor_runner` / `model_runner`**: pass `market` / `topk` / `n_drop` into the qrun environment; macOS-safe thread caps (`OMP_NUM_THREADS=1` and related) around `qrun`.
- **`accumulate_all`** flag (default **False**). RD8 keeps the default: only loops with a positive feedback decision enter the SOTA library. Sector “accumulate-all” reruns are out of scope for RD8.
- **`mlflow==3.9.0`** install step in `rdagent/utils/env.py` for the qlib conda helper.
- **Runtime info parsing** in `get_runtime_info.py`: if stdout has no JSON object, return the raw text instead of crashing.

### `chat_temperature = 1.0`

The patch sets `chat_temperature` from `0.5` to `1.0` in `rdagent/oai/llm_conf.py`. The RD8 runs used **o4-mini**, which only accepts the API default temperature; non-default values are rejected. Matched-rerun session settings therefore use `1.0` (see `matched_rerun/CONFIG_AUDIT.md`).

## Prompts are not modified by the patch

`rdagent/scenarios/qlib/prompts.yaml` is **excluded** from `patches/rdagent_us_rd8.patch`. The July “Maintain Exploration Diversity Across Factor Families” guidance was a separate experiment and did not enter RD8.

RD8 uses the upstream prompt text fixed under `matched_rerun/golden_prompts/`, installed at run time by `matched_rerun/launch_factor_with_prompt_overlay.py` (and `install_golden_prompts.sh` when restoring files on disk).

## Directories in this folder

| Path | Role |
|---|---|
| `matched_rerun/` | Launch / resume scripts, golden prompts, and audits for the matched 20-loop RD8 search (Loops 0–19) under the restored configuration. |
| `loop2_injection/` | Regenerated trajectory: keep Loops 0–1, inject the scale-corrected Loop 2 factor, then continue autonomous search through Loop 19. See `loop2_injection/README.md`. |
