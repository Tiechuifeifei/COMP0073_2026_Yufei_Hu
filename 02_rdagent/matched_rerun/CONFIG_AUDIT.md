# Restored 07-07 configuration + matched-rerun differences

Evidence bases (not HEAD guesses):
- tee `${RDAGENT_ROOT}/tee_rdfactor_run_40loop_20260707_200059.log`
- session `${RDAGENT_ROOT}/log/2026-07-07_12-00-59-480728`
- Loop_0 workspace YAML `.../49ff8ce4.../conf_*.yaml` (birth 2026-07-07)
- git `2f02043b` (2026-06-24; last RD-Agent commit ≤ 07-07 for prompts)
- zsh history launch block (`python -m rdagent.app.qlib_rd_loop.factor --loop_n 40`)

## A. Restored 07-07 configuration summary

| Item | 07-07 value | Evidence |
|---|---|---|
| Launch | `cd RD-Agent && source .env && python -m rdagent.app.qlib_rd_loop.factor --loop_n 40 \| tee ...` | zsh history + tee name |
| `loop_n` | 40 (this matched run uses **20**) | CLI + Loop_0…39 dirs |
| `evolving_n` | 10 | session `RDLOOP_SETTINGS` pickle |
| Chat model | `o4-mini` | tee LiteLLM |
| Embedding | `text-embedding-3-small` | tee / aborted prior run |
| `chat_temperature` | 1.0 | session LITELLM settings |
| Hypothesis / factor / coder prompts | `rdagent/scenarios/qlib/prompts.yaml` (+ proposal/coder/experiment prompts) **without §6 diversity** | session debug_tpl; git `2f02043b`; `bak_before_regime_hint` |
| Execution timeout (configured) | 3600s (`qrun` / file-based) | tee; note: macOS often lacked `timeout` binary |
| Accumulation **runtime** | Only loops with `feedback.decision=True` enter `based_experiments` (`Feedback.__bool__` → decision). With 40/40 Replace=False → **no cross-loop SOTA merge** (`SOTA factor processing` count = 0 for loops 0–19) | tee + `2f02043b` code |
| Accumulation **flag** | **Did not exist** (`QLIB_FACTOR_ACCUMULATE_ALL` added 2026-07-17 `3dcde3a9`) | git log |
| Post-hoc “113 library” | Catalog of all generated factors across loops — **not** each loop’s live Qlib feature set | `SOTA_FACTOR_INVENTORY.md` vs tee |
| Universe | `sp500` | workspace YAML |
| Split | train 2008–2017 / valid 2018–2019 / test 2020–2023 | RDLOOP_SETTINGS + qrun env |
| Portfolio | TopK=20, n_drop=2, deal_price=close, open/close cost=1bp | workspace YAML |
| Benchmark (07-07) | **P10104** | workspace YAML |
| Base features | Alpha20 (20 names) | tee / workspace |
| Discovery data | **only** `daily_pv.h5` cols `$open/$close/$high/$low/$volume/$factor` | tee source_data block |
| Pickle cache | `${RDAGENT_ROOT}/pickle_cache` (contaminated empty-baseline key `d41d8cd…` → CSI300 leak) | session RD_AGENT_SETTINGS + provenance |

## B. Every difference: 07-07 vs this matched run

| Dimension | 07-07 | Matched 2026-09-16 | Why |
|---|---|---|---|
| `loop_n` | 40 | **20** | Explicit matched first-20 design |
| Pickle cache | shared default (leaked) | `${ROOT}/pickle_cache` empty-isolated | Fix incorrect reference/baseline |
| Empty baseline | CSI300-cached `0.031068/0.041045/−0.142199` | Fresh SP500 Alpha20 (anchor ≈ `0.004579/−0.014377/−0.268305`) | Correct reference |
| Benchmark instrument | P10104 | **P84398** (current factor_template) | Intentional: P84398 is dissertation/SPY excess anchor; P10104 not used for preflight anchors |
| Prompt §6 diversity | Absent | Absent after `install_golden_prompts.sh` | Restore 07-07; current worktree still has §6 until install |
| Source data dir | only daily_pv (then) | clean copy under ROOT (daily_pv only) | Block later rd13/fundamental pollution |
| `QLIB_FACTOR_ACCUMULATE_ALL` | n/a | **false** | Match 07-07 runtime decision gate (do **not** copy 09-15 narrative mistakes about “full hist”) |
| Market/topk env params | hard-coded yaml (local edits) | `QLIB_FACTOR_MARKET/TOPK/N_DROP` | Equivalent SP500/20/2 |
| Session / tee | `log/2026-07-07_…` + RD-Agent tee | `${ROOT}/sessions` + `${ROOT}/replication_20loop` | Isolation |
| 09-15 extras NOT copied | — | no diversity prompt; no rd13/fund h5; not using 09-15 workspaces as library seed | Contamination control |
| RD-Agent workspace folder | shared `git_ignore_folder/RD-Agent_workspace` | same shared folder (new UUID dirs only) | Residual: no clean env redirect; mitigated by empty library + no path wiring to old UUIDs |

## C. Accumulation audit (authoritative)

1. **Code at 07-07 (`2f02043b`)**: `based_experiments = [empty] + [t[0] for t in hist if t[1] …]` and `ExperimentFeedback.__bool__` returns `decision`.
2. **Tee fact**: loops 0–19 each have `New factor processing` = 1 and `SOTA factor processing` = 0 ⇒ no prior-loop factor merge while Replace=False.
3. **Therefore intended+actual mechanism for matched run**: `QLIB_FACTOR_ACCUMULATE_ALL=false` (current code: merge `based_experiments` only; when True, merges all successful loops — that is the **07-17 sector extension**, not 07-07).
4. **Initial library**: empty baseline only; may grow only with **this run’s** Replace=yes loops. Must not inherit 07-07 / RD13 / 09-15 factor state.

## D. Cache isolation audit

| Check | Status |
|---|---|
| New cache path | `${ROOT}/pickle_cache` |
| Currently empty | yes (created empty) |
| Default `${RDAGENT_ROOT}/pickle_cache` | not used; old `d41d8cd…` must remain untouched |
| 09-15 cache | not referenced |

## E. Data visibility audit

See `visible_data_inventory.txt` and `llm_visible_columns.txt`.
Clean dirs contain only `daily_pv.h5` + README; columns are the six PV fields.
Default polluted folder still has `rd13_reference.h5` + `fundamental_features.h5` — **must not** be on `FACTOR_CoSTEER_DATA_FOLDER*`.

## F. Prompt hash comparison

See `prompt_hashes.txt` and `prompt_diff.md`.
Until `install_golden_prompts.sh` runs, active `scenarios/qlib/prompts.yaml` **FAILS** hash match (diversity §6 present).
