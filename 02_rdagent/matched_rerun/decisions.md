# Matched reference-corrected RD-Agent run — pre-registered decisions

## 2026-09-16

This run is pre-designated as the principal matched reference-corrected RD-Agent run. The July-07 configuration is restored as closely as possible, while the incorrect reference/baseline and cross-run state contamination are removed. No post-run choice between this run and the 2026-09-15 RD6 run will be made based on performance. The terminal SOTA from this run will be used for the downstream F1C incremental experiment regardless of its performance.

### Scope notes (frozen with this designation)

- Root: `reports/rdagent_matched_reference_rerun_20260916/`
- Loop count: **20** (matched to first 20 of the 07-07 40-loop; not a full 40 replay)
- Optional execution schedule: **pause after loop 10 for read-only audit, then resume the same session for loops 10–19** via `run_first10.sh` + `resume_next10.sh` (or `orchestrate_first10_then_resume.sh`). This remains one principal trajectory; mid-run performance must not decide whether to continue.
- Prompts: git `2f02043b` / `bak_before_regime_hint` (no diversity §6), delivered via **`launch_factor_with_prompt_overlay.py`** when the RD-Agent worktree cannot be rewritten
- Discovery data: `daily_pv` only (no rd13 / fundamental / sentiment / F1C)
- Baseline: fresh S&P500 Alpha20 in an isolated pickle cache (reject CSI300 leak triple)
- Accumulation: `QLIB_FACTOR_ACCUMULATE_ALL=false`, matching 07-07 **runtime** (`Feedback.__bool__` → decision gate; tee showed zero cross-loop SOTA merge while Replace=False)
- The 2026-09-15 corrected 20-loop remains an audit artifact only; it is not a competing principal run
