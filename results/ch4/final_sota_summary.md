# Final SOTA Summary — Matched Reference 20-loop Rerun

**Session:** `sessions/matched_20loop_20260916_090024`  
**Artifacts basis:** `__session__/19/4_record` + `Loop_*/feedback` + tees (not `phase_status` alone)  
**Audit date:** 2026-09-17

## Final SOTA (after Loop 19)

| Field | Value |
|---|---|
| Source loop | **Loop 2** (last `Replace=yes`) |
| Experiment workspace | `61a0e3a7267949579b32513f34c3c930` |
| IC | **0.0045035290804027** |
| ARR (`1day.excess_return_with_cost.annualized_return`) | **0.1663016969585134** |
| MDD (`1day.excess_return_with_cost.max_drawdown`) | **−0.374012941664434** |
| Trace API | `trace.get_sota_experiment()` → same workspace / metrics |

## Accepted loops (`Replace=yes`)

| Loop | IC | ARR | MDD | Exp workspace | New factors |
|---:|---:|---:|---:|---|---|
| 1 | 1.002098552893985e−05 | 0.0395424743714626 | −0.3337506904335999 | `ed833a12…b47659` | `reversal_1d`, `atr_10d`, `volume_ratio_10d` |
| 2 | 0.0045035290804027 | 0.1663016969585134 | −0.374012941664434 | `61a0e3a7…c930` | `atr_20d`, `volume_ratio_20d`, `atr_20d_x_volume_ratio_20d` |

No further Replace=yes in Loops 3–19.

## Incoming baseline (Loop 0)

Fresh Alpha20 empty-baseline (full precision, matches isolated cache):

- IC = `0.0045787190505179`
- ARR = `−0.0143767767379566`
- MDD = `−0.2683045360791799`

## Trajectory one-liner

Baseline → Loop1 accept (low IC / positive ARR) → Loop2 accept (SOTA) → Loops 3–19 all reject → **final SOTA = Loop 2**.

## Companion files

- `loop_summary_0_19.csv`
- `accepted_factor_provenance_0_19.csv`
- `forensic_audit_loops0_19.md`
