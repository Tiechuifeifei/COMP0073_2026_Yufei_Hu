# RD8 Loop-2 correction (20-loop regenerated trajectory)

This folder reconstructs Trajectory A in Appendix B.1 of the report.

## What the regenerated run does

1. **Loops 0–1** — retained from the matched prior run (forked checkpoint; not re-searched).
2. **Loop 2** — inject the scale-corrected ATR–volume interaction (range divided by the previous close). The framework’s unchanged Analysis Unit evaluates it and **rejects** it (report: IC 0.002091, excess ARR +2.88%).
3. **Loops 3–19** — proposed, evaluated and accepted/rejected by the native RD-Agent mechanism with no further manual edits, in two consecutive sessions that continue the same search state (Loops 0–9, then 10–19).
4. **Loops 16–17** — recorded but without a valid result (dependency/interface errors), consistent with Appendix B.1.

Because Loop 2 changes the state from which later loops are proposed, the RD8 factor set is the terminal state of this **regenerated** trajectory, not of the original run. This is the only manual intervention in either RD-Factor trajectory.

## Launch

From the repository root (after `source ./set_env.sh` and a patched `RDAGENT_ROOT`):

```bash
bash 02_rdagent/loop2_injection/run_corrected_20loop.sh
```

Supporting pieces:

- `correction_gate.py` / `corrected_loop_class.py` — define and gate the corrected Loop 2 factor.
- `run_corrected_semantic_20loop.py` — continue from the Loop 9 checkpoint through Loop 19.
- `write_corrected_20loop_audits.py` — write summary audits after the run.

`run_corrected_10loop.sh` / `run_corrected_semantic_10loop.py` are the first session (Loops 0–9) used to produce the Loop 9 checkpoint that the 20-loop continuation resumes from; the dissertation endpoint is Loop 19.
