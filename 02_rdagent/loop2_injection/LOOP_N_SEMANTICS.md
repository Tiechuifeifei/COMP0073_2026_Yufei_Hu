# loop_n semantics (verified before launch)

Source: `rdagent/utils/workflow/loop.py` `LoopBase.run()` and matched `resume_next10.sh`.

## Facts

1. `run(loop_n=N)` **always resets** `loop_idx = 0`.
2. Each kickoff consumes one unit of `loop_n` (decrements after scheduling).
3. If `step_idx[li] >= len(steps)`, that loop is already finished → kickoff is a **no-op**.
4. Therefore CLI `--loop_n N` means **target total loops counted from index 0**, not "N additional loops after resume".

## This experiment

| Item | Value |
|---|---|
| Fork dump | principal `__session__/1/4_record` (loops 0–1 finished, `loop_idx=2`) |
| Required endpoint | **Loop9** (no Loop10+) |
| CLI | `--loop_n 10` |
| Kickoffs 0–1 | no-op |
| Kickoff 2 | corrected Loop2 (implementation override) |
| Kickoffs 3–9 | new LLM search loops |
| Must NOT use | `--loop_n 8` (would stop after Loop7) |

Matched-run precedent: after Loops 0–9, resume used `--loop_n 20` so 0–9 no-op and 10–19 run.

## Continuation Loop10–19 (corrected semantic)

| Item | Value |
|---|---|
| Source dump | `sessions/corrected_semantic_10loop/__session__/9/4_record` (read-only) |
| Destination | `sessions/corrected_semantic_20loop/` (new writes only) |
| Required endpoint | **Loop19** (no Loop20+) |
| CLI | `--loop_n 20` |
| Kickoffs 0–9 | no-op (already finished in hist) |
| Kickoffs 10–19 | newly executed |
| Injection | disabled (`allow_corrected_injection=false`) |
