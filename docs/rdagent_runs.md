# RD-Agent runs (Appendix B.1 / B.2)

## Trajectories

1. Historical mapping trajectory (40-loop SP500 work used for lineage).
2. RD8 matched / corrected semantic rerun — overlay under `02_rdagent/matched_rerun/`
   and Loop-2 correction scripts under `02_rdagent/loop2_injection/`.

## Reproduction (matches Appendix B.2)

1. Clone Microsoft RD-Agent (or fork).
2. `git checkout 4f9ecb00`
3. Apply `patches/rdagent_us_rd8.patch` (generated as `git diff 4f9ecb00..us-market-experiment`).
4. Install golden prompts / overlay launcher from `02_rdagent/matched_rerun/`.
5. For the Loop-2 semantic correction, use `02_rdagent/loop2_injection/`.

The dissertation records a single Loop-2 correction injection; other prompt
edits are out of scope for this matched rerun.
