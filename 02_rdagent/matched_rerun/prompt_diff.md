# Prompt diff: 07-07 golden (git 2f02043b) vs current RD-Agent worktree

Golden source: `git show 2f02043b:...` (byte-identical to `prompts.yaml.bak_before_regime_hint` for scenarios/qlib/prompts.yaml).
Session cross-check: 07-07 Loop_* debug_tpl has **no** `Maintain Exploration Diversity` (40/40 hits diversity=False).

## Hash comparison

| File | Golden SHA256 | Active SHA256 | Match |
|---|---|---|---|
| `rdagent/scenarios/qlib/prompts.yaml` | `f8a213cb35c6677a9a4a38ccbdfb4b05fe4798090da83764b8fafe0f1dc98a1c` | `ff135d246448b80eb9a238210762ed7b6f90ab27329bdac3c28af3bf2b290fc8` | **NO** |
| `rdagent/scenarios/qlib/experiment/prompts.yaml` | `6325f0366075ffc0519a983ac0049c5bf30714479932a7b4d9ba5651ae86d004` | `6325f0366075ffc0519a983ac0049c5bf30714479932a7b4d9ba5651ae86d004` | **YES** |
| `rdagent/components/proposal/prompts.yaml` | `a5187101d3b281ea0be9e0d974f3bd65706a9e7235221e108a92f689c4e5e44c` | `a5187101d3b281ea0be9e0d974f3bd65706a9e7235221e108a92f689c4e5e44c` | **YES** |
| `rdagent/components/coder/factor_coder/prompts.yaml` | `8c6cfc174915bff8b173e959611ffc4e946a19aa406634f0a1eb261725a0efce` | `8c6cfc174915bff8b173e959611ffc4e946a19aa406634f0a1eb261725a0efce` | **YES** |

## Substantive diff (scenarios/qlib/prompts.yaml only; others identical)

```diff
--- ${PROJECT_ROOT}/reports/rdagent_matched_reference_rerun_20260916/golden_prompts/scenarios_qlib_prompts.yaml	2026-09-16 08:35:19
+++ ${RDAGENT_ROOT}/rdagent/scenarios/qlib/prompts.yaml	2026-07-10 06:44:39
@@ -110,6 +110,17 @@
   5. Note
     - Highlight that factors surpassing SOTA are included in the library to avoid re-implementation.
     - No matter how many factors you plan to generate, only reply with one set of hypothesis and reason. The hypothesis can include the proposal of multiple factors at the same time.
+  6. **Maintain Exploration Diversity Across Factor Families:**
+    - Previous experiments show that repeatedly generating regime-conditioned or volatility-scaled momentum variants (e.g. switching, weighting, or partitioning momentum signals by market state) has not produced consistent improvements in annualized return, even though some individual attempts succeeded.
+    - To avoid narrowing the search space around a single theme, actively rotate across distinct factor families across iterations, for example:
+      a) Value/quality-style factors (e.g. price-to-moving-average ratios, earnings-based proxies if data allows)
+      b) Liquidity/microstructure factors (e.g. bid-ask proxies, turnover patterns, illiquidity measures)
+      c) Cross-sectional structural signals (e.g. sector-relative or peer-group-relative measures)
+      d) Sparse/regularized statistical models (e.g. lasso, elastic net on diverse feature sets)
+      e) Representation learning (e.g. autoencoders, PCA on non-momentum feature sets)
+      f) Event/anomaly-driven signals (e.g. earnings surprise proxies, volume spikes unrelated to momentum)
+    - If the last 2-3 iterations have all involved momentum as a core signal, deliberately switch to a family without momentum as the primary driver in the next iteration.
+    - Do not treat "regime-awareness" as inherently preferable — evaluate each new hypothesis on its own merits, not on whether it resembles previously successful designs.
 
 factor_experiment_output_format: |-
   The output should follow JSON format. The schema is as follows:
```

## Forbidden guidance strings in ACTIVE worktree

- PRESENT (FAIL until install_golden_prompts): `Maintain Exploration Diversity Across Factor Families`
- PRESENT (FAIL until install_golden_prompts): `actively rotate across distinct factor families`
- PRESENT (FAIL until install_golden_prompts): `Value/quality-style`
- PRESENT (FAIL until install_golden_prompts): `Representation learning`
- PRESENT (FAIL until install_golden_prompts): `Event/anomaly-driven`
- PRESENT (FAIL until install_golden_prompts): `If the last 2-3 iterations have all involved momentum`

## Forbidden guidance strings in GOLDEN

- absent (OK): `Maintain Exploration Diversity Across Factor Families`
- absent (OK): `actively rotate across distinct factor families`
- absent (OK): `Value/quality-style`
- absent (OK): `Representation learning`
- absent (OK): `Event/anomaly-driven`

## Preflight rule
After `install_golden_prompts.sh`, all four active files must hash-match golden. Any residual diversity/guidance string → FAIL.
