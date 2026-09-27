# Decision log — k5d1_posthoc_diagnosis_20260920_110303

### [20260920_110303] PRE-COMPUTE FREEZE — POST-HOC EXPLORATORY / MECHANISM DIAGNOSIS

**Identity:** POST-HOC EXPLORATORY / MECHANISM DIAGNOSIS  
**Not:** confirmatory selection; not automatic static winner; not HMM reselection.

**Forbidden outcomes of this run:**
- modify F_BASE_W010 / w*=0.1
- retrain models; change fundamentals / RD / sentiment
- open 2024–2025 holdout
- auto-replace final portfolio config
- auto-reselect HMM policy

**Identities:**
- K20_D2 = INHERITED_REFERENCE / BENCHMARK
- K5_D1 = VALIDATION-LEADING CONCENTRATED CONFIGURATION (mechanism study only)

**Authority S1 (read-only):** `reports/portfolio_topk_drop_joint_20260919_141851/`

**Signal:** F_BASE_W010 = 0.1·z(B1-R50)+0.9·z(F1C-R50)  
**Engine:** TopkDropoutStrategy; deal_price=close; costs=1bp; risk_degree=0.95; hold_thresh=1; benchmark=P84398; method_sell=bottom; method_buy=top; score_t → t+1 close.

**Holdout:** CLOSED. Every loaded series filtered and asserted `max_date <= 2023-12-31`. Placebo `end_time=2023-12-31`.

---

### Interpretation rules (frozen before results)

#### Rank 1–5 vs 6–20 (D_top)
- Moving-block bootstrap B=2000, block=20.
- If valid CI lower > 0 **and** test CI lower > 0 →  
  "consistent evidence of incremental predictive information in the extreme top ranks"
- If valid supports but test does not → "top-rank advantage is not stable across periods"
- If neither supports → "rank 1–5 cannot be statistically distinguished from rank 6–20"
- Forbidden language: proved / true alpha proved / causal / definitely / K5 is optimal

#### Contribution
- Use EOD weight at t−1 × stock_return_t; cash_return_t = 0 if engine cash earns 0.
- Reconcile vs **GROSS** portfolio return; STOP if max_abs_error > 1e-6.
- Counterfactual top1/top2 removal = CONTRIBUTION-REMOVAL APPROXIMATION (not rebalanced backtest).

#### Time concentration
- Ex-best-10-days = path-concentration diagnostic, not tradable strategy.
- TEMPORALLY_CONCENTRATED vs TEMPORALLY_BROAD labels as specified in protocol.

#### Placebo (Top20-internal rank randomisation)
- Membership of Top20 fixed; permute scores inside Top20 only; continuous TopkDropout K5_D1 path; seeds 0…299; 6 workers; deterministic per run_id.
- If true K5 percentile ≥ 95% → "the within-Top20 ranking contributes materially to the K5 outcome"
- If < 95% → "K5 performance cannot be clearly distinguished from random selection within the Top20 set"
- If turnover differs materially from placebo, note path/trading-intensity channel.

#### Replacement neighbourhood (K=5; D1…D5)
- Nominal capacities 20/40/60/80/100%; K5_D5 = "100% nominal replacement capacity" only (not "daily full replacement").
- Interpretations: concentration vs retention interaction vs faster replacement vs no clear pattern (protocol §6A.2).

#### Breadth neighbourhood (n_drop=1; K=3…7)
- Local stability / isolated peak / period-specific / cross-period stability (protocol §6B.2).
- **No auto-selection** even if neighbour point estimates exceed K5_D1.

#### Style
- SPY beta / up-down beta; holding characteristics only if PIT-safe; FF/MOM = NOT_AVAILABLE if absent (no download).

#### Language
Use: consistent with / supports / suggests / cannot distinguish / exploratory evidence.  
Avoid: proved / causal / definitely / K5 is optimal (except describing sample point estimates).

---

### Commit gate
This decision_log is frozen **before** formal computation. Git commit hash of this freeze will be recorded in `run_manifest.json`.

### [POST-RUN] Results appended (rules were pre-frozen)
- topdiff interp: rank 1–5 cannot be statistically distinguished from rank 6–20
- placebo valid: the within-Top20 ranking contributes materially to the K5 outcome
- replacement: Top5 concentration appears more important than the exact replacement intensity
- breadth: cross-period local concentration stability
