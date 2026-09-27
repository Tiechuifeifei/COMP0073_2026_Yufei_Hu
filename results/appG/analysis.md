# K5_D1 post-hoc exploratory diagnosis — k5d1_posthoc_diagnosis_20260920_110303

**Identity:** POST-HOC EXPLORATORY / MECHANISM DIAGNOSIS. Not confirmatory selection.

## 1. Original result reproduction

- K5_D1 valid: excess_arr_net=0.184532 (expected≈0.184532) OK=True
- K5_D1 test: excess_arr_net=0.110896 (expected≈0.110896) OK=True
- K20_D2 valid: excess_arr_net=0.071628 (expected≈0.071628) OK=True
- K20_D2 test: excess_arr_net=0.100720 (expected≈0.10072) OK=True

## 2. Rank-bucket evidence

See `rank_bucket_returns.csv`. Higher buckets use LABEL0 on the same date as score (no extra shift).

## 3. Rank1–5 vs Rank6–20 bootstrap

- Valid: mean D_top=0.000561, CI=[-0.000263,0.001480], one-sided p=0.0960
- Test: mean D_top=-0.000051, CI=[-0.000852,0.000770], one-sided p=0.5530
- Interpretation: rank 1–5 cannot be statistically distinguished from rank 6–20

## 4. Contribution concentration

- Reconciliation: K5_D1/valid max_err=3.75e-08 pass=True, K5_D1/test max_err=4.28e-08 pass=True, K20_D2/valid max_err=2.15e-08 pass=True, K20_D2/test max_err=2.45e-08 pass=True
- K5 valid top1: share_total_pnl=0.162, share_pos=0.055, share_abs=0.113
- K5 valid top3: share_total_pnl=0.447, share_pos=0.271, share_abs=0.310
- K5 valid top5: share_total_pnl=0.669, share_pos=0.529, share_abs=0.464
- Approx ex-top1 ARR (valid): 0.1375167428435849
- Approx ex-top2 ARR (valid): 0.09518276202528857
- Label: COUNTERFACTUAL CONTRIBUTION-REMOVAL APPROXIMATION

## 5. Time concentration

- K5_D1 valid: best_month=2018-02 share=0.203, best10_share=0.589, ex_best10_arr≈0.0775, label=TEMPORALLY_CONCENTRATED
- K5_D1 test: best_month=2020-04 share=0.429, best10_share=1.689, ex_best10_arr≈-0.0772, label=nan
- K20_D2 valid: best_month=2018-03 share=0.236, best10_share=0.896, ex_best10_arr≈0.0076, label=TEMPORALLY_CONCENTRATED

## 6. Top20-internal placebo

- Valid: true percentile=98.3%, p=0.0167; the within-Top20 ranking contributes materially to the K5 outcome
- Test: true percentile=66.7%, p=0.3333; K5 performance cannot be clearly distinguished from random selection within the Top20 set
- Valid turnover diff (true−placebo mean)=-0.1513

## 7. K5 replacement neighbourhood

- K5_D1: ARR=0.1845, IR=1.371, TO=0.245, realised_repl=0.127
- K5_D2: ARR=0.1631, IR=1.231, TO=0.292, realised_repl=0.147
- K5_D3: ARR=0.1620, IR=1.225, TO=0.294, realised_repl=0.149
- K5_D4: ARR=0.1620, IR=1.225, TO=0.294, realised_repl=0.149
- K5_D5: ARR=0.1620, IR=1.225, TO=0.294, realised_repl=0.149
- Interpretation: Top5 concentration appears more important than the exact replacement intensity

## 8. K3–K7 breadth neighbourhood

- K3_D1: ARR=0.1497, IR=0.926, TO=0.315
- K4_D1: ARR=0.2021, IR=1.428, TO=0.252
- K5_D1: ARR=0.1845, IR=1.371, TO=0.245
- K6_D1: ARR=0.1641, IR=1.354, TO=0.208
- K7_D1: ARR=0.1192, IR=1.051, TO=0.196
- Interpretation: cross-period local concentration stability
- K5_D1_minus_K4_D1 valid: ΔARR=-0.0176, ΔIR=-0.0565
- K5_D1_minus_K6_D1 valid: ΔARR=0.0205, ΔIR=0.0173
- K5_D1_minus_K4_D1 test: ΔARR=0.0310, ΔIR=0.1331
- K5_D1_minus_K6_D1 test: ΔARR=0.0285, ΔIR=0.0763

## 9. Style exposure

- K5_D1 valid: beta=1.224, alpha=0.000706, R2=0.652
- K5_D1 test: beta=1.349, alpha=0.000318, R2=0.594
- K20_D2 valid: beta=1.154, alpha=0.000253, R2=0.852
- K20_D2 test: beta=1.193, alpha=0.000346, R2=0.737
- FF/MOM: NOT_AVAILABLE; size log mcap: NOT_AVAILABLE
- K5_D1 valid: vol20 port−uni=0.00050, mom12-1 port−uni=0.09296
- K5_D1 test: vol20 port−uni=0.00249, mom12-1 port−uni=-0.03417
- K20_D2 valid: vol20 port−uni=0.00001, mom12-1 port−uni=0.10908
- K20_D2 test: vol20 port−uni=0.00262, mom12-1 port−uni=-0.00275

## 10. Integrated interpretation

K5_D1's validation advantage is best read as **MIXED EVIDENCE**:
- Top-rank signal: rank 1–5 cannot be statistically distinguished from rank 6–20.
- Placebo (valid): the within-Top20 ranking contributes materially to the K5 outcome.
- Replacement axis: Top5 concentration appears more important than the exact replacement intensity.
- Breadth axis: cross-period local concentration stability.
- Time: TEMPORALLY_CONCENTRATED.
- This study does **not** declare K5_D1 an automatic final static winner.

- Tags: mixed, A/ranking, D, G_MIXED
