# Unified downstream specification tables A–D

**Generated:** 2026-09-23T11:37:22+00:00  
**Directory:** `rd13_l20_unified_ABCD_20260923_193417`

Setup: fixed LightGBM OFFICIAL; R∈{10,20,50}; seeds 42/2026/3407; no early stopping;
delayed common sample; train 2008–2017. Alpha20_RD8 ≡ B1.

## A. Price–volume model table (valid 2018–19 / development 2020–23 / appendix 2024–25)

Full numbers are in `A_period_metrics_seed_mean_std.csv` (seed mean±std and ensemble),
`A_yearly_rankic.csv`, and `A_paired_delta_rankic_bootstrap.csv`.

### R50 validation / development (ensemble + seed mean±std of RankIC)

| spec | period | RankIC_ens | RankIC seed μ±σ | IC_ens | ICIR_ens |
|---|---|---:|---|---:|---:|
| Alpha20 | valid_2018_2019 | 0.003962 | 0.003953±0.002564 | 0.004025 | 0.0562 |
| Alpha20_RD8 | valid_2018_2019 | 0.003468 | 0.002637±0.001054 | 0.000220 | 0.0027 |
| Alpha20_RD13_v2 | valid_2018_2019 | 0.003090 | 0.003010±0.001651 | 0.001490 | 0.0166 |
| Alpha20_RD4 | valid_2018_2019 | 0.004897 | 0.004472±0.001198 | 0.004205 | 0.0501 |
| Alpha20 | test_2020_2023 | 0.006171 | 0.005606±0.001066 | 0.007010 | 0.0771 |
| Alpha20_RD8 | test_2020_2023 | 0.009448 | 0.008571±0.001273 | 0.009452 | 0.0875 |
| Alpha20_RD13_v2 | test_2020_2023 | 0.015116 | 0.013674±0.001561 | 0.014739 | 0.1126 |
| Alpha20_RD4 | test_2020_2023 | 0.012040 | 0.010931±0.000483 | 0.013286 | 0.1124 |
| Alpha20 | holdout_2024_2025 | -0.002024 | -0.001588±0.001287 | -0.000566 | -0.0081 |
| Alpha20_RD8 | holdout_2024_2025 | 0.000794 | 0.000635±0.000575 | 0.000278 | 0.0036 |
| Alpha20_RD13_v2 | holdout_2024_2025 | 0.002401 | 0.002023±0.002263 | 0.002433 | 0.0256 |
| Alpha20_RD4 | holdout_2024_2025 | 0.001101 | 0.001065±0.000934 | 0.002467 | 0.0291 |

## B. D2 original vs unified R50

See `B_D2_ORIGINAL_VS_UNIFIED.md` and `B_D2_original_vs_unified_R50.csv`.
Original D2: early_stopping=50, max 1000, three seeds; **actual best_iteration=1**.

- D2_original_earlystop / valid_2018_2019: IC=0.003275, RankIC=0.002965
- unified_R50_no_earlystop / valid_2018_2019: IC=0.001490, RankIC=0.003090
- D2_original_earlystop / test_2020_2023: IC=0.009483, RankIC=0.007291
- unified_R50_no_earlystop / test_2020_2023: IC=0.014739, RankIC=0.015116

## C. Single-factor diagnostics (2020–2023 RankIC top 5)

| family | factor | RankIC | IC | ICIR |
|---|---|---:|---:|---:|
| RD13_v2 | rd13_5d_short_term_reversal | 0.009061 | 0.007785 | 0.0344 |
| RD8 | reversal_1d | 0.005342 | 0.002850 | 0.0137 |
| RD13_v2 | rd13_daily_amihud_illiquidity | 0.005219 | 0.000643 | 0.0077 |
| RD13_v2 | rd13_volatility_adjusted_10d_momentum | 0.000858 | -0.001858 | -0.0096 |
| RD8 | vol_norm_price_momentum_10d | 0.000771 | -0.002290 | -0.0120 |

Maximum |ρ| versus Alpha20: `C_rd_vs_alpha20_max_cs_corr.csv`.

## D. Feature importance (R50, 3-seed mean gain)

See `D_feature_importance_gain_R50.csv`, `D_rd_block_gain_share_R50.csv`, `D_top10_*_R50.csv`.
