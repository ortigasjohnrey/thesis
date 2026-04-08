# GAN Validity Report: gold_RRL_interpolate

## Summary
- dataset: gold_RRL_interpolate
- source_last_date: 2025-04-30
- generated_last_date: 2026-05-08
- future_rows: 267
- expected_future_rows: 267
- duplicate_dates: 0
- null_cells: 0
- avg_vol_ratio: 1.001878
- max_vol_ratio_gap: 0.001878
- avg_mean_gap_abs: 1e-06
- avg_ks_statistic: 0.18413
- max_ks_statistic: 0.41055
- avg_acf_gap_lag1: 0.139954
- max_acf_gap_lag1: 0.380891
- corr_matrix_mae: 0.237061
- corr_matrix_max_gap: 0.648944
- share_features_vol_ratio_pass_0p8_to_1p2: 1.0
- share_features_ks_pvalue_gt_0p05: 0.25
- share_features_acf_gap_lag1_lt_0p15: 0.75

## Highest KS Distance Features
- Employment_Pop_Ratio: ks_statistic=0.4106, ks_pvalue=0.0000, vol_ratio=1.0019, acf_gap_1=0.0433
- Federal_Funds_Rate: ks_statistic=0.3631, ks_pvalue=0.0000, vol_ratio=1.0019, acf_gap_1=0.0608
- gepu: ks_statistic=0.2243, ks_pvalue=0.0000, vol_ratio=1.0019, acf_gap_1=0.1000
- Silver_Futures: ks_statistic=0.1231, ks_pvalue=0.0012, vol_ratio=1.0019, acf_gap_1=0.0830
- Crude_Oil_Futures: ks_statistic=0.1102, ks_pvalue=0.0052, vol_ratio=1.0019, acf_gap_1=0.3809
