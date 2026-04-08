# GAN Validity Report: silver_RRL_interpolate

## Summary
- dataset: silver_RRL_interpolate
- source_last_date: 2025-04-30
- generated_last_date: 2026-05-08
- future_rows: 267
- expected_future_rows: 267
- duplicate_dates: 0
- null_cells: 0
- avg_vol_ratio: 1.001878
- max_vol_ratio_gap: 0.001878
- avg_mean_gap_abs: 0.0
- avg_ks_statistic: 0.088533
- max_ks_statistic: 0.117799
- avg_acf_gap_lag1: 0.102582
- max_acf_gap_lag1: 0.200511
- corr_matrix_mae: 0.370782
- corr_matrix_max_gap: 0.581444
- share_features_vol_ratio_pass_0p8_to_1p2: 1.0
- share_features_ks_pvalue_gt_0p05: 0.333333
- share_features_acf_gap_lag1_lt_0p15: 0.666667

## Highest KS Distance Features
- US30: ks_statistic=0.1178, ks_pvalue=0.0022, vol_ratio=1.0019, acf_gap_1=0.1125
- NASDAQ_100: ks_statistic=0.1015, ks_pvalue=0.0127, vol_ratio=1.0019, acf_gap_1=0.0033
- Silver_Futures: ks_statistic=0.1006, ks_pvalue=0.0138, vol_ratio=1.0019, acf_gap_1=0.1920
- SnP500: ks_statistic=0.0878, ks_pvalue=0.0451, vol_ratio=1.0019, acf_gap_1=0.0666
- Gold_Futures: ks_statistic=0.0856, ks_pvalue=0.0542, vol_ratio=1.0019, acf_gap_1=0.0406
