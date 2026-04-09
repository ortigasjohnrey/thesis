# GAN Validity Report: silver_RRL_interpolate

## Summary
- dataset: silver_RRL_interpolate
- source_last_date: 2025-04-30
- generated_last_date: 2026-05-08
- future_rows: 267
- expected_future_rows: 267
- duplicate_dates: 0
- null_cells: 0
- avg_vol_ratio: 1.001872
- max_vol_ratio_gap: 0.001877
- avg_mean_gap_abs: 0.0
- avg_ks_statistic: 0.074752
- max_ks_statistic: 0.106095
- avg_acf_gap_lag1: 0.045443
- max_acf_gap_lag1: 0.114919
- corr_matrix_mae: 7e-06
- corr_matrix_max_gap: 3.3e-05
- share_features_vol_ratio_pass_0p8_to_1p2: 1.0
- share_features_ks_pvalue_gt_0p05: 0.666667
- share_features_acf_gap_lag1_lt_0p15: 1.0

## Highest KS Distance Features
- Silver_Futures: ks_statistic=0.1061, ks_pvalue=0.0080, vol_ratio=1.0019, acf_gap_1=0.0662
- Gold_Futures: ks_statistic=0.0907, ks_pvalue=0.0349, vol_ratio=1.0019, acf_gap_1=0.1149
- USD_index: ks_statistic=0.0797, ks_pvalue=0.0876, vol_ratio=1.0019, acf_gap_1=0.0247
- US30: ks_statistic=0.0581, ks_pvalue=0.3720, vol_ratio=1.0019, acf_gap_1=0.0114
- NASDAQ_100: ks_statistic=0.0580, ks_pvalue=0.3744, vol_ratio=1.0019, acf_gap_1=0.0394
