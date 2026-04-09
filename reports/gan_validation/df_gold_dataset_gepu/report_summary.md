# GAN Validity Report: df_gold_dataset_gepu

## Summary
- dataset: df_gold_dataset_gepu
- source_last_date: 2025-11-26
- generated_last_date: 2026-12-31
- future_rows: 286
- expected_future_rows: 286
- duplicate_dates: 0
- null_cells: 0
- avg_vol_ratio: 0.98443
- max_vol_ratio_gap: 0.096527
- avg_mean_gap_abs: 0.030611
- avg_ks_statistic: 0.090669
- max_ks_statistic: 0.161924
- avg_acf_gap_lag1: 0.044472
- max_acf_gap_lag1: 0.105712
- corr_matrix_mae: 0.038106
- corr_matrix_max_gap: 0.138436
- share_features_vol_ratio_pass_0p8_to_1p2: 1.0
- share_features_ks_pvalue_gt_0p05: 0.428571
- share_features_acf_gap_lag1_lt_0p15: 1.0

## Highest KS Distance Features
- gepu: ks_statistic=0.1619, ks_pvalue=0.0000, vol_ratio=0.9844, acf_gap_1=0.0121
- Silver_Futures: ks_statistic=0.1280, ks_pvalue=0.0004, vol_ratio=1.0018, acf_gap_1=0.0238
- Gold_Futures: ks_statistic=0.0999, ks_pvalue=0.0105, vol_ratio=1.0018, acf_gap_1=0.1057
- Crude_Oil_Futures: ks_statistic=0.0994, ks_pvalue=0.0110, vol_ratio=1.0018, acf_gap_1=0.0708
- UST10Y_Treasury_Yield: ks_statistic=0.0632, ks_pvalue=0.2398, vol_ratio=1.0018, acf_gap_1=0.0691
