$ErrorActionPreference = 'Stop'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.matplotlib'

Write-Host "===== GOLD: Retraining with stronger anti-laziness loss (v3) =====" -ForegroundColor Cyan

# ── Gold ──────────────────────────────────────────────────────────────────────
$env:GOLD_CLEAN_DF_PATH             = 'df_gold_dataset_gepu_extended_train.csv'
$env:GOLD_TARGET_COL                = 'Gold_Futures'
$env:GOLD_DATE_COL                  = 'Date'
$env:GOLD_HORIZON                   = '1'
$env:GOLD_OUTER_TRAIN_RATIO         = '0.88'
$env:GOLD_FINAL_VAL_RATIO           = '0.10'
$env:GOLD_TUNING_SEED               = '42'
$env:GOLD_REPORTS_DIR               = 'reports/gold_train_only_retrained_v2'
$env:GOLD_SAVE_ROOT_DIR             = 'models/gold_train_only_retrained_v2'
$env:GOLD_FINAL_SEEDS               = '0,1,2,7,11,17,21,42,99,123'

# Optuna budget — generous to find hyperparams that actually diverge from zero
$env:GOLD_INIT_POINTS               = '15'
$env:GOLD_N_ITER                    = '45'
$env:GOLD_N_SPLITS_WALK_FORWARD     = '5'

# Allow longer training so the anti-laziness loss has time to take effect
$env:GOLD_MAX_EPOCHS_TUNING         = '80'
$env:GOLD_MAX_EPOCHS_FINAL          = '120'
$env:GOLD_EARLY_STOPPING_PATIENCE   = '10'
$env:GOLD_EARLY_STOPPING_MIN_DELTA  = '1e-5'

# Hyperparameter search ranges — wider lookback + larger capacity models
$env:GOLD_LOOKBACK_RANGE            = '20,60'
$env:GOLD_FILTERS_RANGE             = '32,96'
$env:GOLD_KERNEL_SIZE_RANGE         = '2,5'
$env:GOLD_LSTM_UNITS_RANGE          = '64,192'
$env:GOLD_DENSE_UNITS_RANGE         = '32,80'
$env:GOLD_DROPOUT_RANGE             = '0.20,0.45'
$env:GOLD_LEARNING_RATE_RANGE       = '5e-4,8e-3'
$env:GOLD_L2_REG_RANGE              = '5e-5,1e-3'
$env:GOLD_BATCH_SIZE_EXP_RANGE      = '4,7'

& .\.venv\Scripts\python.exe train_gold_RRL_interpolate.py

Write-Host "`n===== SILVER: Retraining with stronger anti-laziness loss (v3) =====" -ForegroundColor Cyan

# ── Silver ────────────────────────────────────────────────────────────────────
$env:SILVER_CLEAN_DF_PATH           = 'silver_RRL_interpolate_extended_train.csv'
$env:SILVER_TARGET_COL              = 'Silver_Futures'
$env:SILVER_DATE_COL                = 'Date'
$env:SILVER_HORIZON                 = '1'
$env:SILVER_OUTER_TRAIN_RATIO       = '0.88'
$env:SILVER_FINAL_VAL_RATIO         = '0.10'
$env:SILVER_TUNING_SEED             = '42'
$env:SILVER_REPORTS_DIR             = 'reports/silver_train_only_retrained_v2'
$env:SILVER_SAVE_ROOT_DIR           = 'models/silver_train_only_retrained_v2'
$env:SILVER_FINAL_SEEDS             = '0,1,2,7,11,17,21,42,99,123'

$env:SILVER_INIT_POINTS             = '15'
$env:SILVER_N_ITER                  = '45'
$env:SILVER_N_SPLITS_WALK_FORWARD   = '5'

$env:SILVER_MAX_EPOCHS_TUNING       = '80'
$env:SILVER_MAX_EPOCHS_FINAL        = '120'
$env:SILVER_EARLY_STOPPING_PATIENCE = '10'
$env:SILVER_EARLY_STOPPING_MIN_DELTA= '1e-5'

$env:SILVER_LOOKBACK_RANGE          = '15,50'
$env:SILVER_FILTERS_RANGE           = '32,96'
$env:SILVER_KERNEL_SIZE_RANGE       = '2,5'
$env:SILVER_LSTM_UNITS_RANGE        = '64,192'
$env:SILVER_DENSE_UNITS_RANGE       = '32,80'
$env:SILVER_DROPOUT_RANGE           = '0.20,0.45'
$env:SILVER_LEARNING_RATE_RANGE     = '5e-4,8e-3'
$env:SILVER_L2_REG_RANGE            = '5e-5,1e-3'
$env:SILVER_BATCH_SIZE_EXP_RANGE    = '4,7'

& .\.venv\Scripts\python.exe train_silver_RRL_interpolate.py

Write-Host "`n===== Retraining complete. Running evaluation =====" -ForegroundColor Green
& .\.venv\Scripts\python.exe evaluate_official_test_sets.py
