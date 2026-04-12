import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ============================================================
# Model Architecture — must EXACTLY match training scripts
# (train_gold_RRL_interpolate.py / train_silver_RRL_interpolate.py)
# Key: forward() uses h_n (final hidden state) NOT x[:,-1,:]
# ============================================================
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_shape, params):
        super(CNN_BiLSTM, self).__init__()
        in_channels = input_shape[1]
        dr = params["dropout_rate"]
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=params["filters"],
            kernel_size=params["kernel_size"],
            padding=params["kernel_size"] - 1,
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(params["filters"])
        self.spatial_dropout = nn.Dropout1d(p=dr)
        self.lstm1 = nn.LSTM(
            input_size=params["filters"],
            hidden_size=params["lstm_units"],
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(dr)
        lstm2_units = max(16, params["lstm_units"] // 2)
        self.lstm2 = nn.LSTM(
            input_size=params["lstm_units"] * 2,
            hidden_size=lstm2_units,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(dr)
        self.fc1 = nn.Linear(lstm2_units * 2, params["dense_units"])
        self.fc_dropout = nn.Dropout(dr)
        self.out = nn.Linear(params["dense_units"], 1)

    def forward(self, x):
        # (B, L, F) -> (B, F, L) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        if self.conv1.padding[0] > 0:
            x = x[:, :, :-self.conv1.padding[0]]  # causal crop
        x = self.relu(x)
        x = self.bn1(x)
        x = self.spatial_dropout(x)
        # -> (B, L, Filters) for LSTM
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        # Use FINAL HIDDEN STATE (h_n) — matches training scripts
        _, (h_n, _) = self.lstm2(x)
        h_f = h_n[0, :, :]  # forward hidden state
        h_b = h_n[1, :, :]  # backward hidden state
        x = torch.cat((h_f, h_b), dim=1)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_dropout(x)
        return self.out(x)


class BasisPointScaler:
    """Matches the y_scaler used during training (scale=1000.0)."""
    def __init__(self, scale=10000.0):
        self.scale = scale

    def fit_transform(self, x):
        return x * self.scale

    def transform(self, x):
        return x * self.scale

    def inverse_transform(self, x):
        return x / self.scale


# ============================================================
# Sequence builder — matches create_sequences() in training
# Uses [i - lookback + 1 : i + 1] window (inclusive of current)
# ============================================================
def create_sequences(X_values, y_values, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_values)):
        X_seq.append(X_values[i - lookback + 1: i + 1, :])
        y_seq.append(y_values[i])
    return np.array(X_seq), np.array(y_seq)


def evaluate_asset(asset_name, model_dir, reports_dir, train_csv, test_csv, target_col):
    print(f"\n--- Evaluating {asset_name} ---")

    # ------------------------------------------------------------------
    # 1. Load raw level data
    # ------------------------------------------------------------------
    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)

    # Parse dates so we can align returns back to level prices
    for df in [df_train, df_test]:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # ------------------------------------------------------------------
    # 2. Build returns on the FULL combined frame (train + test)
    #    This exactly mirrors the training preprocessing pipeline:
    #      pct_change → lag1/lag2 → dropna → shift target by -1
    # ------------------------------------------------------------------
    full_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    numeric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()

    returns_df = full_df[numeric_cols].pct_change().replace([np.inf, -np.inf], np.nan)

    # Carry the level price alongside the returns frame so we can
    # recover the correct P_t anchor after all the row drops.
    returns_df["_level_price"] = full_df[target_col].values

    # Add explicit lag features — must match training
    for col in numeric_cols:
        returns_df[f"{col}_lag1"] = returns_df[col].shift(1)
        returns_df[f"{col}_lag2"] = returns_df[col].shift(2)

    returns_df = returns_df.dropna()

    # Shift target to get next-day return label
    TARGET_NAME = "target_t_plus_1"
    returns_df[TARGET_NAME] = returns_df[target_col].shift(-1)

    # Also keep the NEXT day's level price for ground-truth reconstruction
    returns_df["_next_level_price"] = returns_df["_level_price"].shift(-1)

    # Drop the last row (NaN target / next price)
    returns_df = returns_df.dropna(subset=[TARGET_NAME, "_next_level_price"])

    # ------------------------------------------------------------------
    # 3. Chronological split back into train / test returns
    #    We know df_test has N rows; after pct_change + 2 lags the very
    #    first row of df_train is consumed, so test rows in returns_df
    #    are the last len(df_test) rows whose TARGET is not NaN.
    # ------------------------------------------------------------------
    n_test = len(df_test)
    test_returns_df  = returns_df.tail(n_test).copy()
    train_returns_df = returns_df.iloc[:-n_test].copy()

    # ------------------------------------------------------------------
    # 4. Load model artifacts
    # ------------------------------------------------------------------
    params_file = os.path.join(reports_dir, f"{asset_name.lower()}_best_params_optimized.json")
    with open(params_file, "r") as f:
        params = json.load(f)

    seed0_path = os.path.join(model_dir, "seed_0")
    with open(os.path.join(seed0_path, "model_metadata.json"), "r") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_cols"]
    lookback     = metadata["lookback"]

    with open(os.path.join(seed0_path, "x_scaler.pkl"), "rb") as f:
        x_scaler = pickle.load(f)
    with open(os.path.join(seed0_path, "y_scaler.pkl"), "rb") as f:
        y_scaler = pickle.load(f)

    # ------------------------------------------------------------------
    # 5. Build sequences
    #    Prepend the last `lookback` train rows as context window,
    #    then slice exactly `n_test` sequences from the test block.
    # ------------------------------------------------------------------
    context_X = train_returns_df[feature_cols].tail(lookback)
    context_y = train_returns_df[[TARGET_NAME]].tail(lookback)

    test_X = test_returns_df[feature_cols].copy()
    test_y = test_returns_df[[TARGET_NAME]].copy()

    combined_X = pd.concat([context_X, test_X], axis=0, ignore_index=True)
    combined_y = pd.concat([context_y, test_y], axis=0, ignore_index=True)

    # Scale using the saved scalers (no refit)
    X_scaled = x_scaler.transform(combined_X)
    y_scaled = y_scaler.transform(combined_y.values)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled.flatten(), lookback)

    # Mask any NaN targets (shouldn't occur, but safety net)
    valid_mask = ~np.isnan(y_seq)
    X_seq = X_seq[valid_mask]
    y_seq = y_seq[valid_mask]

    # ------------------------------------------------------------------
    # 6. Ensemble inference across all trained seeds
    # ------------------------------------------------------------------
    seeds = [0, 1, 2, 7, 11, 17, 21, 42, 99, 123]
    all_preds = []

    for seed in seeds:
        seed_dir   = os.path.join(model_dir, f"seed_{seed}")
        model_path = os.path.join(seed_dir, f"cnn_bilstm_seed{seed}.pth")
        if not os.path.exists(model_path):
            continue

        m = CNN_BiLSTM((lookback, len(feature_cols)), params)
        m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        m.eval()

        with torch.no_grad():
            preds = m(torch.tensor(X_seq, dtype=torch.float32)).numpy().flatten()
        all_preds.append(preds)

    if not all_preds:
        print(f"  [ERROR] No seed model files found in {model_dir}")
        return {}

    ensemble_pred_scaled = np.mean(all_preds, axis=0)
    print(f"  Ensemble size: {len(all_preds)} seeds")

    # ------------------------------------------------------------------
    # 7. Inverse-transform to get predicted & true RETURNS
    # ------------------------------------------------------------------
    y_pred_return = y_scaler.inverse_transform(ensemble_pred_scaled.reshape(-1, 1)).flatten()
    y_true_return = y_scaler.inverse_transform(y_seq.reshape(-1, 1)).flatten()

    # ------------------------------------------------------------------
    # 8. Reconstruct PRICE LEVELS
    #
    #    The model predicts r_{t+1} = (P_{t+1} - P_t) / P_t
    #    So:  P_{t+1}_pred = P_t * (1 + r_pred)
    #         P_{t+1}_true = the actual next-day level price
    #
    #    We stored _level_price and _next_level_price alongside
    #    the returns — use them directly for a leak-free reconstruction.
    # ------------------------------------------------------------------
    # test_returns_df is already trimmed to exactly the test window
    # valid_mask aligns with rows from the context-stripped sequences
    P_t      = test_returns_df["_level_price"].values[valid_mask]
    P_t_plus1_true = test_returns_df["_next_level_price"].values[valid_mask]

    price_pred = P_t * (1.0 + y_pred_return)
    price_true = P_t_plus1_true  # ground truth next-day price

    # ------------------------------------------------------------------
    # 9. Metrics
    # ------------------------------------------------------------------
    r2      = r2_score(price_true, price_pred)
    rmse    = np.sqrt(mean_squared_error(price_true, price_pred))
    mae     = mean_absolute_error(price_true, price_pred)
    dir_acc = np.mean(np.sign(y_pred_return) == np.sign(y_true_return))

    # Laziness report
    corr_actual = np.corrcoef(price_pred, price_true)[0, 1]
    corr_lag1   = np.corrcoef(price_pred, P_t)[0, 1]
    pred_return_mag = np.mean(np.abs(y_pred_return))
    true_return_mag = np.mean(np.abs(y_true_return))

    print(f"Final Results for {asset_name}:")
    print(f"  R2 (Price):       {r2:.4f}")
    print(f"  RMSE (Price):     {rmse:.4f}")
    print(f"  MAE (Price):      {mae:.4f}")
    print(f"  Dir Accuracy:     {dir_acc:.4f}")
    print(f"  --- Laziness Report ---")
    print(f"  Corr(Pred_t+1, Actual_t+1): {corr_actual:.4f}")
    lazy_tag = "(LAZY — model echoes today)" if corr_lag1 > corr_actual else "(PROACTIVE)"
    print(f"  Corr(Pred_t+1, Actual_t):   {corr_lag1:.4f} {lazy_tag}")
    print(f"  Avg Pred Return Mag:  {pred_return_mag:.6f}")
    print(f"  Avg True Return Mag:  {true_return_mag:.6f}")

    # ------------------------------------------------------------------
    # 10. Plot — predicted price vs actual next-day price
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    xs = np.arange(len(price_true))
    plt.plot(xs, price_true, label="Actual Price (t+1)", color="black", linewidth=1.2)
    plt.plot(xs, price_pred, label="Ensemble Predicted (t+1)", color="orange",
             linestyle="--", linewidth=1.2)
    plt.title(
        f"{asset_name} Ensemble Test (Extended Set)\n"
        f"R²={r2:.4f} | RMSE={rmse:.2f} | Dir Acc={dir_acc:.2%} | n_seeds={len(all_preds)}"
    )
    plt.xlabel("Test Step Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("reports", exist_ok=True)
    plot_path = f"reports/{asset_name.lower()}_extended_test_ensemble.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Plot saved to {plot_path}")

    return {
        "r2": r2, "rmse": rmse, "mae": mae, "dir_acc": dir_acc,
        "corr_actual": corr_actual, "corr_lag1": corr_lag1,
        "pred_return_mag": pred_return_mag, "true_return_mag": true_return_mag,
    }


if __name__ == "__main__":
    evaluate_asset(
        "Gold",
        "models/gold_train_only_retrained_v2",
        "reports/gold_train_only_retrained_v2",
        "df_gold_dataset_gepu_extended_train.csv",
        "df_gold_dataset_gepu_extended_test.csv",
        "Gold_Futures",
    )
    evaluate_asset(
        "Silver",
        "models/silver_train_only_retrained_v2",
        "reports/silver_train_only_retrained_v2",
        "silver_RRL_interpolate_extended_train.csv",
        "silver_RRL_interpolate_extended_test.csv",
        "Silver_Futures",
    )
