import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score

# --- MODEL DEFINITION ---
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_shape, params):
        super(CNN_BiLSTM, self).__init__()
        in_channels = input_shape[1] 
        dr = params["dropout_rate"]
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=params["filters"],
            kernel_size=params["kernel_size"],
            padding=params["kernel_size"] - 1 
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(params["filters"])
        self.spatial_dropout = nn.Dropout1d(p=dr)
        
        self.lstm1 = nn.LSTM(
            input_size=params["filters"],
            hidden_size=params["lstm_units"],
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dr)
        
        lstm2_units = max(16, params["lstm_units"] // 2)
        self.lstm2 = nn.LSTM(
            input_size=params["lstm_units"] * 2,
            hidden_size=lstm2_units,
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(dr)
        
        self.fc1 = nn.Linear(lstm2_units * 2, params["dense_units"])
        self.fc_dropout = nn.Dropout(dr)
        self.out = nn.Linear(params["dense_units"], 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        if self.conv1.padding[0] > 0:
            x = x[:, :, :-self.conv1.padding[0]]
        x = self.relu(x)
        x = self.bn1(x)
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        _, (h_n, c_n) = self.lstm2(x)
        h_f = h_n[0, :, :]
        h_b = h_n[1, :, :]
        x = torch.cat((h_f, h_b), dim=1)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_dropout(x)
        x = self.out(x)
        return x

def check_laziness(asset, model_dir, test_csv, train_csv, target_col):
    with open(os.path.join(model_dir, "seed_42", "model_metadata.json"), "r") as f:
        meta = json.load(f)
    with open(os.path.join(model_dir, "seed_42", "x_scaler.pkl"), "rb") as f:
        x_scaler = pickle.load(f)
    with open(os.path.join(model_dir, "seed_42", "y_scaler.pkl"), "rb") as f:
        y_scaler = pickle.load(f)
    best_params_path = os.path.join("reports", os.path.basename(model_dir), f"{asset.lower()}_best_params_optimized.json")
    with open(best_params_path, "r") as f:
        params = json.load(f)

    df_test = pd.read_csv(test_csv)
    df_train = pd.read_csv(train_csv)
    
    full_df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    numeric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()
    returns_df = full_df[numeric_cols].pct_change().dropna()
    
    test_returns_df = returns_df.tail(len(df_test)).copy()
    feature_cols = meta["feature_cols"]
    lookback = meta["lookback"]
    
    # Prepare inputs
    X_test_raw = test_returns_df[feature_cols].copy()
    # We need context for lookback
    train_tail = returns_df.iloc[:-len(df_test)].tail(lookback)
    X_full = pd.concat([train_tail[feature_cols], X_test_raw], axis=0)
    X_scaled = x_scaler.transform(X_full)
    
    X_seq = []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i-lookback+1 : i+1, :])
    X_seq = np.array(X_seq)
    
    model = CNN_BiLSTM(input_shape=(lookback, len(feature_cols)), params=params)
    model.load_state_dict(torch.load(os.path.join(model_dir, "seed_42", f"cnn_bilstm_seed42.pth"), map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_seq, dtype=torch.float32)).numpy().reshape(-1, 1)
        
    pred_return = y_scaler.inverse_transform(pred_scaled).flatten()
    
    # Absolute prices
    prices = df_test[target_col].values
    last_prices = pd.concat([df_train.tail(1), df_test.iloc[:-1]])[target_col].values
    
    pred_prices = last_prices * (1 + pred_return)
    
    # Check correlations
    corr_with_actual = np.corrcoef(pred_prices, prices)[0,1]
    corr_with_last = np.corrcoef(pred_prices, last_prices)[0,1]
    
    print(f"--- {asset} Laziness Check ---")
    print(f"Correlation (Pred_t+1, Actual_t+1): {corr_with_actual:.6f}")
    print(f"Correlation (Pred_t+1, Actual_t): {corr_with_last:.6f}")
    
    if corr_with_last > corr_with_actual:
        print("Model is LAZY: It correlates more with the last day than with the target.")
    else:
        print("Model has SOME predictive power.")

    # Check average absolute return
    avg_pred_return = np.mean(np.abs(pred_return))
    avg_actual_return = np.mean(np.abs(test_returns_df[target_col].values))
    print(f"Avg Abs Pred Return: {avg_pred_return:.6f}")
    print(f"Avg Abs Actual Return: {avg_actual_return:.6f}")
    if avg_pred_return < 0.1 * avg_actual_return:
        print("Model is TOO VENTURED (Predicting near zero returns consistently).")

if __name__ == "__main__":
    check_laziness("Gold", "models/gold_train_only_retrained_v2", "df_gold_dataset_gepu_extended_test.csv", "df_gold_dataset_gepu_extended_train.csv", "Gold_Futures")
    check_laziness("Silver", "models/silver_train_only_retrained_v2", "silver_RRL_interpolate_extended_test.csv", "silver_RRL_interpolate_extended_train.csv", "Silver_Futures")
