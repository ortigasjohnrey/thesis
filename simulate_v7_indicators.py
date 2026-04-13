import os
import json
import pickle
import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from copy import deepcopy
from tqdm import tqdm
from technical_indicators import calculate_indicators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL ARCHITECTURE (V7 READY) ---

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / (np.sqrt(q.size(-1)) + 1e-6)
        return torch.bmm(self.softmax(scores), v), None

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, filters, kernel_size, dropout=0.3):
        super(CNN_BiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(dropout)
        self.lstm = nn.LSTM(filters, hidden_dim, 2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = SelfAttention(hidden_dim)
        self.fc, self.out = nn.Linear(hidden_dim * 2, 64), nn.Linear(64, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        attn_out, _ = self.attention(x)
        x = torch.mean(attn_out, dim=1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x

class ProactiveDirectionalLoss(nn.Module):
    def __init__(self, h=30.0, a=20.0, s=15.0):
        super().__init__()
        self.huber, self.h_w, self.a_w, self.s_w = nn.HuberLoss(), h, a, s
    def forward(self, pred, target):
        loss_h = self.huber(pred, target)
        loss_hinge = torch.mean(torch.relu(0.5 - pred * torch.sign(target)))
        loss_anti = torch.mean(torch.relu(0.2 - torch.abs(pred)))
        if pred.size(0) > 1:
            loss_spread = torch.relu((torch.std(target)+1e-6)/(torch.std(pred)+1e-6) - 1.0)
        else:
            loss_spread = 0.0
        return loss_h + self.h_w*loss_hinge + self.a_w*loss_anti + self.s_w*loss_spread

# --- V7 SIMULATION ENGINE ---

def simulate_v7(asset_name, config):
    print(f"\n>>> Starting V7 SEQUENTIAL Simulation for {asset_name.upper()}...")
    
    with open(config["best_params"], "r") as f: params = json.load(f)
    train_df = pd.read_csv(config["train_csv"])
    test_df  = pd.read_csv(config["test_csv"])
    for df in [train_df, test_df]:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    
    with open(os.path.join(config["model_dir"], "scaler_X.pkl"), "rb") as f: x_scaler = pickle.load(f)
    with open(os.path.join(config["model_dir"], "scaler_y.pkl"), "rb") as f: y_scaler = pickle.load(f)

    # V8 feature set (must match technical_indicators.py + raw features)
    tech_cols = ['EMA_Fast', 'EMA_Slow', 'RSI_7', 'MACD_Flash', 'MACD_Signal', 'MACD_Hist', 'BB_Width', 'ROC_2']
    all_feature_cols = config["raw_features"] + tech_cols

    models = []
    for seed in config["seeds"]:
        m = CNN_BiLSTM(len(all_feature_cols), params["lstm_units"], params["filters"], params["kernel_size"]).to(device)
        m.load_state_dict(torch.load(os.path.join(config["model_dir"], f"{asset_name}_model_seed_{seed}.pth"), map_location=device, weights_only=True))
        models.append(m)

    context_df = train_df.copy()
    lookback = params["lookback"]
    target_col = config["target_col"]
    criterion = ProactiveDirectionalLoss()
    
    results = []

    for i in tqdm(range(len(test_df)), desc=f"{asset_name} V7-Learning"):
        # 1. COMPUTE TECH INDICATORS ONL-LINE
        # We need enough history for MACD/RSI instability
        full_df_inds = calculate_indicators(context_df, target_col)
        
        # 2. PREDICT TODAY
        abs_last_price = float(context_df[target_col].iloc[-1])
        
        # Prepare feature vector: Returns + Current Indicators
        numeric_context = full_df_inds[config["raw_features"]].tail(lookback + 1).copy()
        for c in config["raw_features"]: numeric_context[c] = pd.to_numeric(numeric_context[c], errors="coerce")
        returns_context = numeric_context.pct_change().dropna()
        
        # Append latest indicators
        for col in tech_cols:
            returns_context[col] = full_df_inds[col].iloc[-lookback:]

        if len(returns_context) < lookback:
            context_df = pd.concat([context_df, test_df.iloc[[i]]], ignore_index=True)
            continue
            
        X_scaled = x_scaler.transform(returns_context[all_feature_cols])
        X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        ensemble_preds = []
        with torch.no_grad():
            for m in models: ensemble_preds.append(m(X_t).cpu().numpy().item())
        
        pred_return = y_scaler.inverse_transform(np.array([[np.mean(ensemble_preds)]])).item()
        predicted_price = abs_last_price * (1.0 + pred_return)
        
        # 3. REVEAL TODAY
        today_row = test_df.iloc[i]
        actual_price = float(today_row[target_col])
        actual_return = (actual_price - abs_last_price) / (abs_last_price + 1e-8)
        
        results.append({
            "date": today_row["Date"],
            "actual": actual_price,
            "predicted": predicted_price,
            "actual_ret": actual_return,
            "pred_ret": pred_return
        })

        # 4. RETRAIN ON REVEALED DAY
        y_scaled = y_scaler.transform(np.array([[actual_return]])).item()
        y_t = torch.tensor([[y_scaled]], dtype=torch.float32).to(device)
        
        for m in models:
            m.train()
            opt = torch.optim.Adam(m.parameters(), lr=float(params["lr"])*0.5)
            for _ in range(2):
                opt.zero_grad()
                loss = criterion(m(X_t), y_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            m.eval()

        context_df = pd.concat([context_df, test_df.iloc[[i]]], ignore_index=True)

    res_df = pd.DataFrame(results)
    rmse = np.sqrt(mean_squared_error(res_df["actual"], res_df["predicted"]))
    da = np.mean(np.sign(res_df["actual_ret"]) == np.sign(res_df["pred_ret"]))
    
    print(f"\n--- {asset_name.upper()} V7 SEQUENTIAL RESULTS ---")
    print(f"  DA:   {da:.2%}")
    print(f"  RMSE: {rmse:.4f}")
    
    return res_df, {"da": da, "rmse": rmse}

if __name__ == "__main__":
    ASSET_CONFIG = {
        "gold": {
            "train_csv": "df_gold_dataset_gepu_extended_train.csv",
            "test_csv": "df_gold_dataset_gepu_extended_test.csv",
            "best_params": "models/gold_RRL_interpolate/best_params.json",
            "model_dir": "models/gold_RRL_interpolate",
            "seeds": [0, 1, 2, 42, 99, 123],
            "target_col": "Gold_Futures",
            "raw_features": ['Silver_Futures', 'Crude_Oil_Futures', 'UST10Y_Treasury_Yield', 'gepu', 'DFF', 'gpr_daily', 'Gold_Futures']
        }
    }
    # Finalize Silver check
    if os.path.exists("models/silver_RRL_interpolate/best_params.json"):
        ASSET_CONFIG["silver"] = {
            "train_csv": "silver_RRL_interpolate_extended_train.csv",
            "test_csv": "silver_RRL_interpolate_extended_test.csv",
            "best_params": "models/silver_RRL_interpolate/best_params.json",
            "model_dir": "models/silver_RRL_interpolate",
            "seeds": [0, 1, 2, 42, 99, 123],
            "target_col": "Silver_Futures",
            "raw_features": ['Silver_Futures', 'Gold_Futures', 'US30', 'SnP500', 'NASDAQ_100', 'USD_index']
        }

    all_summary = {}
    for asset, cfg in ASSET_CONFIG.items():
        _, stats = simulate_v7(asset, cfg)
        all_summary[asset] = stats
    
    print("\n\nFINAL V7 PERFORMANCE SUMMARY:")
    print(json.dumps(all_summary, indent=2))
