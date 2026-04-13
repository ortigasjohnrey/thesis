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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SHARED COMPONENTS (MUST MATCH api.py) ---

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
        attn_output = torch.bmm(self.softmax(scores), v)
        return attn_output, None

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
        
        # Spread loss only works for batches > 1
        if pred.size(0) > 1:
            pred_std = torch.std(pred) + 1e-6
            target_std = torch.std(target) + 1e-6
            loss_spread = torch.relu(target_std / pred_std - 1.0)
        else:
            loss_spread = 0.0
            
        return loss_h + self.h_w*loss_hinge + self.a_w*loss_anti + self.s_w*loss_spread

# --- SIMULATION ENGINE ---

def simulate_asset(asset_name, config):
    print(f"\n>>> Starting Sequential Simulation for {asset_name.upper()}...")
    
    # Load parameters and data
    with open(config["best_params"], "r") as f: params = json.load(f)
    train_df = pd.read_csv(config["train_csv"])
    test_df  = pd.read_csv(config["test_csv"])
    for df in [train_df, test_df]:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    
    with open(os.path.join(config["model_dir"], "scaler_X.pkl"), "rb") as f: x_scaler = pickle.load(f)
    with open(os.path.join(config["model_dir"], "scaler_y.pkl"), "rb") as f: y_scaler = pickle.load(f)

    # Initialize Ensemble
    models = []
    for seed in config["seeds"]:
        m = CNN_BiLSTM(len(config["features"]), params["lstm_units"], params["filters"], params["kernel_size"], dropout=params["dropout"]).to(device)
        m_path = os.path.join(config["model_dir"], f"{asset_name}_model_seed_{seed}.pth")
        m.load_state_dict(torch.load(m_path, map_location=device, weights_only=True))
        models.append(m)

    context_df = train_df.copy()
    lookback = params["lookback"]
    target_col, features = config["target_col"], config["features"]
    criterion = ProactiveDirectionalLoss()

    # Walk-forward simulation
    results = []
    for i in tqdm(range(len(test_df)), desc=f"{asset_name} Learning"):
        # 1. PREDICT TODAY
        abs_last_price = float(context_df[target_col].iloc[-1])
        numeric_df = context_df[features].tail(lookback + 1).copy()
        for i_col in features: numeric_df[i_col] = pd.to_numeric(numeric_df[i_col], errors="coerce")
        returns_df = numeric_df.pct_change().dropna()
        
        if len(returns_df) < lookback:
            context_df = pd.concat([context_df, test_df.iloc[[i]]], ignore_index=True)
            continue
            
        X_scaled = x_scaler.transform(returns_df[features].iloc[-lookback:])
        X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        ensemble_preds = []
        with torch.no_grad():
            for m in models: ensemble_preds.append(m(X_t).cpu().numpy().item())
        
        pred_return = y_scaler.inverse_transform(np.array([[np.mean(ensemble_preds)]])).item()
        predicted_price = abs_last_price * (1.0 + pred_return)
        
        # 2. REVEAL TODAY
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

        # 3. RETRAIN ON REVEALED DAY
        y_scaled = y_scaler.transform(np.array([[actual_return]])).item()
        y_t = torch.tensor([[y_scaled]], dtype=torch.float32).to(device)
        
        for m_idx, m in enumerate(models):
            m.train()
            opt = torch.optim.Adam(m.parameters(), lr=float(params["lr"])*0.5)
            # Perform 2 epochs of fine-tuning
            for epoch in range(2):
                opt.zero_grad()
                pred = m(X_t)
                loss = criterion(pred, y_t)
                if torch.isnan(loss):
                    raise ValueError(f"NaN Loss at {asset_name} day {i}, model {m_idx}, epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
                opt.step()
            m.eval()

        # Update context
        context_df = pd.concat([context_df, test_df.iloc[[i]]], ignore_index=True)

    # 4. ANALYSIS
    res_df = pd.DataFrame(results)
    rmse = np.sqrt(mean_squared_error(res_df["actual"], res_df["predicted"]))
    r2 = r2_score(res_df["actual"], res_df["predicted"])
    da = np.mean(np.sign(res_df["actual_ret"]) == np.sign(res_df["pred_ret"]))
    
    print(f"\n--- {asset_name.upper()} SEQUENTIAL LEARN RESULTS ---")
    print(f"  Final DA:   {da:.2%}")
    print(f"  Final RMSE: {rmse:.4f}")
    print(f"  Final R2:   {r2:.4f}")
    
    return res_df, {"da": da, "rmse": rmse, "r2": r2}

if __name__ == "__main__":
    ASSET_CONFIG = {
        "gold": {
            "train_csv": "df_gold_dataset_gepu_extended_train.csv",
            "test_csv": "df_gold_dataset_gepu_extended_test.csv",
            "best_params": "models/gold_RRL_interpolate/best_params.json",
            "model_dir": "models/gold_RRL_interpolate",
            "seeds": [0, 1, 2, 42, 99, 123],
            "target_col": "Gold_Futures",
            "features": ['Silver_Futures', 'Crude_Oil_Futures', 'UST10Y_Treasury_Yield', 'gepu', 'DFF', 'gpr_daily', 'Gold_Futures']
        },
        "silver": {
            "train_csv": "silver_RRL_interpolate_extended_train.csv",
            "test_csv": "silver_RRL_interpolate_extended_test.csv",
            "best_params": "models/silver_RRL_interpolate/best_params.json",
            "model_dir": "models/silver_RRL_interpolate",
            "seeds": [0, 1, 2, 42, 99, 123],
            "target_col": "Silver_Futures",
            "features": ['Silver_Futures', 'Gold_Futures', 'US30', 'SnP500', 'NASDAQ_100', 'USD_index']
        }
    }
    
    all_summary = {}
    for asset, cfg in ASSET_CONFIG.items():
        try:
            _, stats = simulate_asset(asset, cfg)
            all_summary[asset] = stats
        except Exception as e:
            print(f"ERROR: Simulation failed for {asset}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n\nFINAL COMPARISON SUMMARY:")
    print(json.dumps(all_summary, indent=2))
