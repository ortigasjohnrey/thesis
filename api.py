import os
import json
import pickle
import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_squared_error, r2_score
from copy import deepcopy
from technical_indicators import calculate_indicators, calculate_indicators_v7

app = FastAPI(title="Rapid-Pivot Forecasting Simulation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SCALER DEFINITION ---
class BasisPointScaler:
    def __init__(self, scale=100.0):
        self.scale = scale
    def fit_transform(self, x):
        return x * self.scale
    def transform(self, x):
        return x * self.scale
    def inverse_transform(self, x):
        return x / self.scale

# --- LOSS FUNCTION ---
class ProactiveDirectionalLoss(nn.Module):
    def __init__(self, hinge_weight=60.0, anti_lag_weight=40.0, spread_weight=20.0, shadow_weight=150.0):
        super().__init__()
        self.huber = nn.HuberLoss()
        self.hinge_weight = hinge_weight
        self.anti_lag_weight = anti_lag_weight
        self.spread_weight = spread_weight
        self.shadow_weight = shadow_weight
        
    def forward(self, pred, target, is_lively=False, last_target=None):
        loss_huber = self.huber(pred, target)
        target_sign = torch.sign(target)
        
        # Adaptive Weights if in Lively Regime (to combat laziness)
        hinge_w = self.hinge_weight
        anti_lag_w = self.anti_lag_weight * (2.5 if is_lively else 1.0)
        spread_w = self.spread_weight * (2.5 if is_lively else 1.0)
        
        loss_hinge = torch.mean(torch.relu(0.5 - pred * target_sign))
        threshold = 0.4 if is_lively else 0.2
        loss_anti_lag = torch.mean(torch.relu(threshold - torch.abs(pred)))
        
        # --- SHADOW BREAKER: Penalize "Shadowing" the previous day ---
        loss_shadow = torch.zeros(1).to(pred.device)
        if last_target is not None:
            # last_target is the return from t-1. 
            # If pred is too close to last_target, it's "Shadowing" (Retracing).
            # We want pred and last_target to be different!
            # Using a wider radius (0.5) to force a more proactive gap from yesterday.
            loss_shadow = torch.mean(torch.relu(0.5 - torch.abs(pred - last_target)))
        
        # Stability: Handle zero-variance batches to prevent NaN
        pred_std = torch.std(pred) if pred.size(0) > 1 else torch.zeros(1).to(pred.device)
        target_std = torch.std(target) if target.size(0) > 1 else torch.zeros(1).to(target.device)
        
        if pred_std > 1e-4:
            loss_spread = torch.relu(target_std / pred_std - 1.0)
        else:
            loss_spread = torch.zeros(1).to(pred.device)
            
        return loss_huber + hinge_w*loss_hinge + anti_lag_w*loss_anti_lag + spread_w*loss_spread + self.shadow_weight*loss_shadow

# --- MODEL DEFINITION ---
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.key = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / (np.sqrt(q.size(-1)) + 1e-6)
        attn_output = torch.bmm(self.softmax(scores), v)
        return attn_output, None

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, filters=64, kernel_size=3, n_layers=2, dropout=0.3):
        super(CNN_BiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(dropout)
        self.lstm = nn.LSTM(filters, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = SelfAttention(hidden_dim)
        self.fc, self.out = nn.Linear(hidden_dim * 2, 64), nn.Linear(64, 1)
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x.transpose(1, 2)))).transpose(1, 2))
        x, _ = self.lstm(x)
        attn_out, _ = self.attention(x)
        return self.out(self.relu(self.fc(torch.mean(attn_out, dim=1))))

# --- ASSET CONFIG ---
ASSET_CONFIG = {
    "gold": {
        "train_csv": "df_gold_dataset_gepu_extended_train.csv",
        "test_csv": "df_gold_dataset_gepu_extended_test.csv",
        "lively_test_csv": "df_gold_dataset_gepu_extended_lively.csv",
        "best_params": "models/gold_RRL_interpolate/best_params.json",
        "model_dir": "models/gold_RRL_interpolate",
        "seeds": [0, 1, 2, 42, 99, 123],
        "target_col": "Gold_Futures",
        "dataset_label": "Gold Rapid-Pivot Engine",
        "features": ['Silver_Futures', 'Crude_Oil_Futures', 'UST10Y_Treasury_Yield', 'gepu', 'DFF', 'gpr_daily'],
        "tech_cols": ['EMA_Fast', 'EMA_Slow', 'RSI_7', 'MACD_Flash', 'MACD_Signal', 'MACD_Hist', 'BB_Width', 'ROC_2']
    },
    "silver": {
        "train_csv": "silver_RRL_interpolate_extended_train.csv",
        "test_csv": "silver_RRL_interpolate_extended_test.csv",
        "lively_test_csv": "silver_RRL_interpolate_extended_lively.csv",
        "best_params": "models/silver_RRL_interpolate/best_params.json",
        "model_dir": "models/silver_RRL_interpolate",
        "seeds": [0, 1, 2, 42, 99, 123],
        "target_col": "Silver_Futures",
        "dataset_label": "Silver Rapid-Pivot Engine",
        "features": ['Gold_Futures', 'US30', 'SnP500', 'NASDAQ_100', 'USD_index'],
        "tech_cols": ['EMA_10', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Width', 'ROC_5']
    }
}

STATE_FILE = "simulation_state.json"
state = {
    "gold": {
        "test_idx": 0, "dataset_mode": "lively", "models": [], "base_weights": [], "x_scaler": None, "y_scaler": None, 
        "lookback": None, "params": None, "current_date": None, "history": {}, "diagnostic_logs": [],
        "seed_errors": {s: [] for s in [0, 1, 2, 42, 99, 123]}, 
        "data_split": None, "test_df": None, "train_df": None, "forecast_rows": None
    },
    "silver": {
        "test_idx": 0, "dataset_mode": "lively", "models": [], "base_weights": [], "x_scaler": None, "y_scaler": None, 
        "lookback": None, "params": None, "current_date": None, "history": {}, "diagnostic_logs": [],
        "seed_errors": {s: [] for s in [0, 1, 2, 42, 99, 123]},
        "data_split": None, "test_df": None, "train_df": None, "forecast_rows": None
    }
}

def save_runtime_state():
    payload = {}
    for asset, st in state.items():
        payload[asset] = {
            "current_date": st["current_date"].isoformat() if st["current_date"] else None,
            "test_idx": st["test_idx"],
            "dataset_mode": st["dataset_mode"],
            "seed_errors": {str(k): [float(v) for v in l] for k,l in st["seed_errors"].items()},
            "history": {date_key: float(pred) for date_key, pred in st["history"].items()},
            "diagnostic_logs": st["diagnostic_logs"]
        }
    with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)

def load_runtime_state():
    if not os.path.exists(STATE_FILE): return
    with open(STATE_FILE, "r") as f:
        data = json.load(f)
        for asset in state:
            if asset in data:
                st = state[asset]
                st["current_date"] = pd.to_datetime(data[asset]["current_date"]).date() if data[asset].get("current_date") else None
                st["test_idx"] = data[asset].get("test_idx", 0)
                st["seed_errors"] = {int(k): v for k, v in data[asset].get("seed_errors", {}).items()}
                st["history"] = data[asset].get("history", {})
                st["diagnostic_logs"] = data[asset].get("diagnostic_logs", [])
                st["dataset_mode"] = data[asset].get("dataset_mode", "lively")

def synchronize_visual_logs(asset, end_idx):
    """Rebuild the visual history logs purely for the Price Chart."""
    st = state[asset]
    # We do NOT wipe diagnostic_logs here anymore as retrain_on_revealed_day handles them
    st["history"] = {}
    
    # Simple pass to populate history
    for i in range(end_idx):
        # 1. Get prediction as it was for this day
        context_df = pd.concat([st["train_df"], st["test_df"].iloc[:i]], ignore_index=True)
        pred_info = predict_from_context_frame(asset, context_df)
        forecast_date = str(st["test_df"].iloc[i]["Date_obj"])
        st["history"][forecast_date] = pred_info["predicted_price"]
        
    save_runtime_state()

def set_simulation_date_state(asset, selected_date):
    st = state[asset]
    test_df = st["test_df"]
    old_test_idx = st["test_idx"]
    new_test_idx = int((test_df["Date_obj"] < selected_date).sum())
    
    # 1. Prediction and Metric Log Replay (TOTAL WIPE for Synchronicity)
    reset_models_to_base(asset)
    st["seed_errors"] = {s: [] for s in st["seed_errors"]}
    st["history"] = {}
    st["diagnostic_logs"] = []
    
    # Full reset and replay chronologically up to the new target
    replay_training_range(asset, 0, new_test_idx)
    
    # Post-Jump Sync: Re-populate any missing history points
    synchronize_visual_logs(asset, new_test_idx)
    
    st["current_date"] = selected_date
    st["test_idx"] = new_test_idx
    save_runtime_state()
    return {"status": "success", "day": st["test_idx"]}

def reset_models_to_base(asset):
    st = state[asset]
    for i, model in enumerate(st["models"]): model.load_state_dict(st["base_weights"][i])

def get_training_batch(asset, day_idx, batch_size=10):
    """Collect a batch of context windows and targets for the last N days."""
    st = state[asset]
    config = ASSET_CONFIG[asset]
    lookback = st["lookback"]
    all_features = config["features"] + config["tech_cols"]
    
    start_i = max(0, day_idx - batch_size + 1)
    X_list, y_list, l_list = [], [], []
    
    for i in range(start_i, day_idx + 1):
        # Window ending at i-1
        temp_df = pd.concat([st["train_df"], st["test_df"].iloc[:i+1]], ignore_index=True)
        if asset == "gold":
            df_inds = calculate_indicators(temp_df, config["target_col"])
        else:
            df_inds = calculate_indicators_v7(temp_df, config["target_col"])
            
        # Feature Window
        try:
            X_s = st["x_scaler"].transform(df_inds[all_features].iloc[-lookback-1:-1])
        except ValueError:
            X_s = df_inds[all_features].iloc[-lookback-1:-1].values * 100.0
            
        # Target Point
        act_today = float(temp_df.iloc[-1][config["target_col"]])
        prev_p = float(temp_df.iloc[-2][config["target_col"]])
        act_ret = (act_today - prev_p) / (prev_p + 1e-8)
        y_scaled = st["y_scaler"].transform(np.array([[act_ret]])).item()
        
        # Last Target (Shadow)
        l_scaled = 0.0
        if i > 0:
            p2_p = float(temp_df.iloc[-3][config["target_col"]])
            l_ret = (prev_p - p2_p) / (p2_p + 1e-8)
            l_scaled = st["y_scaler"].transform(np.array([[l_ret]])).item()
            
        X_list.append(X_s)
        y_list.append([y_scaled])
        l_list.append([l_scaled])
        
    return (torch.tensor(np.array(X_list), dtype=torch.float32).to(device),
            torch.tensor(np.array(y_list), dtype=torch.float32).to(device),
            torch.tensor(np.array(l_list), dtype=torch.float32).to(device))

def replay_training_range(asset, start_idx, end_idx):
    for i in range(start_idx, end_idx): retrain_on_revealed_day(asset, i)

def retrain_on_revealed_day(asset, day_idx):
    config = ASSET_CONFIG[asset]
    st = state[asset]
    total_df = pd.concat([st["train_df"], st["test_df"].iloc[:day_idx+1]], ignore_index=True)
    lookback = st["lookback"]
    
    all_features = config["features"] + config["tech_cols"]
    
    if asset == "gold":
        full_df_inds = calculate_indicators(total_df, config["target_col"])
    else:
        full_df_inds = calculate_indicators_v7(total_df, config["target_col"])
        
    actual_today = float(total_df.iloc[-1][config["target_col"]])
    prev_price = float(total_df.iloc[-2][config["target_col"]])
    actual_return = (actual_today - prev_price) / (prev_price + 1e-8)
    
    # 1. Evaluation (Check directional hit before retraining)
    # Robust Scaling Fallback
    try:
        X_s = st["x_scaler"].transform(full_df_inds[all_features].iloc[-lookback-1:-1])
    except ValueError:
        X_s = full_df_inds[all_features].iloc[-lookback-1:-1].values * 100.0
        
    X_t = torch.tensor(X_s, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        seed_preds = []
        for model in st["models"]:
            p_s = model(X_t).cpu().numpy().item()
            p_r = st["y_scaler"].inverse_transform(np.array([[p_s]])).item()
            seed_preds.append(p_r)
        
        avg_pred_ret = np.mean(seed_preds)
        ensemble_hit = (np.sign(avg_pred_ret) == np.sign(actual_return))

    # 2. Retraining
    y_scaled = st["y_scaler"].transform(np.array([[actual_return]])).item()
    y_t = torch.tensor([[y_scaled]], dtype=torch.float32).to(device)
    
    # 1. Detection: Is the model "Retracing" (Shadowing)?
    prev_logs = st["diagnostic_logs"]
    shadow_trigger = False
    if len(prev_logs) >= 5:
        preds = np.array([l["pred_ret"] for l in prev_logs[-10:]])
        actuals = np.array([l["actual_ret"] for l in prev_logs[-10:]])
        if len(preds) >= 5:
            lag_corr = np.corrcoef(preds[1:], actuals[:-1])[0,1]
            if lag_corr > 0.6: 
                shadow_trigger = True

    # 2. Diversity Batch Collection
    X_batch, y_batch, l_batch = get_training_batch(asset, day_idx, batch_size=10)
    
    # 3. Retraining with Entropy Injection
    num_epochs = 2 if ensemble_hit else 4
    if shadow_trigger: num_epochs = 20 # Full brain reset
    
    is_lively = (st.get("dataset_mode") == "lively")
    lr_mult = 5.0 if shadow_trigger else 1.0
    criterion = ProactiveDirectionalLoss()
    final_loss = 0.0
    
    for model in st["models"]:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(st["params"]["lr"])*0.5 * lr_mult)
        for _ in range(num_epochs):
            optimizer.zero_grad()
            # ENTROPY INJECTION: Add 0.5% noise to inputs to prevent level-obsession
            noise = torch.randn_like(X_batch) * 0.005
            output = model(X_batch + noise)
            loss = criterion(output, y_batch, is_lively=is_lively, last_target=l_batch)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        model.eval()
    
    # 3. History & Diagnostic Logging
    # STABILIZATION: Move from fixed 0.6x to ADAPTIVE DAMPER
    pred_inv_ret = np.mean(seed_preds)
    
    # Calculate rolling dir accuracy from last 10 samples for the log
    acc_bonus = 1.0
    logs = st["diagnostic_logs"]
    if len(logs) >= 5:
        # Use existing logs to decide confidence for this new entry
        hits = sum(1 for l in logs[-10:] if l.get("hit", False))
        rolling_acc = hits / len(logs[-10:])
        if rolling_acc > 0.6: acc_bonus = 1.5
        elif rolling_acc < 0.4: acc_bonus = 0.8
            
    final_damper = 0.5 * acc_bonus
    
    # Capture Diagnostic Stats
    mag_err = abs(actual_return - pred_inv_ret) / (abs(actual_return) + 1e-8)
    
    # Calculate current R2 if we have enough history to be meaningful
    running_r2 = -1.0
    if len(st["history"]) > 3:
        # Full R2 calculation based on all returns captured so far
        chrono_keys = sorted(st["history"].keys())
        all_actuals = []
        all_preds = []
        for dk in chrono_keys:
            # We need to find the actual price for each historical date we predicted
            match = st["test_df"][st["test_df"]["Date_obj"] == datetime.date.fromisoformat(dk)]
            if not match.empty:
                # Find the previous day's price to get the return
                p_idx = int(match.index[0])
                if p_idx > 0:
                    prev_p = float(st["test_df"].iloc[p_idx-1][config["target_col"]])
                    act_p = float(match.iloc[0][config["target_col"]])
                    all_actuals.append((act_p - prev_p) / (prev_p + 1e-8))
                    all_preds.append((st["history"][dk] - prev_p) / (prev_p + 1e-8))
        
        if len(all_actuals) > 2:
            aa = np.array(all_actuals)
            ap = np.array(all_preds)
            ssr = np.sum((aa - ap)**2)
            sst = np.sum((aa - np.mean(aa))**2)
            running_r2 = round(float(1 - ssr/(sst + 1e-9)), 3)

    # Volatility Check for the log
    vol_scaler = 4.0 if st.get("dataset_mode") == "lively" else 1.0
    
    # Shadow Factor for the log
    preds_arr = np.array([l["pred_ret"] for l in logs[-10:]] + [pred_inv_ret])
    acts_arr = np.array([l["actual_ret"] for l in logs[-10:]] + [actual_return])
    current_shadow = 0.0
    if len(preds_arr) >= 5:
        current_shadow = round(float(np.corrcoef(preds_arr[1:], acts_arr[:-1])[0,1]), 3)

    st["is_flush"] = shadow_trigger
    log_entry = {
        "date": str(st["test_df"].iloc[day_idx]["Date_obj"]),
        "loss": round(float(final_loss), 6),
        "actual_ret": round(float(actual_return), 5),
        "pred_ret": round(float(pred_inv_ret), 5),
        "hit": bool(ensemble_hit),
        "mag_err": round(float(mag_err), 3),
        "shadow": current_shadow,
        "is_flush": shadow_trigger,
        "r2": running_r2,
        "confidence": round(final_damper, 3),
        "vol_mult": round(vol_scaler, 1),
        "mode": st["dataset_mode"]
    }
    st["diagnostic_logs"].append(log_entry)
    if len(st["diagnostic_logs"]) > 50: st["diagnostic_logs"].pop(0)

    # 4. Final Calculation with Adaptive Damper
    damped_price = prev_price * (1.0 + pred_inv_ret * final_damper)
    st["history"][str(st["test_df"].iloc[day_idx]["Date_obj"])] = round(damped_price, 6)

def get_adaptive_weights(asset):
    st = state[asset]
    config = ASSET_CONFIG[asset]
    inv_errors = []
    for seed in config["seeds"]:
        e_list = st["seed_errors"].get(seed, [])
        avg_e = np.mean(e_list) if e_list else 1.0
        inv_errors.append(1.0 / (avg_e + 1e-6))
    weights = np.array(inv_errors) / np.sum(inv_errors)
    return weights

def predict_from_context_frame(asset, context_df):
    config = ASSET_CONFIG[asset]
    st = state[asset]
    lookback = st["lookback"]
    
    # 1. Compute Full Context Indicators (Asset Specific)
    if asset == "gold":
        full_df_inds = calculate_indicators(context_df, config["target_col"])
    else:
        full_df_inds = calculate_indicators_v7(context_df, config["target_col"])
    
    abs_last_price = float(context_df[config["target_col"]].iloc[-1])
    
    numeric_df = full_df_inds[config["features"]].tail(lookback + 1).copy()
    for c in config["features"]: numeric_df[c] = pd.to_numeric(numeric_df[c], errors="coerce")
    returns_df = numeric_df.pct_change().dropna()
    
    # Append latest indicators
    for col in config["tech_cols"]:
        returns_df[col] = full_df_inds[col].iloc[-lookback:]

    all_features = config["features"] + config["tech_cols"]
    
    if len(returns_df) < lookback: return {"error": "Insufficient history"}
    
    # Robust Scaling for 'Blind-History' mode
    try:
        X_s = st["x_scaler"].transform(returns_df[all_features])
    except ValueError:
        # Fallback to Universal Percentage Scaling (100x) if columns mismatch
        X_s = returns_df[all_features].values * 100.0
        
    X_t = torch.tensor(X_s, dtype=torch.float32).unsqueeze(0).to(device)
    all_preds = []
    with torch.no_grad():
        for model in st["models"]: all_preds.append(model(X_t).cpu().numpy().item())
    
    weights = get_adaptive_weights(asset)
    weighted_scaled_pred = np.average(all_preds, weights=weights)
    pred_return = st["y_scaler"].inverse_transform(np.array([[weighted_scaled_pred]])).item()
    
    # ADAPTIVE DAMPER: 0.5x BASE + ACCURACY BONUS
    # Calculate rolling dir accuracy from last 10 samples
    acc_bonus = 1.0
    logs = st["diagnostic_logs"]
    if len(logs) >= 5:
        hits = sum(1 for l in logs[-10:] if l.get("hit", False))
        rolling_acc = hits / len(logs[-10:])
        if rolling_acc > 0.6:
            acc_bonus = 1.5 # Boost to ~0.75x-0.9x if we are on a streak
        elif rolling_acc < 0.4:
            acc_bonus = 0.8 # Tighten to 0.4x if the model is failing
            
    # REGIME SCALER: Match the volatility of the Lively GAN (Calibrated Post-Thaw)
    vol_scaler = 1.5 if st.get("dataset_mode") == "lively" else 1.0
    final_damper = 0.5 * acc_bonus * vol_scaler
    
    return {
        "predicted_price": round(abs_last_price * (1.0 + pred_return * final_damper), 6),
        "last_train_date": context_df.iloc[-1]["Date"].strftime("%Y-%m-%d"),
        "adaptive_weights": [round(float(w), 3) for w in weights],
        "confidence": round(final_damper, 3),
        "is_flush": st.get("is_flush", False),
        "target_scaled": round(weighted_scaled_pred, 3) # Added for diagnostics
    }

def load_models():
    for asset, config in ASSET_CONFIG.items():
        with open(config["best_params"], "r") as f: params = json.load(f)
        state[asset].update({"params": params, "lookback": params["lookback"]})
        with open(os.path.join(config["model_dir"], "scaler_X.pkl"), "rb") as f: state[asset]["x_scaler"] = pickle.load(f)
        with open(os.path.join(config["model_dir"], "scaler_y.pkl"), "rb") as f: state[asset]["y_scaler"] = pickle.load(f)
        state[asset]["models"] = []
        state[asset]["base_weights"] = []
        all_features_count = len(config["features"]) + len(config["tech_cols"])
        for seed in config["seeds"]:
            # Hidden dim might be under different names in JSON, using a safe fallback
            h_dim = params.get("hidden_dim", params.get("lstm_units", 128))
            model = CNN_BiLSTM(all_features_count, h_dim, params["filters"], params["kernel_size"]).to(device)
            try:
                model.load_state_dict(torch.load(os.path.join(config["model_dir"], f"{asset}_model_seed_{seed}.pth"), map_location=device, weights_only=True))
                print(f"Loaded existing weights for {asset} seed {seed}")
            except Exception as e:
                # This trigger is intentional for the Blind-History upgrade
                print(f"Neural Reconstruction Required for {asset} (Seed {seed}): Dimensional mismatch. Starting with fresh 'Detective' weights.")
            
            model.eval()
            state[asset]["models"].append(model)
            state[asset]["base_weights"].append(deepcopy(model.state_dict()))
        
        train_df = pd.read_csv(config["train_csv"])
        
        # Load file based on mode
        mode = state[asset].get("dataset_mode", "lively")
        test_file = config["lively_test_csv"] if mode == "lively" else config["test_csv"]
        test_df = pd.read_csv(test_file)
        
        for df in [train_df, test_df]:
            df["Date"] = pd.to_datetime(df["Date"])
            df["Date_obj"] = df["Date"].dt.date
        state[asset].update({"train_df": train_df, "test_df": test_df, "current_date": test_df.iloc[0]["Date_obj"]})

load_models()

@app.get("/")
def get_dashboard(): return FileResponse("dashboard.html")

@app.get("/api/status/{asset}")
def get_status(asset: str):
    config = ASSET_CONFIG[asset]
    st = state[asset]
    idx = st["test_idx"]
    test_df = st["test_df"]
    train_df = st["train_df"]

    if idx >= len(test_df): return {"error": "Finished"}

    # Build context for prediction
    context_df = pd.concat([train_df, test_df.iloc[:idx]], ignore_index=True)
    pred_info = predict_from_context_frame(asset, context_df)

    # --- Last training price (Define early for history_log logic) ---
    last_train_price = float(context_df.iloc[-1][config["target_col"]]) if len(context_df) > 0 else None
    last_train_date = str(context_df.iloc[-1]["Date_obj"]) if len(context_df) > 0 else None

    # --- Initialize metrics to None (prevents UnboundLocalError) ---
    rolling_rmse, daily_reliability, rolling_dir_acc = None, None, None
    overall_rmse, overall_r2, overall_dir_acc = None, None, None

    # --- history_log: compare past predictions vs actuals ---
    history_log = []
    for date_str, pred_price in st["history"].items():
        date_obj = datetime.date.fromisoformat(date_str)
        # Find actual price in test_df
        actual_row = test_df[test_df["Date_obj"] == date_obj]
        if not actual_row.empty:
            actual_price = float(actual_row.iloc[0][config["target_col"]])
            history_log.append({
                "date": date_str,
                "actual": actual_price,
                "predicted": pred_price,
                "context_end_date": date_str,
                "error_pct": round(abs(actual_price - pred_price) / (actual_price + 1e-8) * 100, 2),
                "hit": bool(np.sign(actual_price - last_train_price) == np.sign(pred_price - last_train_price)) if last_train_price else True
            })
    history_log.sort(key=lambda x: x["date"], reverse=True)

    # --- Rolling metrics (Calculated on DAILY RETURNS for stability) ---
    if len(history_log) >= 2:
        # Chronological sort for diff calculations
        chrono_log = sorted(history_log, key=lambda x: x["date"])
        
        # Calculate Returns
        actual_prices = np.array([r["actual"] for r in chrono_log])
        pred_prices = np.array([r["predicted"] for r in chrono_log])
        
        # Prices at T-1 for the first prediction in the log
        actual_rets = np.diff(actual_prices) / (actual_prices[:-1] + 1e-8)
        pred_rets = (pred_prices[1:] - actual_prices[:-1]) / (actual_prices[:-1] + 1e-8)
        
        if len(actual_rets) > 0:
            # --- ROLLING WINDOW (Last 20 Days for RMSE/Accuracy) ---
            window = 20
            actual_rets_w = actual_rets[-window:]
            pred_rets_w = pred_rets[-window:]
            rolling_rmse = round(float(np.sqrt(np.mean((actual_rets_w - pred_rets_w) ** 2))), 5)
            rolling_dir_acc = round(float(np.mean(np.sign(actual_rets_w) == np.sign(pred_rets_w))), 3)

            # --- DAILY METRIC (Latest Day Alone) ---
            latest_actual_ret = actual_rets[-1]
            latest_pred_ret = pred_rets[-1]
            # Daily Reliability = 1 - Relative Error
            daily_reliability = round(max(0.0, float(1.0 - abs(latest_actual_ret - latest_pred_ret) / (abs(latest_actual_ret) + 1e-8))), 3)

            # --- OVERALL METRICS (All-Time) ---
            actual_rets_all = actual_rets
            pred_rets_all = pred_rets
            
            overall_rmse = round(float(np.sqrt(np.mean((actual_rets_all - pred_rets_all) ** 2))), 5)
            ss_res_all = np.sum((actual_rets_all - pred_rets_all) ** 2)
            ss_tot_all = np.sum((actual_rets_all - np.mean(actual_rets_all)) ** 2)
            overall_r2 = round(max(-1.0, float(1 - ss_res_all / (ss_tot_all + 1e-9))), 3) if ss_tot_all > 1e-9 else 0.0
            overall_dir_acc = round(float(np.mean(np.sign(actual_rets_all) == np.sign(pred_rets_all))), 3)
            overall_dir_acc = round(float(np.mean(np.sign(actual_rets_all) == np.sign(pred_rets_all))), 3)

    # --- Yesterday stats ---
    yesterday_actual, yesterday_pred = None, None
    if len(history_log) > 0:
        latest = history_log[0]
        yesterday_actual = latest["actual"]
        yesterday_pred = latest["predicted"]

    # --- Last training price ---
    last_train_price = float(context_df.iloc[-1][config["target_col"]]) if len(context_df) > 0 else None
    last_train_date = str(context_df.iloc[-1]["Date_obj"]) if len(context_df) > 0 else None

    # --- Date bounds ---
    min_date = str(test_df.iloc[0]["Date_obj"])
    max_date = str(test_df.iloc[-1]["Date_obj"])

    return {
        "asset": asset,
        "simulation_day": idx,
        "current_date": str(st["current_date"]),
        "forecast_date": str(test_df.iloc[idx]["Date_obj"]),
        "predicted_price": pred_info["predicted_price"],
        "dataset_mode": st["dataset_mode"],
        "diagnostic_logs": st["diagnostic_logs"],
        "adaptive_weights": pred_info["adaptive_weights"],
        "confidence": pred_info.get("confidence", 0.6),
        "engine": "Rapid-Pivot Adaptive v1.2",
        "dataset_label": config["dataset_label"],
        "history_log": history_log,
        "rolling_rmse": rolling_rmse,
        "rolling_r2": daily_reliability,
        "rolling_dir_acc": rolling_dir_acc,
        "overall_rmse": overall_rmse if 'overall_rmse' in locals() else None,
        "overall_r2": overall_r2 if 'overall_r2' in locals() else None,
        "overall_dir_acc": overall_dir_acc if 'overall_dir_acc' in locals() else None,
        "yesterday_actual": yesterday_actual,
        "yesterday_pred": yesterday_pred,
        "last_train_date": last_train_date,
        "last_train_price": last_train_price,
        "min_date": min_date,
        "max_date": max_date,
        "reference_today": str(datetime.date.today()),
    }

@app.post("/api/next_day/{asset}")
def next_day(asset: str):
    config = ASSET_CONFIG[asset]
    st = state[asset]
    idx = st["test_idx"]
    if idx >= len(st["test_df"]): return {"error": "No more data"}

    # 1. Record TODAY's prediction BEFORE we retrain on the revealed actual
    context_df = pd.concat([st["train_df"], st["test_df"].iloc[:idx]], ignore_index=True)
    pred_info = predict_from_context_frame(asset, context_df)
    forecast_date = str(st["test_df"].iloc[idx]["Date_obj"])
    st["history"][forecast_date] = pred_info["predicted_price"]

    # 2. Retrain on the now-revealed actual price for this day
    retrain_on_revealed_day(asset, idx)
    st["test_idx"] += 1
    st["current_date"] = st["test_df"].iloc[st["test_idx"] - 1]["Date_obj"]
    save_runtime_state()
    return {"message": "Rapid-Pivot update complete."}

@app.post("/api/current_date/{asset}")
def set_current_date(asset: str, date: str):
    """Jump simulation to a specific date (resets or fast-forwards as needed)."""
    try:
        selected_date = datetime.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    return set_simulation_date_state(asset, selected_date)

@app.post("/api/use_today/{asset}")
def use_today(asset: str):
    config = ASSET_CONFIG[asset]
    selected_date = datetime.date.today()
    return set_simulation_date_state(asset, selected_date)

@app.post("/api/dataset_mode/{asset}")
def set_dataset_mode(asset: str, mode: str):
    if mode not in ["standard", "lively"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'standard' or 'lively'.")
    
    state[asset]["dataset_mode"] = mode
    # Reset simulation state when mode changes to prevent date-jump bugs
    config = ASSET_CONFIG[asset]
    st = state[asset]
    
    # Reload test_df based on new mode
    test_file = config["lively_test_csv"] if mode == "lively" else config["test_csv"]
    st["test_df"] = pd.read_csv(test_file)
    for df in [st["test_df"]]:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date_obj"] = df["Date"].dt.date
    
    # Reset progress
    st["test_idx"] = 0
    st["history"] = {}
    st["seed_errors"] = {s: [] for s in config["seeds"]}
    st["current_date"] = st["test_df"].iloc[0]["Date_obj"]
    
    save_runtime_state()
    return {"message": f"Switched to {mode} regime. Simulation reset to Day 0.", "mode": mode}

load_runtime_state()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
