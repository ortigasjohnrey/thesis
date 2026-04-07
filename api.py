import os
import json
import pickle
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Forecasting Simulation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# --- STATE AND CONFIG ---
ASSET_CONFIG = {
    "gold": {
        "train_csv": "gold_RRL_interpolate_train.csv",
        "test_csv": "gold_RRL_interpolate_test.csv",
        "best_params": "reports/gold_RRL_interpolate/gold_best_params_optimized.json",
        "model_dir": "models/gold_RRL_interpolate/seed_42",
        "model_pth": "cnn_bilstm_seed42.pth",
        "target_col": "Gold_Futures"
    },
    "silver": {
        "train_csv": "silver_RRL_interpolate_train.csv",
        "test_csv": "silver_RRL_interpolate_test.csv",
        "best_params": "reports/silver_RRL_interpolate/silver_yahoo_best_params.json",
        "model_dir": "models/silver_RRL_interpolate/seed_42",
        "model_pth": "cnn_bilstm_seed42.pth",
        "target_col": "Silver_Futures"
    }
}

state = {
    "gold": {"test_idx": 0, "model": None, "x_scaler": None, "y_scaler": None, "feature_cols": None, "lookback": None, "params": None, "current_date": None, "history": {}},
    "silver": {"test_idx": 0, "model": None, "x_scaler": None, "y_scaler": None, "feature_cols": None, "lookback": None, "params": None, "current_date": None, "history": {}}
}

def load_models():
    for asset, config in ASSET_CONFIG.items():
        # Load best params
        with open(config["best_params"], "r") as f:
            params = json.load(f)
        state[asset]["params"] = params
        
        # Load metadata
        with open(os.path.join(config["model_dir"], "model_metadata.json"), "r") as f:
            meta = json.load(f)
        state[asset]["feature_cols"] = meta["feature_cols"]
        state[asset]["lookback"] = meta["lookback"]
        
        # Load Scalers
        with open(os.path.join(config["model_dir"], "x_scaler.pkl"), "rb") as f:
            state[asset]["x_scaler"] = pickle.load(f)
        with open(os.path.join(config["model_dir"], "y_scaler.pkl"), "rb") as f:
            state[asset]["y_scaler"] = pickle.load(f)
            
        # Initialize model
        model = CNN_BiLSTM(input_shape=(state[asset]["lookback"], len(state[asset]["feature_cols"])), params=params)
        model.load_state_dict(torch.load(os.path.join(config["model_dir"], config["model_pth"]), map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        state[asset]["model"] = model
        
        # Init calendar to row 0 of test
        test_df = pd.read_csv(config["test_csv"])
        if "Date" in test_df.columns:
            state[asset]["current_date"] = pd.to_datetime(test_df.iloc[0]["Date"]).date()

load_models()

def predict_next_day(asset):
    config = ASSET_CONFIG[asset]
    st = state[asset]
    
    df = pd.read_csv(config["train_csv"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    
    cols_to_keep = list(dict.fromkeys(st["feature_cols"] + [config["target_col"]]))
    numeric_df = df[cols_to_keep].copy()
    
    abs_last_price = numeric_df.iloc[-1][config["target_col"]]
    last_date = df.iloc[-1]["Date"].strftime("%Y-%m-%d")
    
    returns_df = numeric_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    
    lookback = st["lookback"]
    if len(returns_df) < lookback:
        return {"error": "Not enough data"}
        
    recent_returns = returns_df[st["feature_cols"]].iloc[-lookback:].copy()
    recent_scaled = st["x_scaler"].transform(recent_returns)
    recent_tensor = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        st["model"].eval()
        pred_scaled = st["model"](recent_tensor).cpu().numpy().reshape(-1, 1)
        
    pred_return = st["y_scaler"].inverse_transform(pred_scaled).item()
    pred_abs = abs_last_price * (1 + pred_return)
    
    return {
        "predicted_price": round(pred_abs, 2),
        "last_train_date": last_date,
        "last_train_price": round(abs_last_price, 2)
    }

@app.get("/")
def get_dashboard():
    return FileResponse("dashboard.html")

@app.get("/api/status/{asset}")
def get_status(asset: str):
    if asset not in ASSET_CONFIG:
        return {"error": "Invalid asset"}
    
    config = ASSET_CONFIG[asset]
    st = state[asset]
    idx = st["test_idx"]
    
    test_df = pd.read_csv(config["test_csv"])
    test_df["Date_obj"] = pd.to_datetime(test_df["Date"]).dt.date
    if idx >= len(test_df):
        return {"error": "Simulation finished, no more test data"}
        
    market_date = test_df.iloc[idx]["Date_obj"]
    current_calendar_date = st["current_date"]
    
    is_market_day = (current_calendar_date == market_date)
    
    pred_info = {}
    if is_market_day:
        pred_info = predict_next_day(asset)
        if pred_info.get("predicted_price"):
            st["history"][str(current_calendar_date)] = pred_info["predicted_price"]
    
    yesterday_actual = None
    yesterday_pred = None
    yesterday_date = None
    
    if idx > 0:
        y_row = test_df.iloc[idx - 1]
        yesterday_date = str(y_row["Date_obj"])
        yesterday_actual = round(y_row[config["target_col"]], 2)
        yesterday_pred = st["history"].get(yesterday_date)
        
    return {
        "asset": asset,
        "simulation_day": idx,
        "current_date": str(current_calendar_date),
        "is_market_day": is_market_day,
        "predicted_price": pred_info.get("predicted_price"),
        "yesterday_date": yesterday_date,
        "yesterday_actual": yesterday_actual,
        "yesterday_pred": yesterday_pred,
        "last_train_price": pred_info.get("last_train_price")
    }

@app.post("/api/next_day/{asset}")
def next_day(asset: str):
    if asset not in ASSET_CONFIG:
        return {"error": "Invalid asset"}
        
    config = ASSET_CONFIG[asset]
    st = state[asset]
    idx = st["test_idx"]
    
    test_df = pd.read_csv(config["test_csv"])
    test_df["Date_obj"] = pd.to_datetime(test_df["Date"]).dt.date
    if idx >= len(test_df):
        return {"error": "No more test data available."}
        
    market_date = test_df.iloc[idx]["Date_obj"]
    current_calendar_date = st["current_date"]
    
    if current_calendar_date == market_date:
        today_row = test_df.iloc[[idx]].copy()
        if "Date_obj" in today_row:
            del today_row["Date_obj"]
        
        train_df = pd.read_csv(config["train_csv"])
        updated_train = pd.concat([train_df, today_row], ignore_index=True)
        updated_train.to_csv(config["train_csv"], index=False)
        
        st["test_idx"] += 1
        import_train_incremental(asset, config, st, updated_train)
        message = "Assimilated market day and finetuned."
    else:
        message = "Advanced non-market day."
        
    st["current_date"] = current_calendar_date + datetime.timedelta(days=1)
    return {"message": message}

def import_train_incremental(asset, config, st, updated_train):
    cols_to_keep = list(dict.fromkeys(st["feature_cols"] + [config["target_col"]]))
    numeric_df = updated_train[cols_to_keep].copy()
    
    returns_df = numeric_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    
    returns_df["target"] = returns_df[config["target_col"]].shift(-1)
    returns_df = returns_df.dropna()
    
    n_samples = st["lookback"] + 32
    if len(returns_df) > n_samples:
        returns_df = returns_df.iloc[-n_samples:]
        
    X_raw = returns_df[st["feature_cols"]].values
    y_raw = returns_df[["target"]].values
    
    X_scaled = st["x_scaler"].transform(X_raw)
    y_scaled = st["y_scaler"].transform(y_raw)
    
    X_seq, y_seq = [], []
    for i in range(st["lookback"], len(X_scaled)):
        X_seq.append(X_scaled[i - st["lookback"]:i])
        y_seq.append(y_scaled[i])
        
    if len(X_seq) == 0:
        return
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    train_data = torch.utils.data.TensorDataset(torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(st["model"].parameters(), lr=1e-5) 
    
    st["model"].train()
    for _ in range(2):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(st["model"](bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(st["model"].parameters(), 1.0)
            optimizer.step()
            
    torch.save(st["model"].state_dict(), os.path.join(config["model_dir"], config["model_pth"]))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
