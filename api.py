import os
import json
import pickle
import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_squared_error, r2_score
from copy import deepcopy

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- INDICATORS ---
def calculate_indicators(df, price_col):
    """Computes Version 8 'Flash' Indicators directly inside the API."""
    df = df.copy()
    df['EMA_Fast'] = df[price_col].ewm(span=3, adjust=False).mean()
    df['EMA_Slow'] = df[price_col].ewm(span=8, adjust=False).mean()
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_7'] = 100 - (100 / (1 + rs))
    exp1 = df[price_col].ewm(span=6, adjust=False).mean()
    exp2 = df[price_col].ewm(span=13, adjust=False).mean()
    df['MACD_Flash'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD_Flash'].ewm(span=5, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Flash'] - df['MACD_Signal']
    df['BB_Mid'] = df[price_col].rolling(window=5).mean()
    df['BB_Std'] = df[price_col].rolling(window=5).std()
    df['BB_Width'] = (4 * df['BB_Std']) / (df['BB_Mid'] + 1e-8)
    df['ROC_2'] = df[price_col].pct_change(periods=2).replace([np.inf, -np.inf], 0).fillna(0)
    # Cross-Asset Ratio
    if 'Silver_Futures' in df.columns and 'Gold_Futures' in df.columns:
        df['GS_Ratio'] = df['Gold_Futures'] / (df['Silver_Futures'] + 1e-8)
    else:
        df['GS_Ratio'] = 0.0
    return df.ffill().bfill().fillna(0)

def add_indicators_silver(df, price_col):
    df = df.copy()
    df['EMA_10'] = df[price_col].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df[price_col].ewm(span=20, adjust=False).mean()
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    exp1 = df[price_col].ewm(span=12, adjust=False).mean()
    exp2 = df[price_col].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['BB_Mid'] = df[price_col].rolling(window=20).mean()
    df['BB_Std'] = df[price_col].rolling(window=20).std()
    df['BB_Width'] = (4 * df['BB_Std']) / (df['BB_Mid'] + 1e-8)
    df['ROC_5'] = df[price_col].pct_change(periods=5).replace([np.inf, -np.inf], 0).fillna(0)
    # Cross-Asset Ratio
    if 'Silver_Futures' in df.columns and 'Gold_Futures' in df.columns:
        df['GS_Ratio'] = df['Gold_Futures'] / (df['Silver_Futures'] + 1e-8)
    else:
        df['GS_Ratio'] = 0.0
    return df.ffill().bfill().fillna(0)

app = FastAPI(title="Hybrid CNN-BiLSTM Forecasting API")
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
    def __init__(self, input_dim, hidden_dim=20, filters=64, kernel_size=4, n_layers=1, dropout=0.1):
        super(CNN_BiLSTM, self).__init__()
        self.conv1     = nn.Conv1d(input_dim, filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1       = nn.BatchNorm1d(filters)
        self.relu      = nn.ReLU()
        self.pool      = nn.MaxPool1d(2)
        self.dropout   = nn.Dropout(dropout)
        lstm_dropout = dropout if n_layers > 1 else 0
        self.lstm      = nn.LSTM(filters, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.fc        = nn.Linear(hidden_dim * 2, 64)
        self.out       = nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if x.shape[-1] > 1: x = self.pool(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :] 
        x = self.relu(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.out(x)

# --- CONFIG ---
ASSET_CONFIG = {
    "gold": {
        "train_csv": "df_gold_dataset_gepu_extended_train.csv",
        "test_csv": "df_gold_dataset_gepu_extended_test.csv",
        "lively_test_csv": "df_gold_dataset_gepu_extended_lively.csv",
        "model_dir": "models/gold_RRL_interpolate",
        "seeds": [42, 123, 99],
        "target_col": "Gold_Futures",
        "dataset_label": "Gold Engine (Hybrid CNN-BiLSTM)",
        "features": ['Silver_Futures', 'Crude_Oil_Futures', 'UST10Y_Treasury_Yield', 'gepu', 'DFF', 'gpr_daily', 'Gold_Futures'],
        "tech_cols": ['EMA_Fast', 'EMA_Slow', 'RSI_7', 'MACD_Flash', 'MACD_Signal', 'MACD_Hist', 'BB_Width', 'ROC_2', 'GS_Ratio']
    },
    "silver": {
        "train_csv": "silver_RRL_interpolate_extended_train.csv",
        "test_csv": "silver_RRL_interpolate_extended_test.csv",
        "lively_test_csv": "silver_RRL_interpolate_extended_lively.csv",
        "model_dir": "models/silver_RRL_interpolate",
        "seeds": [42, 123, 99],
        "target_col": "Silver_Futures",
        "dataset_label": "Silver Engine (Hybrid CNN-BiLSTM)",
        "features": ['Silver_Futures', 'Gold_Futures', 'US30', 'SnP500', 'NASDAQ_100', 'USD_index'],
        "tech_cols": ['EMA_10', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Width', 'ROC_5', 'GS_Ratio']
    }
}

STATE_FILE = "simulation_state.json"
state = {
    "gold": {
        "test_idx": 0, "dataset_mode": "lively", "models": [], "base_weights": [], "x_scaler": None, "y_scaler": None, 
        "lookback": None, "params": None, "current_date": None, "history": {}, "diagnostic_logs": [],
        "test_df": None, "train_df": None
    },
    "silver": {
        "test_idx": 0, "dataset_mode": "lively", "models": [], "base_weights": [], "x_scaler": None, "y_scaler": None, 
        "lookback": None, "params": None, "current_date": None, "history": {}, "diagnostic_logs": [],
        "test_df": None, "train_df": None
    }
}

def save_runtime_state():
    payload = {}
    for asset, st in state.items():
        payload[asset] = {
            "current_date": st["current_date"].isoformat() if st["current_date"] else None,
            "test_idx": st["test_idx"],
            "dataset_mode": st["dataset_mode"],
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
                st["history"] = data[asset].get("history", {})
                st["diagnostic_logs"] = data[asset].get("diagnostic_logs", [])
                st["dataset_mode"] = data[asset].get("dataset_mode", "lively")

def reset_models_to_base(asset):
    st = state[asset]
    for i, model in enumerate(st["models"]): model.load_state_dict(st["base_weights"][i])

def prepare_feature_tensor(asset, context_df):
    config = ASSET_CONFIG[asset]
    st = state[asset]
    lookback = st["lookback"]
    all_features = config["features"] + config["tech_cols"]
    df_inds = calculate_indicators(context_df, config["target_col"]) if asset == "gold" else add_indicators_silver(context_df, config["target_col"])
    prepped_df = df_inds[all_features].copy()
    if len(prepped_df) < lookback: return None
    window = prepped_df.tail(lookback)
    X_s = st["x_scaler"].transform(window)
    return torch.tensor(X_s, dtype=torch.float32).unsqueeze(0).to(device)

def predict_from_context_frame(asset, context_df):
    st = state[asset]
    X_t = prepare_feature_tensor(asset, context_df)
    if X_t is None: return {"error": "Insufficient history"}
    with torch.no_grad():
        council_preds = []
        for model in st["models"]:
            scaled_pred = model(X_t).cpu().numpy().item()
            pred_ret = st["y_scaler"].inverse_transform(pd.DataFrame([[scaled_pred]], columns=["target_return"])).item()
            council_preds.append(pred_ret)
        directions = [np.sign(p) for p in council_preds]
        consensus_dir = 1 if sum(directions) > 0 else -1
        voted_preds = [p for p in council_preds if np.sign(p) == consensus_dir] or council_preds
        final_ret = np.mean(voted_preds)
        last_price = float(context_df.iloc[-1][ASSET_CONFIG[asset]["target_col"]])
        pred_price = last_price * (1.0 + float(final_ret))
        agreement = (sum([1 for d in directions if d == consensus_dir]) / len(directions)) * 100
    return {"predicted_price": round(pred_price, 6), "target_scaled": round(final_ret, 4), "council_agreement": agreement}

def retrain_on_revealed_day(asset, day_idx):
    config, st = ASSET_CONFIG[asset], state[asset]
    total_df = pd.concat([st["train_df"], st["test_df"].iloc[:day_idx+1]], ignore_index=True)
    actual_today = float(total_df.iloc[-1][config["target_col"]])
    
    # Consensus for metrics
    pred_info = predict_from_context_frame(asset, total_df.iloc[:-1])
    if "error" in pred_info: return
    pred_price = pred_info["predicted_price"]
    
    # Online adaptation (Weighted batch of returns)
    X_list, y_list = [], []
    for i in range(max(1, day_idx - 4), day_idx + 1):
        temp_df = pd.concat([st["train_df"], st["test_df"].iloc[:i+1]], ignore_index=True)
        X_t_raw = prepare_feature_tensor(asset, temp_df.iloc[:-1])
        if X_t_raw is None: continue
        p_prev, p_curr = float(temp_df.iloc[-2][config["target_col"]]), float(temp_df.iloc[-1][config["target_col"]])
        ret = (p_curr - p_prev) / (p_prev + 1e-8)
        y_scaled = st["y_scaler"].transform(pd.DataFrame([[ret]], columns=["target_return"])).item()
        X_list.append(X_t_raw.squeeze(0).cpu().numpy()); y_list.append([y_scaled])
    
    final_loss = 0.0
    if X_list:
        X_b, y_b = torch.tensor(np.array(X_list), dtype=torch.float32).to(device), torch.tensor(np.array(y_list), dtype=torch.float32).to(device)
        weights = torch.tensor(np.linspace(0.5, 1.5, len(X_list)), dtype=torch.float32).to(device).unsqueeze(1)
        for model in st["models"]:
            model.train()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d): m.eval()
            opt = torch.optim.Adam(model.parameters(), lr=st["params"]["lr"] * 0.1)
            for _ in range(5):
                opt.zero_grad(); loss = (torch.pow(model(X_b) - y_b, 2) * weights).mean(); loss.backward(); opt.step()
            model.eval()
        final_loss = loss.item()

    # Rolling Diagnostics
    mag_err = abs(actual_today - pred_price) / (actual_today + 1e-8)
    history_window = st["diagnostic_logs"][-19:] + [{ "actual_price": actual_today, "pred_price": pred_price }]
    aa, ap = np.array([h["actual_price"] for h in history_window]), np.array([h["pred_price"] for h in history_window])
    rolling_rmse = round(float(np.sqrt(np.mean((aa - ap)**2))), 2)
    rolling_mae = round(float(np.mean(np.abs(aa - ap))), 2)
    rolling_r2 = 0.0
    if len(aa) > 5 and np.std(aa) > 1e-9:
        r = np.corrcoef(aa, ap)[0, 1]
        rolling_r2 = round(float(np.sign(r) * r**2), 3)

    st["diagnostic_logs"].append({
        "date": str(st["test_df"].iloc[day_idx]["Date_obj"]), "loss": round(float(final_loss), 6),
        "actual_price": round(float(actual_today), 2), "pred_price": round(float(pred_price), 2),
        "hit": bool(np.sign(actual_today-float(total_df.iloc[-2][config['target_col']])) == np.sign(pred_price-float(total_df.iloc[-2][config['target_col']]))),
        "mag_err": round(float(mag_err), 3), "rolling_rmse": rolling_rmse, "rolling_mae": rolling_mae, "rolling_r2": rolling_r2
    })
    if len(st["diagnostic_logs"]) > 100: st["diagnostic_logs"].pop(0)
    st["history"][str(st["test_df"].iloc[day_idx]["Date_obj"])] = round(pred_price, 6)

def set_simulation_date_state(asset, selected_date):
    st = state[asset]
    new_idx = int((st["test_df"]["Date_obj"] < selected_date).sum())
    reset_models_to_base(asset)
    st["diagnostic_logs"], st["history"] = [], {}
    for i in range(new_idx): retrain_on_revealed_day(asset, i)
    st["current_date"], st["test_idx"] = selected_date, new_idx
    save_runtime_state()
    return {"status": "success", "day": st["test_idx"]}

@app.get("/")
def get_dashboard(): return FileResponse("dashboard.html")

@app.get("/api/status/{asset}")
def get_status(asset: str):
    config, st = ASSET_CONFIG[asset], state[asset]
    idx = st["test_idx"]
    if idx >= len(st["test_df"]): return {"error": "Finished"}
    ctx = pd.concat([st["train_df"], st["test_df"].iloc[:idx]], ignore_index=True)
    pred_info = predict_from_context_frame(asset, ctx)
    last_p = float(ctx.iloc[-1][config["target_col"]]) if len(ctx) > 0 else None
    
    history_log = []
    for d_str, p_p in st["history"].items():
        d_obj = datetime.date.fromisoformat(d_str)
        row = st["test_df"][st["test_df"]["Date_obj"] == d_obj]
        if not row.empty:
            act = float(row.iloc[0][config["target_col"]])
            history_log.append({"date": d_str, "actual": act, "predicted": p_p, 
                                "error_pct": round(abs(act - p_p) / (act + 1e-8) * 100, 2)})
    history_log.sort(key=lambda x: x["date"])
    
    rmse, mae, r2 = None, None, None
    overall_mda, rolling_da = None, None
    
    if len(history_log) >= 2:
        aa, pp = np.array([r["actual"] for r in history_log]), np.array([r["predicted"] for r in history_log])
        rmse, mae = round(float(np.sqrt(np.mean((aa-pp)**2))), 2), round(float(np.mean(np.abs(aa-pp))), 2)
        if np.std(aa) > 1e-9: r2 = round(float(r2_score(aa, pp)), 3)

    # Calculate Directional Metrics from diagnostic_logs
    if st["diagnostic_logs"]:
        hits = [log.get("hit", False) for log in st["diagnostic_logs"]]
        overall_mda = round((sum(hits) / max(1, len(hits))) * 100, 1)
        
        rolling_hits = hits[-20:]
        rolling_da = round((sum(rolling_hits) / max(1, len(rolling_hits))) * 100, 1)
    else:
        # Default to 0.0 if logs are empty (e.g. simulation just started or reset)
        overall_mda, rolling_da = 0.0, 0.0

    return {
        "current_date": str(st["current_date"]), "forecast_date": str(st["test_df"].iloc[idx]["Date_obj"]),
        "predicted_price": pred_info.get("predicted_price", 0), "council_agreement": pred_info.get("council_agreement", 0),
        "diagnostic_logs": st["diagnostic_logs"], "history_log": history_log, 
        "overall_rmse": rmse, "overall_mae": mae, "overall_r2": r2,
        "overall_mda": overall_mda, "rolling_da": rolling_da,
        "dataset_mode": st["dataset_mode"], "min_date": str(st["test_df"].iloc[0]["Date_obj"]), "max_date": str(st["test_df"].iloc[-1]["Date_obj"])
    }

@app.post("/api/next_day/{asset}")
def next_day(asset: str):
    st, idx = state[asset], state[asset]["test_idx"]
    if idx >= len(st["test_df"]): return {"error": "Finished"}
    ctx = pd.concat([st["train_df"], st["test_df"].iloc[:idx]], ignore_index=True)
    pred_info = predict_from_context_frame(asset, ctx)
    retrain_on_revealed_day(asset, idx)
    st["test_idx"] += 1; st["current_date"] = st["test_df"].iloc[st["test_idx"]-1]["Date_obj"]
    save_runtime_state()
    return {"message": "Success", "pred": pred_info.get("predicted_price")}

@app.post("/api/current_date/{asset}")
def set_current_date(asset: str, date: str):
    try: selected_date = datetime.date.fromisoformat(date)
    except: raise HTTPException(status_code=400, detail="Invalid date")
    return set_simulation_date_state(asset, selected_date)

@app.get("/api/init")
def initialize():
    for asset in ASSET_CONFIG:
        config = ASSET_CONFIG[asset]
        with open(os.path.join(config["model_dir"], "best_params.json"), "r") as f: params = json.load(f)
        state[asset].update({"params": params, "lookback": params["lookback"]})
        with open(os.path.join(config["model_dir"], "scaler_X.pkl"), "rb") as f: state[asset]["x_scaler"] = pickle.load(f)
        with open(os.path.join(config["model_dir"], "scaler_y.pkl"), "rb") as f: state[asset]["y_scaler"] = pickle.load(f)
        state[asset]["models"] = []
        state[asset]["base_weights"] = []
        feat_count = len(config["features"]) + len(config["tech_cols"])
        for seed in config["seeds"]:
            model = CNN_BiLSTM(feat_count, params["hidden_dim"], 64, 4).to(device)
            path = os.path.join(config["model_dir"], f"{asset}_model_seed_{seed}.pth")
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval(); state[asset]["models"].append(model); state[asset]["base_weights"].append(deepcopy(model.state_dict()))
        state[asset]["train_df"] = pd.read_csv(config["train_csv"])
        test_file = config["lively_test_csv"] if state[asset]["dataset_mode"] == "lively" else config["test_csv"]
        state[asset]["test_df"] = pd.read_csv(test_file)
        for df in [state[asset]["train_df"], state[asset]["test_df"]]:
            df["Date"] = pd.to_datetime(df["Date"]); df["Date_obj"] = df["Date"].dt.date
        state[asset]["current_date"] = state[asset]["test_df"].iloc[0]["Date_obj"]
    load_runtime_state()
    return {"status": "ready"}

initialize()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
