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
        "train_csv": "df_gold_dataset_gepu_extended_train.csv",
        "test_csv": "df_gold_dataset_gepu_extended_test.csv",
        "best_params": "reports/df_gold_dataset_gepu_datecut_full/gold_best_params_optimized.json",
        "model_dir": "models/df_gold_dataset_gepu_datecut_full/seed_99",
        "model_pth": "cnn_bilstm_seed99.pth",
        "target_col": "Gold_Futures",
        "dataset_label": "GEPU Extended Date-Cut Model",
    },
    "silver": {
        "train_csv": "silver_RRL_interpolate_extended_train.csv",
        "test_csv": "silver_RRL_interpolate_extended_test.csv",
        "best_params": "reports/silver_RRL_interpolate/silver_yahoo_best_params.json",
        "model_dir": "models/silver_RRL_interpolate/seed_42",
        "model_pth": "cnn_bilstm_seed42.pth",
        "target_col": "Silver_Futures",
        "dataset_label": "Silver Extended Model (Real + Synthetic Horizon)",
    }
}

STATE_FILE = "simulation_state.json"

state = {
    "gold": {"test_idx": 0, "model": None, "x_scaler": None, "y_scaler": None, "feature_cols": None, "lookback": None, "params": None, "current_date": None, "history": {}, "data_split": None, "model_seed": None, "test_df": None, "forecast_rows": None},
    "silver": {"test_idx": 0, "model": None, "x_scaler": None, "y_scaler": None, "feature_cols": None, "lookback": None, "params": None, "current_date": None, "history": {}, "data_split": None, "model_seed": None, "test_df": None, "forecast_rows": None}
}

def save_runtime_state():
    payload = {}
    for asset, st in state.items():
        payload[asset] = {
            "test_idx": int(st["test_idx"]),
            "current_date": st["current_date"].isoformat() if st["current_date"] else None,
            "history": {date_key: float(pred) for date_key, pred in st["history"].items()}
        }

    temp_path = f"{STATE_FILE}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, STATE_FILE)

def get_reference_today(asset=None):
    override = os.getenv("SIMULATION_TODAY")
    if override:
        try:
            return datetime.date.fromisoformat(override)
        except ValueError:
            pass
    if asset:
        data_split = state.get(asset, {}).get("data_split") or {}
        split_today = data_split.get("requested_train_end_date")
        if split_today:
            try:
                return datetime.date.fromisoformat(split_today)
            except ValueError:
                pass
    return datetime.date.today()

def clamp_date_to_window(selected_date, min_date, max_date):
    if selected_date < min_date:
        return min_date
    if selected_date > max_date:
        return max_date
    return selected_date

def set_simulation_date_state(asset, selected_date):
    st = state[asset]
    test_df = get_test_dates(asset)

    new_test_idx = int((test_df["Date_obj"] < selected_date).sum())
    st["current_date"] = selected_date
    st["test_idx"] = new_test_idx
    st["history"] = {
        logged_date: pred
        for logged_date, pred in st["history"].items()
        if datetime.date.fromisoformat(logged_date) < selected_date
    }
    save_runtime_state()

    return {
        "current_date": str(st["current_date"]),
        "simulation_day": st["test_idx"]
    }

def load_level_frame(csv_path):
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        df["Date_obj"] = df["Date"].dt.date
    return df

def predict_from_context_frame(asset, context_df):
    config = ASSET_CONFIG[asset]
    st = state[asset]

    cols_to_keep = list(dict.fromkeys(st["feature_cols"] + [config["target_col"]]))
    numeric_df = context_df[cols_to_keep].copy()
    abs_last_price = float(numeric_df.iloc[-1][config["target_col"]])
    last_date = context_df.iloc[-1]["Date"].strftime("%Y-%m-%d")

    returns_df = numeric_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    lookback = st["lookback"]
    if len(returns_df) < lookback:
        return {"error": "Not enough data"}

    recent_returns = returns_df[st["feature_cols"]].iloc[-lookback:].copy()
    recent_scaled = st["x_scaler"].transform(recent_returns)
    recent_tensor = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled = st["model"](recent_tensor).cpu().numpy().reshape(-1, 1)

    pred_return = st["y_scaler"].inverse_transform(pred_scaled).item()
    pred_abs = abs_last_price * (1 + pred_return)
    return {
        "predicted_price": round(float(pred_abs), 2),
        "last_train_date": last_date,
        "last_train_price": round(abs_last_price, 2),
    }

def build_precomputed_forecasts(asset):
    config = ASSET_CONFIG[asset]
    train_df = load_level_frame(config["train_csv"])
    test_df = load_level_frame(config["test_csv"])
    if test_df.empty:
        return test_df, []

    forecast_rows = []
    context_df = train_df.copy()
    for _, test_row in test_df.iterrows():
        pred_info = predict_from_context_frame(asset, context_df)
        if pred_info.get("error"):
            raise ValueError(f"Unable to precompute forecasts for {asset}: {pred_info['error']}")

        forecast_rows.append(
            {
                "date": test_row["Date"].strftime("%Y-%m-%d"),
                "date_obj": test_row["Date_obj"],
                "actual_price": round(float(test_row[config["target_col"]]), 2),
                "predicted_price": pred_info["predicted_price"],
                "context_end_date": pred_info["last_train_date"],
                "context_end_price": pred_info["last_train_price"],
            }
        )
        context_df = pd.concat([context_df, test_row.to_frame().T], ignore_index=True)

    return test_df, forecast_rows

def load_runtime_state():
    if not os.path.exists(STATE_FILE):
        return

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        persisted = json.load(f)

    for asset in ASSET_CONFIG:
        asset_state = persisted.get(asset)
        if not asset_state:
            continue

        test_df = get_test_dates(asset)
        min_date = test_df.iloc[0]["Date_obj"]
        max_date = test_df.iloc[-1]["Date_obj"]
        max_restore_date = max_date + datetime.timedelta(days=1)

        current_date_raw = asset_state.get("current_date")
        try:
            current_date = datetime.date.fromisoformat(current_date_raw) if current_date_raw else min_date
        except (TypeError, ValueError):
            current_date = min_date

        if current_date < min_date:
            current_date = min_date
        if current_date > max_restore_date:
            current_date = max_restore_date

        max_test_idx = len(test_df)
        try:
            test_idx = int(asset_state.get("test_idx", 0))
        except (TypeError, ValueError):
            test_idx = 0
        test_idx = max(0, min(test_idx, max_test_idx))

        raw_history = asset_state.get("history", {})
        cleaned_history = {}
        if isinstance(raw_history, dict):
            for logged_date, pred in raw_history.items():
                try:
                    logged_date_obj = datetime.date.fromisoformat(logged_date)
                    pred_value = float(pred)
                except (TypeError, ValueError):
                    continue
                if min_date <= logged_date_obj <= max_date and logged_date_obj < current_date:
                    cleaned_history[logged_date] = pred_value

        state[asset]["current_date"] = current_date
        state[asset]["test_idx"] = test_idx
        state[asset]["history"] = cleaned_history

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
        state[asset]["data_split"] = dict(meta.get("data_split") or {})
        state[asset]["model_seed"] = meta.get("seed")
        
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

        test_df, forecast_rows = build_precomputed_forecasts(asset)
        state[asset]["test_df"] = test_df
        state[asset]["forecast_rows"] = forecast_rows

        train_df = load_level_frame(config["train_csv"])
        live_split = dict(state[asset]["data_split"] or {})
        live_split["raw_train_rows"] = int(len(train_df))
        live_split["raw_test_rows"] = int(len(test_df))
        if not train_df.empty and "Date_obj" in train_df.columns:
            live_split["raw_train_last_date"] = str(train_df.iloc[-1]["Date_obj"])
        if not test_df.empty and "Date_obj" in test_df.columns:
            live_split["raw_test_first_date"] = str(test_df.iloc[0]["Date_obj"])
            live_split["raw_test_last_date"] = str(test_df.iloc[-1]["Date_obj"])
        state[asset]["data_split"] = live_split

        if not test_df.empty and "Date_obj" in test_df.columns:
            state[asset]["current_date"] = test_df.iloc[0]["Date_obj"]

def predict_next_day(asset):
    st = state[asset]
    idx = st["test_idx"]
    forecast_rows = st.get("forecast_rows") or []
    if idx >= len(forecast_rows):
        return {"error": "No cached forecast available"}

    row = forecast_rows[idx]
    return {
        "forecast_date": row["date"],
        "predicted_price": row["predicted_price"],
        "last_train_date": row["context_end_date"],
        "last_train_price": row["context_end_price"],
    }

def get_test_dates(asset):
    cached_test_df = state[asset].get("test_df")
    if cached_test_df is not None:
        return cached_test_df.copy()
    return load_level_frame(ASSET_CONFIG[asset]["test_csv"])


load_models()
load_runtime_state()

@app.get("/")
def get_dashboard():
    return FileResponse(
        "dashboard.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/dashboard")
def get_dashboard_alias():
    return get_dashboard()

@app.get("/api/status/{asset}")
def get_status(asset: str):
    if asset not in ASSET_CONFIG:
        return {"error": "Invalid asset"}
    
    config = ASSET_CONFIG[asset]
    st = state[asset]
    idx = st["test_idx"]
    
    test_df = get_test_dates(asset)
    if idx >= len(test_df):
        return {"error": "Simulation finished, no more test data"}

    min_date = test_df.iloc[0]["Date_obj"]
    max_date = test_df.iloc[-1]["Date_obj"]
    market_date = test_df.iloc[idx]["Date_obj"]
    current_calendar_date = st["current_date"]

    reference_today = get_reference_today(asset)
    effective_today = clamp_date_to_window(reference_today, min_date, max_date)
    today_alignment_note = None
    if reference_today < min_date:
        today_alignment_note = (
            f"Reference today is {reference_today}. This test window begins on {min_date}, "
            f"so the dashboard is aligned to {effective_today}."
        )
    elif reference_today > max_date:
        today_alignment_note = (
            f"Reference today is {reference_today}. This test window ends on {max_date}, "
            f"so the dashboard is aligned to {effective_today}."
        )

    is_market_day = (current_calendar_date == market_date)
    
    forecast_rows = st.get("forecast_rows") or []
    pred_info = predict_next_day(asset)
    
    # Generate Rolling Metrics Log
    log_arr = []
    y_true = []
    y_pred = []

    for row in forecast_rows[:idx]:
        y_true.append(float(row["actual_price"]))
        y_pred.append(float(row["predicted_price"]))
        log_arr.append({
            "date": row["date"],
            "actual": round(float(row["actual_price"]), 2),
            "predicted": round(float(row["predicted_price"]), 2)
        })
            
    # Calculate Rolling metrics
    rolling_rmse = None
    rolling_r2 = None
    if len(y_true) > 1:
        rolling_rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)
        rolling_r2 = round(r2_score(y_true, y_pred), 4)
        
    yesterday_actual = None
    yesterday_pred = None
    yesterday_date = None
    
    if idx > 0:
        yesterday_actual = y_true[-1] if len(y_true) > 0 else None
        yesterday_pred = y_pred[-1] if len(y_pred) > 0 else None
        yesterday_date = log_arr[-1]["date"] if len(log_arr) > 0 else None
        
    # Send ordered history log (newest at the top visually preferably, but let JS handle it)
    log_arr.reverse()
        
    return {
        "asset": asset,
        "dataset_label": config.get("dataset_label"),
        "model_seed": st.get("model_seed"),
        "data_split": st.get("data_split"),
        "simulation_day": idx,
        "current_date": str(current_calendar_date),
        "min_date": str(min_date),
        "max_date": str(max_date),
        "test_row_count": int(len(test_df)),
        "prediction_mode": "precomputed_frozen_model",
        "reference_today": str(reference_today),
        "effective_today": str(effective_today),
        "today_alignment_note": today_alignment_note,
        "is_market_day": is_market_day,
        "forecast_date": pred_info.get("forecast_date"),
        "predicted_price": pred_info.get("predicted_price"),
        "last_train_date": pred_info.get("last_train_date"),
        "yesterday_date": yesterday_date,
        "yesterday_actual": yesterday_actual,
        "yesterday_pred": yesterday_pred,
        "last_train_price": pred_info.get("last_train_price"),
        "rolling_rmse": rolling_rmse,
        "rolling_r2": rolling_r2,
        "history_log": log_arr
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
        st["test_idx"] += 1
        message = "Advanced to the next precomputed market day. Model weights and forecasts remain frozen."
    else:
        message = "Advanced non-market day. Frozen forecasts were unchanged."
        
    st["current_date"] = current_calendar_date + datetime.timedelta(days=1)
    save_runtime_state()
    return {"message": message}

@app.post("/api/current_date/{asset}")
def set_current_date(asset: str, date: str):
    if asset not in ASSET_CONFIG:
        return {"error": "Invalid asset"}

    try:
        selected_date = datetime.date.fromisoformat(date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Date must use YYYY-MM-DD format.") from exc

    test_df = get_test_dates(asset)

    min_date = test_df.iloc[0]["Date_obj"]
    max_date = test_df.iloc[-1]["Date_obj"]
    if selected_date < min_date or selected_date > max_date:
        raise HTTPException(
            status_code=400,
            detail=f"Date must be between {min_date} and {max_date}."
        )

    state_update = set_simulation_date_state(asset, selected_date)

    return {
        "message": "Simulation date updated.",
        **state_update
    }

@app.post("/api/use_today/{asset}")
def use_reference_today(asset: str):
    if asset not in ASSET_CONFIG:
        return {"error": "Invalid asset"}

    test_df = get_test_dates(asset)
    min_date = test_df.iloc[0]["Date_obj"]
    max_date = test_df.iloc[-1]["Date_obj"]
    requested_today = get_reference_today(asset)
    effective_today = clamp_date_to_window(requested_today, min_date, max_date)
    state_update = set_simulation_date_state(asset, effective_today)

    message = "Dashboard aligned to the reference today."
    if effective_today != requested_today:
        message = (
            f"Reference today is {requested_today}, so the dashboard was aligned to the "
            f"nearest available test date {effective_today}."
        )

    return {
        "message": message,
        "requested_today": str(requested_today),
        "effective_today": str(effective_today),
        **state_update
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
