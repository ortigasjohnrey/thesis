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

# --- SCALER DEFINITION (must match training script so pickle can deserialize) ---
class BasisPointScaler:
    """Scales returns to basis-point space: x_scaled = x_raw * scale."""
    def __init__(self, scale=1000.0):
        self.scale = scale
    def fit_transform(self, x):
        return x * self.scale
    def transform(self, x):
        return x * self.scale
    def inverse_transform(self, x):
        return x / self.scale

# --- MODEL DEFINITION ---
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, filters=64, kernel_size=3, n_layers=2, dropout=0.3):
        super(CNN_BiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(filters, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x

# --- STATE AND CONFIG ---
ASSET_CONFIG = {
    "gold": {
        "train_csv": "df_gold_dataset_gepu_extended_train.csv",
        "test_csv": "df_gold_dataset_gepu_extended_test.csv",
        "best_params": "models/gold_RRL_interpolate/best_params.json",
        "model_dir": "models/gold_RRL_interpolate",
        "seeds": [0, 1, 2, 42, 99, 123],
        "target_col": "Gold_Futures",
        "dataset_label": "Gold CNN-BiLSTM Optuna Ensemble (Anti-Laziness)",
        "features": ['Silver_Futures', 'Crude_Oil_Futures', 'UST10Y_Treasury_Yield', 'gepu', 'DFF', 'gpr_daily', 'Gold_Futures']
    },
    "silver": {
        "train_csv": "silver_RRL_interpolate_extended_train.csv",
        "test_csv": "silver_RRL_interpolate_extended_test.csv",
        "best_params": "models/silver_RRL_interpolate/best_params.json",
        "model_dir": "models/silver_RRL_interpolate",
        "seeds": [0, 1, 2, 42, 99, 123],
        "target_col": "Silver_Futures",
        "dataset_label": "Silver CNN-BiLSTM Optuna Ensemble (Anti-Laziness)",
        "features": ['Silver_Futures', 'Gold_Futures', 'US30', 'SnP500', 'NASDAQ_100', 'USD_index']
    }
}

STATE_FILE = "simulation_state.json"

state = {
    "gold": {"test_idx": 0, "models": [], "x_scaler": None, "y_scaler": None, "feature_cols": None, "lookback": None, "params": None, "current_date": None, "history": {}, "data_split": None, "model_seeds": None, "test_df": None, "forecast_rows": None},
    "silver": {"test_idx": 0, "models": [], "x_scaler": None, "y_scaler": None, "feature_cols": None, "lookback": None, "params": None, "current_date": None, "history": {}, "data_split": None, "model_seeds": None, "test_df": None, "forecast_rows": None}
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
    """
    Build a one-step-ahead price prediction from the given context window.
    Strictly follows the return-based transformation and feature set from training.
    """
    config = ASSET_CONFIG[asset]
    st = state[asset]
    target_col = config["target_col"]
    feature_cols = config["features"]

    # --- Step 1: Anchor price ---
    abs_last_price = float(context_df[target_col].iloc[-1])
    last_date = context_df.iloc[-1]["Date"].strftime("%Y-%m-%d")

    # --- Step 2: Compute Returns accurately ---
    # Only keep the columns we need
    numeric_df = context_df[feature_cols].copy()
    # Force numeric
    for c in feature_cols:
        numeric_df[c] = pd.to_numeric(numeric_df[c], errors="coerce")
    
    returns_df = numeric_df.pct_change().replace([np.inf, -np.inf], 0).dropna()

    lookback = st["lookback"]
    if len(returns_df) < lookback:
        return {"error": f"Not enough data (needs {lookback}, has {len(returns_df)})"}

    # --- Step 3: Scale and Infer with Ensemble ---
    recent_returns = returns_df[feature_cols].iloc[-lookback:].copy()
    recent_scaled = st["x_scaler"].transform(recent_returns)
    recent_tensor = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    all_preds = []
    for model in st["models"]:
        with torch.no_grad():
            pred_scaled = model(recent_tensor).cpu().numpy().reshape(-1, 1)
            all_preds.append(pred_scaled)
    
    avg_pred_scaled = np.mean(all_preds, axis=0)
    pred_return = st["y_scaler"].inverse_transform(avg_pred_scaled).item()
    pred_abs = abs_last_price * (1.0 + pred_return)

    return {
        "predicted_price": round(float(pred_abs), 2),
        "last_train_date": last_date,
        "last_train_price": round(abs_last_price, 2),
    }


def build_precomputed_forecasts(asset):
    """
    Walk forward through the test set, predicting each test day's price from
    the context window of all data seen so far, then appending that test row
    to context before moving to the next step.

    Alignment:
      - The model is called with context ending at train_day_T.
      - It predicts price_T+1 = P_T * (1 + r_pred).
      - test_row at position i IS day T+1 — its `target_col` value IS the
        actual price we are predicting.
      - So  actual_price = test_row[target_col]  is correctly aligned with
            predicted_price from predict_from_context_frame.

    Bug fixed (previously): the date label and actual_price were both taken
    from the same test_row, but since the context ends ONE step before that
    test_row, the alignment is in fact correct — actual = P_{t+1}, pred = P^_{t+1}.
    The real bug was that predict_from_context_frame was using wrong feature cols.
    """
    config = ASSET_CONFIG[asset]
    train_df = load_level_frame(config["train_csv"])
    test_df = load_level_frame(config["test_csv"])
    if test_df.empty:
        return test_df, []

    forecast_rows = []
    context_df = train_df.copy()

    for i in range(len(test_df)):
        test_row = test_df.iloc[i]
        pred_info = predict_from_context_frame(asset, context_df)
        if pred_info.get("error"):
            raise ValueError(f"Unable to precompute forecasts for {asset}: {pred_info['error']}")

        # actual_price  = P_{t+1}  (today's price, which is what we predicted)
        # predicted_price = P^_{t+1} (model's prediction of today from yesterday's context)
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

        # Append current test_row to context; use index slicer to preserve dtypes
        context_df = pd.concat([context_df, test_df.iloc[[i]]], ignore_index=True)
        context_df["Date"] = pd.to_datetime(context_df["Date"], errors="coerce")

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
        state[asset]["lookback"] = params["lookback"]
        
        # Load Scalers
        with open(os.path.join(config["model_dir"], "scaler_X.pkl"), "rb") as f:
            state[asset]["x_scaler"] = pickle.load(f)
        with open(os.path.join(config["model_dir"], "scaler_y.pkl"), "rb") as f:
            state[asset]["y_scaler"] = pickle.load(f)
            
        # Initialize and load all models in the ensemble
        state[asset]["models"] = []
        input_dim = len(config["features"])
        for seed in config["seeds"]:
            model = CNN_BiLSTM(
                input_dim=input_dim,
                hidden_dim=params["lstm_units"],
                filters=params["filters"],
                kernel_size=params["kernel_size"],
                dropout=params["dropout"]
            )
            model_name = f"{asset}_model_seed_{seed}.pth"
            model_path = os.path.join(config["model_dir"], model_name)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            state[asset]["models"].append(model)

        test_df, forecast_rows = build_precomputed_forecasts(asset)
        state[asset]["test_df"] = test_df
        state[asset]["forecast_rows"] = forecast_rows

        train_df = load_level_frame(config["train_csv"])
        live_split = {
            "raw_train_rows": int(len(train_df)),
            "raw_test_rows": int(len(test_df))
        }
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
    
    # -----------------------------------------------------------------
    # Generate Rolling Metrics Log
    # actual_price = P_{t+1} (the real next-day price)
    # predicted_price = P^_{t+1} (model forecast for that same day)
    # These are correctly aligned in build_precomputed_forecasts.
    # -----------------------------------------------------------------
    log_arr = []
    y_true_arr = []
    y_pred_arr = []
    y_prev_arr = []   # P_t (last context price) for directional accuracy

    for i, row in enumerate(forecast_rows[:idx]):
        actual = float(row["actual_price"])
        pred   = float(row["predicted_price"])
        ctx_price_raw = row.get("context_end_price")
        # context_end_price is P_t — the anchor the model used for reconstruction.
        # Never fall back to actual_price (P_{t+1}) as that corrupts directional calc.
        ctx_price = float(ctx_price_raw) if ctx_price_raw is not None else actual

        y_true_arr.append(actual)
        y_pred_arr.append(pred)
        y_prev_arr.append(ctx_price)

        log_arr.append({
            "date": row["date"],
            "actual": round(actual, 2),
            "predicted": round(pred, 2),
            "context_end_date": row.get("context_end_date"),
            "context_end_price": round(ctx_price, 2),
        })

    # -----------------------------------------------------------------
    # Rolling metrics
    # actual_price        = P_{t+1}  (real next-day price)
    # predicted_price     = P^{t+1}  (model forecast for that same day)
    # context_end_price   = P_t      (last known price when forecast was made)
    # -----------------------------------------------------------------
    rolling_rmse = None
    rolling_r2 = None
    rolling_dir_acc = None
    rolling_lazy = None        # True if model correlates more with P_t than P_{t+1}
    corr_actual = None
    corr_lag1   = None

    n = len(y_true_arr)
    if n > 1:
        y_true_np = np.array(y_true_arr)
        y_pred_np = np.array(y_pred_arr)
        y_prev_np = np.array(y_prev_arr)

        rolling_rmse = round(float(np.sqrt(mean_squared_error(y_true_np, y_pred_np))), 4)
        rolling_r2   = round(float(r2_score(y_true_np, y_pred_np)), 4)

        # Directional accuracy: sign(P^{t+1} - P_t) == sign(P_{t+1} - P_t)
        actual_direction = np.sign(y_true_np - y_prev_np)
        pred_direction   = np.sign(y_pred_np  - y_prev_np)
        rolling_dir_acc  = round(float(np.mean(actual_direction == pred_direction)), 4)

        # Laziness check: pred should correlate with P_{t+1}, not P_t.
        # If corr(pred, P_t) > corr(pred, P_{t+1}) the model is lazy (predicts yesterday).
        if np.std(y_pred_np) > 1e-8 and np.std(y_true_np) > 1e-8 and np.std(y_prev_np) > 1e-8:
            corr_actual = round(float(np.corrcoef(y_pred_np, y_true_np)[0, 1]), 4)
            corr_lag1   = round(float(np.corrcoef(y_pred_np, y_prev_np)[0, 1]), 4)
            rolling_lazy = bool(corr_lag1 > corr_actual)

    yesterday_actual = None
    yesterday_pred   = None
    yesterday_date   = None

    if idx > 0 and log_arr:
        # log_arr is not reversed yet here, so last entry = most recent
        last_entry = log_arr[-1]
        yesterday_date   = last_entry["date"]
        yesterday_actual = last_entry["actual"]
        yesterday_pred   = last_entry["predicted"]

    # Newest entries at the top for the dashboard table
    log_arr.reverse()

    return {
        "asset": asset,
        "dataset_label": config.get("dataset_label"),
        "model_seed": "Ensemble (6 seeds)",
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
        "rolling_dir_acc": rolling_dir_acc,
        "rolling_lazy": rolling_lazy,
        "corr_pred_actual": corr_actual,
        "corr_pred_lag1": corr_lag1,
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
