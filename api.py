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
        "best_params": "reports/gold_train_only_retrained_v2/gold_best_params_optimized.json",
        "model_dir": "models/gold_train_only_retrained_v2/seed_42",
        "model_pth": "cnn_bilstm_seed42.pth",
        "target_col": "Gold_Futures",
        "dataset_label": "GEPU Extended Train-Only Retrained Model V2",
    },
    "silver": {
        "train_csv": "silver_RRL_interpolate_extended_train.csv",
        "test_csv": "silver_RRL_interpolate_extended_test.csv",
        "best_params": "reports/silver_train_only_retrained_v2/silver_best_params_optimized.json",
        "model_dir": "models/silver_train_only_retrained_v2/seed_42",
        "model_pth": "cnn_bilstm_seed42.pth",
        "target_col": "Silver_Futures",
        "dataset_label": "Silver Extended Model V2 (Real + Synthetic Horizon)",
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
    """
    Build a one-step-ahead price prediction from the given context window.

    Preprocessing must exactly mirror the training pipeline:
      raw level prices  →  pct_change()  →  lag1 / lag2 features  →  dropna
      →  select feature_cols in order  →  scale  →  infer  →  inverse_scale
      →  reconstruct price level

    Bug fixed: the old code tried to select precomputed lag columns
    (e.g. 'Silver_Futures_lag1') directly from the CSV, which doesn't have them.
    We now take only BASE numeric columns from the raw frame, compute all
    transformations from scratch, then select feature_cols from the result.
    """
    config = ASSET_CONFIG[asset]
    st = state[asset]
    target_col = config["target_col"]

    # --- Step 1: select only base (non-lag) numeric columns from the raw frame ---
    # Drop any non-numeric columns except Date so pct_change works cleanly.
    base_numeric_cols = context_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_df = context_df[base_numeric_cols].copy()

    # The last level price is the anchor for price-level reconstruction.
    # We capture it from the LAST row BEFORE any returns/lag computation.
    abs_last_price = float(numeric_df.iloc[-1][target_col])
    last_date = context_df.iloc[-1]["Date"].strftime("%Y-%m-%d")

    # --- Step 2: compute returns exactly as in training ---
    returns_df = numeric_df.pct_change().replace([np.inf, -np.inf], np.nan)

    # --- Step 3: add explicit lag features matching training ---
    for col in base_numeric_cols:
        returns_df[f"{col}_lag1"] = returns_df[col].shift(1)
        returns_df[f"{col}_lag2"] = returns_df[col].shift(2)

    returns_df = returns_df.dropna()

    lookback = st["lookback"]
    if len(returns_df) < lookback:
        return {"error": "Not enough data"}

    # --- Step 4: validate that all expected feature cols are present ---
    feature_cols = st["feature_cols"]
    missing = [c for c in feature_cols if c not in returns_df.columns]
    if missing:
        return {"error": f"Feature mismatch — missing: {missing}"}

    # --- Step 5: scale and infer ---
    recent_returns = returns_df[feature_cols].iloc[-lookback:].copy()
    recent_scaled = st["x_scaler"].transform(recent_returns)
    recent_tensor = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    st["model"].eval()
    with torch.no_grad():
        pred_scaled = st["model"](recent_tensor).cpu().numpy().reshape(-1, 1)

    pred_return = st["y_scaler"].inverse_transform(pred_scaled).item()
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

    for _, test_row in test_df.iterrows():
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

        # Append current test_row to context; re-parse Date so strftime() keeps working
        new_row = test_row.to_frame().T.reset_index(drop=True)
        context_df = pd.concat([context_df, new_row], ignore_index=True)
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
    
    # -----------------------------------------------------------------
    # Generate Rolling Metrics Log
    # actual_price = P_{t+1} (the real next-day price)
    # predicted_price = P^_{t+1} (model forecast for that same day)
    # These are correctly aligned in build_precomputed_forecasts.
    # -----------------------------------------------------------------
    log_arr = []
    y_true_arr = []
    y_pred_arr = []
    y_prev_arr = []   # P_t (yesterday's actual price) for laziness check

    for i, row in enumerate(forecast_rows[:idx]):
        y_true_arr.append(float(row["actual_price"]))
        y_pred_arr.append(float(row["predicted_price"]))
        # context_end_price is the last price the model saw (= P_t)
        y_prev_arr.append(float(row.get("context_end_price") or row["actual_price"]))
        log_arr.append({
            "date": row["date"],
            "actual": round(float(row["actual_price"]), 2),
            "predicted": round(float(row["predicted_price"]), 2),
            "context_end_date": row.get("context_end_date"),
            "context_end_price": round(float(row.get("context_end_price") or row["actual_price"]), 2),
        })

    # -----------------------------------------------------------------
    # Rolling metrics
    # -----------------------------------------------------------------
    rolling_rmse = None
    rolling_r2 = None
    rolling_dir_acc = None
    rolling_lazy = None        # True if model correlates more with P_t than P_{t+1}

    n = len(y_true_arr)
    if n > 1:
        y_true_np = np.array(y_true_arr)
        y_pred_np = np.array(y_pred_arr)
        y_prev_np = np.array(y_prev_arr)

        rolling_rmse = round(float(np.sqrt(mean_squared_error(y_true_np, y_pred_np))), 4)
        rolling_r2   = round(float(r2_score(y_true_np, y_pred_np)), 4)

        # Directional accuracy: was the predicted move in the right direction?
        # direction = sign(P_{t+1}_pred - P_t) vs sign(P_{t+1}_actual - P_t)
        actual_direction = np.sign(y_true_np - y_prev_np)
        pred_direction   = np.sign(y_pred_np  - y_prev_np)
        rolling_dir_acc  = round(float(np.mean(actual_direction == pred_direction)), 4)

        # Laziness: does pred correlate more with P_t than P_{t+1}?
        if np.std(y_pred_np) > 0 and np.std(y_true_np) > 0 and np.std(y_prev_np) > 0:
            corr_actual = float(np.corrcoef(y_pred_np, y_true_np)[0, 1])
            corr_lag1   = float(np.corrcoef(y_pred_np, y_prev_np)[0, 1])
            rolling_lazy = bool(corr_lag1 > corr_actual)
        else:
            corr_actual = None
            corr_lag1   = None

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
        "rolling_dir_acc": rolling_dir_acc,
        "rolling_lazy": rolling_lazy,
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
