#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import exchange_calendars as xcals
except Exception:
    xcals = None

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATE_COL = "Date"
RAW_TARGET_COL = "Gold_Futures"
TARGET_CALENDAR_NAME = "COMEX"
DEFAULT_QUERY_DATE = "2025-08-15"

COMEX_HOLIDAY_MANUAL_OVERRIDES = pd.DataFrame({
    "Date": pd.to_datetime([
        "2022-06-20",
        "2023-06-19",
        "2024-06-19",
        "2025-06-19",
    ]),
    "holiday_name": [
        "Juneteenth National Independence Day",
        "Juneteenth National Independence Day",
        "Juneteenth National Independence Day",
        "Juneteenth National Independence Day",
    ],
    "holiday_type": [
        "manual_override",
        "manual_override",
        "manual_override",
        "manual_override",
    ],
    "session_effect": [
        "holiday_schedule_notice",
        "holiday_schedule_notice",
        "holiday_schedule_notice",
        "holiday_schedule_notice",
    ],
    "close_time": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
    "open_time": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
    "tag_source": [
        "manual_cme_notice",
        "manual_cme_notice",
        "manual_cme_notice",
        "manual_cme_notice",
    ],
    "source_note": [
        "Juneteenth holiday schedule manually added because exchange_calendars did not surface metadata for this date.",
        "Juneteenth holiday schedule manually added because exchange_calendars did not surface metadata for this date.",
        "Juneteenth holiday schedule manually added because exchange_calendars did not surface metadata for this date.",
        "Juneteenth holiday schedule manually added because exchange_calendars did not surface metadata for this date.",
    ],
})


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

        _, (h_n, _) = self.lstm2(x)
        h_f = h_n[0, :, :]
        h_b = h_n[1, :, :]
        x = torch.cat((h_f, h_b), dim=1)

        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_dropout(x)
        x = self.out(x)
        return x


def infer_model_params_from_state_dict(state_dict):
    return {
        "filters": int(state_dict["conv1.weight"].shape[0]),
        "kernel_size": int(state_dict["conv1.weight"].shape[2]),
        "lstm_units": int(state_dict["lstm1.weight_ih_l0"].shape[0] // 4),
        "dense_units": int(state_dict["fc1.weight"].shape[0]),
        "dropout_rate": 0.0,
    }


def create_sequences(X_df, y_df, lookback, abs_y_df=None):
    X_values = X_df.values
    y_values = y_df.values.reshape(-1)

    if abs_y_df is not None:
        abs_y_values = abs_y_df.values.reshape(-1)

    X_seq, y_seq, abs_y_seq = [], [], []
    for i in range(lookback, len(X_df)):
        X_seq.append(X_values[i - lookback:i, :])
        y_seq.append(y_values[i])
        if abs_y_df is not None:
            abs_y_seq.append(abs_y_values[i])

    if abs_y_df is not None:
        return np.array(X_seq), np.array(y_seq), np.array(abs_y_seq)
    return np.array(X_seq), np.array(y_seq)


def load_and_prepare_raw_dataframe(csv_path, date_col):
    df = pd.read_csv(csv_path)

    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not found in {csv_path}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df


def _empty_holiday_frame():
    return pd.DataFrame(
        columns=[
            "Date",
            "holiday_name",
            "holiday_type",
            "session_effect",
            "close_time",
            "open_time",
            "tag_source",
            "source_note",
        ]
    )


def build_calendar_holiday_tables(calendar, start_date, end_date, manual_overrides=None):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    def _frames_from_rules(rules, holiday_type, session_effect, close_time=pd.NaT, open_time=pd.NaT):
        frames = []
        for rule in rules:
            rule_dates = pd.DatetimeIndex(rule.dates(start, end))
            if len(rule_dates) == 0:
                continue
            frames.append(pd.DataFrame({
                "Date": rule_dates,
                "holiday_name": getattr(rule, "name", str(rule)),
                "holiday_type": holiday_type,
                "session_effect": session_effect,
                "close_time": close_time,
                "open_time": open_time,
                "tag_source": "exchange_calendars",
                "source_note": "Tagged from exchange_calendars holiday rules.",
            }))
        return frames

    full_frames = []
    regular_holiday_rules = getattr(getattr(calendar, "regular_holidays", None), "rules", [])
    full_frames.extend(
        _frames_from_rules(
            regular_holiday_rules,
            holiday_type="full_closure_regular",
            session_effect="closed",
        )
    )

    adhoc_dates = pd.DatetimeIndex(getattr(calendar, "adhoc_holidays", []))
    if len(adhoc_dates) > 0:
        adhoc_dates = adhoc_dates[(adhoc_dates >= start) & (adhoc_dates <= end)]
        if len(adhoc_dates) > 0:
            full_frames.append(pd.DataFrame({
                "Date": adhoc_dates,
                "holiday_name": "Ad hoc full-closure holiday",
                "holiday_type": "full_closure_adhoc",
                "session_effect": "closed",
                "close_time": pd.NaT,
                "open_time": pd.NaT,
                "tag_source": "exchange_calendars",
                "source_note": "Tagged from exchange_calendars ad hoc holiday list.",
            }))

    full_closures = pd.concat(full_frames, ignore_index=True) if full_frames else _empty_holiday_frame()

    def _special_frames(special_rules, session_effect):
        frames = []
        for tm, holiday_calendar in special_rules:
            rules = getattr(holiday_calendar, "rules", [])
            if rules:
                frames.extend(
                    _frames_from_rules(
                        rules,
                        holiday_type=session_effect,
                        session_effect=session_effect,
                        close_time=str(tm) if session_effect == "special_close" else pd.NaT,
                        open_time=str(tm) if session_effect == "special_open" else pd.NaT,
                    )
                )
            else:
                cal_dates = pd.DatetimeIndex(holiday_calendar.holidays(start, end))
                if len(cal_dates) == 0:
                    continue
                frames.append(pd.DataFrame({
                    "Date": cal_dates,
                    "holiday_name": f"{session_effect} holiday",
                    "holiday_type": session_effect,
                    "session_effect": session_effect,
                    "close_time": str(tm) if session_effect == "special_close" else pd.NaT,
                    "open_time": str(tm) if session_effect == "special_open" else pd.NaT,
                    "tag_source": "exchange_calendars",
                    "source_note": "Tagged from exchange_calendars special session rules.",
                }))
        return frames

    special_close_frames = _special_frames(getattr(calendar, "special_closes", []), "special_close")
    special_open_frames = _special_frames(getattr(calendar, "special_opens", []), "special_open")

    special_closes = pd.concat(special_close_frames, ignore_index=True) if special_close_frames else _empty_holiday_frame()
    special_opens = pd.concat(special_open_frames, ignore_index=True) if special_open_frames else _empty_holiday_frame()

    manual_df = _empty_holiday_frame()
    if manual_overrides is not None and len(manual_overrides) > 0:
        manual_df = manual_overrides.copy()
        manual_df["Date"] = pd.to_datetime(manual_df["Date"]).dt.normalize()
        manual_df = manual_df[(manual_df["Date"] >= start) & (manual_df["Date"] <= end)]

    holiday_schedule = pd.concat(
        [full_closures, special_closes, special_opens, manual_df],
        ignore_index=True
    )
    if not holiday_schedule.empty:
        holiday_schedule["Date"] = pd.to_datetime(holiday_schedule["Date"]).dt.normalize()
        holiday_schedule = (
            holiday_schedule
            .sort_values(["Date", "holiday_type", "holiday_name"])
            .drop_duplicates(subset=["Date", "holiday_type", "holiday_name"])
            .reset_index(drop=True)
        )

    return full_closures, special_closes, special_opens, manual_df, holiday_schedule


def nearest_available_forecast_bounds(query_date, available_dates):
    available_idx = pd.DatetimeIndex(pd.to_datetime(available_dates)).normalize().sort_values().unique()
    query_date = pd.Timestamp(query_date).normalize()

    prev_dates = available_idx[available_idx < query_date]
    next_dates = available_idx[available_idx > query_date]

    prev_date = prev_dates.max() if len(prev_dates) else pd.NaT
    next_date = next_dates.min() if len(next_dates) else pd.NaT
    return prev_date, next_date


def build_unavailable_forecast_dates(available_forecast_dates, start_date, end_date, calendar_name="COMEX", manual_overrides=None):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    available_idx = pd.DatetimeIndex(pd.to_datetime(available_forecast_dates)).normalize().sort_values().unique()
    daily_range = pd.date_range(start, end, freq="D")
    unavailable_idx = daily_range.difference(available_idx)

    rows = []
    holiday_schedule = _empty_holiday_frame()
    calendar_sessions = None

    if xcals is not None:
        calendar = xcals.get_calendar(calendar_name)
        calendar_sessions = pd.DatetimeIndex(calendar.sessions_in_range(start, end)).normalize()
        _, _, _, _, holiday_schedule = build_calendar_holiday_tables(
            calendar,
            start,
            end,
            manual_overrides=manual_overrides,
        )
        if not holiday_schedule.empty:
            holiday_schedule["Date"] = pd.to_datetime(holiday_schedule["Date"]).dt.normalize()

    for dt in unavailable_idx:
        dt = pd.Timestamp(dt).normalize()
        prev_date, next_date = nearest_available_forecast_bounds(dt, available_idx)

        schedule_rows = pd.DataFrame()
        if not holiday_schedule.empty:
            schedule_rows = holiday_schedule.loc[holiday_schedule["Date"] == dt].copy()

        if calendar_sessions is None:
            if dt.dayofweek >= 5:
                market_status = "non-market day"
                reason = "Weekend / not available in dataset"
            else:
                market_status = "not available in dataset"
                reason = "Date not present in forecast table"
        else:
            is_session = dt in calendar_sessions
            if not is_session:
                market_status = "non-market day"
                if not schedule_rows.empty:
                    reason = " | ".join(
                        schedule_rows["holiday_name"].astype(str) + " (" + schedule_rows["session_effect"].astype(str) + ")"
                    )
                elif dt.dayofweek >= 5:
                    reason = "Weekend / COMEX closed"
                else:
                    reason = "COMEX closed / no market session"
            else:
                market_status = "calendar session but unavailable in dataset"
                if not schedule_rows.empty:
                    reason = " | ".join(
                        schedule_rows["holiday_name"].astype(str) + " (" + schedule_rows["session_effect"].astype(str) + ")"
                    ) + " | date still unavailable in dataset"
                else:
                    reason = "Valid COMEX session but missing from forecast table"

        rows.append({
            "Date": dt,
            "market_status": market_status,
            "reason": reason,
            "nearest_previous_available_forecast": prev_date,
            "nearest_next_available_forecast": next_date,
        })

    unavailable_df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return unavailable_df


def build_simulation_table(historical_df, new_df, metadata, x_scaler, y_scaler, model):
    feature_cols = list(metadata["feature_cols"])
    lookback = int(metadata["lookback"])
    model_target_col = metadata["target_col"]

    required_cols = [DATE_COL] + feature_cols
    if RAW_TARGET_COL not in required_cols:
        required_cols.append(RAW_TARGET_COL)

    hist = historical_df[required_cols].copy()
    new = new_df[required_cols].copy()

    full_raw = pd.concat([hist, new], ignore_index=True)
    full_raw = (
        full_raw
        .drop_duplicates(subset=[DATE_COL], keep="last")
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    numeric_cols = [c for c in full_raw.columns if c != DATE_COL]
    for col in numeric_cols:
        full_raw[col] = pd.to_numeric(full_raw[col], errors="coerce")

    work = full_raw.copy().set_index(DATE_COL)

    abs_p_t = "P_t_abs"
    abs_target_col = "P_t_plus_1_abs"

    work[abs_p_t] = work[RAW_TARGET_COL]

    returns_df = work[numeric_cols].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    returns_df[abs_p_t] = work[abs_p_t].loc[returns_df.index]

    returns_df[model_target_col] = returns_df[RAW_TARGET_COL].shift(-1)
    returns_df[abs_target_col] = returns_df[abs_p_t].shift(-1)
    returns_df["forecast_date"] = returns_df.index.to_series().shift(-1)

    returns_df = returns_df.dropna().reset_index()

    X_df = returns_df[feature_cols].copy()
    y_df = returns_df[[model_target_col]].copy()
    p_t_df = returns_df[[abs_p_t]].copy()
    p_t_plus_1_df = returns_df[[abs_target_col]].copy()

    X_scaled = pd.DataFrame(x_scaler.transform(X_df), columns=feature_cols)
    y_scaled = pd.DataFrame(y_scaler.transform(y_df), columns=[model_target_col])

    X_seq, y_seq, p_t_seq = create_sequences(X_scaled, y_scaled, lookback, abs_y_df=p_t_df)
    _, _, p_t_plus_1_seq = create_sequences(X_scaled, y_scaled, lookback, abs_y_df=p_t_plus_1_df)

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
        pred_scaled = model(X_t).cpu().numpy().reshape(-1, 1)

    pred_return = y_scaler.inverse_transform(pred_scaled).reshape(-1)
    actual_return = y_scaler.inverse_transform(y_seq.reshape(-1, 1)).reshape(-1)

    predicted_price = p_t_seq.reshape(-1) * (1 + pred_return)
    actual_price = p_t_plus_1_seq.reshape(-1)

    results_df = pd.DataFrame({
        "anchor_date": returns_df[DATE_COL].iloc[lookback:].reset_index(drop=True),
        "forecast_date": returns_df["forecast_date"].iloc[lookback:].reset_index(drop=True),
        "predicted_return": pred_return,
        "actual_return": actual_return,
        "predicted_price": predicted_price,
        "actual_price": actual_price,
        "absolute_error": np.abs(actual_price - predicted_price),
        "squared_error": (actual_price - predicted_price) ** 2,
    })

    return results_df, returns_df


def evaluate_window(results_df, start_date, end_date):
    picked = results_df[
        (results_df["forecast_date"] >= pd.Timestamp(start_date)) &
        (results_df["forecast_date"] <= pd.Timestamp(end_date))
    ].copy()

    if picked.empty:
        raise ValueError("No forecast rows found inside the selected simulation window.")

    rmse = np.sqrt(mean_squared_error(picked["actual_price"], picked["predicted_price"]))
    mae = mean_absolute_error(picked["actual_price"], picked["predicted_price"])
    r2 = r2_score(picked["actual_price"], picked["predicted_price"])

    return picked, {
        "rows": int(len(picked)),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def lookup_forecast_by_date(results_df, query_date):
    q = pd.Timestamp(query_date).normalize()
    tmp = results_df.copy()
    tmp["forecast_date"] = pd.to_datetime(tmp["forecast_date"]).dt.normalize()
    return tmp[tmp["forecast_date"] == q].copy()


def nearest_available_dates(results_df, query_date, n=10):
    q = pd.Timestamp(query_date).normalize()
    tmp = results_df.copy()
    tmp["forecast_date"] = pd.to_datetime(tmp["forecast_date"]).dt.normalize()
    tmp["abs_days_diff"] = (tmp["forecast_date"] - q).abs().dt.days
    return tmp.sort_values(["abs_days_diff", "forecast_date"]).head(n).reset_index(drop=True)


def validate_input_frames(history_df, new_df, metadata, sim_start, sim_end):
    required_cols = [DATE_COL] + list(metadata["feature_cols"])
    missing_history = [c for c in required_cols if c not in history_df.columns]
    missing_new = [c for c in required_cols if c not in new_df.columns]

    if missing_history:
        raise ValueError(f"Historical CSV is missing required columns: {missing_history}")
    if missing_new:
        raise ValueError(f"New-data CSV is missing required columns: {missing_new}")

    new_window = new_df[
        (new_df[DATE_COL] >= pd.Timestamp(sim_start)) &
        (new_df[DATE_COL] <= pd.Timestamp(sim_end))
    ].copy()

    if new_window.empty:
        raise ValueError(
            "The new-data CSV does not contain any rows inside the requested simulation window "
            f"{sim_start} to {sim_end}."
        )

    if new_window[list(metadata["feature_cols"])].isna().any().any():
        bad_cols = new_window[list(metadata["feature_cols"])].columns[
            new_window[list(metadata["feature_cols"])].isna().any()
        ].tolist()
        raise ValueError(
            "The new-data CSV contains missing values in the simulation window. "
            f"Columns with missing values: {bad_cols}"
        )

    print("Input validation passed.")
    print(f"Historical range: {history_df[DATE_COL].min().date()} to {history_df[DATE_COL].max().date()}")
    print(f"New-data range   : {new_df[DATE_COL].min().date()} to {new_df[DATE_COL].max().date()}")
    print(f"Rows in window   : {len(new_window)}")


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the local gold CNN-BiLSTM new-data simulation."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=root,
        help="Project root. Defaults to the parent of the scripts folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Seed folder to load from models/gold_RRL_interpolate_6am/seed_<seed>.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=root / "data" / "historical" / "gold_RRL_interpolate_history_to_2025_04_30.csv",
        help="Historical CSV path.",
    )
    parser.add_argument(
        "--new-data-csv",
        type=Path,
        default=root / "data" / "new" / "gold_RRL_interpolate_2025_05_01_to_2025_11_26.csv",
        help="Prepared new-data CSV path.",
    )
    parser.add_argument(
        "--sim-start",
        type=str,
        default="2025-05-01",
        help="Simulation window start date (inclusive).",
    )
    parser.add_argument(
        "--sim-end",
        type=str,
        default="2025-11-26",
        help="Simulation window end date (inclusive).",
    )
    parser.add_argument(
        "--query-date",
        type=str,
        default=DEFAULT_QUERY_DATE,
        help="Forecast date to inspect after the simulation runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs",
        help="Output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = args.project_root / "models" / "gold_RRL_interpolate_6am" / f"seed_{args.seed}"
    model_path = base_dir / f"cnn_bilstm_seed{args.seed}.pth"
    x_scaler_path = base_dir / "x_scaler.pkl"
    y_scaler_path = base_dir / "y_scaler.pkl"
    metadata_path = base_dir / "model_metadata.json"

    required_files = [args.history_csv, args.new_data_csv, model_path, x_scaler_path, y_scaler_path, metadata_path]
    missing = [str(p) for p in required_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Some required files are missing.\n" +
            "\n".join(missing) +
            "\n\nPlace the prepared new-data CSV at the expected local path or pass --new-data-csv."
        )

    history_df = load_and_prepare_raw_dataframe(args.history_csv, DATE_COL)
    new_df = load_and_prepare_raw_dataframe(args.new_data_csv, DATE_COL)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    validate_input_frames(history_df, new_df, metadata, args.sim_start, args.sim_end)

    with open(x_scaler_path, "rb") as f:
        x_scaler = pickle.load(f)

    with open(y_scaler_path, "rb") as f:
        y_scaler = pickle.load(f)

    state_dict = torch.load(model_path, map_location=DEVICE)

    if all(k in metadata for k in ["filters", "kernel_size", "lstm_units", "dense_units", "dropout_rate"]):
        params = {
            "filters": int(metadata["filters"]),
            "kernel_size": int(metadata["kernel_size"]),
            "lstm_units": int(metadata["lstm_units"]),
            "dense_units": int(metadata["dense_units"]),
            "dropout_rate": float(metadata["dropout_rate"]),
        }
    else:
        params = infer_model_params_from_state_dict(state_dict)

    input_shape = (int(metadata["lookback"]), len(metadata["feature_cols"]))
    model = CNN_BiLSTM(input_shape=input_shape, params=params).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    results_df, combined_work_df = build_simulation_table(
        historical_df=history_df,
        new_df=new_df,
        metadata=metadata,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        model=model,
    )

    sim_results, sim_metrics = evaluate_window(results_df, args.sim_start, args.sim_end)

    all_results_path = args.output_dir / f"gold_simulation_all_results_seed_{args.seed}.csv"
    window_results_path = args.output_dir / f"gold_simulation_window_{args.sim_start}_to_{args.sim_end}_seed_{args.seed}.csv"
    metrics_path = args.output_dir / f"gold_simulation_metrics_{args.sim_start}_to_{args.sim_end}_seed_{args.seed}.json"
    plot_path = args.output_dir / f"gold_simulation_plot_{args.sim_start}_to_{args.sim_end}_seed_{args.seed}.png"

    results_df.to_csv(all_results_path, index=False)
    sim_results.to_csv(window_results_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(sim_metrics, f, indent=4)

    plt.figure(figsize=(14, 6))
    plt.plot(sim_results["forecast_date"], sim_results["actual_price"], label="Actual Price", linewidth=2)
    plt.plot(sim_results["forecast_date"], sim_results["predicted_price"], label="Predicted Price", linewidth=2)
    plt.title(
        f"Actual vs Predicted Gold Price\n"
        f"RMSE = {sim_metrics['rmse']:.6f} | MAE = {sim_metrics['mae']:.6f} | R² = {sim_metrics['r2']:.6f}"
    )
    plt.xlabel("Forecast Date")
    plt.ylabel("Gold Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("\nSimulation metrics:")
    print(json.dumps(sim_metrics, indent=4))
    print(f"Saved full results : {all_results_path}")
    print(f"Saved window CSV   : {window_results_path}")
    print(f"Saved metrics JSON : {metrics_path}")
    print(f"Saved plot         : {plot_path}")

    query_ts = pd.to_datetime(args.query_date, errors="coerce")
    if pd.isna(query_ts):
        print(f"\nQuery date '{args.query_date}' is invalid. Skipping lookup.")
        return

    query_ts = pd.Timestamp(query_ts).normalize()
    picked = lookup_forecast_by_date(results_df, query_ts)

    if picked.empty:
        print(f"\nNo forecast row found for {query_ts.date()}.")
        forecast_min = pd.to_datetime(results_df["forecast_date"]).min().normalize()
        forecast_max = pd.to_datetime(results_df["forecast_date"]).max().normalize()

        if query_ts < forecast_min or query_ts > forecast_max:
            print("That date is outside the available forecast range shown below.")
            print(f"Available forecast range: {forecast_min.date()} to {forecast_max.date()}")
        else:
            unavailable = build_unavailable_forecast_dates(
                available_forecast_dates=results_df["forecast_date"],
                start_date=forecast_min,
                end_date=forecast_max,
                calendar_name=TARGET_CALENDAR_NAME,
                manual_overrides=COMEX_HOLIDAY_MANUAL_OVERRIDES,
            )
            match = unavailable.loc[unavailable["Date"] == query_ts].copy()
            if not match.empty:
                print(match.to_string(index=False))
            else:
                nearest = nearest_available_dates(results_df, query_ts, n=10)
                print("\nNearest available forecast dates:")
                print(nearest[["anchor_date", "forecast_date", "predicted_price", "actual_price"]].to_string(index=False))
    else:
        one_step_rmse = float(np.sqrt(picked["squared_error"].iloc[0]))
        lookup_path = args.output_dir / f"gold_lookup_{query_ts.date()}_seed_{args.seed}.csv"
        picked.to_csv(lookup_path, index=False)
        print(f"\nForecast row for {query_ts.date()}:")
        print(
            picked[
                [
                    "anchor_date",
                    "forecast_date",
                    "predicted_price",
                    "actual_price",
                    "absolute_error",
                    "squared_error",
                ]
            ].to_string(index=False)
        )
        print(f"One-step RMSE: {one_step_rmse:.6f}")
        print(f"Overall simulation RMSE: {sim_metrics['rmse']:.6f}")
        print(f"Overall simulation R²: {sim_metrics['r2']:.6f}")
        print(f"Saved lookup CSV: {lookup_path}")


if __name__ == "__main__":
    main()
