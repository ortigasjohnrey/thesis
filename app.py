#!/usr/bin/env python3
"""Flask live simulation API for the Gold CNN-BiLSTM model.

This app loads the trained PyTorch CNN-BiLSTM model already included in the
bundle, builds the out-of-sample simulation table, then exposes a browser GUI
and JSON API for rolling one-step-ahead simulation.
"""
from __future__ import annotations

import json
import pickle
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.metrics import mean_squared_error, r2_score
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_simulation import (  # noqa: E402
    CNN_BiLSTM,
    COMEX_HOLIDAY_MANUAL_OVERRIDES,
    DATE_COL,
    TARGET_CALENDAR_NAME,
    build_simulation_table,
    build_unavailable_forecast_dates,
    infer_model_params_from_state_dict,
    load_and_prepare_raw_dataframe,
    validate_input_frames,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = ROOT / "simulation_config.json"
EXPECTED_NEW_DATA_NAME = "gold_RRL_interpolate_2025_05_01_to_2025_11_26.csv"
TEMPLATE_NEW_DATA_NAME = "gold_RRL_interpolate_2025_05_01_to_2025_11_26_TEMPLATE.csv"

# Local-only simulation state. One entry is created every time the user starts a run.
ACTIVE_RUNS: dict[str, dict[str, Any]] = {}


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def expected_new_data_path() -> Path:
    cfg = load_config()
    local_paths = cfg.get("local_paths", {})
    path = local_paths.get("new_data_csv_expected")
    return ROOT / path if path else ROOT / "data" / "new" / EXPECTED_NEW_DATA_NAME


def template_new_data_path() -> Path:
    return ROOT / "data" / "new" / TEMPLATE_NEW_DATA_NAME


def json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return None
    if pd.isna(value):
        return None
    return value


def frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for col in out.columns:
        if np.issubdtype(out[col].dtype, np.datetime64):
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    out = out.replace([np.inf, -np.inf], np.nan)
    records = out.where(pd.notna(out), None).to_dict(orient="records")
    return [{k: json_safe(v) for k, v in row.items()} for row in records]


def row_to_record(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    return {k: json_safe(v) for k, v in data.items()}


def forecast_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "rmse": None, "r2": None}

    rmse = float(np.sqrt(mean_squared_error(df["actual_price"], df["predicted_price"])))
    r2_value = None
    if len(df) >= 2:
        actual = df["actual_price"].astype(float)
        # r2_score is undefined when the actual values have zero variance.
        if float(np.var(actual)) > 0:
            r2_value = float(r2_score(df["actual_price"], df["predicted_price"]))
    return {"rows": int(len(df)), "rmse": rmse, "r2": r2_value}


def chart_payload(df: pd.DataFrame) -> dict[str, list[Any]]:
    if df.empty:
        return {"dates": [], "actual": [], "predicted": []}
    return {
        "dates": pd.to_datetime(df["forecast_date"]).dt.strftime("%Y-%m-%d").tolist(),
        "actual": [json_safe(x) for x in df["actual_price"].tolist()],
        "predicted": [json_safe(x) for x in df["predicted_price"].tolist()],
    }


def missing_required_files(seed: int) -> list[str]:
    cfg = load_config()
    local_paths = cfg["local_paths"]
    history_csv = ROOT / local_paths["history_csv"]
    new_data_csv = expected_new_data_path()
    seed_dir = ROOT / local_paths["model_root"] / f"seed_{seed}"
    required = [
        history_csv,
        new_data_csv,
        seed_dir / f"cnn_bilstm_seed{seed}.pth",
        seed_dir / "x_scaler.pkl",
        seed_dir / "y_scaler.pkl",
        seed_dir / "model_metadata.json",
    ]
    return [rel(p) for p in required if not p.exists()]


@lru_cache(maxsize=16)
def load_results(seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load model + data once per seed and return full simulation table."""
    cfg = load_config()
    local_paths = cfg["local_paths"]
    sim_start = cfg["simulation_window"]["start"]
    sim_end = cfg["simulation_window"]["end"]

    history_csv = ROOT / local_paths["history_csv"]
    new_data_csv = expected_new_data_path()
    model_root = ROOT / local_paths["model_root"]
    seed_dir = model_root / f"seed_{seed}"

    model_path = seed_dir / f"cnn_bilstm_seed{seed}.pth"
    x_scaler_path = seed_dir / "x_scaler.pkl"
    y_scaler_path = seed_dir / "y_scaler.pkl"
    metadata_path = seed_dir / "model_metadata.json"

    missing = missing_required_files(seed)
    if missing:
        raise FileNotFoundError(
            "Missing required file(s): " + ", ".join(missing) + ". "
            "Use the Upload CSV button or place the completed new-data CSV at "
            f"{rel(expected_new_data_path())}."
        )

    history_df = load_and_prepare_raw_dataframe(history_csv, DATE_COL)
    new_df = load_and_prepare_raw_dataframe(new_data_csv, DATE_COL)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    validate_input_frames(history_df, new_df, metadata, sim_start, sim_end)

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
            "dropout_rate": float(metadata.get("dropout_rate", 0.0)),
        }
    else:
        params = infer_model_params_from_state_dict(state_dict)

    input_shape = (int(metadata["lookback"]), len(metadata["feature_cols"]))
    model = CNN_BiLSTM(input_shape=input_shape, params=params).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    results_df, _ = build_simulation_table(history_df, new_df, metadata, x_scaler, y_scaler, model)
    results_df["forecast_date"] = pd.to_datetime(results_df["forecast_date"]).dt.normalize()
    results_df["anchor_date"] = pd.to_datetime(results_df["anchor_date"]).dt.normalize()
    results_df = results_df[
        (results_df["forecast_date"] >= pd.Timestamp(sim_start))
        & (results_df["forecast_date"] <= pd.Timestamp(sim_end))
    ].copy().sort_values(["anchor_date", "forecast_date"]).reset_index(drop=True)

    if results_df.empty:
        raise ValueError("No forecast rows were produced inside the configured simulation window.")

    meta = {
        "seed": seed,
        "simulation_start": sim_start,
        "simulation_end": sim_end,
        "available_rows": int(len(results_df)),
        "first_anchor_date": pd.to_datetime(results_df["anchor_date"].min()).strftime("%Y-%m-%d"),
        "last_anchor_date": pd.to_datetime(results_df["anchor_date"].max()).strftime("%Y-%m-%d"),
        "first_forecast_date": pd.to_datetime(results_df["forecast_date"].min()).strftime("%Y-%m-%d"),
        "last_forecast_date": pd.to_datetime(results_df["forecast_date"].max()).strftime("%Y-%m-%d"),
        "device": str(DEVICE),
        "model_type": "CNN-BiLSTM fixed-model rolling one-step-ahead simulation",
        "target": "Gold_Futures next-day price",
    }
    return results_df, meta


def explain_unavailable_date(results_df: pd.DataFrame, date_value: pd.Timestamp, mode: str) -> str:
    if mode == "anchor":
        return "Selected today/anchor date is not available. The app will start from the next available anchor date."
    try:
        unavailable = build_unavailable_forecast_dates(
            available_forecast_dates=results_df["forecast_date"],
            start_date=results_df["forecast_date"].min(),
            end_date=results_df["forecast_date"].max(),
            calendar_name=TARGET_CALENDAR_NAME,
            manual_overrides=COMEX_HOLIDAY_MANUAL_OVERRIDES,
        )
        unavailable["Date"] = pd.to_datetime(unavailable["Date"]).dt.normalize()
        match = unavailable.loc[unavailable["Date"] == date_value]
        if not match.empty:
            return str(match["reason"].iloc[0])
    except Exception:
        pass
    return "Selected forecast date is not available. The app will start from the next available forecast date."


@app.route("/")
def index():
    cfg = load_config()
    sim_window = cfg["simulation_window"]
    default_seed = cfg.get("default_seed", 2)
    return render_template(
        "index.html",
        default_seed=default_seed,
        sim_start=sim_window["start"],
        sim_end=sim_window["end"],
    )


@app.route("/api/status", methods=["GET"])
def api_status():
    cfg = load_config()
    seed = int(cfg.get("default_seed", 2))
    expected = expected_new_data_path()
    template = template_new_data_path()
    return jsonify({
        "ok": True,
        "device": str(DEVICE),
        "simulation_window": cfg["simulation_window"],
        "default_seed": seed,
        "new_data_exists": expected.exists(),
        "expected_new_data_path": rel(expected),
        "template_exists": template.exists(),
        "template_path": rel(template),
        "missing_required_files": missing_required_files(seed),
        "api_routes": {
            "GET /": "browser dashboard",
            "GET /api/status": "check files and configuration",
            "POST /api/start": "start a rolling simulation run",
            "POST /api/next": "advance one forecast row",
            "POST /api/reset": "reset an active run",
        },
    })


@app.route("/api/template", methods=["GET"])
def api_template():
    path = template_new_data_path()
    if not path.exists():
        return jsonify({"ok": False, "error": f"Template file not found: {rel(path)}"}), 404
    return send_file(path, as_attachment=True, download_name=TEMPLATE_NEW_DATA_NAME)


@app.route("/api/upload-new-data", methods=["POST"])
def api_upload_new_data():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file field found. Upload using form field name 'file'."}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "No CSV file selected."}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name.lower().endswith(".csv"):
        return jsonify({"ok": False, "error": "Upload must be a CSV file."}), 400

    target = expected_new_data_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    file.save(target)
    load_results.cache_clear()
    ACTIVE_RUNS.clear()

    # Validate basic readability immediately.
    try:
        df = pd.read_csv(target, nrows=5)
    except Exception as exc:
        target.unlink(missing_ok=True)
        return jsonify({"ok": False, "error": f"Uploaded file could not be read as CSV: {exc}"}), 400

    return jsonify({
        "ok": True,
        "message": "New-data CSV uploaded successfully.",
        "saved_as": rel(target),
        "uploaded_filename": safe_name,
        "preview_columns": list(df.columns),
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    payload = request.get_json(silent=True) or {}
    cfg = load_config()
    # Use fixed default seed - no longer accepting seed from frontend
    seed = int(cfg.get("default_seed", 2))
    start_raw = payload.get("start_date")
    mode = payload.get("mode", "anchor")
    if mode not in {"anchor", "forecast"}:
        return jsonify({"ok": False, "error": "mode must be either 'anchor' or 'forecast'."}), 400

    start_date = pd.to_datetime(start_raw, errors="coerce")
    if pd.isna(start_date):
        return jsonify({"ok": False, "error": "Invalid start_date. Use YYYY-MM-DD."}), 400
    start_date = pd.Timestamp(start_date).normalize()

    try:
        results_df, meta = load_results(seed)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    date_col = "anchor_date" if mode == "anchor" else "forecast_date"
    filtered = results_df[results_df[date_col] >= start_date].copy().reset_index(drop=True)
    if filtered.empty:
        return jsonify({
            "ok": False,
            "error": f"No rows found on or after {start_date.date()} using mode={mode}.",
            "available_range": {
                "first_anchor_date": meta["first_anchor_date"],
                "last_anchor_date": meta["last_anchor_date"],
                "first_forecast_date": meta["first_forecast_date"],
                "last_forecast_date": meta["last_forecast_date"],
            },
        }), 404

    exact_start_available = bool((results_df[date_col] == start_date).any())
    start_note = None
    if not exact_start_available:
        start_note = explain_unavailable_date(results_df, start_date, mode)

    run_id = uuid.uuid4().hex
    ACTIVE_RUNS[run_id] = {
        "seed": seed,
        "mode": mode,
        "requested_start_date": start_date.strftime("%Y-%m-%d"),
        "rows": filtered,
        "pointer": 0,
        "shown": pd.DataFrame(columns=filtered.columns),
        "meta": meta,
    }

    return jsonify({
        "ok": True,
        "run_id": run_id,
        "meta": meta,
        "mode": mode,
        "requested_start_date": start_date.strftime("%Y-%m-%d"),
        "exact_start_available": exact_start_available,
        "start_note": start_note,
        "first_anchor_date": pd.to_datetime(filtered["anchor_date"].iloc[0]).strftime("%Y-%m-%d"),
        "first_forecast_date": pd.to_datetime(filtered["forecast_date"].iloc[0]).strftime("%Y-%m-%d"),
        "total_rows": int(len(filtered)),
        "remaining_rows": int(len(filtered)),
        "metrics": forecast_metrics(pd.DataFrame(columns=filtered.columns)),
        "chart": chart_payload(pd.DataFrame(columns=filtered.columns)),
    })


@app.route("/api/next", methods=["POST"])
def api_next():
    payload = request.get_json(silent=True) or {}
    run_id = str(payload.get("run_id", ""))
    if run_id not in ACTIVE_RUNS:
        return jsonify({"ok": False, "error": "Invalid or expired run_id. Start the simulation again."}), 404

    state = ACTIVE_RUNS[run_id]
    rows: pd.DataFrame = state["rows"]
    pointer = int(state["pointer"])
    if pointer >= len(rows):
        shown: pd.DataFrame = state["shown"]
        return jsonify({
            "ok": True,
            "finished": True,
            "message": "Simulation finished. No more forecast rows are available.",
            "current_row": None,
            "metrics": forecast_metrics(shown),
            "chart": chart_payload(shown),
            "shown_rows": int(len(shown)),
            "remaining_rows": 0,
        })

    row = rows.iloc[pointer].copy()
    state["pointer"] = pointer + 1
    state["shown"] = pd.concat([state["shown"], row.to_frame().T], ignore_index=True)
    shown = state["shown"]
    finished = state["pointer"] >= len(rows)

    return jsonify({
        "ok": True,
        "finished": finished,
        "current_row": row_to_record(row),
        "metrics": forecast_metrics(shown),
        "chart": chart_payload(shown),
        "shown_rows": int(len(shown)),
        "remaining_rows": int(len(rows) - state["pointer"]),
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    payload = request.get_json(silent=True) or {}
    run_id = str(payload.get("run_id", ""))
    if run_id not in ACTIVE_RUNS:
        return jsonify({"ok": False, "error": "Invalid or expired run_id. Start the simulation again."}), 404
    state = ACTIVE_RUNS[run_id]
    state["pointer"] = 0
    state["shown"] = pd.DataFrame(columns=state["rows"].columns)
    return jsonify({
        "ok": True,
        "message": "Simulation reset.",
        "shown_rows": 0,
        "remaining_rows": int(len(state["rows"])),
        "metrics": forecast_metrics(state["shown"]),
        "chart": chart_payload(state["shown"]),
    })


@app.route("/api/run/<run_id>/revealed", methods=["GET"])
def api_revealed(run_id: str):
    if run_id not in ACTIVE_RUNS:
        return jsonify({"ok": False, "error": "Invalid or expired run_id."}), 404
    shown: pd.DataFrame = ACTIVE_RUNS[run_id]["shown"]
    return jsonify({
        "ok": True,
        "run_id": run_id,
        "rows": frame_to_records(shown),
        "metrics": forecast_metrics(shown),
        "chart": chart_payload(shown),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
