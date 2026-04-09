import base64
import html
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
GAN_DIR = ROOT / "gan"
sys.path.insert(0, str(GAN_DIR))

from dataset_catalog import get_all_dataset_configs  # noqa: E402


REPORT_ROOT = ROOT / "reports" / "gan_validation"
OUTPUT_HTML = REPORT_ROOT / "GAN_Model.html"
ALIAS_OUTPUTS = [
    REPORT_ROOT / "all_gan_system_documentation.html",
    ROOT / "reports" / "df_gold_dataset_gepu_datecut_full" / "gold_gan_system_documentation.html",
]

FORECAST_CONFIG = {
    "gold": {
        "label": "GEPU Extended Date-Cut Model",
        "train_csv": ROOT / "df_gold_dataset_gepu_extended_train.csv",
        "test_csv": ROOT / "df_gold_dataset_gepu_extended_test.csv",
        "extended_csv": ROOT / "df_gold_dataset_gepu_extended.csv",
        "model_dir": ROOT / "models" / "df_gold_dataset_gepu_datecut_full" / "seed_99",
        "params_path": ROOT / "reports" / "df_gold_dataset_gepu_datecut_full" / "gold_best_params_optimized.json",
        "report_dir": ROOT / "reports" / "df_gold_dataset_gepu_datecut_full",
        "target_col": "Gold_Futures",
        "plot_name": "gold_seed_99_predictions.png",
        "eval_prefix": "gold",
    },
    "silver": {
        "label": "Baseline Silver Model",
        "train_csv": ROOT / "silver_RRL_interpolate_train.csv",
        "test_csv": None,
        "extended_csv": ROOT / "silver_RRL_interpolate_extended.csv",
        "model_dir": ROOT / "models" / "silver_RRL_interpolate" / "seed_42",
        "params_path": ROOT / "reports" / "silver_RRL_interpolate" / "silver_yahoo_best_params.json",
        "report_dir": ROOT / "reports" / "silver_RRL_interpolate",
        "target_col": "Silver_Futures",
        "plot_name": "silver_yahoo_seed_42_predictions.png",
        "eval_prefix": "silver",
    },
}

REFERENCE_LINKS = [
    (
        "Rama Cont (2001), Empirical properties of asset returns: stylized facts and statistical issues",
        "https://www.stat.rice.edu/~dobelman/courses/texts/stylized.cont.2001.pdf",
        "Motivates evaluating synthetic financial data in return space and preserving dependence structure.",
    ),
    (
        "Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar (2019), Time-series Generative Adversarial Networks",
        "https://papers.neurips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf",
        "Supports GAN-based sequence generation when temporal dynamics matter.",
    ),
    (
        "Magnus Wiese et al. (2020), Quant GANs: Deep Generation of Financial Time Series",
        "https://arxiv.org/abs/1907.06673",
        "Direct financial-time-series support for GANs that preserve volatility structure and dependence.",
    ),
    (
        "Justin Hellermann et al. (2022), Financial Time Series Data Augmentation with Generative Adversarial Networks and Extended Intertemporal Return Plots",
        "https://arxiv.org/abs/2205.08924",
        "Supports GAN-generated financial data as augmentation for downstream forecasting experiments.",
    ),
]


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_shape, params):
        super().__init__()
        in_channels = input_shape[1]
        dr = params["dropout_rate"]
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=params["filters"],
            kernel_size=params["kernel_size"],
            padding=params["kernel_size"] - 1,
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(params["filters"])
        self.spatial_dropout = nn.Dropout1d(p=dr)
        self.lstm1 = nn.LSTM(
            input_size=params["filters"],
            hidden_size=params["lstm_units"],
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(dr)
        lstm2_units = max(16, params["lstm_units"] // 2)
        self.lstm2 = nn.LSTM(
            input_size=params["lstm_units"] * 2,
            hidden_size=lstm2_units,
            batch_first=True,
            bidirectional=True,
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
        return self.out(x)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_level_frame(path: Path):
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        df["Date_obj"] = df["Date"].dt.date
    return df


def image_uri(path: Path):
    if not path.exists():
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_image(path: Path, caption: str):
    uri = image_uri(path)
    if uri is None:
        return f"<div class='missing-asset'>Missing image: {html.escape(path.name)}</div>"
    return (
        "<figure class='plot'>"
        f"<img src='{uri}' alt='{html.escape(caption)}'>"
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def table_html(df: pd.DataFrame):
    return df.to_html(index=False, border=0, classes="data-table")


def badge(text: str, cls: str):
    return f"<span class='badge {cls}'>{html.escape(text)}</span>"


def build_test_frame(asset: str, forecast_cfg: dict):
    if forecast_cfg["test_csv"] is not None:
        return load_level_frame(forecast_cfg["test_csv"])
    train_df = load_level_frame(forecast_cfg["train_csv"])
    extended_df = load_level_frame(forecast_cfg["extended_csv"])
    cutoff = train_df["Date"].max()
    return extended_df.loc[extended_df["Date"] > cutoff].copy().reset_index(drop=True)


def build_model_frame(level_df: pd.DataFrame, target_col: str):
    numeric_df = level_df.select_dtypes(include=[np.number]).copy()
    work_df = numeric_df.copy()
    work_df["P_t_abs"] = work_df[target_col]
    returns_df = work_df[numeric_df.columns].pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
    returns_df["P_t_abs"] = work_df["P_t_abs"].loc[returns_df.index]
    returns_df["target_t_plus_1"] = returns_df[target_col].shift(-1)
    returns_df["P_t_plus_1_abs"] = returns_df["P_t_abs"].shift(-1)
    model_df = returns_df.dropna().copy()
    return model_df


def safe_subset_metrics(eval_df: pd.DataFrame, label: str):
    if eval_df.empty:
        return {
            "subset": label,
            "rows": 0,
            "start_date": "n/a",
            "end_date": "n/a",
            "rmse": "n/a",
            "mae": "n/a",
            "mape_percent": "n/a",
            "r2": "n/a",
            "directional_accuracy": "n/a",
        }

    r2_value = "n/a" if len(eval_df) < 2 else round(float(r2_score(eval_df["actual_price"], eval_df["predicted_price"])), 4)
    return {
        "subset": label,
        "rows": int(len(eval_df)),
        "start_date": str(eval_df["date"].min().date()),
        "end_date": str(eval_df["date"].max().date()),
        "rmse": round(float(mean_squared_error(eval_df["actual_price"], eval_df["predicted_price"]) ** 0.5), 4),
        "mae": round(float(mean_absolute_error(eval_df["actual_price"], eval_df["predicted_price"])), 4),
        "mape_percent": round(float(eval_df["pct_error"].mean()), 4),
        "r2": r2_value,
        "directional_accuracy": round(float(eval_df["direction_match"].mean()) * 100.0, 2),
    }


def make_eval_plot(eval_df: pd.DataFrame, output_path: Path, title: str):
    plt.figure(figsize=(14, 6))
    plt.plot(eval_df["date"], eval_df["actual_price"], linewidth=2, label="Actual")
    plt.plot(eval_df["date"], eval_df["predicted_price"], linewidth=2, linestyle="--", label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_error_plot(eval_df: pd.DataFrame, output_path: Path, title: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(eval_df["date"], eval_df["signed_error"], color="#b91c1c", linewidth=1.2)
    axes[0].axhline(0.0, color="#1f2937", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Signed error")
    axes[0].set_title(title)
    axes[1].plot(eval_df["date"], eval_df["abs_error"], color="#1d4ed8", linewidth=1.2)
    axes[1].set_ylabel("Absolute error")
    axes[1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_lifecycle_plot(
    source_df: pd.DataFrame,
    extended_df: pd.DataFrame,
    observed_end_date: pd.Timestamp,
    train_end_date: pd.Timestamp,
    target_col: str,
    output_path: Path,
    title: str,
):
    future_df = extended_df.loc[extended_df["Date"] > observed_end_date].copy()
    test_df = extended_df.loc[extended_df["Date"] > train_end_date].copy()
    observed_train_df = extended_df.loc[extended_df["Date"] <= min(train_end_date, observed_end_date)].copy()
    generated_train_df = extended_df.loc[(extended_df["Date"] > observed_end_date) & (extended_df["Date"] <= train_end_date)].copy()
    unused_real_df = test_df.loc[test_df["Date"] <= observed_end_date].copy()
    unused_generated_df = test_df.loc[test_df["Date"] > observed_end_date].copy()
    plt.figure(figsize=(14, 6))
    plt.plot(extended_df["Date"], extended_df[target_col], color="#4a4a4a", linewidth=1.2, label="Full extended series")
    plt.plot(future_df["Date"], future_df[target_col], color="#d97706", linewidth=1.6, label="GAN-generated extension")
    if not observed_train_df.empty:
        plt.axvspan(observed_train_df["Date"].min(), observed_train_df["Date"].max(), color="#dcfce7", alpha=0.35, label="Observed rows used in forecast training")
    if not generated_train_df.empty:
        plt.axvspan(generated_train_df["Date"].min(), generated_train_df["Date"].max(), color="#fed7aa", alpha=0.45, label="GAN rows used in forecast training")
    if not unused_real_df.empty:
        plt.axvspan(unused_real_df["Date"].min(), unused_real_df["Date"].max(), color="#dbeafe", alpha=0.35, label="Observed rows not used in forecast training")
    if not unused_generated_df.empty:
        plt.axvspan(unused_generated_df["Date"].min(), unused_generated_df["Date"].max(), color="#fce7f3", alpha=0.35, label="GAN rows not used in forecast training")
    plt.axvline(observed_end_date, color="#b45309", linestyle="--", linewidth=1.1, label="Last observed date used by GAN")
    plt.axvline(train_end_date, color="#2563eb", linestyle="--", linewidth=1.1, label="Forecast train end")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_frozen_forecast(asset: str, dataset_cfg: dict):
    forecast_cfg = FORECAST_CONFIG[asset]
    report_dir = forecast_cfg["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    params = read_json(forecast_cfg["params_path"])
    metadata = read_json(forecast_cfg["model_dir"] / "model_metadata.json")
    train_df = load_level_frame(forecast_cfg["train_csv"])
    test_df = build_test_frame(asset, forecast_cfg)
    source_df = load_level_frame(ROOT / dataset_cfg["source_file"])
    prepared_path = ROOT / dataset_cfg["prepared_file"] if dataset_cfg.get("prepared_file") else ROOT / dataset_cfg["source_file"]
    prepared_df = load_level_frame(prepared_path)
    extended_df = load_level_frame(forecast_cfg["extended_csv"])

    model = CNN_BiLSTM((metadata["lookback"], len(metadata["feature_cols"])), params)
    state_dict = torch.load(forecast_cfg["model_dir"] / metadata["model_file"], map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with open(forecast_cfg["model_dir"] / "x_scaler.pkl", "rb") as handle:
        x_scaler = __import__("pickle").load(handle)
    with open(forecast_cfg["model_dir"] / "y_scaler.pkl", "rb") as handle:
        y_scaler = __import__("pickle").load(handle)

    context_df = train_df.copy()
    rows = []
    observed_end_date = prepared_df["Date"].max()
    for _, row in test_df.iterrows():
        cols = list(dict.fromkeys(metadata["feature_cols"] + [forecast_cfg["target_col"]]))
        numeric_df = context_df[cols].copy()
        abs_last_price = float(numeric_df.iloc[-1][forecast_cfg["target_col"]])
        last_date = context_df.iloc[-1]["Date"]
        returns_df = numeric_df.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
        recent_returns = returns_df[metadata["feature_cols"]].iloc[-metadata["lookback"]:].copy()
        recent_scaled = x_scaler.transform(recent_returns)
        tensor = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(tensor).cpu().numpy().reshape(-1, 1)
        pred_return = float(y_scaler.inverse_transform(pred_scaled).item())
        pred_abs = abs_last_price * (1.0 + pred_return)
        actual_price = float(row[forecast_cfg["target_col"]])
        signed_error = pred_abs - actual_price
        rows.append(
            {
                "date": row["Date"],
                "actual_price": actual_price,
                "predicted_price": float(pred_abs),
                "context_end_date": last_date,
                "context_end_price": abs_last_price,
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
                "pct_error": abs(signed_error) / actual_price * 100.0 if actual_price else 0.0,
                "is_synthetic_actual": bool(row["Date"] > observed_end_date),
            }
        )
        context_df = pd.concat([context_df, row.to_frame().T], ignore_index=True)

    eval_df = pd.DataFrame(rows)
    eval_df["actual_direction"] = (eval_df["actual_price"] - eval_df["context_end_price"]).apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
    eval_df["pred_direction"] = (eval_df["predicted_price"] - eval_df["context_end_price"]).apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
    eval_df["direction_match"] = eval_df["actual_direction"] == eval_df["pred_direction"]

    metrics = {
        "asset": asset,
        "dataset_label": forecast_cfg["label"],
        "test_rows": int(len(eval_df)),
        "test_start_date": str(eval_df["date"].min().date()),
        "test_end_date": str(eval_df["date"].max().date()),
        "real_rows_in_eval": int((~eval_df["is_synthetic_actual"]).sum()),
        "synthetic_rows_in_eval": int(eval_df["is_synthetic_actual"].sum()),
        "rmse": float(mean_squared_error(eval_df["actual_price"], eval_df["predicted_price"]) ** 0.5),
        "mae": float(mean_absolute_error(eval_df["actual_price"], eval_df["predicted_price"])),
        "mape_percent": float(eval_df["pct_error"].mean()),
        "r2": float(r2_score(eval_df["actual_price"], eval_df["predicted_price"])),
        "directional_accuracy": float(eval_df["direction_match"].mean()),
        "max_abs_error": float(eval_df["abs_error"].max()),
        "worst_date": str(eval_df.loc[eval_df["abs_error"].idxmax(), "date"].date()),
    }

    prefix = forecast_cfg["eval_prefix"]
    csv_path = report_dir / f"{prefix}_actual_vs_prediction_extended_test.csv"
    metrics_path = report_dir / f"{prefix}_actual_vs_prediction_extended_metrics.json"
    plot_path = report_dir / f"{prefix}_actual_vs_prediction_extended_test.png"
    error_plot_path = report_dir / f"{prefix}_prediction_error_over_time.png"
    lifecycle_plot_path = report_dir / f"{prefix}_dataset_lifecycle.png"

    eval_df_out = eval_df.copy()
    eval_df_out["date"] = eval_df_out["date"].dt.strftime("%Y-%m-%d")
    eval_df_out["context_end_date"] = eval_df_out["context_end_date"].dt.strftime("%Y-%m-%d")
    eval_df_out.to_csv(csv_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    make_eval_plot(eval_df, plot_path, f"{asset.title()} actual vs predicted prices")
    make_error_plot(eval_df, error_plot_path, f"{asset.title()} prediction error over time")
    make_lifecycle_plot(
        source_df=source_df,
        extended_df=extended_df,
        observed_end_date=observed_end_date,
        train_end_date=train_df["Date"].max(),
        target_col=forecast_cfg["target_col"],
        output_path=lifecycle_plot_path,
        title=f"{asset.title()} dataset lifecycle",
    )

    return {
        "forecast_cfg": forecast_cfg,
        "metadata": metadata,
        "params": params,
        "eval_df": eval_df,
        "metrics": metrics,
        "csv_path": csv_path,
        "metrics_path": metrics_path,
        "plot_path": plot_path,
        "error_plot_path": error_plot_path,
        "lifecycle_plot_path": lifecycle_plot_path,
        "source_df": source_df,
        "prepared_df": prepared_df,
        "extended_df": extended_df,
        "train_df": train_df,
        "test_df": test_df,
        "observed_end_date": observed_end_date,
    }


def build_asset_section(dataset_cfg: dict, eval_bundle: dict):
    asset = dataset_cfg["asset"]
    report_dir = ROOT / "reports" / "gan_validation" / dataset_cfg["name"]
    selected_metrics = read_json(report_dir / "selected_candidate_metrics.json")
    report_summary = (report_dir / "report_summary.md").read_text(encoding="utf-8")
    source_path = ROOT / dataset_cfg["source_file"]
    source_df = load_level_frame(source_path)
    prepared_path = ROOT / dataset_cfg["prepared_file"] if dataset_cfg.get("prepared_file") else source_path
    prepared_df = load_level_frame(prepared_path)
    extended_df = eval_bundle["extended_df"]
    train_df = eval_bundle["train_df"]
    test_df = eval_bundle["test_df"]
    metadata = eval_bundle["metadata"]
    params = eval_bundle["params"]
    forecast_metrics = eval_bundle["metrics"]
    eval_df = eval_bundle["eval_df"]
    observed_end_date = eval_bundle["observed_end_date"]
    train_model_df = build_model_frame(train_df.set_index("Date"), eval_bundle["forecast_cfg"]["target_col"])
    test_model_df = build_model_frame(test_df.set_index("Date"), eval_bundle["forecast_cfg"]["target_col"]) if not test_df.empty else pd.DataFrame()

    prep_summary = None
    prep_summary_path = report_dir / "source_preparation_summary.json"
    if prep_summary_path.exists():
        prep_summary = read_json(prep_summary_path)

    stage_rows = [
        ["Raw source dataset", source_path.name, len(source_df), str(source_df["Date"].min().date()), str(source_df["Date"].max().date()), "Original source panel used by the GAN pipeline."],
    ]
    if dataset_cfg.get("prepared_file"):
        stage_rows.append(
            ["Prepared GAN-ready dataset", prepared_path.name, len(prepared_df), str(prepared_df["Date"].min().date()), str(prepared_df["Date"].max().date()), "Business-day or cleaned panel used directly by the GAN."]
        )
    stage_rows.extend(
        [
            ["Extended dataset", eval_bundle["forecast_cfg"]["extended_csv"].name, len(extended_df), str(extended_df["Date"].min().date()), str(extended_df["Date"].max().date()), "Prepared history plus GAN-generated future rows."],
            ["Forecast train split", eval_bundle["forecast_cfg"]["train_csv"].name, len(train_df), str(train_df["Date"].min().date()), str(train_df["Date"].max().date()), "Rows available to the frozen forecasting model."],
            ["Forecast evaluation window", "derived from extended dataset", len(test_df), str(test_df["Date"].min().date()), str(test_df["Date"].max().date()), "Rows used for actual-vs-prediction evaluation in this report."],
        ]
    )
    stage_df = pd.DataFrame(stage_rows, columns=["Stage", "File", "Rows", "Start date", "End date", "Notes"])

    observed_rows_used = train_df.loc[train_df["Date"] <= observed_end_date]
    generated_rows_used = train_df.loc[train_df["Date"] > observed_end_date]
    unused_real_rows = test_df.loc[test_df["Date"] <= observed_end_date]
    unused_generated_rows = test_df.loc[test_df["Date"] > observed_end_date]

    partition_df = pd.DataFrame(
        [
            [
                "Observed rows used in forecast training",
                int(len(observed_rows_used)),
                str(observed_rows_used["Date"].min().date()) if not observed_rows_used.empty else "n/a",
                str(observed_rows_used["Date"].max().date()) if not observed_rows_used.empty else "n/a",
                "Observed price-level rows from the prepared/source dataset that were available to the frozen forecaster.",
            ],
            [
                "GAN-generated rows used in forecast training",
                int(len(generated_rows_used)),
                str(generated_rows_used["Date"].min().date()) if not generated_rows_used.empty else "n/a",
                str(generated_rows_used["Date"].max().date()) if not generated_rows_used.empty else "n/a",
                "Synthetic rows appended before the forecasting train cutoff.",
            ],
            [
                "Observed rows not used in forecast training",
                int(len(unused_real_rows)),
                str(unused_real_rows["Date"].min().date()) if not unused_real_rows.empty else "n/a",
                str(unused_real_rows["Date"].max().date()) if not unused_real_rows.empty else "n/a",
                "Held-out observed rows used only for prediction/evaluation, not for fitting the frozen forecaster.",
            ],
            [
                "GAN-generated rows not used in forecast training",
                int(len(unused_generated_rows)),
                str(unused_generated_rows["Date"].min().date()) if not unused_generated_rows.empty else "n/a",
                str(unused_generated_rows["Date"].max().date()) if not unused_generated_rows.empty else "n/a",
                "Held-out synthetic rows used only for prediction/evaluation, not for fitting the frozen forecaster.",
            ],
        ],
        columns=["Partition", "Rows", "Start date", "End date", "Meaning"],
    )

    model_window_df = pd.DataFrame(
        [
            [
                "Forecast-train model rows",
                int(len(train_model_df)),
                str(train_model_df.index.min().date()) if not train_model_df.empty else "n/a",
                str(train_model_df.index.max().date()) if not train_model_df.empty else "n/a",
                "These are the actual supervised rows after pct_change() and the t+1 target shift.",
            ],
            [
                "Forecast-eval model rows",
                int(len(test_model_df)),
                str(test_model_df.index.min().date()) if not test_model_df.empty else "n/a",
                str(test_model_df.index.max().date()) if not test_model_df.empty else "n/a",
                "These rows were not used for fitting; they are forecast contexts/targets only.",
            ],
        ],
        columns=["Model partition", "Rows", "First model date", "Last model date", "Notes"],
    )

    thresholds_df = pd.DataFrame(
        [
            ["avg_ks_stat", round(selected_metrics["avg_ks_stat"], 6), "<= 0.12", "Yes" if selected_metrics["avg_ks_stat"] <= 0.12 else "No"],
            ["avg_acf_gap", round(selected_metrics["avg_acf_gap"], 6), "<= 0.15", "Yes" if selected_metrics["avg_acf_gap"] <= 0.15 else "No"],
            ["corr_gap", round(selected_metrics["corr_gap"], 6), "<= 0.25", "Yes" if selected_metrics["corr_gap"] <= 0.25 else "No"],
        ],
        columns=["Metric", "Current value", "Threshold", "Pass"],
    )

    model_df = pd.DataFrame(
        [
            ["Model label", eval_bundle["forecast_cfg"]["label"]],
            ["Frozen model seed", metadata["seed"]],
            ["Lookback", metadata["lookback"]],
            ["Feature count", len(metadata["feature_cols"])],
            ["Batch size", params["batch_size"]],
            ["Filters", params["filters"]],
            ["Kernel size", params["kernel_size"]],
            ["LSTM units", params["lstm_units"]],
            ["Dense units", params["dense_units"]],
            ["Dropout", round(params["dropout_rate"], 6)],
            ["Learning rate", round(params["learning_rate"], 9)],
            ["L2 regularization", round(params["l2_reg"], 9)],
        ],
        columns=["Setting", "Value"],
    )

    missingness_html = ""
    notes_html = ""
    if prep_summary is not None:
        missingness_df = pd.DataFrame(
            [
                [col, prep_summary["raw_missing_pct"][col], prep_summary["prepared_missing_cells_before_fill"][col], prep_summary["prepared_missing_cells_after_fill"][col]]
                for col in prep_summary["raw_missing_pct"]
            ],
            columns=["Column", "Raw missing %", "Prepared missing cells before fill", "Prepared missing cells after fill"],
        )
        missingness_html = "<h4>Preparation Missingness</h4>" + table_html(missingness_df)
        notes_html = "<ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in prep_summary["notes"]) + "</ul>"
    else:
        notes_html = "<p>No extra preparation step was configured for this dataset; the GAN used the source file directly.</p>"

    preview_df = eval_df[["date", "actual_price", "predicted_price", "abs_error", "pct_error", "is_synthetic_actual"]].head(12).copy()
    preview_df["date"] = preview_df["date"].dt.strftime("%Y-%m-%d")
    top_errors_df = eval_df.nlargest(10, "abs_error")[["date", "actual_price", "predicted_price", "abs_error", "pct_error", "is_synthetic_actual"]].copy()
    top_errors_df["date"] = top_errors_df["date"].dt.strftime("%Y-%m-%d")

    subset_metrics_df = pd.DataFrame(
        [
            safe_subset_metrics(eval_df, "All rows not used in forecast training"),
            safe_subset_metrics(eval_df.loc[~eval_df["is_synthetic_actual"]].copy(), "Observed rows not used in forecast training"),
            safe_subset_metrics(eval_df.loc[eval_df["is_synthetic_actual"]].copy(), "GAN-generated rows not used in forecast training"),
        ]
    )

    quality_cls = "ok" if selected_metrics["quality_label"] == "good" else ("warn" if selected_metrics["quality_label"] == "usable_with_caution" else "danger")
    readiness_text = "READY" if thresholds_df["Pass"].eq("Yes").all() and selected_metrics["quality_label"] != "reject" else "NOT_READY"
    readiness_cls = "ok" if readiness_text == "READY" else "danger"

    validity_points = [
        "The extended file is structurally valid: no duplicate dates, no null cells, and the generated horizon reaches the intended end date.",
        f"The current selected GAN candidate is rated {selected_metrics['quality_label']} and clears the repo gate with avg_ks_stat={selected_metrics['avg_ks_stat']:.4f}, avg_acf_gap={selected_metrics['avg_acf_gap']:.4f}, and corr_gap={selected_metrics['corr_gap']:.4f}."
        if readiness_text == "READY"
        else f"The current selected GAN candidate is structurally usable but does not fully clear the repo gate because one or more metrics miss the thresholds.",
        "The forecasting model predictions shown below are produced on rows that were not used to fit the frozen forecasting model.",
        "This makes the report valid for internal model-behavior analysis, dataset-extension validation, and simulation testing.",
        "The key limitation is that any evaluation rows marked as GAN-generated are synthetic actuals, so they support internal validation rather than real-market performance claims.",
    ]

    return f"""
<section class="asset-section">
  <h2 id="{asset}">{asset.title()} GAN and Forecasting</h2>
  <div class="grid-4">
    <div class="metric-card"><div class="label">GAN quality label</div><div class="value">{badge(selected_metrics["quality_label"], quality_cls)}</div></div>
    <div class="metric-card"><div class="label">Strict gate</div><div class="value">{badge(readiness_text, readiness_cls)}</div></div>
    <div class="metric-card"><div class="label">Forecast eval rows</div><div class="value">{forecast_metrics["test_rows"]}</div></div>
    <div class="metric-card"><div class="label">Synthetic rows in eval</div><div class="value">{forecast_metrics["synthetic_rows_in_eval"]}</div></div>
  </div>
  <p>This section covers the current configured GAN dataset <span class="mono">{dataset_cfg["name"]}</span> and the frozen forecasting model attached to the <span class="mono">{asset}</span> pipeline. It separates the observed history, the GAN-generated extension, the part used in forecast training, and the part left unused for prediction-only evaluation.</p>
  <h3>Dataset Lifecycle</h3>
  {table_html(stage_df)}
  <h3>What Was Used For Forecast Training And What Was Not</h3>
  {table_html(partition_df)}
  {table_html(model_window_df)}
  <h3>Preparation Notes</h3>
  {notes_html}
  {missingness_html}
  {render_image(eval_bundle["lifecycle_plot_path"], f"{asset.title()} dataset lifecycle showing observed training data, GAN-generated training data, and rows not used in forecast training.")}
  <h3>GAN Readiness Metrics</h3>
  {table_html(thresholds_df)}
  <ul>
    <li><strong>avg_vol_gap:</strong> {selected_metrics["avg_vol_gap"]:.6f}</li>
    <li><strong>avg_mean_gap_z:</strong> {selected_metrics["avg_mean_gap_z"]:.6f}</li>
    <li><strong>avg_ks_stat:</strong> {selected_metrics["avg_ks_stat"]:.6f}</li>
    <li><strong>avg_acf_gap:</strong> {selected_metrics["avg_acf_gap"]:.6f}</li>
    <li><strong>corr_gap:</strong> {selected_metrics["corr_gap"]:.6f}</li>
    <li><strong>selected seed / candidate:</strong> {selected_metrics["seed"]} / {selected_metrics["candidate"]}</li>
  </ul>
  <pre class="report-summary">{html.escape(report_summary)}</pre>
  {render_image(report_dir / f"training_loss_curve_seed_{selected_metrics['seed']}.png", f"{asset.title()} GAN training losses for the selected seed.")}
  {render_image(report_dir / "plots" / "recent_history_vs_generated.png", f"{asset.title()} recent history vs generated future.")}
  {render_image(report_dir / "plots" / "stationary_distribution_comparison.png", f"{asset.title()} stationary distribution comparison.")}
  {render_image(report_dir / "plots" / "stationary_autocorrelation.png", f"{asset.title()} stationary autocorrelation comparison.")}
  {render_image(report_dir / "plots" / "stationary_correlation_heatmaps.png", f"{asset.title()} stationary correlation heatmaps.")}
  <h3>Frozen Forecast Model</h3>
  {table_html(model_df)}
  <p>Forecast features: <span class="mono">{", ".join(metadata["feature_cols"])}</span></p>
  {render_image(eval_bundle["forecast_cfg"]["report_dir"] / "plots" / eval_bundle["forecast_cfg"]["plot_name"], f"{asset.title()} original saved-model prediction plot.")}
  <h3>Prediction On Data Not Used In Forecast Training</h3>
  <div class="grid-4">
    <div class="metric-card"><div class="label">RMSE</div><div class="value">{forecast_metrics["rmse"]:.2f}</div></div>
    <div class="metric-card"><div class="label">MAE</div><div class="value">{forecast_metrics["mae"]:.2f}</div></div>
    <div class="metric-card"><div class="label">R2</div><div class="value">{forecast_metrics["r2"]:.4f}</div></div>
    <div class="metric-card"><div class="label">MAPE</div><div class="value">{forecast_metrics["mape_percent"]:.3f}%</div></div>
  </div>
  <p class="small">Evaluation window: <span class="mono">{forecast_metrics["test_start_date"]}</span> to <span class="mono">{forecast_metrics["test_end_date"]}</span>. Real rows in evaluation: <span class="mono">{forecast_metrics["real_rows_in_eval"]}</span>. Synthetic rows in evaluation: <span class="mono">{forecast_metrics["synthetic_rows_in_eval"]}</span>.</p>
  {table_html(subset_metrics_df)}
  {render_image(eval_bundle["plot_path"], f"{asset.title()} actual vs predicted prices over the extended evaluation window.")}
  {render_image(eval_bundle["error_plot_path"], f"{asset.title()} prediction error over time.")}
  <h4>First 12 Evaluation Rows</h4>
  {table_html(preview_df)}
  <h4>Top 10 Worst Absolute Errors</h4>
  {table_html(top_errors_df)}
  <h3>Discussion And Validity Justification</h3>
  <ul>
    {''.join(f"<li>{html.escape(point)}</li>" for point in validity_points)}
  </ul>
</section>
"""


def build_html(asset_sections: list[str], summary_df: pd.DataFrame):
    references_html = "".join(
        f"<li><a href='{html.escape(url)}' target='_blank' rel='noreferrer'>{html.escape(title)}</a> - {html.escape(reason)}</li>"
        for title, url, reason in REFERENCE_LINKS
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GAN Model Documentation</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;color:#111827;background:#fff;margin:0;line-height:1.55}} .page{{max-width:1220px;margin:0 auto;padding:28px 20px 60px}} h1,h2,h3,h4{{margin-top:0;color:#111827}} h1{{font-size:30px}} h2{{font-size:24px;border-bottom:1px solid #e5e7eb;padding-bottom:8px;margin-top:34px}} h3{{font-size:18px;margin-top:24px}} p,li{{font-size:15px}} .muted{{color:#4b5563}} .summary-box,.warning-box{{padding:16px 18px;margin:16px 0 24px;border:1px solid #d1d5db;background:#f9fafb}} .warning-box{{border-color:#f59e0b;background:#fffbeb}} .grid-4{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}} .metric-card{{border:1px solid #d1d5db;padding:14px;background:#fff}} .metric-card .label{{font-size:13px;color:#6b7280;text-transform:uppercase;letter-spacing:.03em}} .metric-card .value{{font-size:24px;font-weight:700;margin-top:6px}} .badge{{display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700}} .badge.ok{{background:#dcfce7;color:#166534}} .badge.warn{{background:#fef3c7;color:#92400e}} .badge.danger{{background:#fee2e2;color:#991b1b}} .data-table{{width:100%;border-collapse:collapse;margin:12px 0 20px;font-size:14px}} .data-table th,.data-table td{{border:1px solid #d1d5db;padding:8px 10px;text-align:left;vertical-align:top}} .data-table th{{background:#f3f4f6}} .plot{{margin:18px 0 28px;border:1px solid #d1d5db;padding:12px;background:#fff}} .plot img{{width:100%;height:auto;display:block}} .plot figcaption{{margin-top:10px;color:#4b5563;font-size:14px}} .report-summary{{white-space:pre-wrap;background:#f9fafb;border:1px solid #d1d5db;padding:12px;font-size:13px;overflow-x:auto}} .mono{{font-family:Consolas,Menlo,monospace}} .small{{font-size:13px}} .missing-asset{{border:1px dashed #9ca3af;padding:16px;color:#6b7280;margin:16px 0}} @media (max-width:900px){{.grid-4{{grid-template-columns:1fr}}}}
</style></head><body><div class="page">
<h1>GAN Model Documentation</h1>
<p class="muted">This file supersedes the old gold-only HTML and combines the current gold and silver GAN pipelines into one readable report. It explains why GAN is being used in this forecasting system, what each GAN produced, what the forecasters were trained on, what was left out of training, and how the predictions behaved on the unused data.</p>
<div class="summary-box"><strong>System-level interpretation.</strong> In this repo, GANs are scenario builders and dataset extenders, not the final forecasting models. They create future-dated multivariate panels so the downstream CNN-BiLSTM forecasters can be evaluated on longer horizons. A GAN run is considered valid here when the extended file is structurally sound and the selected candidate preserves return-space distribution, autocorrelation, and cross-feature correlation closely enough to pass the repo thresholds.</div>
<h2>Why GAN Is Used In This System</h2>
<ul><li>The forecasters consume multiple interacting market and macro variables, not only one target series.</li><li>The training pipelines operate in return space, so synthetic data should preserve return distributions and dependence structure rather than only smooth price levels.</li><li>Simple interpolation is usually too smooth for financial time series and does not preserve cross-feature joint behavior.</li><li>GANs are being used here as multivariate future panel generators for augmentation, synthetic holdouts, and dashboard simulations.</li></ul>
<div class="warning-box"><strong>Important boundary.</strong> Good synthetic behavior does not automatically prove real-market forecasting skill. In the sections below, any evaluation rows marked as GAN-generated are synthetic actuals. They support internal validation and simulation, not real-market performance claims.</div>
<h2>Cross-Asset Summary</h2>
{table_html(summary_df)}
{''.join(asset_sections)}
<h2>References</h2>
<ul>{references_html}</ul>
</div></body></html>"""


def main():
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    asset_sections = []
    summary_rows = []
    for dataset_cfg in get_all_dataset_configs():
        asset = dataset_cfg["asset"]
        if asset not in FORECAST_CONFIG:
            continue
        eval_bundle = evaluate_frozen_forecast(asset, dataset_cfg)
        selected_metrics = read_json(ROOT / "reports" / "gan_validation" / dataset_cfg["name"] / "selected_candidate_metrics.json")
        forecast_metrics = eval_bundle["metrics"]
        observed_end_date = eval_bundle["observed_end_date"]
        train_df = eval_bundle["train_df"]
        test_df = eval_bundle["test_df"]
        strict_ready = (
            selected_metrics["quality_label"] != "reject"
            and selected_metrics["avg_ks_stat"] <= 0.12
            and selected_metrics["avg_acf_gap"] <= 0.15
            and selected_metrics["corr_gap"] <= 0.25
        )
        summary_rows.append(
            {
                "Asset": asset,
                "GAN dataset": dataset_cfg["name"],
                "GAN quality": selected_metrics["quality_label"],
                "Strict gate": "READY" if strict_ready else "NOT_READY",
                "GAN avg_ks_stat": round(selected_metrics["avg_ks_stat"], 4),
                "GAN avg_acf_gap": round(selected_metrics["avg_acf_gap"], 4),
                "GAN corr_gap": round(selected_metrics["corr_gap"], 4),
                "Observed rows used in training": int((train_df["Date"] <= observed_end_date).sum()),
                "GAN rows used in training": int((train_df["Date"] > observed_end_date).sum()),
                "Rows not used in training": int(len(test_df)),
                "GAN rows not used in training": int((test_df["Date"] > observed_end_date).sum()),
                "Forecast RMSE": round(forecast_metrics["rmse"], 4),
                "Forecast R2": round(forecast_metrics["r2"], 4),
            }
        )
        asset_sections.append(build_asset_section(dataset_cfg, eval_bundle))

    summary_df = pd.DataFrame(summary_rows)
    html_text = build_html(asset_sections, summary_df)
    OUTPUT_HTML.write_text(html_text, encoding="utf-8")
    for alias_path in ALIAS_OUTPUTS:
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        alias_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
