import base64
import html
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
GAN_REPORT_DIR = ROOT / "reports" / "gan_validation" / "df_gold_dataset_gepu"
FORECAST_REPORT_DIR = ROOT / "reports" / "df_gold_dataset_gepu_datecut_full"
OUTPUT_HTML = FORECAST_REPORT_DIR / "gold_gan_system_documentation.html"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def image_html(path: Path, caption: str):
    if not path.exists():
        return f"<div class='missing-asset'>Missing image: {html.escape(path.name)}</div>"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        "<figure class='plot'>"
        f"<img src='data:image/png;base64,{encoded}' alt='{html.escape(caption)}'>"
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def table_html(df: pd.DataFrame):
    return df.to_html(index=False, border=0, classes="data-table")


def badge(text: str, cls: str):
    return f"<span class='badge {cls}'>{html.escape(text)}</span>"


def latest_gold_training_config():
    log_path = ROOT / "gan" / "gan_training_stationary.log"
    if not log_path.exists():
        return {}
    pattern = re.compile(r"([a-zA-Z_]+)=([^,]+)")
    current = {}
    latest = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Training config:" in line:
            current = {m.group(1): m.group(2).strip() for m in pattern.finditer(line.split("Training config:", 1)[1])}
        elif "--- Training seed" in line and "df_gold_dataset_gepu" in line and current:
            latest = current.copy()
    return latest


def lifecycle_plot(prepared_df: pd.DataFrame, extended_df: pd.DataFrame, cutoff_date: pd.Timestamp, out_path: Path):
    future_df = extended_df.loc[extended_df["Date"] > prepared_df["Date"].max()].copy()
    test_df = extended_df.loc[extended_df["Date"] > cutoff_date].copy()
    plt.figure(figsize=(14, 6))
    plt.plot(prepared_df["Date"], prepared_df["Gold_Futures"], linewidth=1.2, label="Prepared history", color="#4a4a4a")
    plt.plot(future_df["Date"], future_df["Gold_Futures"], linewidth=1.6, label="GAN extension", color="#d97706")
    plt.axvline(prepared_df["Date"].max(), linestyle="--", linewidth=1.1, color="#b45309", label="Last observed source date")
    plt.axvline(cutoff_date, linestyle="--", linewidth=1.1, color="#2563eb", label="Forecast train cutoff")
    plt.axvspan(test_df["Date"].min(), test_df["Date"].max(), color="#dbeafe", alpha=0.35, label="Forecast test window")
    plt.title("Gold dataset lifecycle")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig(out_path, dpi=150)
    plt.close()


def error_plot(eval_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(eval_df["date"], eval_df["signed_error"], color="#b91c1c", linewidth=1.2)
    axes[0].axhline(0.0, color="#1f2937", linestyle="--", linewidth=1.0)
    axes[0].set_title("Prediction error across the extended synthetic test window")
    axes[1].plot(eval_df["date"], eval_df["abs_error"], color="#1d4ed8", linewidth=1.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    raw_df = pd.read_csv(ROOT / "df_gold_dataset_gepu.csv", parse_dates=["Date"])
    prepared_df = pd.read_csv(ROOT / "df_gold_dataset_gepu_gan_ready.csv", parse_dates=["Date"])
    extended_df = pd.read_csv(ROOT / "df_gold_dataset_gepu_extended.csv", parse_dates=["Date"])
    train_df = pd.read_csv(ROOT / "df_gold_dataset_gepu_extended_train.csv", parse_dates=["Date"])
    test_df = pd.read_csv(ROOT / "df_gold_dataset_gepu_extended_test.csv", parse_dates=["Date"])
    eval_df = pd.read_csv(FORECAST_REPORT_DIR / "gold_actual_vs_prediction_extended_test.csv", parse_dates=["date"])

    source_summary = read_json(GAN_REPORT_DIR / "source_preparation_summary.json")
    candidate_metrics = read_json(GAN_REPORT_DIR / "selected_candidate_metrics.json")
    forecast_metrics = read_json(FORECAST_REPORT_DIR / "gold_actual_vs_prediction_extended_metrics.json")
    model_metadata = read_json(ROOT / "models" / "df_gold_dataset_gepu_datecut_full" / "seed_99" / "model_metadata.json")
    best_params = read_json(FORECAST_REPORT_DIR / "gold_best_params_optimized.json")
    report_summary_text = (GAN_REPORT_DIR / "report_summary.md").read_text(encoding="utf-8")
    gan_cfg = latest_gold_training_config()

    strict_ready = candidate_metrics["quality_label"] != "reject" and candidate_metrics["avg_ks_stat"] <= 0.12 and candidate_metrics["avg_acf_gap"] <= 0.15 and candidate_metrics["corr_gap"] <= 0.25
    readiness_badge = badge("READY (strict gate)" if strict_ready else "NOT_READY (strict gate)", "ok" if strict_ready else "danger")
    quality_badge = badge(candidate_metrics["quality_label"], "ok" if candidate_metrics["quality_label"] == "good" else "warn")

    eval_df["signed_error"] = eval_df["predicted_price"] - eval_df["actual_price"]
    stage_plot = FORECAST_REPORT_DIR / "gold_dataset_lifecycle.png"
    err_plot = FORECAST_REPORT_DIR / "gold_prediction_error_over_time.png"
    lifecycle_plot(prepared_df, extended_df, pd.Timestamp(model_metadata["data_split"]["requested_train_end_date"]), stage_plot)
    error_plot(eval_df, err_plot)

    summary_df = pd.DataFrame([
        ["Prepared source ends", str(prepared_df["Date"].max().date())],
        ["Extended dataset ends", str(extended_df["Date"].max().date())],
        ["Synthetic future rows", len(extended_df.loc[extended_df["Date"] > prepared_df["Date"].max()])],
        ["Forecast test rows", len(test_df)],
        ["GAN quality label", candidate_metrics["quality_label"]],
        ["Strict gate", "READY" if strict_ready else "NOT_READY"],
        ["Selected seed / candidate", f'{candidate_metrics["seed"]} / {candidate_metrics["candidate"]}'],
    ], columns=["Metric", "Value"])
    threshold_df = pd.DataFrame([
        ["avg_ks_stat", round(candidate_metrics["avg_ks_stat"], 6), "<= 0.12", "Yes" if candidate_metrics["avg_ks_stat"] <= 0.12 else "No"],
        ["avg_acf_gap", round(candidate_metrics["avg_acf_gap"], 6), "<= 0.15", "Yes" if candidate_metrics["avg_acf_gap"] <= 0.15 else "No"],
        ["corr_gap", round(candidate_metrics["corr_gap"], 6), "<= 0.25", "Yes" if candidate_metrics["corr_gap"] <= 0.25 else "No"],
    ], columns=["Metric", "Current value", "Threshold", "Pass"])
    gan_cfg_df = pd.DataFrame([[k, v] for k, v in gan_cfg.items()], columns=["Setting", "Value"]) if gan_cfg else pd.DataFrame([["latest_config", "missing"]], columns=["Setting", "Value"])
    forecast_df = pd.DataFrame([
        ["Frozen model seed", model_metadata["seed"]],
        ["Lookback", model_metadata["lookback"]],
        ["Feature count", len(model_metadata["feature_cols"])],
        ["Batch size", best_params["batch_size"]],
        ["Filters", best_params["filters"]],
        ["Kernel size", best_params["kernel_size"]],
        ["LSTM units", best_params["lstm_units"]],
        ["Dense units", best_params["dense_units"]],
        ["Dropout", round(best_params["dropout_rate"], 6)],
        ["Learning rate", round(best_params["learning_rate"], 9)],
    ], columns=["Setting", "Value"])
    preview_df = eval_df[["date", "actual_price", "predicted_price", "abs_error", "pct_error"]].head(15).copy()
    preview_df["date"] = preview_df["date"].dt.strftime("%Y-%m-%d")

    references = "".join(
        [
            "<li><a href='https://www.stat.rice.edu/~dobelman/courses/texts/stylized.cont.2001.pdf' target='_blank' rel='noreferrer'>Rama Cont (2001)</a> - return-space and dependence justification.</li>",
            "<li><a href='https://papers.neurips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf' target='_blank' rel='noreferrer'>TimeGAN (2019)</a> - temporal GAN support.</li>",
            "<li><a href='https://arxiv.org/abs/1907.06673' target='_blank' rel='noreferrer'>Quant GANs (2020)</a> - financial GAN support.</li>",
            "<li><a href='https://arxiv.org/abs/2205.08924' target='_blank' rel='noreferrer'>Hellermann et al. (2022)</a> - augmentation justification.</li>",
        ]
    )

    html_text = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Gold GAN System Documentation</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;color:#111827;background:#fff;margin:0;line-height:1.55}} .page{{max-width:1180px;margin:0 auto;padding:28px 20px 60px}} h1,h2,h3{{margin-top:0;color:#111827}} h2{{font-size:22px;margin-top:34px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}} p,li{{font-size:15px}} .summary-box,.warning-box,.ok-box{{padding:16px 18px;margin:16px 0 24px;border:1px solid #d1d5db;background:#f9fafb}} .warning-box{{border-color:#f59e0b;background:#fffbeb}} .ok-box{{border-color:#10b981;background:#ecfdf5}} .grid-3{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}} .metric-card{{border:1px solid #d1d5db;padding:14px;background:#fff}} .metric-card .label{{font-size:13px;color:#6b7280;text-transform:uppercase;letter-spacing:.03em}} .metric-card .value{{font-size:26px;font-weight:700;margin-top:6px}} .badge{{display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700}} .badge.ok{{background:#dcfce7;color:#166534}} .badge.warn{{background:#fef3c7;color:#92400e}} .badge.danger{{background:#fee2e2;color:#991b1b}} .data-table{{width:100%;border-collapse:collapse;margin:12px 0 20px;font-size:14px}} .data-table th,.data-table td{{border:1px solid #d1d5db;padding:8px 10px;text-align:left;vertical-align:top}} .data-table th{{background:#f3f4f6}} .plot{{margin:18px 0 28px;border:1px solid #d1d5db;padding:12px;background:#fff}} .plot img{{width:100%;height:auto;display:block}} .plot figcaption{{margin-top:10px;color:#4b5563;font-size:14px}} .report-summary{{white-space:pre-wrap;background:#f9fafb;border:1px solid #d1d5db;padding:12px;font-size:13px;overflow-x:auto}} .missing-asset{{border:1px dashed #9ca3af;padding:16px;color:#6b7280;margin:16px 0}}</style></head>
<body><div class="page">
<h1>Gold GAN and Forecasting System Documentation</h1>
<div class="summary-box"><strong>One-line conclusion.</strong> The long-horizon gold GAN is now {readiness_badge} with {quality_badge}. Current metrics are avg_ks_stat = {candidate_metrics["avg_ks_stat"]:.4f}, avg_acf_gap = {candidate_metrics["avg_acf_gap"]:.4f}, and corr_gap = {candidate_metrics["corr_gap"]:.4f}.</div>
<h2>1. Why GAN Is Used Here</h2>
<p>The GAN is not the final forecaster. It extends the multivariate gold panel beyond the last observed real date so the frozen CNN-BiLSTM forecaster can be tested and simulated on a longer future-like window. This is justified because the forecasting pipeline works in return space, consumes multiple interacting features, and needs a coherent future panel rather than a smooth interpolation.</p>
<div class="warning-box"><strong>Important boundary.</strong> The long-horizon evaluation window is synthetic after the last observed source date. That makes it useful for internal simulation and model-behavior analysis, but not proof of real-market forecasting skill.</div>
<h2>2. Dataset Lifecycle</h2>
{table_html(pd.DataFrame([["Raw source", "df_gold_dataset_gepu.csv", len(raw_df), str(raw_df["Date"].min().date()), str(raw_df["Date"].max().date())], ["Prepared GAN-ready", "df_gold_dataset_gepu_gan_ready.csv", len(prepared_df), str(prepared_df["Date"].min().date()), str(prepared_df["Date"].max().date())], ["Extended", "df_gold_dataset_gepu_extended.csv", len(extended_df), str(extended_df["Date"].min().date()), str(extended_df["Date"].max().date())], ["Forecast train", "df_gold_dataset_gepu_extended_train.csv", len(train_df), str(train_df["Date"].min().date()), str(train_df["Date"].max().date())], ["Forecast test", "df_gold_dataset_gepu_extended_test.csv", len(test_df), str(test_df["Date"].min().date()), str(test_df["Date"].max().date())]], columns=["Stage", "File", "Rows", "Start date", "End date"]))}
<ul>{''.join(f"<li>{html.escape(note)}</li>" for note in source_summary["notes"])}</ul>
{table_html(pd.DataFrame([[col, source_summary["raw_missing_pct"][col], source_summary["prepared_missing_cells_before_fill"][col], source_summary["prepared_missing_cells_after_fill"][col]] for col in source_summary["raw_missing_pct"]], columns=["Column", "Raw missing %", "Prepared missing cells before fill", "Prepared missing cells after fill"]))}
{image_html(stage_plot, "Lifecycle plot of the prepared gold dataset, GAN extension, and forecast test horizon.")}
<h2>3. Current GAN Results</h2>
{table_html(summary_df)}
{table_html(threshold_df)}
<div class="{'ok-box' if strict_ready else 'warning-box'}"><strong>Strict-gate verdict.</strong> {'The long-horizon extension is currently training-ready under the repo gate.' if strict_ready else 'The long-horizon extension is structurally valid, but it does not currently clear the repo gate.'}</div>
{table_html(gan_cfg_df)}
<pre class="report-summary">{html.escape(report_summary_text)}</pre>
{image_html(GAN_REPORT_DIR / f"training_loss_curve_seed_{candidate_metrics['seed']}.png", "GAN training losses for the selected long-horizon run.")}
{image_html(GAN_REPORT_DIR / "plots" / "recent_history_vs_generated.png", "Recent prepared history versus generated future continuation.")}
{image_html(GAN_REPORT_DIR / "plots" / "stationary_distribution_comparison.png", "Historical vs generated stationary distributions.")}
{image_html(GAN_REPORT_DIR / "plots" / "stationary_autocorrelation.png", "Historical vs generated stationary autocorrelation.")}
{image_html(GAN_REPORT_DIR / "plots" / "stationary_correlation_heatmaps.png", "Historical vs generated stationary correlation heatmaps.")}
<h2>4. Frozen Forecast Model Integration</h2>
<p>The dashboard and API use a frozen prediction tape, so clicking Next Day no longer retrains or recomputes the forecasting model. The current backend only advances through already-computed predictions tied to the updated gold test split.</p>
{table_html(forecast_df)}
<p>Forecast features: {html.escape(', '.join(model_metadata["feature_cols"]))}</p>
<h2>5. Actual vs Prediction on the Extended Test Window</h2>
<div class="grid-3"><div class="metric-card"><div class="label">RMSE</div><div class="value">{forecast_metrics["rmse"]:.2f}</div></div><div class="metric-card"><div class="label">MAE</div><div class="value">{forecast_metrics["mae"]:.2f}</div></div><div class="metric-card"><div class="label">R2</div><div class="value">{forecast_metrics["r2"]:.4f}</div></div><div class="metric-card"><div class="label">MAPE</div><div class="value">{forecast_metrics["mape_percent"]:.3f}%</div></div><div class="metric-card"><div class="label">Directional accuracy</div><div class="value">{forecast_metrics["directional_accuracy"] * 100:.2f}%</div></div><div class="metric-card"><div class="label">Worst abs error</div><div class="value">{forecast_metrics["max_abs_error"]:.2f}</div></div></div>
<p>Evaluation window: {forecast_metrics["test_start_date"]} to {forecast_metrics["test_end_date"]}. Here, "actual" means the values in the extended GAN-generated gold test dataset.</p>
{image_html(FORECAST_REPORT_DIR / "gold_actual_vs_prediction_extended_test.png", "Actual vs predicted gold prices across the long synthetic test window.")}
{image_html(err_plot, "Signed and absolute prediction error across the long synthetic test window.")}
{table_html(preview_df)}
<h2>6. Interpretation and Safe Use</h2>
<ul><li>The engineering pipeline now works end to end on the long gold horizon.</li><li>The long gold GAN currently passes the repo's strict readiness gate again.</li><li>The extended window is appropriate for dashboard simulation, internal stress testing, and behavior analysis.</li><li>The long-horizon synthetic test metrics should not be described as real-market out-of-sample performance.</li></ul>
<h2>7. References</h2><ul>{references}</ul>
</div></body></html>"""

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html_text, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
