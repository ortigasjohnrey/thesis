import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import logging
import json
from scipy.stats import ks_2samp
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TARGET_END_DATE = pd.Timestamp("2026-05-08")
SOURCE_FILES = ["gold_RRL_interpolate.csv", "silver_RRL_interpolate.csv"]
GAN_REPORTS_ROOT = os.path.join(PROJECT_ROOT, "reports", "gan_validation")

# Set up logging to file with more detail
logging.basicConfig(
    filename=os.path.join(SCRIPT_DIR, 'gan_training_stationary.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for Stationary Returns training: {device}")
logging.info(f"Initialized GAN Training on {device}")

if device.type == "cuda":
    # Warm up CUDA once so cuBLAS/cuDNN contexts are created before training logs begin.
    _cuda_warmup = torch.zeros(1, device=device)
    _cuda_warmup = _cuda_warmup + 1
    torch.cuda.synchronize()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    logging.info("CUDA context warmup completed.")

# High Quality Hyperparameters
WINDOW_SIZE = 15
NOISE_DIM = 32
BATCH_SIZE = int(os.getenv("GAN_BATCH_SIZE", "256" if device.type == "cuda" else "64"))
EPOCHS = int(os.getenv("GAN_EPOCHS", "250"))
LR_G = float(os.getenv("GAN_LR_G", "0.0002"))
LR_D = float(os.getenv("GAN_LR_D", "0.0002"))
N_CRITIC = int(os.getenv("GAN_N_CRITIC", "3"))
GEN_HIDDEN_SIZE = int(os.getenv("GAN_GEN_HIDDEN_SIZE", "128"))
DISC_HIDDEN_SIZE = int(os.getenv("GAN_DISC_HIDDEN_SIZE", "128"))
GEN_DROPOUT = float(os.getenv("GAN_GEN_DROPOUT", "0.10"))
MSE_LOSS_WEIGHT = float(os.getenv("GAN_MSE_LOSS_WEIGHT", "1.0"))
MOMENT_LOSS_WEIGHT = float(os.getenv("GAN_MOMENT_LOSS_WEIGHT", "0.75"))
DRIFT_LOSS_WEIGHT = float(os.getenv("GAN_DRIFT_LOSS_WEIGHT", "0.25"))
NUM_WORKERS = int(os.getenv("GAN_NUM_WORKERS", "0"))
PIN_MEMORY = device.type == "cuda"
USE_AMP = device.type == "cuda"
NUM_CANDIDATES = int(os.getenv("GAN_NUM_CANDIDATES", "32" if device.type == "cuda" else "12"))
RECENT_REF_DAYS = int(os.getenv("GAN_RECENT_REF_DAYS", "504"))


def parse_seed_list(env_name, default_value):
    raw = os.getenv(env_name, default_value)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


GOLD_SEEDS = parse_seed_list("GAN_GOLD_SEEDS", "0,1,2,3")
SILVER_SEEDS = parse_seed_list("GAN_SILVER_SEEDS", "0")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        channels = input_size + NOISE_DIM
        self.backbone = nn.Sequential(
            ConvBlock(channels, hidden_size, dilation=1),
            ConvBlock(hidden_size, hidden_size, dilation=2),
            ConvBlock(hidden_size, hidden_size, dilation=4),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(GEN_DROPOUT),
            nn.Linear(hidden_size, output_size),
            nn.Tanh(),
        )

    def forward(self, history, noise):
        x = torch.cat((history, noise), dim=2).transpose(1, 2)
        hidden = self.backbone(x).transpose(1, 2)
        generated = self.head(hidden[:, -1, :])
        return generated.unsqueeze(1)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.features = nn.Sequential(
            sn(nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            sn(nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, dilation=2)),
            nn.LeakyReLU(0.2),
            sn(nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=4, dilation=4)),
            nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            sn(nn.Linear(hidden_size, hidden_size // 2)),
            nn.LeakyReLU(0.2),
            sn(nn.Linear(hidden_size // 2, 1)),
        )

    def forward(self, seq):
        x = seq.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x)
        return self.fc(x)


def compute_moment_loss(fake_next, actual_next):
    fake_values = fake_next.squeeze(1)
    actual_values = actual_next.squeeze(1)
    fake_mean = fake_values.mean(dim=0)
    actual_mean = actual_values.mean(dim=0)
    fake_std = fake_values.std(dim=0, unbiased=False)
    actual_std = actual_values.std(dim=0, unbiased=False)
    mean_loss = nn.functional.l1_loss(fake_mean, actual_mean)
    std_loss = nn.functional.l1_loss(fake_std, actual_std)
    return mean_loss + std_loss


def compute_drift_loss(history, fake_next, actual_next):
    last_history = history[:, -1, :]
    fake_delta = fake_next.squeeze(1) - last_history
    actual_delta = actual_next.squeeze(1) - last_history
    return nn.functional.l1_loss(fake_delta, actual_delta)


def discriminator_hinge_loss(real_score, fake_score):
    real_loss = torch.relu(1.0 - real_score).mean()
    fake_loss = torch.relu(1.0 + fake_score).mean()
    return real_loss + fake_loss

def plot_metrics(d_losses, g_losses, title, filepath):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title(f"Stationary Training - {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filepath)
    plt.close()

def plot_price_reconstruction(hist_df, gen_df, feature_name, filepath):
    plt.figure(figsize=(12, 6))
    plt.plot(hist_df['Date'], hist_df[feature_name], label="Historical", color='blue')
    plt.plot(gen_df['Date'], gen_df[feature_name], label="Generated (Stationary)", color='orange', linestyle='--')
    plt.title(f"Price Path Reconstruction: {feature_name}")
    plt.legend()
    plt.savefig(filepath)
    plt.close()


def calibrate_generated_stationary(gen_stat, hist_stat_df, feature_names):
    calibrated = gen_stat.copy()
    hist_means = hist_stat_df[feature_names].mean().to_numpy()
    hist_stds = hist_stat_df[feature_names].std().to_numpy()
    gen_means = calibrated.mean(axis=0)
    gen_stds = calibrated.std(axis=0)

    for idx, feature in enumerate(feature_names):
        hist_std = hist_stds[idx]
        gen_std = gen_stds[idx]
        hist_mean = hist_means[idx]
        gen_mean = gen_means[idx]

        if hist_std <= 1e-12:
            calibrated[:, idx] = hist_mean
            continue

        standardized = calibrated[:, idx] - gen_mean
        if gen_std > 1e-12:
            standardized = standardized / gen_std
        else:
            standardized = np.zeros_like(standardized)

        calibrated[:, idx] = standardized * hist_std + hist_mean
        logging.info(
            f"Calibrated {feature}: raw_mean={gen_mean:.6f}, raw_std={gen_std:.6f}, "
            f"target_mean={hist_mean:.6f}, target_std={hist_std:.6f}"
        )

    return calibrated


def make_stationary(df, price_cols, rate_cols):
    stat_df = df.copy()
    for col in price_cols:
        stat_df[col] = np.log(df[col] / df[col].shift(1))
    for col in rate_cols:
        stat_df[col] = df[col].diff()
    return stat_df.dropna().reset_index(drop=True)


def reconstruct_future_rows(last_known_vals, gen_stat, feature_names, price_cols):
    current_vals = last_known_vals.copy()
    rows = []
    for row_stat in gen_stat:
        for idx, col in enumerate(feature_names):
            if col in price_cols:
                current_vals[idx] = current_vals[idx] * np.exp(row_stat[idx])
            else:
                current_vals[idx] = current_vals[idx] + row_stat[idx]
        rows.append(current_vals.copy())
    return np.array(rows)


def candidate_quality_metrics(reference_stat_df, candidate_stat, feature_names):
    reference_values = reference_stat_df[feature_names]
    candidate_df = pd.DataFrame(candidate_stat, columns=feature_names)

    vol_gaps = []
    mean_gaps = []
    ks_stats = []
    acf_gaps = []

    for feature in feature_names:
        ref_series = reference_values[feature].astype(float)
        cand_series = candidate_df[feature].astype(float)
        ref_std = float(ref_series.std())
        cand_std = float(cand_series.std())
        vol_gaps.append(abs((cand_std / (ref_std + 1e-12)) - 1.0))
        mean_gaps.append(abs(float(cand_series.mean()) - float(ref_series.mean())) / (ref_std + 1e-12))
        ks_stats.append(float(ks_2samp(ref_series, cand_series).statistic))
        for lag in (1, 2, 3):
            ref_acf = float(ref_series.autocorr(lag=lag)) if len(ref_series) > lag else 0.0
            cand_acf = float(cand_series.autocorr(lag=lag)) if len(cand_series) > lag else 0.0
            acf_gaps.append(abs(ref_acf - cand_acf))

    ref_corr = reference_values.corr().to_numpy()
    cand_corr = candidate_df.corr().to_numpy()
    upper_mask = np.triu(np.ones_like(ref_corr, dtype=bool), k=1)
    corr_gap = float(np.abs(ref_corr - cand_corr)[upper_mask].mean()) if upper_mask.any() else 0.0

    return {
        "avg_vol_gap": float(np.mean(vol_gaps)),
        "max_vol_gap": float(np.max(vol_gaps)),
        "avg_mean_gap_z": float(np.mean(mean_gaps)),
        "avg_ks_stat": float(np.mean(ks_stats)),
        "avg_acf_gap": float(np.mean(acf_gaps)) if acf_gaps else 0.0,
        "corr_gap": corr_gap,
    }


def quality_label_from_metrics(metrics):
    if (
        metrics["avg_ks_stat"] <= 0.12
        and metrics["avg_acf_gap"] <= 0.10
        and metrics["corr_gap"] <= 0.18
    ):
        return "good"
    if (
        metrics["avg_ks_stat"] <= 0.22
        and metrics["avg_acf_gap"] <= 0.22
        and metrics["corr_gap"] <= 0.32
    ):
        return "usable_with_caution"
    return "reject"


def quality_score(metrics):
    return (
        3.0 * metrics["avg_ks_stat"]
        + 2.0 * metrics["avg_acf_gap"]
        + 2.0 * metrics["corr_gap"]
        + 2.0 * metrics["avg_vol_gap"]
        + 2.0 * metrics["max_vol_gap"]
        + 0.5 * metrics["avg_mean_gap_z"]
    )


def generate_best_candidate(netG, scaled_stat, scaler, hist_stat_df, feature_names, price_cols, future_dates):
    reference_len = min(len(hist_stat_df), max(RECENT_REF_DAYS, len(future_dates) * 2))
    recent_reference_df = hist_stat_df.tail(reference_len).reset_index(drop=True)
    current_window_template = torch.FloatTensor(scaled_stat[-WINDOW_SIZE:]).unsqueeze(0).to(device)
    last_known_vals = hist_stat_df.attrs["last_known_vals"].copy()

    best = None
    candidate_rows = []

    netG.eval()
    for candidate_idx in range(NUM_CANDIDATES):
        gen_stat_scaled = []
        current_window = current_window_template.clone()

        for _ in range(len(future_dates)):
            with torch.no_grad():
                noise = torch.randn(1, WINDOW_SIZE, NOISE_DIM, device=device)
                next_stat = netG(current_window, noise)
                gen_stat_scaled.append(next_stat.detach().cpu().numpy()[0, 0, :])
                current_window = torch.cat((current_window[:, 1:, :], next_stat), dim=1)

        gen_stat = scaler.inverse_transform(np.array(gen_stat_scaled))
        gen_stat = calibrate_generated_stationary(gen_stat, hist_stat_df, feature_names)
        full_metrics = candidate_quality_metrics(hist_stat_df, gen_stat, feature_names)
        recent_metrics = candidate_quality_metrics(recent_reference_df, gen_stat, feature_names)
        metrics = {
            "avg_vol_gap": full_metrics["avg_vol_gap"],
            "max_vol_gap": full_metrics["max_vol_gap"],
            "avg_mean_gap_z": full_metrics["avg_mean_gap_z"],
            "avg_ks_stat": 0.6 * recent_metrics["avg_ks_stat"] + 0.4 * full_metrics["avg_ks_stat"],
            "avg_acf_gap": 0.6 * recent_metrics["avg_acf_gap"] + 0.4 * full_metrics["avg_acf_gap"],
            "corr_gap": 0.6 * recent_metrics["corr_gap"] + 0.4 * full_metrics["corr_gap"],
            "full_avg_ks_stat": full_metrics["avg_ks_stat"],
            "recent_avg_ks_stat": recent_metrics["avg_ks_stat"],
            "full_avg_acf_gap": full_metrics["avg_acf_gap"],
            "recent_avg_acf_gap": recent_metrics["avg_acf_gap"],
            "full_corr_gap": full_metrics["corr_gap"],
            "recent_corr_gap": recent_metrics["corr_gap"],
        }
        metrics["quality_score"] = quality_score(metrics)
        metrics["quality_label"] = quality_label_from_metrics(metrics)
        metrics["candidate"] = candidate_idx

        recon_array = reconstruct_future_rows(last_known_vals.copy(), gen_stat, feature_names, price_cols)
        for idx, feature in enumerate(feature_names):
            if feature in price_cols and np.any(recon_array[:, idx] <= 0):
                metrics["quality_score"] += 10.0
                metrics["quality_label"] = "reject"
                break
        if metrics["max_vol_gap"] > 0.5:
            metrics["quality_score"] += 10.0
            metrics["quality_label"] = "reject"

        candidate_rows.append(metrics.copy())
        if best is None or metrics["quality_score"] < best["metrics"]["quality_score"]:
            best = {
                "gen_stat": gen_stat,
                "recon_array": recon_array,
                "metrics": metrics.copy(),
            }

    return best, pd.DataFrame(candidate_rows).sort_values("quality_score").reset_index(drop=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_seed_list(filepath):
    if "gold" in filepath.lower():
        return GOLD_SEEDS
    return SILVER_SEEDS

def process_file(filepath):
    print(f"\n--- Overhauling Pipeline (Stationary Returns) for {filepath} ---")
    logging.info(f"--- Starting Stationary Returns Pipeline for {filepath} ---")
    
    input_path = os.path.join(PROJECT_ROOT, filepath)
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Identify feature types dynamically
    price_keywords = ['Futures', 'US30', 'SnP500', 'NASDAQ_100']
    rate_keywords = ['Yield', 'Rate', 'Ratio', 'USD_index', 'gepu', 'gpr_daily']
    
    price_cols = [c for c in df.columns if any(kw in c for kw in price_keywords)]
    rate_cols = [c for c in df.columns if any(kw in c for kw in rate_keywords)]
    leftovers = [c for c in df.columns if c not in price_cols and c != 'Date' and c not in rate_cols]
    rate_cols += leftovers
    all_features = price_cols + rate_cols
    
    logging.info(f"Detected Price columns: {price_cols}")
    logging.info(f"Detected Rate/Index columns: {rate_cols}")
    
    # 1. Transform to Stationary Data
    stat_df = make_stationary(df, price_cols, rate_cols)
    stat_df.attrs["last_known_vals"] = df[all_features].iloc[-1].astype(float).to_numpy(copy=True)
    
    # 2. Scale stationary returns
    scaler = StandardScaler()
    scaled_stat = scaler.fit_transform(stat_df[all_features].values)
    logging.info(f"Normalized stationary data with mean: {scaler.mean_} and scale: {scaler.scale_}")
    
    # Sequence building
    X, Y = [], []
    for i in range(len(scaled_stat) - WINDOW_SIZE):
        X.append(scaled_stat[i:i+WINDOW_SIZE])
        Y.append(scaled_stat[i+WINDOW_SIZE])
    
    dataset = TensorDataset(torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y)).unsqueeze(1))
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )
    
    base_name = os.path.splitext(filepath)[0]
    report_dir = os.path.join(GAN_REPORTS_ROOT, base_name)
    os.makedirs(report_dir, exist_ok=True)
    seed_results = []
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=TARGET_END_DATE, freq='B')
    logging.info(
        f"Generating future synthetic data from {last_date.date()} "
        f"through {TARGET_END_DATE.date()} for {len(future_dates)} business days."
    )

    if len(future_dates) == 0:
        logging.info(f"No future dates required for {filepath}; last date already reaches target.")
        return

    print("Training on Stationary Returns...")
    logging.info("Starting training loop...")
    seed_list = dataset_seed_list(filepath)
    logging.info(
        f"Training config: batch_size={BATCH_SIZE}, epochs={EPOCHS}, "
        f"n_critic={N_CRITIC}, amp={USE_AMP}, workers={NUM_WORKERS}, "
        f"candidates={NUM_CANDIDATES}, seeds={seed_list}, "
        f"gen_hidden={GEN_HIDDEN_SIZE}, disc_hidden={DISC_HIDDEN_SIZE}, "
        f"gen_dropout={GEN_DROPOUT}, lr_g={LR_G}, lr_d={LR_D}"
    )

    best_seed_run = None
    best_model_state = None

    for seed in seed_list:
        set_global_seed(seed)
        print(f"Training seed {seed}...")
        logging.info(f"--- Training seed {seed} for {filepath} ---")

        num_features = len(all_features)
        netG = Generator(num_features, GEN_HIDDEN_SIZE, num_features).to(device)
        netD = Discriminator(num_features, DISC_HIDDEN_SIZE).to(device)
        optG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.9))
        optD = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.9))
        scaler_g = torch.amp.GradScaler("cuda", enabled=USE_AMP)
        scaler_d = torch.amp.GradScaler("cuda", enabled=USE_AMP)

        d_hist, g_hist = [], []
        for epoch in range(EPOCHS):
            epoch_d, epoch_g = 0, 0
            for i, (history, actual_next) in enumerate(dataloader):
                history = history.to(device, non_blocking=PIN_MEMORY)
                actual_next = actual_next.to(device, non_blocking=PIN_MEMORY)
                for _ in range(N_CRITIC):
                    optD.zero_grad(set_to_none=True)
                    noise = torch.randn(history.size(0), WINDOW_SIZE, NOISE_DIM, device=device)
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        real_seq = torch.cat((history, actual_next), dim=1)
                        fake_next = netG(history, noise)
                        fake_seq = torch.cat((history, fake_next.detach()), dim=1)
                        real_score = netD(real_seq)
                        fake_score = netD(fake_seq)
                        d_loss = discriminator_hinge_loss(real_score, fake_score)
                    scaler_d.scale(d_loss).backward()
                    scaler_d.step(optD)
                    scaler_d.update()
                    epoch_d += d_loss.item()

                optG.zero_grad(set_to_none=True)
                noise = torch.randn(history.size(0), WINDOW_SIZE, NOISE_DIM, device=device)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    fake_next = netG(history, noise)
                    fake_seq = torch.cat((history, fake_next), dim=1)
                    g_adv_loss = -netD(fake_seq).mean()
                    mse_loss = nn.MSELoss()(fake_next, actual_next)
                    moment_loss = compute_moment_loss(fake_next, actual_next)
                    drift_loss = compute_drift_loss(history, fake_next, actual_next)
                    g_total = (
                        g_adv_loss
                        + MSE_LOSS_WEIGHT * mse_loss
                        + MOMENT_LOSS_WEIGHT * moment_loss
                        + DRIFT_LOSS_WEIGHT * drift_loss
                    )
                scaler_g.scale(g_total).backward()
                scaler_g.step(optG)
                scaler_g.update()
                epoch_g += g_total.item()

            d_hist.append(epoch_d / (len(dataloader) * N_CRITIC))
            g_hist.append(epoch_g / len(dataloader))
            if (epoch + 1) % 50 == 0:
                status_msg = (
                    f"Seed {seed} Epoch {epoch+1}/{EPOCHS}: "
                    f"D_Loss={d_hist[-1]:.6f}, Total_G_Loss={g_hist[-1]:.6f}"
                )
                print(status_msg)
                logging.info(status_msg)

        training_history = pd.DataFrame(
            {
                "epoch": np.arange(1, len(d_hist) + 1),
                "discriminator_loss": d_hist,
                "generator_total_loss": g_hist,
                "seed": seed,
            }
        )
        training_history.to_csv(os.path.join(report_dir, f"training_loss_history_seed_{seed}.csv"), index=False)
        plot_metrics(
            d_hist,
            g_hist,
            f"{base_name} seed {seed}",
            os.path.join(report_dir, f"training_loss_curve_seed_{seed}.png"),
        )

        best_candidate, candidate_score_df = generate_best_candidate(
            netG=netG,
            scaled_stat=scaled_stat,
            scaler=scaler,
            hist_stat_df=stat_df,
            feature_names=all_features,
            price_cols=price_cols,
            future_dates=future_dates,
        )
        candidate_score_df["seed"] = seed
        metrics = best_candidate["metrics"].copy()
        metrics["seed"] = seed
        seed_results.append(
            {
                "seed": seed,
                "metrics": metrics,
                "candidate_scores": candidate_score_df,
                "training_history": training_history,
                "gen_stat": best_candidate["gen_stat"],
                "recon_array": best_candidate["recon_array"],
                "model_state": {k: v.detach().cpu().clone() for k, v in netG.state_dict().items()},
            }
        )

        if best_seed_run is None or metrics["quality_score"] < best_seed_run["metrics"]["quality_score"]:
            best_seed_run = seed_results[-1]
            best_model_state = seed_results[-1]["model_state"]

    candidate_score_df = pd.concat([run["candidate_scores"] for run in seed_results], ignore_index=True)
    candidate_score_df.to_csv(os.path.join(report_dir, "candidate_quality_scores.csv"), index=False)
    seed_summary_df = pd.DataFrame([run["metrics"] for run in seed_results]).sort_values("quality_score")
    seed_summary_df.to_csv(os.path.join(report_dir, "seed_quality_summary.csv"), index=False)
    with open(os.path.join(report_dir, "selected_candidate_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_seed_run["metrics"], f, indent=2)
    logging.info(f"Selected best seed metrics: {best_seed_run['metrics']}")
    gen_stat = best_seed_run["gen_stat"]

    # 4. Price Path Reconstruction
    last_known_vals = df[all_features].iloc[-1].astype(float).to_numpy(copy=True)
    logging.info(f"Reconstructing price path starting from: {dict(zip(all_features, last_known_vals))}")
    recon_df = pd.DataFrame(best_seed_run["recon_array"], columns=all_features)
    recon_df.insert(0, 'Date', future_dates)
    
    # Validation Logging
    logging.info("--- VALIDATION METRICS (Additional Detail) ---")
    for idx, feature in enumerate(all_features):
        h_std = np.std(stat_df[feature])
        g_std = np.std(gen_stat[:, idx])
        h_mean = np.mean(stat_df[feature])
        g_mean = np.mean(gen_stat[:, idx])
        vol_err = abs(h_std - g_std) / (h_std + 1e-9) * 100
        log_line = f"[{feature}] Hist_Std: {h_std:.6f}, Gen_Std: {g_std:.6f}, Err: {vol_err:.2f}%, Hist_Mean: {h_mean:.6f}, Gen_Mean: {g_mean:.6f}"
        logging.info(log_line)
        if feature in price_cols[:1]: # Print first target only to console
             print(f"[{feature}] Volatility Error: {vol_err:.2f}%")
    
    # Save results
    plot_price_reconstruction(
        df,
        recon_df,
        all_features[0],
        os.path.join(SCRIPT_DIR, f"{base_name}_stationary_path.png")
    )
    output_path = os.path.join(PROJECT_ROOT, f"{base_name}_extended.csv")
    pd.concat([df, recon_df], ignore_index=True).to_csv(output_path, index=False)
    torch.save(best_model_state, os.path.join(SCRIPT_DIR, f"{base_name}_stationary_gen.pth"))
    logging.info(f"Finalized and saved extended data to {output_path}")
    print(
        f"Saved {os.path.basename(output_path)} with {len(recon_df)} generated rows "
        f"through {TARGET_END_DATE.date()}"
    )

if __name__ == "__main__":
    for f in SOURCE_FILES:
        if os.path.exists(os.path.join(PROJECT_ROOT, f)):
            process_file(f)
