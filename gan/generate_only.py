import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import logging
import pickle

from dataset_catalog import (
    PROJECT_ROOT,
    TARGET_END_DATE,
    ensure_prepared_source,
    get_dataset_config_by_asset,
    get_model_output_path,
    get_output_base_name,
)

# Configuration matching generate_gan_data.py
WINDOW_SIZE = 15
NOISE_DIM = 32
GEN_HIDDEN_SIZE = 128
GEN_DROPOUT = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)

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

def make_stationary(df, price_cols, rate_cols):
    stat_df = df.copy()
    for col in price_cols:
        stat_df[col] = np.log(df[col] / df[col].shift(1))
    for col in rate_cols:
        stat_df[col] = df[col].diff()
    return stat_df.dropna().reset_index(drop=True)

def generate_extension(asset):
    print(f"--- Extending {asset} targeting {TARGET_END_DATE.date()} ---")
    dataset_config = get_dataset_config_by_asset(asset)
    input_path, _ = ensure_prepared_source(dataset_config)
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    price_keywords = ['Futures', 'US30', 'SnP500', 'NASDAQ_100']
    rate_keywords = ['Yield', 'Rate', 'Ratio', 'USD_index', 'gepu', 'gpr_daily']
    price_cols = [c for c in df.columns if any(kw in c for kw in price_keywords)]
    rate_cols = [c for c in df.columns if any(kw in c for kw in rate_keywords)]
    leftovers = [c for c in df.columns if c not in price_cols and c != 'Date' and c not in rate_cols]
    rate_cols += leftovers
    all_features = price_cols + rate_cols
    
    stat_df = make_stationary(df, price_cols, rate_cols)
    last_known_vals = df[all_features].iloc[-1].astype(float).to_numpy(copy=True)
    
    scaler = StandardScaler()
    scaled_stat = scaler.fit_transform(stat_df[all_features].values)
    
    # Load model
    model_path = get_model_output_path(dataset_config)
    num_features = len(all_features)
    netG = Generator(num_features, GEN_HIDDEN_SIZE, num_features).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=TARGET_END_DATE, freq='B')
    
    if len(future_dates) == 0:
        print(f"No extension needed for {asset}")
        return

    # Generation loop
    current_window = torch.FloatTensor(scaled_stat[-WINDOW_SIZE:]).unsqueeze(0).to(device)
    gen_stat_scaled = []
    
    for _ in range(len(future_dates)):
        with torch.no_grad():
            noise = torch.randn(1, WINDOW_SIZE, NOISE_DIM, device=device)
            next_stat = netG(current_window, noise)
            gen_stat_scaled.append(next_stat.detach().cpu().numpy()[0, 0, :])
            current_window = torch.cat((current_window[:, 1:, :], next_stat), dim=1)
            
    gen_stat = scaler.inverse_transform(np.array(gen_stat_scaled))
    
    # Price path reconstruction
    current_vals = last_known_vals.copy()
    recon_rows = []
    for row_stat in gen_stat:
        for idx, col in enumerate(all_features):
            if col in price_cols:
                current_vals[idx] = current_vals[idx] * np.exp(row_stat[idx])
            else:
                current_vals[idx] = current_vals[idx] + row_stat[idx]
        recon_rows.append(current_vals.copy())
        
    recon_df = pd.DataFrame(recon_rows, columns=all_features)
    recon_df.insert(0, 'Date', future_dates)
    
    output_path = PROJECT_ROOT / f"{get_output_base_name(dataset_config)}_extended.csv"
    pd.concat([df, recon_df], ignore_index=True).to_csv(output_path, index=False)
    print(f"Saved extended data to {output_path}")

if __name__ == "__main__":
    generate_extension("gold")
    generate_extension("silver")
