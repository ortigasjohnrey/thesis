import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from generate_gan_data import Generator, WINDOW_SIZE, NOISE_DIM
from dataset_catalog import get_dataset_config_by_asset, ensure_prepared_source

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProxyDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def calculate_fidelity(asset_name, model_pth):
    print(f"Analyzing fidelity for {asset_name}...")
    config = get_dataset_config_by_asset(asset_name)
    source_path, _ = ensure_prepared_source(config)
    df = pd.read_csv(source_path)
    
    # Preprocess (Stationary)
    price_keywords = ['Futures', 'US30', 'SnP500', 'NASDAQ_100']
    price_cols = [c for c in df.columns if any(kw in c for kw in price_keywords)]
    other_cols = [c for c in df.columns if c not in price_cols and c != 'Date']
    all_features = price_cols + other_cols
    
    stat_df = df.copy()
    for col in price_cols:
        stat_df[col] = np.log(df[col] / df[col].shift(1))
    for col in other_cols:
        stat_df[col] = df[col].diff()
    stat_df = stat_df.dropna()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(stat_df[all_features].values)
    
    num_features = len(all_features)
    netG = Generator(num_features, 128, num_features).to(device)
    netG.load_state_dict(torch.load(model_pth, map_location=device))
    netG.eval()
    
    # 1. Prepare Real and Fake Datasets
    X_real = []
    for i in range(len(scaled_data) - WINDOW_SIZE):
        X_real.append(scaled_data[i:i+WINDOW_SIZE])
    X_real = np.array(X_real)
    
    X_fake = []
    with torch.no_grad():
        # Generate fake sequences of length WINDOW_SIZE
        # We'll use the same history but generate the NEXT step
        for i in range(len(scaled_data) - WINDOW_SIZE):
            history = torch.FloatTensor(scaled_data[i:i+WINDOW_SIZE-1]).unsqueeze(0).to(device)
            # To make a full window of size 15, we need 14 history + 1 gen
            noise = torch.randn(1, WINDOW_SIZE-1, NOISE_DIM, device=device)
            # Re-using history of length 14 to predict 15th
            # This is a bit simplified, but captures the "local" fidelity
            next_val = netG(history, noise).cpu().numpy()[0, 0, :]
            fake_seq = np.vstack([scaled_data[i:i+WINDOW_SIZE-1], next_val])
            X_fake.append(fake_seq)
    X_fake = np.array(X_fake)
    
    # 2. Train Proxy Discriminator (Classifier)
    # Balanced dataset
    num_samples = len(X_real)
    X = np.vstack([X_real, X_fake])
    y = np.concatenate([np.ones(num_samples), np.zeros(num_samples)])
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    split = int(0.8 * len(X))
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X[:split]), torch.FloatTensor(y[:split])), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X[split:]), torch.FloatTensor(y[split:])), batch_size=64)
    
    proxy_d = ProxyDiscriminator(num_features, 64).to(device)
    optimizer = optim.Adam(proxy_d.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    print("Training proxy discriminator (independent auditor)...")
    for epoch in range(10): # Quick audit
        proxy_d.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(proxy_d(bx), by)
            loss.backward()
            optimizer.step()
            
    proxy_d.eval()
    all_scores = []
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device).unsqueeze(1)
            scores = proxy_d(bx)
            all_scores.append(scores.cpu().numpy())
            preds = (scores > 0.5).float()
            correct += (preds == by).sum().item()
            total += by.size(0)
            
    accuracy = correct / total
    print(f"Discriminative Accuracy: {accuracy:.4f} (Goal: 0.5 for perfect indistinguishability)")
    
    # 3. Plot Score Distributions
    real_test_scores = []
    fake_test_scores = []
    with torch.no_grad():
        # Get scores for real and fake separately for plotting
        real_t = torch.FloatTensor(X_real).to(device)
        fake_t = torch.FloatTensor(X_fake).to(device)
        # Process in batches to avoid OOM
        for i in range(0, len(X_real), 512):
            real_test_scores.append(proxy_d(real_t[i:i+512]).cpu().numpy())
            fake_test_scores.append(proxy_d(fake_t[i:i+512]).cpu().numpy())
            
    real_test_scores = np.vstack(real_test_scores).flatten()
    fake_test_scores = np.vstack(fake_test_scores).flatten()
    
    plt.figure(figsize=(10, 5))
    plt.hist(real_test_scores, bins=50, alpha=0.5, label='Real (Proxy D Scores)', color='blue')
    plt.hist(fake_test_scores, bins=50, alpha=0.5, label='Fake (Proxy D Scores)', color='orange')
    plt.axvline(0.5, color='red', linestyle='--', label='Decision Boundary')
    plt.title(f"Fidelity Audit: {asset_name} (Accuracy: {accuracy:.2%})")
    plt.xlabel("Probability of belonging to Real class")
    plt.ylabel("Frequency")
    plt.legend()
    plot_path = f"gan/{asset_name}_fidelity_audit.png"
    plt.savefig(plot_path)
    plt.close()
    
    return accuracy, plot_path

if __name__ == "__main__":
    results = []
    # Gold
    acc, path = calculate_fidelity("gold", "gan/df_gold_dataset_gepu_stationary_gen.pth")
    results.append({"asset": "gold", "accuracy": acc, "plot": path})
    
    # Silver
    acc, path = calculate_fidelity("silver", "gan/silver_RRL_interpolate_stationary_gen.pth")
    results.append({"asset": "silver", "accuracy": acc, "plot": path})
    
    import json
    with open("gan/fidelity_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)
