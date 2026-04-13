import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters from the research paper (Table 2)
LOOKBACK = 30
COUNCIL_SEEDS = [42, 123, 99]
FILTERS = 64
KERNEL_SIZE = 4
HIDDEN_DIM = 20
DROPOUT = 0.1
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 150

# Seed for reproducibility
FINAL_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=20, filters=64, kernel_size=4, n_layers=1, dropout=0.1):
        super(CNN_BiLSTM, self).__init__()
        # Paper alignment: Filters=64, Kernel=4
        self.conv1     = nn.Conv1d(input_dim, filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1       = nn.BatchNorm1d(filters)
        self.relu      = nn.ReLU()
        self.pool      = nn.MaxPool1d(2)
        self.dropout   = nn.Dropout(dropout)
        
        # PyTorch warns if dropout > 0 but n_layers = 1. Silence it.
        lstm_dropout = dropout if n_layers > 1 else 0
        self.lstm      = nn.LSTM(filters, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.fc        = nn.Linear(hidden_dim * 2, 64)
        self.out       = nn.Linear(64, 1)

    def forward(self, x):
        # Input shape: [Batch, SeqLen, Features]
        x = x.transpose(1, 2) # [Batch, Features, SeqLen]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if x.shape[-1] > 1: x = self.pool(x)
        x = self.dropout(x)
        x = x.transpose(1, 2) # [Batch, SeqLen/2, Filters]
        x, _ = self.lstm(x)
        
        # Use the last hidden state of the BiLSTM sequence to focus on current price trends
        x = x[:, -1, :] 
        
        x = self.relu(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.out(x)

def create_sequences(data, target, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)

def train_model(X_train, y_train, X_test, y_test, input_dim, seed):
    set_seed(seed)
    model = CNN_BiLSTM(input_dim, HIDDEN_DIM, FILTERS, KERNEL_SIZE, 1, DROPOUT).to(device)
    criterion = nn.L1Loss() # MAE as suggested by paper context
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_y_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_X_t)
                test_loss = criterion(test_outputs, test_y_t.unsqueeze(1))
                logger.info(f"Seed {seed} | Epoch {epoch} | Train Loss {epoch_loss/len(train_loader):.6f} | Test Loss {test_loss.item():.6f}")
                
    return model

def main():
    logger.info("Starting Gold Price Training (Single Model - No Ensemble)")
    
    # Load data
    train_df = pd.read_csv("df_gold_dataset_gepu_extended_train.csv")
    test_df = pd.read_csv("df_gold_dataset_gepu_extended_test.csv")
    
    features = ['Silver_Futures', 'Crude_Oil_Futures', 'UST10Y_Treasury_Yield', 'gepu', 'DFF', 'gpr_daily', 'Gold_Futures']
    target_col = 'Gold_Futures'
    
    # Simple Technical Indicators (matches api.py v8)
    def add_indicators(df):
        df = df.copy()
        df['EMA_Fast'] = df[target_col].ewm(span=3, adjust=False).mean()
        df['EMA_Slow'] = df[target_col].ewm(span=8, adjust=False).mean()
        delta = df[target_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / (loss + 1e-8)
        df['RSI_7'] = 100 - (100 / (1 + rs))
        exp1 = df[target_col].ewm(span=6, adjust=False).mean()
        exp2 = df[target_col].ewm(span=13, adjust=False).mean()
        df['MACD_Flash'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD_Flash'].ewm(span=5, adjust=False).mean()
        df['MACD_Hist'] = df['MACD_Flash'] - df['MACD_Signal']
        df['BB_Mid'] = df[target_col].rolling(window=5).mean()
        df['BB_Std'] = df[target_col].rolling(window=5).std()
        df['BB_Width'] = (4 * df['BB_Std']) / (df['BB_Mid'] + 1e-8)
        df['ROC_2'] = df[target_col].pct_change(periods=2).replace([np.inf, -np.inf], 0).fillna(0)
        # Cross-Asset Ratio
        if 'Silver_Futures' in df.columns and 'Gold_Futures' in df.columns:
            df['GS_Ratio'] = df['Gold_Futures'] / (df['Silver_Futures'] + 1e-8)
        else:
            df['GS_Ratio'] = 0.0
        return df.ffill().bfill().fillna(0)

    train_df = add_indicators(train_df)
    test_df = add_indicators(test_df)
    
    tech_cols = ['EMA_Fast', 'EMA_Slow', 'RSI_7', 'MACD_Flash', 'MACD_Signal', 'MACD_Hist', 'BB_Width', 'ROC_2', 'GS_Ratio']
    all_features = features + tech_cols
    
    # Target: Percentage Change (Stationary)
    # We predict the RETURN for the next day
    train_df['target_return'] = train_df[target_col].pct_change().shift(-1).fillna(0)
    test_df['target_return'] = test_df[target_col].pct_change().shift(-1).fillna(0)
    
    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    train_X_scaled = scaler_X.fit_transform(train_df[all_features])
    train_y_scaled = scaler_y.fit_transform(train_df[['target_return']]).flatten()
    
    test_X_scaled = scaler_X.transform(test_df[all_features])
    test_y_scaled = scaler_y.transform(test_df[['target_return']]).flatten()
    
    # Create Sequences
    X_train, y_train = create_sequences(train_X_scaled, train_y_scaled, LOOKBACK)
    X_test, y_test = create_sequences(test_X_scaled, test_y_scaled, LOOKBACK)
    
    # Council Training
    SAVE_DIR = "models/gold_RRL_interpolate"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for seed in COUNCIL_SEEDS:
        logger.info(f"--- Training Council Member: Seed {seed} ---")
        model = train_model(X_train, y_train, X_test, y_test, len(all_features), seed)
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"gold_model_seed_{seed}.pth"))

    # Save shared assets
    with open(os.path.join(SAVE_DIR, "scaler_X.pkl"), "wb") as f: pickle.dump(scaler_X, f)
    with open(os.path.join(SAVE_DIR, "scaler_y.pkl"), "wb") as f: pickle.dump(scaler_y, f)
    with open(os.path.join(SAVE_DIR, "best_params.json"), "w") as f:
        json.dump({
            "lookback": LOOKBACK,
            "filters": FILTERS,
            "kernel_size": KERNEL_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "lr": LEARNING_RATE
        }, f)
    
    logger.info("Gold Council Training Complete.")

if __name__ == "__main__":
    main()
