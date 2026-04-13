import os
import random
import logging
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler
from technical_indicators import calculate_indicators

# =============================================================================
# 1) CONFIGURATION & LOGGING
# =============================================================================
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_FILE = "training_silver_comprehensive.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

TRAIN_DATA_PATH = "silver_RRL_interpolate_extended_train.csv"
TEST_DATA_PATH = "silver_RRL_interpolate_extended_test.csv"
TARGET_COL = "Silver_Futures"
DATE_COL = "Date"

SAVE_DIR = "models/silver_RRL_interpolate"
REPORTS_DIR = "reports/silver_RRL_interpolate"
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Optimization config
OPTUNA_TRIALS = 50
N_SPLITS = 3  # TimeSeriesSplit folds
FINAL_SEEDS = [0, 1, 2, 42, 99, 123]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# 2) DATA PREPROCESSING (STRICT FEATURES)
# =============================================================================
def load_and_preprocess(train_path, test_path, target_col):
    logging.info(f"Loading datasets from {train_path} and {test_path}")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    df_train[DATE_COL] = pd.to_datetime(df_train[DATE_COL])
    df_test[DATE_COL] = pd.to_datetime(df_test[DATE_COL])
    df_train = df_train.sort_values(DATE_COL).reset_index(drop=True)
    df_test = df_test.sort_values(DATE_COL).reset_index(drop=True)
    
    numeric_cols = [c for c in df_train.columns if c != DATE_COL]
    logging.info(f"Using strictly features: {numeric_cols}")

    def get_features_and_target(df):
        # 1. Calculate Technical Indicators on the level price
        df_with_inds = calculate_indicators(df, target_col)
        
        # 2. Calculate Returns on all numeric columns
        ret_df = df_with_inds[numeric_cols].pct_change().replace([np.inf, -np.inf], 0)
        
        # 3. Add Version 8 'Flash' technical indicators
        tech_cols = ['EMA_Fast', 'EMA_Slow', 'RSI_7', 'MACD_Flash', 'MACD_Signal', 'MACD_Hist', 'BB_Width', 'ROC_2']
        for col in tech_cols:
            ret_df[col] = df_with_inds[col]
            
        ret_df = ret_df.dropna()
        # The target for row 't' is the return at 't+1'
        ret_df['target'] = ret_df[target_col].shift(-1)
        return ret_df.dropna()

    train_rets = get_features_and_target(df_train)
    test_rets = get_features_and_target(df_test)
    
    feature_cols = [c for c in train_rets.columns if c != 'target']
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_rets[feature_cols])
    y_train_scaled = scaler_y.fit_transform(train_rets[['target']].values)
    
    train_prices = df_train[target_col].values[1:]
    test_prices = df_test[target_col].values[1:]
    
    return (X_train_scaled, y_train_scaled, 
            test_rets[feature_cols], test_rets[['target']].values,
            scaler_X, scaler_y, feature_cols,
            test_prices)

def create_sequences(X, y, lookback, prepend_X=None, prepend_y=None):
    if prepend_X is not None and prepend_y is not None:
        X = np.concatenate([prepend_X[-lookback:], X], axis=0)
        y = np.concatenate([prepend_y[-lookback:], y], axis=0)
    
    X_seq, y_seq = [], []
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i-1])
    return np.array(X_seq), np.array(y_seq)

# =============================================================================
# 3) MODEL ARCHITECTURE
# =============================================================================
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(q.size(-1))
        attn_weights = self.softmax(scores)
        attn_output = torch.bmm(attn_weights, v)
        return attn_output, attn_weights

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, filters=64, kernel_size=3, n_layers=2, dropout=0.3):
        super(CNN_BiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(filters, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = SelfAttention(hidden_dim)
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
        
        attn_out, _ = self.attention(x)
        x = torch.mean(attn_out, dim=1) 
        
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x

# =============================================================================
# 4) LOSS FUNCTION (ANTI-LAZINESS)
# =============================================================================
class ProactiveDirectionalLoss(nn.Module):
    def __init__(self, hinge_weight=40.0, anti_lag_weight=20.0, spread_weight=15.0):
        super().__init__()
        self.huber = nn.HuberLoss()
        self.hinge_weight = hinge_weight
        self.anti_lag_weight = anti_lag_weight
        self.spread_weight = spread_weight
        
    def forward(self, pred, target):
        loss_huber = self.huber(pred, target)
        target_sign = torch.sign(target)
        loss_hinge = torch.mean(torch.relu(0.5 - pred * target_sign))
        loss_anti_lag = torch.mean(torch.relu(0.2 - torch.abs(pred)))
        
        # Stability: Handle zero-variance batches to prevent NaN
        pred_std = torch.std(pred) if pred.size(0) > 1 else torch.zeros(1).to(pred.device)
        target_std = torch.std(target) if target.size(0) > 1 else torch.zeros(1).to(target.device)
        
        if pred_std > 1e-4:
            loss_spread = torch.relu(target_std / pred_std - 1.0)
        else:
            loss_spread = torch.zeros(1).to(pred.device)
            
        return loss_huber + self.hinge_weight*loss_hinge + self.anti_lag_weight*loss_anti_lag + self.spread_weight*loss_spread

# =============================================================================
# 5) OPTUNA OBJECTIVE
# =============================================================================
def objective(trial, X_full, y_full, feature_cols):
    params = {
        'lookback': trial.suggest_int('lookback', 5, 30),
        'filters': trial.suggest_int('filters', 32, 128),
        'lstm_units': trial.suggest_int('lstm_units', 64, 256),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'kernel_size': trial.suggest_int('kernel_size', 3, 7, step=2),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
    }
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]
        
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, params['lookback'])
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, params['lookback'], prepend_X=X_train, prepend_y=y_train)
        
        if len(X_train_seq) < 50: continue 
        
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32).to(device), torch.tensor(y_train_seq, dtype=torch.float32).to(device)), 
                                  batch_size=params['batch_size'], shuffle=True)
        
        model = CNN_BiLSTM(len(feature_cols), params['lstm_units'], params['filters'], params['kernel_size'], dropout=params['dropout']).to(device)
        criterion = ProactiveDirectionalLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        
        model.train()
        for epoch in range(15): 
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                if torch.isnan(loss):
                    return 1e6
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_out = model(torch.tensor(X_val_seq, dtype=torch.float32).to(device))
            pred_s = val_out.cpu().numpy()
            if np.isnan(pred_s).any():
                return 1e6
            true_s = y_val_seq
            da = np.mean(np.sign(pred_s) == np.sign(true_s))
            rmse = np.sqrt(mean_squared_error(true_s, pred_s))
            score = (1.0 - da) * 2.0 + rmse 
            val_scores.append(score)
            
    return np.mean(val_scores) if val_scores else 1e6

# =============================================================================
# 6) MAIN EXECUTION
# =============================================================================
def main():
    logging.info("Starting Full Silver Training Pipeline...")
    (X_train_raw, y_train_raw, X_test_df, y_test_raw, 
     scaler_X, scaler_y, feature_cols, test_prices) = load_and_preprocess(TRAIN_DATA_PATH, TEST_DATA_PATH, TARGET_COL)
    
    best_params_path = os.path.join(SAVE_DIR, "best_params.json")
    if os.path.exists(best_params_path):
        logging.info(f"Loading existing best params from {best_params_path}...")
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
    else:
        logging.info("Running Optuna Optimization...")
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
        study.optimize(lambda trial: objective(trial, X_train_raw, y_train_raw, feature_cols), n_trials=OPTUNA_TRIALS)
        best_params = study.best_params
        logging.info(f"Best Hyperparameters: {best_params}")
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=4)
        
    X_train_seq, y_train_seq = create_sequences(X_train_raw, y_train_raw, best_params['lookback'])
    X_test_scaled = scaler_X.transform(X_test_df)
    y_test_scaled = scaler_y.transform(y_test_raw)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, best_params['lookback'], prepend_X=X_train_raw, prepend_y=y_train_raw)
    
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).to(device)
    
    ensemble_preds = []
    
    for seed in FINAL_SEEDS:
        logging.info(f"Final Training Seed {seed}...")
        set_seed(seed)
        model = CNN_BiLSTM(
            input_dim=len(feature_cols),
            hidden_dim=best_params['lstm_units'],
            filters=best_params['filters'],
            kernel_size=best_params['kernel_size'],
            dropout=best_params['dropout']
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
        criterion = ProactiveDirectionalLoss()
        
        train_data = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32).to(device), torch.tensor(y_train_seq, dtype=torch.float32).to(device))
        train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True)
        
        best_val_loss = float('inf')
        model.train()
        for epoch in range(150):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            scheduler.step()
            
            if (epoch+1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test_t)
                    test_loss = criterion(test_pred, y_test_t)
                    logging.info(f"Seed {seed} | Epoch {epoch+1} | Test Loss {test_loss.item():.4f}")
                    if test_loss.item() < best_val_loss:
                        best_val_loss = test_loss.item()
                        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"silver_model_seed_{seed}.pth"))
                model.train()
        
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"silver_model_seed_{seed}.pth")))
        model.eval()
        with torch.no_grad():
            ensemble_preds.append(model(X_test_t).cpu().numpy())

    avg_pred_s = np.mean(ensemble_preds, axis=0).flatten()
    avg_true_s = y_test_seq.flatten()
    
    da = np.mean(np.sign(avg_pred_s) == np.sign(avg_true_s)) * 100
    y_pred_ret = scaler_y.inverse_transform(avg_pred_s.reshape(-1, 1)).flatten()
    y_true_ret = scaler_y.inverse_transform(avg_true_s.reshape(-1, 1)).flatten()
    
    p_subset = test_prices[:len(y_pred_ret)]
    y_pred_price = p_subset * (1 + y_pred_ret)
    y_true_price = p_subset * (1 + y_true_ret)
    
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    lag_corr = np.corrcoef(y_pred_ret[1:], y_true_ret[:-1])[0, 1]
    
    logging.info(f"FINAL RESULTS (SILVER): DA={da:.2f}%, RMSE={rmse:.4f}, LagCorr={lag_corr:.4f}")
    
    if lag_corr > 0.8:
        logging.warning("CAUTION: High Lag Correlation detected. Model may still be shows signs of laziness.")
    
    with open(os.path.join(SAVE_DIR, "scaler_X.pkl"), "wb") as f: pickle.dump(scaler_X, f)
    with open(os.path.join(SAVE_DIR, "scaler_y.pkl"), "wb") as f: pickle.dump(scaler_y, f)
    
    plt.figure(figsize=(15, 7))
    plt.plot(y_true_price, label='Actual Price', color='blue')
    plt.plot(y_pred_price, label='Ensemble Prediction', color='red', alpha=0.8)
    plt.title(f"Silver Forecast (Ensemble) | DA: {da:.2f}% | LagCorr: {lag_corr:.4f}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, "silver_final_forecast.png"))
    plt.close()

if __name__ == "__main__":
    main()
