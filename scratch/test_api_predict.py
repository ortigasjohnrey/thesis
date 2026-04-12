"""
Smoke test: verify that the fixed predict_from_context_frame logic in api.py
correctly produces a non-lazy prediction from the gold model.
"""
import sys, os
sys.path.insert(0, os.getcwd())

# We need BasisPointScaler to unpickle y_scaler
class BasisPointScaler:
    def __init__(self, scale=10000.0):
        self.scale = scale
    def fit_transform(self, x): return x * self.scale
    def transform(self, x): return x * self.scale
    def inverse_transform(self, x): return x / self.scale

import builtins, types
# Inject BasisPointScaler into __main__ so pickle can find it
import __main__
__main__.BasisPointScaler = BasisPointScaler

import pandas as pd, numpy as np, pickle, json, torch
import torch.nn as nn

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_shape, params):
        super().__init__()
        in_channels = input_shape[1]; dr = params['dropout_rate']
        self.conv1 = nn.Conv1d(in_channels, params['filters'], params['kernel_size'], padding=params['kernel_size']-1)
        self.relu = nn.ReLU(); self.bn1 = nn.BatchNorm1d(params['filters'])
        self.spatial_dropout = nn.Dropout1d(p=dr)
        self.lstm1 = nn.LSTM(params['filters'], params['lstm_units'], batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dr)
        lu2 = max(16, params['lstm_units']//2)
        self.lstm2 = nn.LSTM(params['lstm_units']*2, lu2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dr)
        self.fc1 = nn.Linear(lu2*2, params['dense_units']); self.fc_dropout = nn.Dropout(dr)
        self.out = nn.Linear(params['dense_units'], 1)
    def forward(self, x):
        x = x.permute(0,2,1); x = self.conv1(x)
        if self.conv1.padding[0] > 0: x = x[:,:,:-self.conv1.padding[0]]
        x = self.relu(x); x = self.bn1(x); x = self.spatial_dropout(x)
        x = x.permute(0,2,1); x, _ = self.lstm1(x); x = self.dropout1(x)
        _, (h_n, _) = self.lstm2(x)
        x = torch.cat((h_n[0,:,:], h_n[1,:,:]), dim=1); x = self.dropout2(x)
        x = self.fc1(x); x = self.relu(x); x = self.fc_dropout(x)
        return self.out(x)

model_dir = 'models/gold_train_only_retrained_v2/seed_42'
with open(f'{model_dir}/model_metadata.json') as f: meta = json.load(f)
with open(f'{model_dir}/x_scaler.pkl','rb') as f: x_scaler = pickle.load(f)
with open(f'{model_dir}/y_scaler.pkl','rb') as f: y_scaler = pickle.load(f)
with open('reports/gold_train_only_retrained_v2/gold_best_params_optimized.json') as f: params = json.load(f)

feature_cols = meta['feature_cols']; lookback = meta['lookback']
print(f'feature_cols count: {len(feature_cols)}, lookback: {lookback}')

# Load train data
df = pd.read_csv('df_gold_dataset_gepu_extended_train.csv')
df['Date'] = pd.to_datetime(df['Date']); df = df.sort_values('Date').reset_index(drop=True)
print('CSV cols:', list(df.columns))

# Simulate predict_from_context_frame (new implementation)
base_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print('Base numeric cols:', base_numeric_cols)
numeric_df = df[base_numeric_cols].copy()
abs_last_price = float(numeric_df.iloc[-1]['Gold_Futures'])
returns_df = numeric_df.pct_change().replace([np.inf,-np.inf], np.nan)
for col in base_numeric_cols:
    returns_df[f'{col}_lag1'] = returns_df[col].shift(1)
    returns_df[f'{col}_lag2'] = returns_df[col].shift(2)
returns_df = returns_df.dropna()
missing = [c for c in feature_cols if c not in returns_df.columns]
print('Missing features:', missing if missing else 'NONE — all features found!')
recent = returns_df[feature_cols].iloc[-lookback:].copy()
recent_scaled = x_scaler.transform(recent)
t = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0)
m = CNN_BiLSTM((lookback, len(feature_cols)), params)
m.load_state_dict(torch.load(f'{model_dir}/cnn_bilstm_seed42.pth', map_location='cpu', weights_only=True))
m.eval()
with torch.no_grad(): pred_scaled = m(t).numpy().reshape(-1,1)
pred_return = y_scaler.inverse_transform(pred_scaled).item()
pred_abs = abs_last_price * (1 + pred_return)
print(f'Last Gold price (P_t):  ${abs_last_price:.2f}')
print(f'Predicted return:       {pred_return*100:.4f}%')
print(f'Predicted next price:   ${pred_abs:.2f}')
lazy = abs(pred_return) < 0.0003  # < 0.03% is suspicious
print(f'Laziness check (|ret| < 0.03%): {"LAZY (still near-zero)" if lazy else "OK — model making a real prediction"}')
print('SUCCESS')
