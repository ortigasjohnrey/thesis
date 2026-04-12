import os
import random
import logging
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
from optuna.samplers import TPESampler
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training_gold_rrl.log"),
        logging.StreamHandler()
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logging.info(f"GPU found: {torch.cuda.get_device_name(0)}")
else:
    logging.warning("No GPU found. Training will bottleneck on CPU.")


def parse_int_list_env(env_name: str, default_values):
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return list(default_values)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_range_env(env_name: str, default_values, cast):
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return tuple(default_values)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{env_name} must contain exactly two comma-separated values.")
    lower, upper = cast(parts[0]), cast(parts[1])
    if lower >= upper:
        raise ValueError(f"{env_name} must have an increasing range, got {raw!r}.")
    return (lower, upper)

# =========================
# 2) User Inputs
# =========================
CLEAN_DF_PATH = os.getenv("GOLD_CLEAN_DF_PATH", "gold_RRL_interpolate_train.csv")
TARGET_COL = os.getenv("GOLD_TARGET_COL", "Gold_Futures")
DATE_COL = os.getenv("GOLD_DATE_COL", "Date")
HORIZON = int(os.getenv("GOLD_HORIZON", "1"))
OUTER_TRAIN_RATIO = float(os.getenv("GOLD_OUTER_TRAIN_RATIO", "0.80"))
FINAL_VAL_RATIO_WITHIN_OUTER_TRAIN = float(os.getenv("GOLD_FINAL_VAL_RATIO", "0.10"))
TUNING_SEED = int(os.getenv("GOLD_TUNING_SEED", "42"))
N_SPLITS_WALK_FORWARD = int(os.getenv("GOLD_N_SPLITS_WALK_FORWARD", "5"))
INIT_POINTS = int(os.getenv("GOLD_INIT_POINTS", "15"))
N_ITER = int(os.getenv("GOLD_N_ITER", "40"))
FINAL_SEEDS = parse_int_list_env("GOLD_FINAL_SEEDS", [0, 1, 2, 42, 99, 123])
MAX_EPOCHS_TUNING = int(os.getenv("GOLD_MAX_EPOCHS_TUNING", "50"))
MAX_EPOCHS_FINAL = int(os.getenv("GOLD_MAX_EPOCHS_FINAL", "50"))
EARLY_STOPPING_PATIENCE = int(os.getenv("GOLD_EARLY_STOPPING_PATIENCE", "5"))
EARLY_STOPPING_MIN_DELTA = float(os.getenv("GOLD_EARLY_STOPPING_MIN_DELTA", "1e-4"))
TRAIN_END_DATE = os.getenv("GOLD_TRAIN_END_DATE")
EXPORT_DATE_SPLITS = os.getenv("GOLD_EXPORT_DATE_SPLITS", "1") != "0"

SAVE_ALL_SEED_MODELS = os.getenv("GOLD_SAVE_ALL_SEED_MODELS", "1") != "0"
SAVE_ROOT_DIR = os.getenv("GOLD_SAVE_ROOT_DIR", "models/gold_RRL_interpolate")
REPORTS_DIR = os.getenv("GOLD_REPORTS_DIR", "reports/gold_RRL_interpolate")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
DATASET_BASENAME = os.path.splitext(os.path.basename(CLEAN_DF_PATH))[0]
TRAIN_SPLIT_EXPORT_PATH = os.getenv("GOLD_TRAIN_SPLIT_EXPORT_PATH", f"{DATASET_BASENAME}_train.csv")
TEST_SPLIT_EXPORT_PATH = os.getenv("GOLD_TEST_SPLIT_EXPORT_PATH", f"{DATASET_BASENAME}_test.csv")

PBOUNDS = {
    "lookback": parse_range_env("GOLD_LOOKBACK_RANGE", (20, 100), int),
    "filters": parse_range_env("GOLD_FILTERS_RANGE", (32, 256), int),
    "kernel_size": parse_range_env("GOLD_KERNEL_SIZE_RANGE", (2, 7), int),
    "lstm_units": parse_range_env("GOLD_LSTM_UNITS_RANGE", (64, 512), int),
    "dense_units": parse_range_env("GOLD_DENSE_UNITS_RANGE", (32, 128), int),
    "dropout_rate": parse_range_env("GOLD_DROPOUT_RANGE", (0.1, 0.5), float),
    "learning_rate": parse_range_env("GOLD_LEARNING_RATE_RANGE", (1.0e-5, 5.0e-3), float),
    "l2_reg": parse_range_env("GOLD_L2_REG_RANGE", (1.0e-6, 1.0e-3), float),
    "batch_size_exp": parse_range_env("GOLD_BATCH_SIZE_EXP_RANGE", (5, 9), int),
}

# =========================
# 3) Reproducibility Helper
# =========================
def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# 4) Load Cleaned DataFrame
# =========================
df = pd.read_csv(CLEAN_DF_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"TARGET_COL '{TARGET_COL}' is not present.")

if DATE_COL is not None and DATE_COL in df.columns:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    df = df.set_index(DATE_COL)

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if TARGET_COL not in numeric_cols:
    raise ValueError("Target must be numeric.")

df = df[numeric_cols].copy()
level_df = df.copy()

ABS_P_T = "P_t_abs"
df[ABS_P_T] = df[TARGET_COL]

returns_df = df[numeric_cols].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
returns_df[ABS_P_T] = df[ABS_P_T].loc[returns_df.index]

# FEATURE ENGINEERING: Add explicit lags to the return series
for col in numeric_cols:
    returns_df[f"{col}_lag1"] = returns_df[col].shift(1)
    returns_df[f"{col}_lag2"] = returns_df[col].shift(2)

returns_df = returns_df.dropna()

MODEL_TARGET_COL = "target_t_plus_1"
returns_df[MODEL_TARGET_COL] = returns_df[TARGET_COL].shift(-HORIZON)

ABS_TARGET_COL = "P_t_plus_1_abs"
returns_df[ABS_TARGET_COL] = returns_df[ABS_P_T].shift(-HORIZON)

df = returns_df.dropna().copy()

# =========================
# 5) Chronological Outer Split
# =========================
split_summary = {
    "source_path": CLEAN_DF_PATH,
    "target_col": TARGET_COL,
    "horizon": HORIZON,
}

if TRAIN_END_DATE:
    split_cutoff = pd.Timestamp(TRAIN_END_DATE)
    if DATE_COL is None or not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("GOLD_TRAIN_END_DATE requires a valid datetime index based on DATE_COL.")

    if split_cutoff <= df.index.min():
        raise ValueError(
            f"GOLD_TRAIN_END_DATE must be later than the first model row date {df.index.min().date()}."
        )
    if split_cutoff > level_df.index.max():
        raise ValueError(
            f"GOLD_TRAIN_END_DATE must be on or before the last level date {level_df.index.max().date()}."
        )

    if EXPORT_DATE_SPLITS:
        level_train = level_df.loc[level_df.index <= split_cutoff].reset_index()
        level_test = level_df.loc[level_df.index > split_cutoff].reset_index()
        level_train.to_csv(TRAIN_SPLIT_EXPORT_PATH, index=False)
        level_test.to_csv(TEST_SPLIT_EXPORT_PATH, index=False)
        logging.info(
            f"Exported date split files using cutoff {split_cutoff.date()}: "
            f"{TRAIN_SPLIT_EXPORT_PATH} ({len(level_train)} rows), "
            f"{TEST_SPLIT_EXPORT_PATH} ({len(level_test)} rows)"
        )

    # Each row predicts t+1, so rows dated on the cutoff predict beyond the cutoff.
    # Keep model-train rows strictly before the requested cutoff to avoid leakage.
    train_mask = df.index < split_cutoff
    test_mask = df.index >= split_cutoff

    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"Date split at {split_cutoff.date()} produced an empty train or test partition "
            f"after return/target construction."
        )

    df_train_outer = df.loc[train_mask].copy().reset_index(drop=True)
    df_test = df.loc[test_mask].copy().reset_index(drop=True)
    split_summary.update(
        {
            "split_mode": "date_cutoff",
            "requested_train_end_date": str(split_cutoff.date()),
            "raw_train_rows": int((level_df.index <= split_cutoff).sum()),
            "raw_test_rows": int((level_df.index > split_cutoff).sum()),
            "model_train_rows": int(train_mask.sum()),
            "model_test_rows": int(test_mask.sum()),
            "model_train_last_input_date": str(df.index[train_mask].max().date()),
            "model_test_first_input_date": str(df.index[test_mask].min().date()),
            "split_note": (
                "Because horizon=1, rows dated on the requested cutoff predict the next date "
                "and are assigned to test."
            ),
        }
    )
    logging.info(
        f"Using date cutoff split at {split_cutoff.date()}: "
        f"model_train_rows={train_mask.sum()}, model_test_rows={test_mask.sum()}, "
        f"train_last_input_date={df.index[train_mask].max().date()}, "
        f"test_first_input_date={df.index[test_mask].min().date()}"
    )
else:
    outer_train_size = int(len(df) * OUTER_TRAIN_RATIO)
    df_train_outer = df.iloc[:outer_train_size].copy().reset_index(drop=True)
    df_test = df.iloc[outer_train_size:].copy().reset_index(drop=True)
    split_summary.update(
        {
            "split_mode": "ratio",
            "outer_train_ratio": OUTER_TRAIN_RATIO,
            "model_train_rows": int(len(df_train_outer)),
            "model_test_rows": int(len(df_test)),
        }
    )

feature_cols = [col for col in df_train_outer.columns if col not in [MODEL_TARGET_COL, ABS_TARGET_COL, ABS_P_T]]

X_train_outer_raw = df_train_outer[feature_cols].copy()
y_train_outer_raw = df_train_outer[[MODEL_TARGET_COL]].copy()
X_test_raw = df_test[feature_cols].copy()
y_test_raw = df_test[[MODEL_TARGET_COL]].copy()

# =========================
# 7) Sequence Builder
# =========================
def create_sequences(X_df, y_df, lookback, abs_y_df=None):
    X_values = X_df.values
    y_values = y_df.values.reshape(-1)
    if abs_y_df is not None:
        abs_y_values = abs_y_df.values.reshape(-1)
    X_seq, y_seq, abs_y_seq = [], [], []
    for i in range(lookback, len(X_df)):
        X_seq.append(X_values[i - lookback + 1 : i + 1, :])
        y_seq.append(y_values[i])
        if abs_y_df is not None:
            abs_y_seq.append(abs_y_values[i])
    if abs_y_df is not None:
        return np.array(X_seq), np.array(y_seq), np.array(abs_y_seq)
    return np.array(X_seq), np.array(y_seq)

def create_sequences_with_context(X_prev, y_prev, X_block, y_block, lookback, abs_y_prev=None, abs_y_block=None):
    X_full = pd.concat([X_prev.tail(lookback), X_block], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_prev.tail(lookback), y_block], axis=0).reset_index(drop=True)
    if abs_y_prev is not None and abs_y_block is not None:
        abs_full = pd.concat([abs_y_prev.tail(lookback), abs_y_block], axis=0).reset_index(drop=True)
        return create_sequences(X_full, y_full, lookback, abs_full)
    return create_sequences(X_full, y_full, lookback)

# =========================
# 9) PyTorch Model Builder
# =========================
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_shape, params):
        super(CNN_BiLSTM, self).__init__()
        in_channels = input_shape[1] # Features
        dr = params["dropout_rate"]
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=params["filters"],
            kernel_size=params["kernel_size"],
            padding=params["kernel_size"] - 1 # causal padding right-crop
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
        # x is (B, L, F) -> permute to (B, F, L) for Conv1d
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        if self.conv1.padding[0] > 0:
            x = x[:, :, :-self.conv1.padding[0]]
            
        x = self.relu(x)
        x = self.bn1(x)
        x = self.spatial_dropout(x)
        
        # permute to (B, L, Filters) for LSTM
        x = x.permute(0, 2, 1)
        
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        _, (h_n, c_n) = self.lstm2(x)
        # Bidirectional h_n shape: (2, B, hidden_size)
        h_f = h_n[0, :, :]
        h_b = h_n[1, :, :]
        x = torch.cat((h_f, h_b), dim=1)
        
        x = self.dropout2(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_dropout(x)
        x = self.out(x)
        return x

class BasisPointScaler:
    def __init__(self, scale=10000.0):
        self.scale = scale
    def fit_transform(self, x):
        return x * self.scale
    def transform(self, x):
        return x * self.scale
    def inverse_transform(self, x):
        return x / self.scale

def get_scalers():
    return StandardScaler(), BasisPointScaler(scale=1000.0)

class ProactiveHuberLoss(nn.Module):
    """
    Anti-laziness loss for BasisPointScaler-scaled targets (scale=1000).

    Gold/Silver daily returns are roughly ±0.3-1% → ±3-10 BPS in scaled space.
    The old dead_zone_threshold=0.5 BPS was essentially harmless (most predictions
    exceeded it just by expressing any signal at all).  We raise it to 3.0 BPS so
    the model is penalised any time it hides in an effectively-flat prediction.

    Components
    ----------
    1. Volatility-weighted Huber  — big-move errors cost more.
    2. Hinge directional loss     — penalises wrong-sign predictions with margin 2.
    3. Dead-zone penalty          — penalises |pred| < dead_zone_threshold (3.0 BPS).
    4. Return-spread penalty      — penalises var(pred) << var(target); forces the
                                    model to spread its predictions like the real data.
    """
    def __init__(self, hinge_weight=8.0, dead_zone_weight=20.0,
                 spread_weight=5.0, vol_weight_power=1.5):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='none')
        self.hinge_margin = 2.0          # 2 BPS directional margin
        self.hinge_weight = hinge_weight
        self.dead_zone_weight = dead_zone_weight
        self.dead_zone_threshold = 3.0   # 3 BPS — roughly 0.3% daily return
        self.spread_weight = spread_weight
        self.vol_weight_power = vol_weight_power

    def forward(self, pred, target):
        # 1. Volatility-weighted Huber
        weights = 1.0 + torch.pow(torch.abs(target), self.vol_weight_power)
        loss_huber = torch.mean(self.huber(pred, target) * weights)

        # 2. Hinge loss — directional enforcement
        sign_target = torch.sign(target)
        hinge_loss = torch.mean(torch.relu(self.hinge_margin - pred * sign_target))

        # 3. Dead-zone penalty — punish flat / near-zero predictions
        dead_zone_penalty = torch.mean(torch.relu(self.dead_zone_threshold - torch.abs(pred)))

        # 4. Return-spread penalty — model variance should track target variance
        pred_std   = torch.std(pred,   unbiased=False) + 1e-8
        target_std = torch.std(target, unbiased=False) + 1e-8
        spread_penalty = torch.relu(target_std / pred_std - 1.0)  # 0 when pred_std >= target_std

        return (
            loss_huber
            + self.hinge_weight    * hinge_loss
            + self.dead_zone_weight * dead_zone_penalty
            + self.spread_weight   * spread_penalty
        )

# Alias for compatibility
DirectionalHuberLoss = ProactiveHuberLoss

def train_pytorch_model(model, X_train, y_train, X_val, y_val, params, max_epochs, patience, min_delta):
    """
    Two-phase training to prevent dead-zone/hinge penalties from causing immediate
    gradient instability that triggers early stopping in just 2-3 epochs.

    Phase 1 (warmup_epochs):  plain HuberLoss — gives the model a smooth loss landscape
                               and reasonable initial weights.
    Phase 2 (remaining epochs): full ProactiveHuberLoss with early stopping on val loss.
    """
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device))
    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=False)

    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device))
    val_loader = DataLoader(val_data, batch_size=params["batch_size"], shuffle=False)

    model.to(device)
    warmup_criterion  = nn.HuberLoss()   # smooth warmup — no directional pressure
    main_criterion    = ProactiveHuberLoss(hinge_weight=8.0, dead_zone_weight=20.0, spread_weight=5.0)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params.get("l2_reg", 1e-5))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

    WARMUP_EPOCHS = min(15, max(5, max_epochs // 6))  # ~15% of budget, min 5

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {"loss": [], "val_loss": []}

    # ── Phase 1: Warmup with plain HuberLoss (no early stopping) ──────────────
    for epoch in range(WARMUP_EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = warmup_criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                val_loss += warmup_criterion(model(bx), by).item() * bx.size(0)
        val_loss /= len(val_loader.dataset)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    # Capture warmup best state as starting point for phase 2
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    best_val_loss = float('inf')
    patience_counter = 0

    # ── Phase 2: Full anti-laziness loss + early stopping ─────────────────────
    for epoch in range(WARMUP_EPOCHS, max_epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = main_criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                val_loss += main_criterion(model(bx), by).item() * bx.size(0)
        val_loss /= len(val_loader.dataset)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history



optuna_trial_logs = []

def optuna_objective(trial):
    set_global_seed(TUNING_SEED)

    params = {
        "lookback": trial.suggest_int("lookback", int(PBOUNDS["lookback"][0]), int(PBOUNDS["lookback"][1])),
        "filters": trial.suggest_int("filters", int(PBOUNDS["filters"][0]), int(PBOUNDS["filters"][1])),
        "kernel_size": trial.suggest_int("kernel_size", int(PBOUNDS["kernel_size"][0]), int(PBOUNDS["kernel_size"][1])),
        "lstm_units": trial.suggest_int("lstm_units", int(PBOUNDS["lstm_units"][0]), int(PBOUNDS["lstm_units"][1])),
        "dense_units": trial.suggest_int("dense_units", int(PBOUNDS["dense_units"][0]), int(PBOUNDS["dense_units"][1])),
        "dropout_rate": trial.suggest_float("dropout_rate", float(PBOUNDS["dropout_rate"][0]), float(PBOUNDS["dropout_rate"][1])),
        "learning_rate": trial.suggest_float("learning_rate", float(PBOUNDS["learning_rate"][0]), float(PBOUNDS["learning_rate"][1]), log=True),
        "l2_reg": trial.suggest_float("l2_reg", float(PBOUNDS["l2_reg"][0]), float(PBOUNDS["l2_reg"][1]), log=True),
        "batch_size_exp": trial.suggest_int("batch_size_exp", int(PBOUNDS["batch_size_exp"][0]), int(PBOUNDS["batch_size_exp"][1])),
    }

    params["batch_size"] = int(2 ** params["batch_size_exp"])
    params["lookback"] = max(2, params["lookback"])
    params["filters"] = max(8, params["filters"])
    params["kernel_size"] = max(1, params["kernel_size"])
    params["lstm_units"] = max(8, params["lstm_units"])
    params["dense_units"] = max(4, params["dense_units"])
    params["batch_size"] = max(8, params["batch_size"])

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_WALK_FORWARD)
    fold_rmses = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df_train_outer), start=1):
        fold_train_raw = df_train_outer.iloc[train_idx].copy().reset_index(drop=True)
        fold_val_raw = df_train_outer.iloc[val_idx].copy().reset_index(drop=True)

        X_fold_train_raw = fold_train_raw[feature_cols].copy()
        y_fold_train_raw = fold_train_raw[[MODEL_TARGET_COL]].copy()
        X_fold_val_raw = fold_val_raw[feature_cols].copy()
        y_fold_val_raw = fold_val_raw[[MODEL_TARGET_COL]].copy()

        scaler_X, scaler_y = get_scalers()

        X_fold_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_fold_train_raw), columns=feature_cols)
        X_fold_val_scaled = pd.DataFrame(scaler_X.transform(X_fold_val_raw), columns=feature_cols)

        y_fold_train_scaled = pd.DataFrame(scaler_y.fit_transform(y_fold_train_raw.values), columns=[MODEL_TARGET_COL])
        y_fold_val_scaled = pd.DataFrame(scaler_y.transform(y_fold_val_raw.values), columns=[MODEL_TARGET_COL])

        X_train_seq, y_train_seq = create_sequences(X_fold_train_scaled, y_fold_train_scaled, params["lookback"])
        X_val_seq, y_val_seq = create_sequences_with_context(X_fold_train_scaled, y_fold_train_scaled, X_fold_val_scaled, y_fold_val_scaled, params["lookback"])

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            return float("inf")

        model = CNN_BiLSTM(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), params=params).to(device)

        history = train_pytorch_model(
            model,
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            params=params,
            max_epochs=MAX_EPOCHS_TUNING,
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA
        )

        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
            y_val_pred_scaled = model(X_val_t).cpu().numpy().reshape(-1, 1)
            
        y_val_true_scaled = y_val_seq.reshape(-1, 1)

        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled).reshape(-1)
        y_val_true = scaler_y.inverse_transform(y_val_true_scaled).reshape(-1)

        dir_acc = np.mean(np.sign(y_val_pred) == np.sign(y_val_true))
        fold_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
        
        # Composite score: We want high directional accuracy and low RMSE.
        # We use (1 - dir_acc) as a loss term.
        fold_score = (1.0 - dir_acc) * 100.0 + fold_rmse
        fold_rmses.append(fold_score)

        trial.report(float(fold_score), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_rmse = float(np.mean(fold_rmses))
    optuna_trial_logs.append({
        "trial_number": trial.number,
        "params": params.copy(),
        "fold_rmses": fold_rmses.copy(),
        "mean_rmse": mean_rmse,
    })
    return mean_rmse


set_global_seed(TUNING_SEED)
sampler = TPESampler(seed=TUNING_SEED)
pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, INIT_POINTS), n_warmup_steps=2)

study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, study_name="cnn_bilstm_wf_optuna_torch")

study.optimize(
    optuna_objective,
    n_trials=INIT_POINTS + N_ITER,
    show_progress_bar=True
)

print("Best Optuna result:", study.best_value)

best_raw = study.best_trial.params
best_params = {
    "lookback": int(best_raw["lookback"]),
    "filters": int(best_raw["filters"]),
    "kernel_size": int(best_raw["kernel_size"]),
    "lstm_units": int(best_raw["lstm_units"]),
    "dense_units": int(best_raw["dense_units"]),
    "dropout_rate": float(best_raw["dropout_rate"]),
    "learning_rate": float(best_raw["learning_rate"]),
    "l2_reg": float(best_raw["l2_reg"]),
    "batch_size": int(2 ** int(best_raw["batch_size_exp"])),
}

# =========================
# 13) Final Split Setup
# =========================
n_train_outer_raw = len(df_train_outer)
n_val_raw_final = max(1, int(n_train_outer_raw * FINAL_VAL_RATIO_WITHIN_OUTER_TRAIN))
n_fit_raw_final = n_train_outer_raw - n_val_raw_final

df_fit_final_raw = df_train_outer.iloc[:n_fit_raw_final].copy().reset_index(drop=True)
df_val_final_raw = df_train_outer.iloc[n_fit_raw_final:].copy().reset_index(drop=True)

X_fit_final_raw = df_fit_final_raw[feature_cols].copy()
y_fit_final_raw = df_fit_final_raw[[MODEL_TARGET_COL]].copy()
X_val_final_raw = df_val_final_raw[feature_cols].copy()
y_val_final_raw = df_val_final_raw[[MODEL_TARGET_COL]].copy()

scaler_X_epoch, scaler_y_epoch = get_scalers()

X_fit_final_scaled = pd.DataFrame(scaler_X_epoch.fit_transform(X_fit_final_raw), columns=feature_cols)
y_fit_final_scaled = pd.DataFrame(scaler_y_epoch.fit_transform(y_fit_final_raw.values), columns=[MODEL_TARGET_COL])
X_val_final_scaled = pd.DataFrame(scaler_X_epoch.transform(X_val_final_raw), columns=feature_cols)
y_val_final_scaled = pd.DataFrame(scaler_y_epoch.transform(y_val_final_raw.values), columns=[MODEL_TARGET_COL])

LOOKBACK = best_params["lookback"]

X_fit_final, y_fit_final = create_sequences(X_fit_final_scaled, y_fit_final_scaled, LOOKBACK)
X_val_final, y_val_final = create_sequences_with_context(X_fit_final_scaled, y_fit_final_scaled, X_val_final_scaled, y_val_final_scaled, LOOKBACK)

scaler_X_final, scaler_y_final = get_scalers()
X_train_outer_scaled = pd.DataFrame(scaler_X_final.fit_transform(X_train_outer_raw), columns=feature_cols)
y_train_outer_scaled = pd.DataFrame(scaler_y_final.fit_transform(y_train_outer_raw.values), columns=[MODEL_TARGET_COL])
X_test_scaled = pd.DataFrame(scaler_X_final.transform(X_test_raw), columns=feature_cols)
y_test_scaled = pd.DataFrame(scaler_y_final.transform(y_test_raw.values), columns=[MODEL_TARGET_COL])

X_train_outer_seq, y_train_outer_seq = create_sequences(X_train_outer_scaled, y_train_outer_scaled, LOOKBACK)

abs_P_t_test_raw = df_test[[ABS_P_T]].copy()
abs_target_test_raw = df_test[[ABS_TARGET_COL]].copy()

_, _, P_t_test_seq = create_sequences_with_context(
    X_train_outer_scaled, y_train_outer_scaled, X_test_scaled, y_test_scaled, LOOKBACK,
    abs_y_prev=df_train_outer[[ABS_P_T]], abs_y_block=abs_P_t_test_raw
)

_, _, P_t_plus_1_test_seq = create_sequences_with_context(
    X_train_outer_scaled, y_train_outer_scaled, X_test_scaled, y_test_scaled, LOOKBACK,
    abs_y_prev=df_train_outer[[ABS_TARGET_COL]], abs_y_block=abs_target_test_raw
)

X_test_seq, y_test_seq = create_sequences_with_context(
    X_train_outer_scaled, y_train_outer_scaled, X_test_scaled, y_test_scaled, LOOKBACK
)

# =========================
# 16) Save Artifacts Helper
# =========================
def save_forecasting_artifacts(model, save_dir: str, x_scaler, feature_cols, target_col: str, lookback: int, y_scaler=None, model_name: str = "final_model.pth", extra_metadata=None):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, model_name))
    with open(os.path.join(save_dir, "x_scaler.pkl"), "wb") as f:
        pickle.dump(x_scaler, f)
    if y_scaler is not None:
        with open(os.path.join(save_dir, "y_scaler.pkl"), "wb") as f:
            pickle.dump(y_scaler, f)
    metadata = {
        "feature_cols": list(feature_cols),
        "target_col": target_col,
        "lookback": int(lookback),
        "model_file": model_name,
        "has_y_scaler": y_scaler is not None,
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    with open(os.path.join(save_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

split_summary_path = os.path.join(REPORTS_DIR, "data_split_summary.json")
with open(split_summary_path, "w", encoding="utf-8") as f:
    json.dump(split_summary, f, indent=4)
logging.info(f"Saved split summary to {split_summary_path}")

# =========================
# 17) Train Final Seeds
# =========================
final_results = []
seed_histories = {}
seed_predictions = {}

for seed in FINAL_SEEDS:
    logging.info(f"SEED {seed} EVALUATION")
    set_global_seed(seed)

    selection_model = CNN_BiLSTM(input_shape=(X_fit_final.shape[1], X_fit_final.shape[2]), params=best_params).to(device)
    
    history = train_pytorch_model(
        selection_model,
        X_fit_final, y_fit_final,
        X_val_final, y_val_final,
        params=best_params,
        max_epochs=MAX_EPOCHS_FINAL,
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA
    )
    
    seed_histories[seed] = history

    best_epoch_idx = int(np.argmin(history["val_loss"]))
    best_epoch = best_epoch_idx + 1
    best_loss = float(history["loss"][best_epoch_idx])
    best_val_loss = float(history["val_loss"][best_epoch_idx])

    set_global_seed(seed)
    refit_model = CNN_BiLSTM(input_shape=(X_train_outer_seq.shape[1], X_train_outer_seq.shape[2]), params=best_params).to(device)
    
    d_train = TensorDataset(torch.tensor(X_train_outer_seq, dtype=torch.float32).to(device), torch.tensor(y_train_outer_seq, dtype=torch.float32).unsqueeze(1).to(device))
    l_train = DataLoader(d_train, batch_size=best_params["batch_size"], shuffle=False)
    warmup_crit = nn.HuberLoss()
    main_crit   = ProactiveHuberLoss(hinge_weight=8.0, dead_zone_weight=20.0, spread_weight=5.0)
    optimizer   = optim.Adam(refit_model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params.get("l2_reg", 1e-5))

    warmup_ep = min(15, max(5, best_epoch // 6))  # match train_pytorch_model logic

    refit_model.train()
    for e in range(best_epoch):
        crit = warmup_crit if e < warmup_ep else main_crit
        for bx, by in l_train:
            optimizer.zero_grad()
            loss = crit(refit_model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refit_model.parameters(), 1.0)
            optimizer.step()



    refit_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        y_pred_test_scaled = refit_model(X_test_t).cpu().numpy().reshape(-1, 1)
        
    y_pred_test_return = scaler_y_final.inverse_transform(y_pred_test_scaled).reshape(-1)
    
    y_pred_test = P_t_test_seq.reshape(-1) * (1 + y_pred_test_return)
    y_true_test = P_t_plus_1_test_seq.reshape(-1)

    rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    r2 = r2_score(y_true_test, y_pred_test)

    final_results.append({
        "seed": seed, "best_epoch": best_epoch, "best_loss": best_loss,
        "best_val_loss": best_val_loss, "rmse_test": rmse, "r2_test": r2
    })
    
    seed_predictions[seed] = {"y_true_test": y_true_test, "y_pred_test": y_pred_test}

    if SAVE_ALL_SEED_MODELS:
        seed_dir = os.path.join(SAVE_ROOT_DIR, f"seed_{seed}")
        save_forecasting_artifacts(
            model=refit_model, save_dir=seed_dir, x_scaler=scaler_X_final,
            y_scaler=scaler_y_final, feature_cols=feature_cols,
            target_col=MODEL_TARGET_COL, lookback=best_params["lookback"],
            model_name=f"cnn_bilstm_seed{seed}.pth",
            extra_metadata={
                "seed": seed,
                "rmse_test": float(rmse),
                "r2_test": float(r2),
                "best_epoch": int(best_epoch),
                "data_split": split_summary,
            }
        )

final_results_df = pd.DataFrame(final_results)
print(final_results_df)

final_results_path = os.path.join(REPORTS_DIR, "gold_final_seed_results_optimized.csv")
optuna_report_path = os.path.join(REPORTS_DIR, "gold_optuna_walkforward_report_optimized.csv")
best_params_path = os.path.join(REPORTS_DIR, "gold_best_params_optimized.json")

optuna_report_df = pd.DataFrame(optuna_trial_logs)
optuna_report_df.to_csv(optuna_report_path, index=False)
final_results_df.to_csv(final_results_path, index=False)

with open(best_params_path, "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=4)

# =========================
# 18) Export Plots
# =========================
for seed in FINAL_SEEDS:
    history = seed_histories[seed]
    
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"Seed {seed} - Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"gold_seed_{seed}_loss.png"))
    plt.close()

    y_true_test = seed_predictions[seed]["y_true_test"]
    y_pred_test = seed_predictions[seed]["y_pred_test"]
    
    row = final_results_df[final_results_df["seed"] == seed].iloc[0]
    rmse = row["rmse_test"]
    r2 = row["r2_test"]

    plt.figure(figsize=(10, 4))
    plt.plot(y_true_test, label="Actual Gold Price", color='black')
    plt.plot(y_pred_test, label="Predicted Price", linestyle="--", color='goldenrod')
    plt.title(f"Seed {seed} - Walkforward Test | RMSE=${rmse:.2f}, R²={r2:.4f}")
    plt.xlabel("Test Sequence Index")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"gold_seed_{seed}_predictions.png"))
    plt.close()

print("Training Pipeline Complete!")
