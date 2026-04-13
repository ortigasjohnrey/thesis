import pandas as pd
import numpy as np

def calculate_indicators(df, price_col):
    """
    Computes Version 8 'Flash' Indicators.
    Optimized for high-speed, 1-day-ahead directional flips.
    """
    df = df.copy()
    
    # 1. Flash EMA (Micro-Trend)
    # Using 3/8 day crosses instead of 10/20 for instant reaction
    df['EMA_Fast'] = df[price_col].ewm(span=3, adjust=False).mean()
    df['EMA_Slow'] = df[price_col].ewm(span=8, adjust=False).mean()
    
    # 2. Flash RSI (High Sensitivity)
    # Using 7-day window to detect over-extensions within an active week
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_7'] = 100 - (100 / (1 + rs))
    
    # 3. Flash MACD (Fast Convergence)
    # Standard is 12,26,9. Flash is 6,13,5.
    exp1 = df[price_col].ewm(span=6, adjust=False).mean()
    exp2 = df[price_col].ewm(span=13, adjust=False).mean()
    df['MACD_Flash'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD_Flash'].ewm(span=5, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Flash'] - df['MACD_Signal']
    
    # 4. Flash Bollinger (Volatility Squeeze)
    # 5-day window for detecting sudden breakout pressure
    df['BB_Mid'] = df[price_col].rolling(window=5).mean()
    df['BB_Std'] = df[price_col].rolling(window=5).std()
    df['BB_Width'] = (4 * df['BB_Std']) / (df['BB_Mid'] + 1e-8)
    
    # 5. Flash Momentum (2-Day ROC)
    # Does the pivot start today?
    df['ROC_2'] = df[price_col].pct_change(periods=2).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Fill any NaNs from rolling windows
    df = df.ffill().bfill().fillna(0)
    
    return df

def calculate_indicators_v7(df, price_col):
    """
    Computes Version 7 'Standard' Indicators.
    Baseline for the high-precision Silver engine.
    """
    df = df.copy()
    df['EMA_10'] = df[price_col].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df[price_col].ewm(span=20, adjust=False).mean()
    
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    exp1 = df[price_col].ewm(span=12, adjust=False).mean()
    exp2 = df[price_col].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['BB_Mid'] = df[price_col].rolling(window=20).mean()
    df['BB_Std'] = df[price_col].rolling(window=20).std()
    df['BB_Width'] = (4 * df['BB_Std']) / (df['BB_Mid'] + 1e-8)
    df['ROC_5'] = df[price_col].pct_change(periods=5).fillna(0)
    
    df = df.ffill().bfill().fillna(0)
    return df
