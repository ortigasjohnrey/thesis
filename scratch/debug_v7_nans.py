import pandas as pd
import numpy as np
from technical_indicators import calculate_indicators

def debug_data():
    df_train = pd.read_csv("df_gold_dataset_gepu_extended_train.csv")
    target_col = "Gold_Futures"
    numeric_cols = [c for c in df_train.columns if c != "Date"]
    
    df_with_inds = calculate_indicators(df_train, target_col)
    ret_df = df_with_inds[numeric_cols].pct_change().replace([np.inf, -np.inf], 0)
    
    tech_cols = ['EMA_10', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Width', 'ROC_5']
    for col in tech_cols:
        ret_df[col] = df_with_inds[col]
        
    ret_df = ret_df.dropna()
    ret_df['target'] = ret_df[target_col].shift(-1)
    train_rets = ret_df.dropna()
    
    print("NaN counts in train_rets:")
    print(train_rets.isna().sum())
    
    print("\nShape:", train_rets.shape)

if __name__ == "__main__":
    debug_data()
