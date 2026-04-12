import pandas as pd
import numpy as np

def load_level_frame(csv_path):
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        df["Date_obj"] = df["Date"].dt.date
    return df

csv_path = "df_gold_dataset_gepu_extended_train.csv"
context_df = load_level_frame(csv_path)

_non_feature = {"Date", "Date_obj"}
base_numeric_cols = [
    c for c in context_df.columns
    if c not in _non_feature and pd.api.types.is_numeric_dtype(context_df[c])
]

print(f"base_numeric_cols: {base_numeric_cols}")

numeric_df = context_df[base_numeric_cols].copy()
returns_df = numeric_df.pct_change().replace([np.inf, -np.inf], np.nan)

for col in base_numeric_cols:
    returns_df[f"{col}_lag1"] = returns_df[col].shift(1)
    returns_df[f"{col}_lag2"] = returns_df[col].shift(2)

returns_df = returns_df.dropna()

print(f"returns_df columns: {returns_df.columns.tolist()}")
print(f"returns_df length: {len(returns_df)}")
